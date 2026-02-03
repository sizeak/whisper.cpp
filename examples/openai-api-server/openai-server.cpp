// OpenAI API Compatible Server for whisper.cpp
// Implements /v1/audio/transcriptions endpoint compatible with OpenAI's Audio API

#include "common.h"
#include "common-whisper.h"

#include "whisper.h"
#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace httplib;
using json = nlohmann::ordered_json;

namespace fs = std::filesystem;

namespace {

// Response format constants
constexpr std::string_view FORMAT_JSON         = "json";
constexpr std::string_view FORMAT_TEXT         = "text";
constexpr std::string_view FORMAT_VERBOSE_JSON = "verbose_json";

// Signal handling
std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}

// Server parameters
struct server_params {
    std::string hostname   = "127.0.0.1";
    std::string models_dir = "./models";
    std::string api_key;  // Empty means no authentication required

    int32_t port          = 8080;
    int32_t read_timeout  = 600;
    int32_t write_timeout = 600;
};

// Whisper parameters (defaults)
struct whisper_params {
    int32_t n_threads  = std::min(4, static_cast<int32_t>(std::thread::hardware_concurrency()));
    int32_t gpu_device = 0;

    bool use_gpu    = true;
    bool flash_attn = true;

    std::string language = "en";
};

// Custom deleter for whisper_context
struct whisper_context_deleter {
    void operator()(whisper_context* ctx) const {
        if (ctx) {
            whisper_free(ctx);
        }
    }
};

using whisper_context_ptr = std::unique_ptr<whisper_context, whisper_context_deleter>;

// Model manager with caching
class ModelManager {
    std::unordered_map<std::string, whisper_context_ptr> cache_;
    mutable std::shared_mutex mutex_;
    fs::path models_dir_;
    bool use_gpu_;
    int gpu_device_;
    bool flash_attn_;

public:
    ModelManager(fs::path models_dir, bool use_gpu, int gpu_device, bool flash_attn)
        : models_dir_(std::move(models_dir))
        , use_gpu_(use_gpu)
        , gpu_device_(gpu_device)
        , flash_attn_(flash_attn) {}

    whisper_context* get_or_load(std::string_view model_name) {
        std::string key{model_name};

        // Try read lock first (fast path for cached models)
        {
            std::shared_lock lock(mutex_);
            if (auto it = cache_.find(key); it != cache_.end()) {
                return it->second.get();
            }
        }

        // Upgrade to write lock for loading
        std::unique_lock lock(mutex_);

        // Double-check after acquiring write lock
        if (auto it = cache_.find(key); it != cache_.end()) {
            return it->second.get();
        }

        auto model_path = models_dir_ / ("ggml-" + key + ".bin");
        fprintf(stderr, "Loading model: %s\n", model_path.c_str());

        auto cparams       = whisper_context_default_params();
        cparams.use_gpu    = use_gpu_;
        cparams.gpu_device = gpu_device_;
        cparams.flash_attn = flash_attn_;

        whisper_context_ptr ctx{whisper_init_from_file_with_params(model_path.c_str(), cparams)};

        if (!ctx) {
            fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
            return nullptr;
        }

        fprintf(stderr, "Model loaded successfully: %s\n", model_path.c_str());

        auto* raw_ptr = ctx.get();
        cache_.emplace(std::move(key), std::move(ctx));
        return raw_ptr;
    }

    std::vector<std::string> list_available() const {
        std::vector<std::string> models;

        if (!fs::exists(models_dir_) || !fs::is_directory(models_dir_)) {
            return models;
        }

        for (const auto& entry : fs::directory_iterator(models_dir_)) {
            if (entry.is_regular_file()) {
                auto filename = entry.path().filename().string();
                if (filename.starts_with("ggml-") && filename.ends_with(".bin")) {
                    // Extract model name: "ggml-base.en.bin" -> "base.en"
                    models.push_back(filename.substr(5, filename.size() - 9));
                }
            }
        }
        return models;
    }
};

// OpenAI-style error response
json make_error_response(std::string_view message, std::string_view type, int code) {
    return json{
        {"error", json{
            {"message", message},
            {"type", type},
            {"code", code}
        }}
    };
}

void send_error(Response& res, int status, std::string_view message,
                std::string_view type = "invalid_request_error") {
    res.status = status;
    res.set_content(make_error_response(message, type, status).dump(), "application/json");
}

// Check Bearer token authentication
bool check_auth(const Request& req, const std::string& api_key, Response& res) {
    if (api_key.empty()) {
        return true;  // No auth required
    }

    auto auth_header = req.get_header_value("Authorization");
    if (auth_header.empty()) {
        send_error(res, 401, "Missing Authorization header", "authentication_error");
        return false;
    }

    constexpr std::string_view bearer_prefix = "Bearer ";
    if (!auth_header.starts_with(bearer_prefix)) {
        send_error(res, 401, "Invalid Authorization header format", "authentication_error");
        return false;
    }

    auto token = auth_header.substr(bearer_prefix.size());
    if (token != api_key) {
        send_error(res, 401, "Invalid API key", "authentication_error");
        return false;
    }

    return true;
}

// Parse optional parameter from multipart form
std::optional<std::string> get_optional_param(const Request& req, const std::string& name) {
    if (req.has_file(name)) {
        return req.get_file_value(name).content;
    }
    return std::nullopt;
}

// Parse float parameter with default
float get_float_param(const Request& req, const std::string& name, float default_val) {
    if (auto val = get_optional_param(req, name)) {
        try {
            return std::stof(*val);
        } catch (...) {
            return default_val;
        }
    }
    return default_val;
}

void print_usage(int /*argc*/, char** argv, const whisper_params& params, const server_params& sparams) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "OpenAI API compatible whisper.cpp server\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,   --help              show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N         [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "        --models-dir PATH   [%-7s] models directory path\n", sparams.models_dir.c_str());
    fprintf(stderr, "  -l,   --language LANG     [%-7s] default spoken language\n", params.language.c_str());
    fprintf(stderr, "        --host HOST         [%-7s] hostname for the server\n", sparams.hostname.c_str());
    fprintf(stderr, "        --port PORT         [%-7d] port number for the server\n", sparams.port);
    fprintf(stderr, "        --api-key KEY       [%-7s] require Bearer token authentication\n", sparams.api_key.empty() ? "none" : "***");
    fprintf(stderr, "  -ng,  --no-gpu            [%-7s] disable GPU acceleration\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -dev, --device N          [%-7d] GPU device ID\n", params.gpu_device);
    fprintf(stderr, "  -fa,  --flash-attn        [%-7s] enable flash attention\n", params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa, --no-flash-attn     [%-7s] disable flash attention\n", params.flash_attn ? "false" : "true");
    fprintf(stderr, "\n");
}

bool params_parse(int argc, char** argv, whisper_params& params, server_params& sparams) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params, sparams);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")     { params.n_threads  = std::stoi(argv[++i]); }
        else if (                  arg == "--models-dir")  { sparams.models_dir = argv[++i]; }
        else if (arg == "-l"    || arg == "--language")    { params.language   = argv[++i]; }
        else if (                  arg == "--host")        { sparams.hostname  = argv[++i]; }
        else if (                  arg == "--port")        { sparams.port      = std::stoi(argv[++i]); }
        else if (                  arg == "--api-key")     { sparams.api_key   = argv[++i]; }
        else if (arg == "-ng"   || arg == "--no-gpu")      { params.use_gpu    = false; }
        else if (arg == "-dev"  || arg == "--device")      { params.gpu_device = std::stoi(argv[++i]); }
        else if (arg == "-fa"   || arg == "--flash-attn")  { params.flash_attn = true; }
        else if (arg == "-nfa"  || arg == "--no-flash-attn") { params.flash_attn = false; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params, sparams);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    ggml_backend_load_all();

    whisper_params params;
    server_params sparams;

    if (!params_parse(argc, argv, params, sparams)) {
        return 1;
    }

    // Create model manager
    ModelManager model_manager(sparams.models_dir, params.use_gpu, params.gpu_device, params.flash_attn);

    // Mutex for serializing transcription requests (whisper context is not thread-safe)
    std::mutex transcription_mutex;

    auto svr = std::make_unique<Server>();

    // Set default headers
    svr->set_default_headers({
        {"Server", "whisper.cpp"},
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "content-type, authorization"}
    });

    // OPTIONS handler for CORS preflight
    svr->Options("/v1/audio/transcriptions", [](const Request&, Response& res) {
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    });

    svr->Options("/v1/models", [](const Request&, Response& res) {
        res.set_header("Access-Control-Allow-Methods", "GET, OPTIONS");
    });

    // GET /health - Health check endpoint
    svr->Get("/health", [](const Request&, Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // GET /v1/models - List available models
    svr->Get("/v1/models", [&](const Request& req, Response& res) {
        if (!check_auth(req, sparams.api_key, res)) {
            return;
        }

        auto models = model_manager.list_available();

        json data = json::array();
        for (const auto& model : models) {
            data.push_back(json{
                {"id", model},
                {"object", "model"},
                {"owned_by", "whisper.cpp"}
            });
        }

        json response = json{
            {"object", "list"},
            {"data", data}
        };

        res.set_content(response.dump(), "application/json");
    });

    // POST /v1/audio/transcriptions - Main transcription endpoint
    svr->Post("/v1/audio/transcriptions", [&](const Request& req, Response& res) {
        if (!check_auth(req, sparams.api_key, res)) {
            return;
        }

        // Check for required 'file' field
        if (!req.has_file("file")) {
            send_error(res, 400, "Missing required parameter: 'file'");
            return;
        }

        // Check for required 'model' field
        if (!req.has_file("model")) {
            send_error(res, 400, "Missing required parameter: 'model'");
            return;
        }

        auto audio_file = req.get_file_value("file");
        auto model_name = req.get_file_value("model").content;

        // Parse optional parameters
        auto language        = get_optional_param(req, "language").value_or(params.language);
        auto prompt          = get_optional_param(req, "prompt").value_or("");
        auto response_format = get_optional_param(req, "response_format").value_or(std::string(FORMAT_JSON));
        auto temperature     = get_float_param(req, "temperature", 0.0f);

        fprintf(stderr, "Transcription request: model=%s, language=%s, format=%s\n",
                model_name.c_str(), language.c_str(), response_format.c_str());

        // Load audio data
        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;

        if (!read_audio_data(audio_file.content, pcmf32, pcmf32s, false)) {
            send_error(res, 400, "Failed to read audio data. Ensure the file is a valid WAV format.");
            return;
        }

        fprintf(stderr, "Audio loaded: %zu samples (%.1f sec)\n",
                pcmf32.size(), static_cast<float>(pcmf32.size()) / WHISPER_SAMPLE_RATE);

        // Acquire lock for transcription
        std::lock_guard<std::mutex> lock(transcription_mutex);

        // Get or load the model
        whisper_context* ctx = model_manager.get_or_load(model_name);
        if (!ctx) {
            send_error(res, 400, "Failed to load model: " + model_name);
            return;
        }

        // Check language validity
        if (language != "auto" && whisper_lang_id(language.c_str()) == -1) {
            send_error(res, 400, "Invalid language: " + language);
            return;
        }

        // Set up whisper parameters
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        wparams.print_realtime   = false;
        wparams.print_progress   = false;
        wparams.print_timestamps = false;
        wparams.print_special    = false;
        wparams.translate        = false;
        wparams.language         = language.c_str();
        wparams.detect_language  = (language == "auto");
        wparams.n_threads        = params.n_threads;
        wparams.initial_prompt   = prompt.empty() ? nullptr : prompt.c_str();
        wparams.temperature      = temperature;
        wparams.no_context       = true;

        // Enable token timestamps for verbose_json
        if (response_format == FORMAT_VERBOSE_JSON) {
            wparams.token_timestamps = true;
            wparams.max_len          = 60;
            wparams.split_on_word    = true;
        }

        // Tell whisper to abort if the HTTP connection closed
        wparams.abort_callback = [](void* user_data) {
            auto req_ptr = static_cast<const Request*>(user_data);
            return req_ptr->is_connection_closed();
        };
        wparams.abort_callback_user_data = const_cast<void*>(static_cast<const void*>(&req));

        // Run inference
        if (whisper_full(ctx, wparams, pcmf32.data(), static_cast<int>(pcmf32.size())) != 0) {
            if (req.is_connection_closed()) {
                fprintf(stderr, "Client disconnected, aborted processing\n");
                res.status = 499;  // Client Closed Request
                res.set_content(R"({"error":"client disconnected"})", "application/json");
                return;
            }
            send_error(res, 500, "Failed to process audio", "server_error");
            return;
        }

        // Build response based on format
        const int n_segments = whisper_full_n_segments(ctx);

        if (response_format == FORMAT_TEXT) {
            // Plain text response
            std::string result;
            for (int i = 0; i < n_segments; ++i) {
                result += whisper_full_get_segment_text(ctx, i);
            }
            res.set_content(result, "text/plain; charset=utf-8");
        }
        else if (response_format == FORMAT_VERBOSE_JSON) {
            // Verbose JSON response (OpenAI-style)
            std::string full_text;
            for (int i = 0; i < n_segments; ++i) {
                full_text += whisper_full_get_segment_text(ctx, i);
            }

            json jres = json{
                {"task", "transcribe"},
                {"language", whisper_lang_str_full(whisper_full_lang_id(ctx))},
                {"duration", static_cast<float>(pcmf32.size()) / WHISPER_SAMPLE_RATE},
                {"text", full_text},
                {"segments", json::array()}
            };

            for (int i = 0; i < n_segments; ++i) {
                json segment = json{
                    {"id", i},
                    {"text", whisper_full_get_segment_text(ctx, i)},
                    {"start", whisper_full_get_segment_t0(ctx, i) * 0.01},
                    {"end", whisper_full_get_segment_t1(ctx, i) * 0.01}
                };

                // Add token-level details
                float total_logprob = 0.0f;
                int token_count = 0;
                const int n_tokens = whisper_full_n_tokens(ctx, i);

                segment["tokens"] = json::array();
                segment["words"] = json::array();

                for (int j = 0; j < n_tokens; ++j) {
                    whisper_token_data token = whisper_full_get_token_data(ctx, i, j);

                    if (token.id >= whisper_token_eot(ctx)) {
                        continue;
                    }

                    segment["tokens"].push_back(token.id);

                    json word = json{
                        {"word", whisper_full_get_token_text(ctx, i, j)},
                        {"start", token.t0 * 0.01},
                        {"end", token.t1 * 0.01},
                        {"probability", token.p}
                    };
                    segment["words"].push_back(word);

                    total_logprob += token.plog;
                    token_count++;
                }

                segment["temperature"] = temperature;
                segment["avg_logprob"] = token_count > 0 ? total_logprob / token_count : 0.0f;
                segment["no_speech_prob"] = whisper_full_get_segment_no_speech_prob(ctx, i);

                jres["segments"].push_back(segment);
            }

            res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        }
        else {
            // Default JSON response
            std::string full_text;
            for (int i = 0; i < n_segments; ++i) {
                full_text += whisper_full_get_segment_text(ctx, i);
            }

            json jres = json{{"text", full_text}};
            res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        }

        fprintf(stderr, "Transcription completed successfully\n");
    });

    // Error handlers
    svr->set_exception_handler([](const Request&, Response& res, std::exception_ptr ep) {
        std::string error_message = "Unknown error";
        try {
            std::rethrow_exception(std::move(ep));
        } catch (const std::exception& e) {
            error_message = e.what();
        } catch (...) {
            // Keep default message
        }
        res.status = 500;
        res.set_content(make_error_response(error_message, "server_error", 500).dump(), "application/json");
    });

    svr->set_error_handler([](const Request& req, Response& res) {
        if (res.status == 400) {
            res.set_content(make_error_response("Invalid request", "invalid_request_error", 400).dump(), "application/json");
        } else if (res.status == 404) {
            res.set_content(make_error_response("Not found: " + req.path, "invalid_request_error", 404).dump(), "application/json");
        } else if (res.status != 500) {
            res.set_content(make_error_response("Error", "server_error", res.status).dump(), "application/json");
        }
    });

    // Set timeouts
    svr->set_read_timeout(sparams.read_timeout);
    svr->set_write_timeout(sparams.write_timeout);

    if (!svr->bind_to_port(sparams.hostname, sparams.port)) {
        fprintf(stderr, "Error: couldn't bind to server socket: hostname=%s port=%d\n",
                sparams.hostname.c_str(), sparams.port);
        return 1;
    }

    printf("\nwhisper.cpp OpenAI-compatible server listening at http://%s:%d\n", sparams.hostname.c_str(), sparams.port);
    printf("Models directory: %s\n", sparams.models_dir.c_str());
    if (!sparams.api_key.empty()) {
        printf("Authentication: enabled (Bearer token)\n");
    } else {
        printf("Authentication: disabled\n");
    }
    printf("\nEndpoints:\n");
    printf("  POST /v1/audio/transcriptions  - Transcribe audio\n");
    printf("  GET  /v1/models                - List available models\n");
    printf("  GET  /health                   - Health check\n");
    printf("\n");

    shutdown_handler = [&](int signal) {
        printf("\nCaught signal %d, shutting down gracefully...\n", signal);
        if (svr) {
            svr->stop();
        }
    };

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, nullptr);
    sigaction(SIGTERM, &sigint_action, nullptr);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    std::thread server_thread([&] {
        if (!svr->listen_after_bind()) {
            fprintf(stderr, "Error: server listen failed\n");
        }
    });

    svr->wait_until_ready();
    server_thread.join();

    return 0;
}
