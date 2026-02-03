#!/bin/bash
#
# Benchmark script for comparing whisper.cpp models
# Usage: ./scripts/benchmark-models.sh <path-to-wav-file>
#
# This script will:
# 1. Build whisper-cli if not present
# 2. Download missing models
# 3. Run transcription benchmarks
# 4. Display comparison results
#

set -e

# Configuration
MODELS=("base.en" "medium.en" "large-v3-turbo")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$ROOT_DIR/models"
BUILD_DIR="$ROOT_DIR/build"
WHISPER_CLI="$BUILD_DIR/bin/whisper-cli"

# Colors for output (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    BOLD=''
    NC=''
fi

# Results storage
declare -A RESULTS_TOTAL
declare -A RESULTS_ENCODE
declare -A RESULTS_LOAD
declare -A RESULTS_RTF

# Backend info (detected from first benchmark run)
BACKEND_INFO=""
THREADS_INFO=""

usage() {
    echo "Usage: $0 <path-to-wav-file> [--csv output.csv]"
    echo ""
    echo "Benchmark whisper.cpp models and compare transcription performance."
    echo ""
    echo "Arguments:"
    echo "  <path-to-wav-file>   Path to a WAV audio file to transcribe"
    echo "  --csv <file>         Optional: save results to CSV file"
    echo ""
    echo "Models benchmarked: ${MODELS[*]}"
    exit 1
}

error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${BLUE}$1${NC}"
}

success() {
    echo -e "${GREEN}$1${NC}"
}

warning() {
    echo -e "${YELLOW}$1${NC}"
}

check_dependencies() {
    local missing=()

    if ! command -v cmake &> /dev/null; then
        missing+=("cmake")
    fi

    if ! command -v ffprobe &> /dev/null; then
        missing+=("ffprobe (ffmpeg)")
    fi

    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        missing+=("curl or wget")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing required dependencies: ${missing[*]}\nPlease install them and try again."
    fi
}

validate_input() {
    local wav_file="$1"

    if [ -z "$wav_file" ]; then
        usage
    fi

    if [ ! -f "$wav_file" ]; then
        error "File not found: $wav_file"
    fi

    if [ ! -r "$wav_file" ]; then
        error "File not readable: $wav_file"
    fi

    # Check if it's a valid audio file
    if ! ffprobe -v error "$wav_file" &> /dev/null; then
        error "Invalid or unsupported audio file: $wav_file"
    fi
}

get_audio_duration() {
    local wav_file="$1"
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$wav_file"
}

detect_gpu_backend() {
    local cmake_args=""
    local backend_name=""

    # Check for CUDA
    if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
        backend_name="CUDA"
        cmake_args="-DGGML_CUDA=ON"
    # Check for Vulkan
    elif command -v vulkaninfo &> /dev/null || [ -f "/usr/include/vulkan/vulkan.h" ]; then
        backend_name="Vulkan"
        cmake_args="-DGGML_VULKAN=ON"
    # Check for Metal (macOS)
    elif [ "$(uname)" = "Darwin" ]; then
        backend_name="Metal"
        cmake_args="-DGGML_METAL=ON"
    fi

    if [ -n "$backend_name" ]; then
        # Print to stderr so it doesn't get captured by command substitution
        echo -e "${BLUE}Detected $backend_name - enabling GPU acceleration${NC}" >&2
    fi

    echo "$cmake_args"
}

build_whisper() {
    if [ -x "$WHISPER_CLI" ]; then
        info "whisper-cli already built"
        return 0
    fi

    info "Building whisper-cli..."

    cd "$ROOT_DIR"

    local gpu_args
    gpu_args=$(detect_gpu_backend)

    # Configure
    cmake -B build $gpu_args -DCMAKE_BUILD_TYPE=Release

    # Build only whisper-cli target
    cmake --build build --target whisper-cli -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

    if [ ! -x "$WHISPER_CLI" ]; then
        error "Failed to build whisper-cli"
    fi

    success "whisper-cli built successfully"
}

download_model() {
    local model="$1"
    local model_file="$MODELS_DIR/ggml-$model.bin"

    if [ -f "$model_file" ]; then
        info "Model $model already exists"
        return 0
    fi

    info "Downloading model: $model"

    if [ -x "$MODELS_DIR/download-ggml-model.sh" ]; then
        "$MODELS_DIR/download-ggml-model.sh" "$model" "$MODELS_DIR"
    else
        error "Model download script not found: $MODELS_DIR/download-ggml-model.sh"
    fi

    if [ ! -f "$model_file" ]; then
        error "Failed to download model: $model"
    fi

    success "Model $model downloaded"
}

ensure_models() {
    info "Checking models..."
    for model in "${MODELS[@]}"; do
        download_model "$model"
    done
}

parse_backend_info() {
    local system_info="$1"
    local backends=""

    # Check for GPU backends
    if [[ $system_info == *"CUDA = 1"* ]]; then
        backends="CUDA"
    elif [[ $system_info == *"VULKAN = 1"* ]]; then
        backends="Vulkan"
    elif [[ $system_info == *"METAL = 1"* ]] || [[ $system_info == *"Metal : EMBED_LIBRARY = 1"* ]]; then
        backends="Metal"
    elif [[ $system_info == *"SYCL = 1"* ]]; then
        backends="SYCL"
    elif [[ $system_info == *"COREML = 1"* ]]; then
        backends="CoreML"
    else
        backends="CPU"
    fi

    # Check for SIMD features
    local simd=""
    if [[ $system_info == *"AVX2 = 1"* ]]; then
        simd="AVX2"
    elif [[ $system_info == *"AVX = 1"* ]]; then
        simd="AVX"
    elif [[ $system_info == *"NEON = 1"* ]]; then
        simd="NEON"
    fi

    if [ -n "$simd" ] && [ "$backends" = "CPU" ]; then
        backends="CPU ($simd)"
    elif [ -n "$simd" ]; then
        backends="$backends + $simd"
    fi

    echo "$backends"
}

run_benchmark() {
    local model="$1"
    local wav_file="$2"
    local model_file="$MODELS_DIR/ggml-$model.bin"

    info "Benchmarking: $model"

    # Run whisper-cli and capture output (stderr contains timing info)
    local output
    output=$("$WHISPER_CLI" -m "$model_file" -f "$wav_file" 2>&1) || true

    # Parse timing values
    local total_time encode_time load_time

    # Extract total time: "total time = %8.2f ms"
    total_time=$(echo "$output" | grep "total time" | sed -n 's/.*total time = *\([0-9.]*\) ms.*/\1/p' | tail -1)

    # Extract encode time: "encode time = %8.2f ms"
    encode_time=$(echo "$output" | grep "encode time" | sed -n 's/.*encode time = *\([0-9.]*\) ms.*/\1/p' | tail -1)

    # Extract load time: "load time = %8.2f ms"
    load_time=$(echo "$output" | grep "load time" | sed -n 's/.*load time = *\([0-9.]*\) ms.*/\1/p' | tail -1)

    # Extract backend info from system_info (only on first run)
    if [ -z "$BACKEND_INFO" ]; then
        local system_info
        system_info=$(echo "$output" | grep "system_info")
        if [ -n "$system_info" ]; then
            BACKEND_INFO=$(parse_backend_info "$system_info")
            # Extract thread count
            THREADS_INFO=$(echo "$system_info" | sed -n 's/.*n_threads = *\([0-9]*\).*/\1/p')
        fi
    fi

    # Store results
    RESULTS_TOTAL[$model]="${total_time:-N/A}"
    RESULTS_ENCODE[$model]="${encode_time:-N/A}"
    RESULTS_LOAD[$model]="${load_time:-N/A}"

    # Calculate real-time factor if we have valid data
    if [ -n "$total_time" ] && [ -n "$AUDIO_DURATION" ]; then
        # RTF = audio_duration_seconds / (total_time_ms / 1000)
        # Higher is better (faster than real-time)
        local rtf
        rtf=$(awk "BEGIN {printf \"%.1f\", $AUDIO_DURATION / ($total_time / 1000)}" 2>/dev/null || echo "N/A")
        RESULTS_RTF[$model]="$rtf"
    else
        RESULTS_RTF[$model]="N/A"
    fi
}

format_time() {
    local ms="$1"
    if [ "$ms" = "N/A" ] || [ -z "$ms" ]; then
        echo "N/A"
        return
    fi

    # Format with comma separators for readability
    printf "%'.0f ms" "$ms"
}

print_results() {
    local wav_file="$1"
    local csv_file="$2"

    echo ""
    echo -e "${BOLD}=== Whisper Model Benchmark ===${NC}"
    echo "Audio file: $wav_file"
    printf "Duration: %.1f seconds\n" "$AUDIO_DURATION"
    if [ -n "$BACKEND_INFO" ]; then
        echo "Backend: $BACKEND_INFO"
    fi
    if [ -n "$THREADS_INFO" ]; then
        echo "Threads: $THREADS_INFO"
    fi
    echo ""

    # Print table header
    printf "| %-17s | %12s | %12s | %11s | %16s |\n" "Model" "Load Time" "Encode Time" "Total Time" "Real-time Factor"
    printf "|%s|%s|%s|%s|%s|\n" "-------------------" "--------------" "--------------" "-------------" "------------------"

    # Print each model's results
    for model in "${MODELS[@]}"; do
        local total="${RESULTS_TOTAL[$model]}"
        local encode="${RESULTS_ENCODE[$model]}"
        local load="${RESULTS_LOAD[$model]}"
        local rtf="${RESULTS_RTF[$model]}"

        # Format times
        local total_fmt encode_fmt load_fmt rtf_fmt

        if [ "$total" != "N/A" ] && [ -n "$total" ]; then
            total_fmt=$(printf "%'.0f ms" "$total")
        else
            total_fmt="N/A"
        fi

        if [ "$encode" != "N/A" ] && [ -n "$encode" ]; then
            encode_fmt=$(printf "%'.0f ms" "$encode")
        else
            encode_fmt="N/A"
        fi

        if [ "$load" != "N/A" ] && [ -n "$load" ]; then
            load_fmt=$(printf "%'.0f ms" "$load")
        else
            load_fmt="N/A"
        fi

        if [ "$rtf" != "N/A" ] && [ -n "$rtf" ]; then
            rtf_fmt="${rtf}x"
        else
            rtf_fmt="N/A"
        fi

        printf "| %-17s | %12s | %12s | %11s | %16s |\n" "$model" "$load_fmt" "$encode_fmt" "$total_fmt" "$rtf_fmt"
    done

    echo ""

    # Save to CSV if requested
    if [ -n "$csv_file" ]; then
        {
            echo "# Backend: ${BACKEND_INFO:-unknown}, Threads: ${THREADS_INFO:-unknown}"
            echo "model,load_time_ms,encode_time_ms,total_time_ms,realtime_factor"
            for model in "${MODELS[@]}"; do
                echo "$model,${RESULTS_LOAD[$model]},${RESULTS_ENCODE[$model]},${RESULTS_TOTAL[$model]},${RESULTS_RTF[$model]}"
            done
        } > "$csv_file"
        success "Results saved to: $csv_file"
    fi
}

# Main script
main() {
    local wav_file=""
    local csv_file=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --csv)
                csv_file="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                if [ -z "$wav_file" ]; then
                    wav_file="$1"
                else
                    error "Unknown argument: $1"
                fi
                shift
                ;;
        esac
    done

    # Validate input
    if [ -z "$wav_file" ]; then
        usage
    fi

    # Convert to absolute path
    wav_file="$(cd "$(dirname "$wav_file")" && pwd)/$(basename "$wav_file")"

    echo -e "${BOLD}Whisper Model Benchmark${NC}"
    echo "========================"
    echo ""

    # Check dependencies
    info "Checking dependencies..."
    check_dependencies
    success "All dependencies found"

    # Validate input file
    info "Validating input file..."
    validate_input "$wav_file"

    # Get audio duration
    AUDIO_DURATION=$(get_audio_duration "$wav_file")
    printf "Audio duration: %.1f seconds\n" "$AUDIO_DURATION"
    echo ""

    # Build whisper-cli if needed
    build_whisper
    echo ""

    # Download models if needed
    ensure_models
    echo ""

    # Run benchmarks
    info "Running benchmarks..."
    echo ""
    for model in "${MODELS[@]}"; do
        run_benchmark "$model" "$wav_file"
        echo ""
    done

    # Print results
    print_results "$wav_file" "$csv_file"
}

main "$@"
