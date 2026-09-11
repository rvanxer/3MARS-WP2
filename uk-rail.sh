#!/usr/bin/env bash
set -euo pipefail

R_VERSION="4.5.2"
R_SERIES="4.5"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R_SCRIPT="$SCRIPT_DIR/uk-rail.r"
LOCKFILE="$SCRIPT_DIR/renv.lock"

fail() {
    printf 'uk-rail: %s\n' "$*" >&2
    exit 1
}

output_exists() {
    local data_dir output_file
    data_dir="$(sed -n 's/^[[:space:]]*DATA_DIR:[[:space:]]*//p' \
        "$SCRIPT_DIR/env.yml" | head -n 1)"
    data_dir="${data_dir%$'\r'}"

    if [[ "$data_dir" =~ ^\"(.*)\"$ ]]; then
        data_dir="${BASH_REMATCH[1]}"
    elif [[ "$data_dir" =~ ^\'(.*)\'$ ]]; then
        data_dir="${BASH_REMATCH[1]}"
    fi
    [ -n "$data_dir" ] || return 1

    case "$data_dir" in
        "~") data_dir="$HOME" ;;
        "~/"*) data_dir="$HOME/${data_dir#\~/}" ;;
        /*) ;;
        *) data_dir="$SCRIPT_DIR/$data_dir" ;;
    esac
    output_file="$data_dir/gtfs/feeds/ext-UK_rail.zip"
    [ -f "$output_file" ] || return 1
    printf 'Output already exists; skipping: %s\n' "$output_file"
}

run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        fail "Administrator access is required, but sudo is unavailable."
    fi
}

confirm_install() {
    if [ "${UK_RAIL_YES:-0}" = "1" ]; then
        return
    fi
    if [ ! -t 0 ]; then
        fail "R $R_VERSION is missing. Re-run interactively or set UK_RAIL_YES=1 to permit installation."
    fi

    printf 'R %s is missing. Install it and its system requirements? [y/N] ' "$R_VERSION"
    read -r answer
    case "$answer" in
        y|Y|yes|YES) ;;
        *) fail "Installation cancelled." ;;
    esac
}

r_version() {
    "$1" --vanilla --quiet -e \
        'cat(paste(R.version$major, R.version$minor, sep = "."))' \
        2>/dev/null
}

r_home() {
    "$1" --vanilla --quiet -e 'cat(R.home())' 2>/dev/null
}

find_r() {
    local candidate version home os_name
    local -a candidates
    os_name="$(uname -s)"

    if [ "$os_name" = "Darwin" ]; then
        candidates=(
            /Library/Frameworks/R.framework/Resources/bin/Rscript
            "/usr/local/bin/R-$R_VERSION"
            "R-$R_VERSION"
            Rscript
        )
    else
        candidates=(
            "/usr/local/bin/R-$R_VERSION"
            "R-$R_VERSION"
            Rscript
        )
    fi

    for candidate in "${candidates[@]}"; do
        if [[ "$candidate" = */* ]]; then
            [ -x "$candidate" ] || continue
        elif command -v "$candidate" >/dev/null 2>&1; then
            candidate="$(command -v "$candidate")"
        else
            continue
        fi
        version="$(r_version "$candidate" || true)"
        home="$(r_home "$candidate" || true)"
        if [ "$version" != "$R_VERSION" ] || [[ "$home" = *conda* ]] || [[ "$home" = *'/envs/'* ]]; then
            continue
        fi
        if [ "$os_name" = "Darwin" ] && [[ "$home" != /Library/Frameworks/R.framework/* ]]; then
            continue
        fi
        if [ -n "$home" ]; then
            R_BIN="$candidate"
            return 0
        fi
    done
    return 1
}

install_homebrew() {
    if command -v brew >/dev/null 2>&1; then
        return
    fi

    command -v curl >/dev/null 2>&1 || fail "curl is required to install Homebrew."
    printf 'Installing Homebrew so that rig can install the requested R version.\n'
    NONINTERACTIVE=1 /bin/bash -c \
        "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    if [ -x /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    else
        fail "Homebrew installation completed, but brew was not found. Open a new terminal and run this script again."
    fi
}

install_rig_macos() {
    install_homebrew
    brew tap r-lib/rig
    brew trust --cask r-lib/rig/rig
    brew install --cask rig
}

install_rig_debian() {
    local bootstrap_tmp
    bootstrap_tmp="$(mktemp -d)"

    run_as_root apt-get update
    run_as_root apt-get install -y ca-certificates curl gnupg
    curl -fsSL https://rig.r-pkg.org/deb/rig.gpg -o "$bootstrap_tmp/rig.gpg"
    printf '%s\n' 'deb http://rig.r-pkg.org/deb rig main' > "$bootstrap_tmp/rig.list"
    run_as_root install -m 0644 "$bootstrap_tmp/rig.gpg" /etc/apt/trusted.gpg.d/rig.gpg
    run_as_root install -m 0644 "$bootstrap_tmp/rig.list" /etc/apt/sources.list.d/rig.list
    run_as_root apt-get update
    run_as_root apt-get install -y r-rig
    rm -rf "$bootstrap_tmp"
}

install_linux_requirements() {
    local packages missing package
    packages=(
        build-essential
        cmake
        gfortran
        git
        libcurl4-openssl-dev
        libgdal-dev
        libgeos-dev
        libicu-dev
        libproj-dev
        libssl-dev
        libudunits2-dev
        libxml2-dev
        zlib1g-dev
    )
    missing=()
    for package in "${packages[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'ok installed'; then
            missing+=("$package")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        run_as_root apt-get update
        run_as_root apt-get install -y "${missing[@]}"
    fi
}

validate_linux_distribution() {
    local linux_id
    [ -r /etc/os-release ] || fail "Cannot identify this Linux distribution."
    # shellcheck disable=SC1091
    . /etc/os-release
    linux_id="${ID:-}"
    case "$linux_id" in
        ubuntu|debian) ;;
        *) fail "Unsupported Linux distribution: ${PRETTY_NAME:-$linux_id}. Supported: Ubuntu and Debian." ;;
    esac
}

install_r() {
    local os_name
    os_name="$(uname -s)"
    confirm_install

    case "$os_name" in
        Darwin)
            if ! command -v rig >/dev/null 2>&1; then
                install_rig_macos
            fi
            ;;
        Linux)
            validate_linux_distribution
            if ! command -v rig >/dev/null 2>&1; then
                install_rig_debian
            fi
            ;;
        *)
            fail "Unsupported operating system: $os_name. Supported: macOS, Ubuntu and Debian."
            ;;
    esac

    if ! rig ls --plain | grep -Fxq "$R_VERSION"; then
        rig add "$R_VERSION"
    fi
    hash -r
}

configure_library() {
    local os_name arch platform
    os_name="$(uname -s)"
    arch="$(uname -m)"
    platform="$($R_BIN --vanilla --quiet -e 'cat(R.version$platform)' 2>/dev/null)"

    if [ "$os_name" = "Darwin" ]; then
        [ "$arch" = "aarch64" ] && arch="arm64"
        R_LIB="$HOME/Library/R/$arch/$R_SERIES-3mars"
    else
        R_LIB="$HOME/R/${platform}-library/$R_SERIES-3mars"
    fi
    mkdir -p "$R_LIB"
    INSTALL_LOG="$R_LIB/uk-rail-install.log"
}

restore_packages() {
    [ -f "$LOCKFILE" ] || fail "Package lockfile not found: $LOCKFILE"

    printf 'Restoring the locked R packages; the first run may take several minutes.\n'
    printf 'Installation details: %s\n' "$INSTALL_LOG"
    : > "$INSTALL_LOG"

    if ! R_LIBS_USER="$R_LIB" "$R_BIN" --vanilla --quiet -e \
        'if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv", repos = "https://cloud.r-project.org", quiet = TRUE)' \
        >> "$INSTALL_LOG" 2>&1; then
        tail -n 80 "$INSTALL_LOG" >&2
        fail "Could not install renv. See $INSTALL_LOG"
    fi

    if ! R_LIBS_USER="$R_LIB" UK_RAIL_PROJECT_DIR="$SCRIPT_DIR" \
        "$R_BIN" --vanilla --quiet -e \
        'renv::restore(project = Sys.getenv("UK_RAIL_PROJECT_DIR"), library = Sys.getenv("R_LIBS_USER"), prompt = FALSE)' \
        >> "$INSTALL_LOG" 2>&1; then
        tail -n 80 "$INSTALL_LOG" >&2
        fail "Could not restore the locked packages. See $INSTALL_LOG"
    fi
    printf 'R packages are ready.\n'
}

main() {
    local check_only
    check_only=0
    case "${1:-}" in
        "") ;;
        --check) check_only=1 ;;
        -h|--help)
            printf 'Usage: bash uk-rail.sh [--check]\n'
            printf '  --check  Install/restore dependencies and validate inputs without converting.\n'
            exit 0
            ;;
        *) fail "Unknown argument: $1" ;;
    esac

    [ -f "$R_SCRIPT" ] || fail "R script not found: $R_SCRIPT"
    [ -f "$SCRIPT_DIR/env.yml" ] || fail "Configuration file not found: $SCRIPT_DIR/env.yml"

    if [ "$check_only" -eq 0 ] && output_exists; then
        return
    fi

    if ! find_r; then
        install_r
        find_r || fail "R $R_VERSION was installed, but its executable was not found on PATH. Open a new terminal and retry."
    fi

    if [ "$(uname -s)" = "Linux" ]; then
        validate_linux_distribution
        install_linux_requirements
    fi

    configure_library
    printf 'Using R %s at %s\n' "$R_VERSION" "$R_BIN"
    printf 'Using R package library %s\n' "$R_LIB"
    restore_packages

    R_LIBS_USER="$R_LIB" UK_RAIL_PROJECT_DIR="$SCRIPT_DIR" UK_RAIL_SCRIPT="$R_SCRIPT" \
        UK_RAIL_CHECK_ONLY="$check_only" \
        "$R_BIN" --vanilla --quiet -e \
        'source(Sys.getenv("UK_RAIL_SCRIPT"), chdir = TRUE)'
}

main "$@"
