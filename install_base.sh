#! /usr/bin/env bash

# Script to install the depenencies for LISA on an Ubuntu-like system.

# This is intended to be used for setting up containers and virtual machines to
# run LISA (e.g. for CI infrastructure or for Vagrant installation).
# This can also work for a fresh LISA install on a workstation.

# Read standard /etc/os-release file and extract the needed field lsb_release
# binary is not installed on all distro, but that file is found pretty much
# anywhere.
read_os_release() {
    field_name=$1
    (source /etc/os-release &> /dev/null && printf "%s" "${!field_name}")
}

# Test the value of a field in /etc/os-release
test_os_release(){
    field_name=$1
    value=$2

    if [[ "$(read_os_release "$field_name")" == "$value" ]]; then
        # same as "true" command
        return 0
    else
        # same as "false" commnad
        return 1
    fi
}

LISA_HOME=${LISA_HOME:-$(dirname "${BASH_SOURCE[0]}")}
cd "$LISA_HOME" || (echo "LISA_HOME ($LISA_HOME) does not exists" && exit 1)

# Must be kept in sync with shell/lisa_shell
ANDROID_HOME="$LISA_HOME/tools/android-sdk-linux/"
mkdir -p "$ANDROID_HOME"

# No need for the whole SDK for this one
install_android_platform_tools() {
    return
    echo "Installing Android platform tools ..."

    local url="https://dl.google.com/android/repository/platform-tools-latest-linux.zip"
    local archive="$ANDROID_HOME/android-platform-tools.zip"

    wget --no-verbose "$url" -O "$archive" &&
    echo "Extracting $archive ..." &&
    unzip -q -o "$archive" -d "$ANDROID_HOME"
}

cleanup_android_home() {
    echo "Cleaning up Android SDK: $ANDROID_HOME"
    rm -r "$ANDROID_HOME"
    mkdir -p "$ANDROID_HOME"
}

install_android_sdk_manager() {
    echo "Installing Android SDK manager ..."

    # URL taken from "Command line tools only": https://developer.android.com/studio
    local url="https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip"
    local archive="$ANDROID_HOME/android-sdk-manager.zip"
    rm "$archive" &>/dev/null

    echo "Downloading Android SDK manager from: $url"
    wget --no-verbose "$url" -O "$archive" &&
    echo "Extracting $archive ..." &&
    unzip -q -o "$archive" -d "$ANDROID_HOME"

    call_android_sdk sdkmanager --list
}

ANDROID_SDK_JAVA_VERSION=8
call_android_sdk() {
    tool="$ANDROID_HOME/tools/bin/$1"
    shift

    # Android SDK is picky on Java version, so we need to set JAVA_HOME
    # according to the distribution
    case $(read_os_release NAME) in
        Ubuntu)
            local JAVA_HOME=/usr/lib/jvm/java-$ANDROID_SDK_JAVA_VERSION-openjdk-amd64/
            ;;
        "Arch Linux")
            local JAVA_HOME=/usr/lib/jvm/java-$ANDROID_SDK_JAVA_VERSION-openjdk/jre
            ;;
    esac
    JAVA_HOME=$JAVA_HOME "$tool" "$@"
}

# Needs install_android_sdk_manager first
install_android_tools() {
    yes | call_android_sdk sdkmanager "build-tools;29.0.1"
    # We could use install_android_platform_tools here if the SDK starts being annoying
    yes | call_android_sdk sdkmanager "platform-tools"
}

# Install nodejs from snap packages instead of the main package manager
install_nodejs_snap() {
    # sanity check to make sure snap is up and running
    if ! which snap >/dev/null 2>&1; then
        echo 'Snap not installed on that system, not installing nodejs'
        return 1
    elif snap list >/dev/null 2>&1; then
        echo 'Snap not usable on that system, not installing nodejs'
        return 1
    else
        echo "Installing snap nodejs package ..."
        snap install node --classic --channel=8
    fi
}

install_apt() {
    echo "Installing apt packages ..."
    sudo apt-get update &&
    sudo apt-get install -y "${apt_packages[@]}"
}

install_pacman() {
    echo "Installing pacman packages ..."
    sudo pacman -Sy --needed --noconfirm "${pacman_packages[@]}"
}

set -eu

# APT-based distributions like Ubuntu or Debian
apt_packages=(
    build-essential
    git
    openssh-client
    wget
    unzip
    expect
    kernelshark
    python3
    python3-dev
    python3-pip
    # venv is not installed by default on Ubuntu, even though it is part of the
    # Python standard library
    python3-venv
    python3-setuptools
    python3-tk
    gobject-introspection
    libcairo2-dev
    libgirepository1.0-dev
    gir1.2-gtk-3.0
)

# pacman-based distributions like Archlinux or its derivatives
pacman_packages=(
    git
    openssh
    base-devel
    wget
    unzip
    expect
    python
    python-pip
    python-setuptools
    gobject-introspection

    # These packages can be installed from AUR
    # kernelshark
)

# Array of functions to call in order
install_functions=()


# Detection based on the package-manager, so that it works on derivatives of
# distributions we expect. Matching on distro name would prevent that.
if which apt-get &>/dev/null; then
    install_functions+=(install_apt)
    package_manager='apt-get'
    expected_distro="Ubuntu"

elif which pacman &>/dev/null; then
    install_functions+=(install_pacman)
    package_manager="pacman"
    expected_distro="Arch Linux"
else
    echo "The package manager of distribution $(read_os_release NAME) is not supported, will only install distro-agnostic code"
fi

if [[ ! -z "$package_manager" ]] && ! test_os_release NAME "$expected_distro"; then
    echo
    echo "INFO: the distribution seems based on $package_manager but is not $expected_distro, some package names might not be right"
    echo
fi

usage() {
    echo "Usage: $0 [--help] [--cleanup-android-sdk] [--install-android-tools] [--install-doc-extras] [--install-nodejs] [--install-all]"
    echo "Install distribution packages and other bits that don't fit in the Python venv managed by lisa-install."
    echo "Archlinux and Ubuntu are supported, although derivative distributions will probably work as well."
}

# Use conditional fall-through ;;& to all matching all branches with
# --install-all
for arg in "$@"; do
    # We need this flag since *) does not play well with fall-through ;;&
    handled=0
    case "$arg" in

    "--cleanup-android-sdk")
        install_functions+=(cleanup_android_home)
        handled=1
        ;;&

    # TODO: remove --install-android-sdk, since it is only temporarily there to
    # give some time to migrate CI scripts
    "--install-android-sdk" | "--install-android-tools" | "--install-all")
        install_functions+=(
            install_android_sdk_manager # Needed by install_android_build_tools
            install_android_tools
        )
        apt_packages+=(openjdk-$ANDROID_SDK_JAVA_VERSION-jre openjdk-$ANDROID_SDK_JAVA_VERSION-jdk)
        pacman_packages+=(jre$ANDROID_SDK_JAVA_VERSION-openjdk jdk$ANDROID_SDK_JAVA_VERSION-openjdk)
        handled=1;
        ;;&

    "--install-doc-extras" | "--install-all")
        apt_packages+=(plantuml graphviz)
        # plantuml can be installed from the AUR
        pacman_packages+=(graphviz)
        handled=1;
        ;;&

    "--install-nodejs" | "--install-all")
        # NodeJS v8+ is required, Ubuntu 16.04 LTS supports only an older version.
        # As a special case we can install it as a snap package
        if test_os_release NAME Ubuntu && test_os_release VERSION_ID 16.04; then
            install_functions+=(install_nodejs_snap)
        else
            apt_packages+=(nodejs npm)
            pacman_packages+=(nodejs npm)
        fi
        handled=1;
        ;;&

    "--help")
        usage
        exit 0
        ;;&

    *)
        if [[ $handled != 1 ]]; then
            echo "Unrecognised argument: $arg"
            usage
            exit 2
        fi
        ;;
    esac
done

# In order in which they will be executed if specified in command line
ordered_functions=(
    # Distro package managers before anything else, so all the basic
    # pre-requisites are there
    install_apt
    install_nodejs_snap
    install_pacman

    # cleanup must be done BEFORE installing
    cleanup_android_home
    install_android_sdk_manager # Needed by install_android_build_tools
    install_android_tools
    install_android_platform_tools
)

# Remove duplicates in the list
install_functions=($(echo "${install_functions[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# Call all the hooks in the order of available_functions
ret=0
for _func in "${ordered_functions[@]}"; do
    for func in "${install_functions[@]}"; do
        if [[ $func == $_func ]]; then
           # If one hook returns non-zero, we keep going but return an overall failure
           # code
            $func || ret=$?
            echo
        fi
    done
done

exit $ret

# vim: set tabstop=4 shiftwidth=4 textwidth=80 expandtab:
