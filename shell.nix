{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  libs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    glib
    glibc
    libGL
    libGLU
    xorg.libX11
  ];
in pkgs.mkShell {
  buildInputs = with pkgs; [
    raylib
    python311
    python311Packages.pip
    python311Packages.virtualenv
    python311Packages.matplotlib
    gcc
    gnumake
    cmake
    git
  ];

  shellHook = ''
    # Link host GPU driver (key for CUDA)
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH

    # Link other libs (excluding cudatoolkit and nvidia_x11 to avoid stub conflicts)
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libs}:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)

    # No CUDA_PATH needed for prebuilt wheels

    # Allow wheel builds if needed
    export SOURCE_DATE_EPOCH=$(date +%s)

    # Setup venv if it doesn't exist
    VENV="./.venv"
    if [ ! -d $VENV ]; then
      echo "Creating virtual environment..."
      virtualenv $VENV
    fi
    source $VENV/bin/activate

    echo "Environment ready. To install GPU-enabled Torch, run:"
    echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
  '';
}