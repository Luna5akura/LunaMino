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

    VENV="./.venv"
    if [ ! -d $VENV ]; then
      echo "Creating virtual environment..."
      virtualenv $VENV
    fi
    source $VENV/bin/activate

    # 🤖 自动给 Triton 打 NixOS 专属补丁！
    TRITON_DRIVER="$VENV/lib/python3.11/site-packages/triton/backends/nvidia/driver.py"
    if [ -f "$TRITON_DRIVER" ]; then
      sed -i 's|\["/sbin/ldconfig", "-p"\]|["echo", "libcuda.so (libc6,x86-64) => /run/opengl-driver/lib/libcuda.so"]|g' "$TRITON_DRIVER"
    fi
  '';
}