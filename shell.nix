{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    poetry
    git
    pkg-config
    ffmpeg
    libjpeg
    libpng
    # if you want to use opencv-python:
    python312Packages.opencv4
    python312Packages.sentence-transformers
    python312Packages.scikit-learn
    python312Packages.pip
  ];

  # Environment variables to help pip & Poetry use binaries, not source builds
  env = {
    # Prevent attempts to use CUDA
    PYTORCH_CUDA_VERSION = "none";
    # This ensures you’re not linking to system CUDA accidentally
    FORCE_CUDA = "0";
    # Set this if you’re building anything image-related
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib";
  };

  # Optional: force Poetry to use Python 3.12
  shellHook = ''
    poetry env use ${pkgs.python312}/bin/python3.12
  '';
}

