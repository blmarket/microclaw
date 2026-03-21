{ pkgs }:
pkgs.rustPlatform.buildRustPackage {
  pname = "microclaw";
  version = "0.0.163";
  src = ./.;
  cargoLock.lockFile = ./Cargo.lock;

  buildFeatures = pkgs.lib.optionals pkgs.stdenv.isLinux [
    "journald"
    "sqlite-vec"
  ];

  nativeBuildInputs = with pkgs; [
    pkg-config
  ];

  buildInputs = with pkgs; [
    openssl.out
    sqlite
    libsodium
  ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
    udev
  ];

  OPENSSL_DIR = "${pkgs.openssl.dev}";
  OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
  OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
  LD_LIBRARY_PATH = "${pkgs.openssl.out}/lib:${pkgs.sqlite}/lib:${pkgs.libsodium}/lib";

  doCheck = false;
}
