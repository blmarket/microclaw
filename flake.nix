{
  description = "MicroClaw - Multi-channel agent runtime";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pkg-config
            openssl
            sqlite
            libsodium
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            udev
          ];

          LD_LIBRARY_PATH = "${pkgs.openssl}/lib:${pkgs.sqlite}/lib:${pkgs.libsodium}/lib";

          shellHook = ''
            export OPENSSL_DIR=${pkgs.openssl.dev}
            export OPENSSL_LIB_DIR=${pkgs.openssl.out}/lib
            export OPENSSL_INCLUDE_DIR=${pkgs.openssl.dev}/include
            export PKG_CONFIG_PATH=${pkgs.openssl.out}/lib/pkgconfig:$PKG_CONFIG_PATH
          '';
        };

        packages = {
          microclaw = import ./default.nix { inherit pkgs; };
          default = self.packages.${system}.microclaw;
        };
      }
    );
}
