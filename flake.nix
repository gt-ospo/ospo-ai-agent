{
  description =
    "GT OSPO AI Agent";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }@attrs:
    flake-utils.lib.eachSystem flake-utils.lib.defaultSystems (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in rec {
        devShells.default = pkgs.mkShell { buildInputs = with pkgs; [
          gnumake
          nixfmt
          poppler-utils
          (python3.withPackages (p: with p; [
            ollama
            nltk
            numpy
            chromadb

            jupytext
            jupyterlab
          ]))
        ];
      };
      });
}

