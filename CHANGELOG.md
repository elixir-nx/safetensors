# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.3](https://github.com/elixir-nx/safetensors/tree/v0.1.3) (2024-02-23)

### Added

- Added `Safetensors.write!/2` for memory-efficient writing to file ([#8](https://github.com/elixir-nx/safetensors/pull/8))
- Added `:lazy` option to `Safetensors.read!/2` for reading tensors lazily ([#9](https://github.com/elixir-nx/safetensors/pull/9))

## [v0.1.2](https://github.com/elixir-nx/safetensors/tree/v0.1.2) (2023-09-27)

### Added

- Added `Safetensors.read!/1` for more efficient loading from file ([#4](https://github.com/elixir-nx/safetensors/pull/4))

## [v0.1.1](https://github.com/elixir-nx/safetensors/tree/v0.1.1) (2023-08-04)

### Added

- Added `Safetensors.dump/1` for serializing tensors

### Fixed

- Fixed `Safetensors.load!/1` for big-endian systems ([#2](https://github.com/elixir-nx/safetensors/pull/2))

## [v0.1.0](https://github.com/elixir-nx/safetensors/tree/v0.1.0) (2023-08-01)

Initial release.
