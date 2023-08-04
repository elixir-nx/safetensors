# Safetensors

[![ci](https://github.com/mimiquate/safetensors/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mimiquate/safetensors/actions?query=branch%3Amain)
[![Hex.pm](https://img.shields.io/hexpm/v/safetensors.svg)](https://hex.pm/packages/safetensors)
[![Docs](https://img.shields.io/badge/docs-gray.svg)](https://hexdocs.pm/safetensors)

[Safetensors](https://huggingface.co/docs/safetensors/index) implementation for [Nx](https://github.com/elixir-nx/nx).

Safetensors is a simple format for storing tensors in a language-agnostic manner. This packages allows loading and storing data in this format.

## Installation

You can add the `:safetensors` dependency to your `mix.exs`:

```elixir
def deps do
  [
    {:safetensors, "~> 0.1.0"}
  ]
end
```

## License

Copyright 2023 Mimiquate

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
