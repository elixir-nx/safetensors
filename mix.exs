defmodule Safetensors.MixProject do
  use Mix.Project

  @version "0.1.3"
  @description "Safetensors implementation for Nx"

  def project do
    [
      app: :safetensors,
      version: @version,
      description: @description,
      name: "Safetensors",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      package: package()
    ]
  end

  def application do
    []
  end

  defp deps do
    [
      {:jason, "~> 1.4"},
      nx_dep(),
      {:ex_doc, "~> 0.37", only: :dev, runtime: false}
    ]
  end

  defp nx_dep do
    # Allow local Nx development for fp8 testing
    if path = System.get_env("NX_PATH") do
      {:nx, path: path, override: true}
    else
      {:nx, "~> 0.5"}
    end
  end

  defp docs do
    [
      main: "Safetensors",
      source_url: "https://github.com/elixir-nx/safetensors",
      source_ref: "v#{@version}"
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => "https://github.com/elixir-nx/safetensors"
      }
    ]
  end
end
