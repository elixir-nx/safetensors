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
      {:nx, "~> 0.5"},

      # Dev
      {:blend, "~> 0.3.0", only: :dev},
      {:ex_doc, "~> 0.30.3", only: :dev, runtime: false}
    ]
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
