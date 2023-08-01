defmodule Safetensors.MixProject do
  use Mix.Project

  @description "Safetensors load/deserialization in Elixir"

  def project do
    [
      app: :safetensors,
      version: "0.1.0",
      description: @description,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:jason, "~> 1.4"},
      {:nx, "~> 0.5.3"},
      {:ex_doc, "~> 0.30.3", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => "https://github.com/mimiquate/safetensors"
      }
    ]
  end
end
