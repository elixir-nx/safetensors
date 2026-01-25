defmodule Safetensors.Shared do
  @moduledoc false

  @doc """
  Builds Nx tensor from the given safetensors binary.
  """
  @spec build_tensor(binary(), tuple(), Nx.Type.t()) :: Nx.Tensor.t()
  def build_tensor(binary, shape, type) do
    {_, elem_size} = type

    binary
    |> new_byte_order(elem_size, :little)
    |> IO.iodata_to_binary()
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
  end

  @doc """
  Changes endianness `binary` if `endianness` does not match system.
  """
  @spec new_byte_order(binary(), pos_integer(), :little | :big) :: iodata()
  def new_byte_order(binary, size, endianness) do
    if System.endianness() == endianness do
      binary
    else
      data =
        for <<data::size(size)-binary <- binary>> do
          data
          |> :binary.decode_unsigned()
          |> :binary.encode_unsigned(endianness)
        end

      IO.iodata_to_binary(data)
    end
  end
end
