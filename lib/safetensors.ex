defmodule Safetensors do
  @moduledoc """
  [Safetensors](https://huggingface.co/docs/safetensors/index) implementation for `Nx`.

  ## Examples

      iex> x = Nx.tensor([1, 2, 3])
      iex> y = Nx.tensor([1.0, 2.0, 3.0])
      iex> tensors = %{"x" => x, "y" => y}
      iex> data = Safetensors.dump(tensors)
      iex> tensors = Safetensors.load!(data)
      iex> tensors["x"]
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >
      iex> tensors["y"]
      #Nx.Tensor<
        f32[3]
        [1.0, 2.0, 3.0]
      >

  """

  @header_metadata_key "__metadata__"

  @type_to_dtype %{
    {:bf, 16} => "BF16",
    {:f, 64} => "F64",
    {:f, 32} => "F32",
    {:f, 16} => "F16",
    {:s, 64} => "I64",
    {:s, 32} => "I32",
    {:s, 16} => "I16",
    {:s, 8} => "I8",
    {:u, 64} => "U64",
    {:u, 32} => "U32",
    {:u, 16} => "U16",
    {:u, 8} => "U8"
  }

  @dtype_to_type for {k, v} <- @type_to_dtype, into: %{}, do: {v, k}

  @doc """
  Serializes the given map of tensors to iodata.

  `iodata` is a list of binaries that can be written to any io device,
  such as a file or a socket. You can ensure the result is a binary by
  calling `IO.iodata_to_binary/1`.
  """
  @spec dump(%{String.t() => Nx.Tensor.t()}) :: iodata()
  def dump(tensors) when is_map(tensors) do
    {header_entries, {buffer, _offset}} =
      Enum.map_reduce(tensors, {[], 0}, fn {tensor_name, tensor}, {buffer, offset} ->
        {_, elem_size} = Nx.type(tensor)

        binary =
          tensor
          |> Nx.to_binary()
          |> new_byte_order(elem_size, :little)

        end_offset = offset + byte_size(binary)

        header_entry = {
          tensor_name,
          Jason.OrderedObject.new(
            dtype: tensor |> Nx.type() |> type_to_dtype(),
            shape: tensor |> Nx.shape() |> Tuple.to_list(),
            data_offsets: [offset, end_offset]
          )
        }

        {header_entry, {[buffer, binary], end_offset}}
      end)

    header_json =
      header_entries
      |> Jason.OrderedObject.new()
      |> Jason.encode!()

    [
      <<byte_size(header_json)::unsigned-64-integer-little>>,
      header_json,
      buffer
    ]
  end

  @doc """
  Reads a safe tensor from file.

  Tensors are loaded into Nx one by one,
  without loading the whole file into disk.
  """
  @spec read!(path :: Path.t()) :: %{String.t() => Nx.Tensor.t()}
  def read!(path) do
    File.open!(path, [:read, :raw], fn file ->
      {:ok, <<header_size::unsigned-64-integer-little>>} = :file.read(file, 8)
      {:ok, header_json} = :file.read(file, header_size)

      header = decode_header!(header_json)

      for {tensor_name, tensor_info} <- header, into: %{} do
        %{"data_offsets" => [offset_start, offset_end]} = tensor_info

        {:ok, binary} =
          :file.pread(file, header_size + 8 + offset_start, offset_end - offset_start)

        {tensor_name, build_tensor(binary, tensor_info)}
      end
    end)
  end

  @doc """
  Loads a serialized map of tensors.

  It is the opposite of `dump/1`.
  """
  @spec load!(iodata()) :: %{String.t() => Nx.Tensor.t()}
  def load!(data) when is_binary(data) or is_list(data) do
    data = IO.iodata_to_binary(data)

    <<
      header_size::unsigned-64-integer-little,
      header_json::binary-size(header_size),
      buffer::binary
    >> = data

    header = decode_header!(header_json)

    for {tensor_name, tensor_info} <- header, into: %{} do
      %{"data_offsets" => [offset_start, offset_end]} = tensor_info

      tensor =
        buffer
        |> binary_slice(offset_start, offset_end - offset_start)
        |> build_tensor(tensor_info)

      {tensor_name, tensor}
    end
  end

  defp decode_header!(header_json) do
    {_metadata, header} =
      header_json
      |> Jason.decode!()
      |> Map.pop(@header_metadata_key)

    header
  end

  defp build_tensor(binary, tensor_info) do
    %{"dtype" => dtype, "shape" => shape} = tensor_info
    {_, elem_size} = type = dtype_to_type(dtype)

    binary
    |> new_byte_order(elem_size, :little)
    |> Nx.from_binary(type)
    |> Nx.reshape(List.to_tuple(shape))
  end

  defp type_to_dtype(type) do
    @type_to_dtype[type] || raise "unrecognized type #{inspect(type)}"
  end

  defp dtype_to_type(dtype) do
    @dtype_to_type[dtype] || raise "unrecognized dtype #{inspect(dtype)}"
  end

  defp new_byte_order(binary, size, endianness) do
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
