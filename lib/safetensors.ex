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
        s32[3]
        [1, 2, 3]
      >
      iex> tensors["y"]
      #Nx.Tensor<
        f32[3]
        [1.0, 2.0, 3.0]
      >

  """

  alias Safetensors.Shared

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
  Writes a map of tensors to a file.

  Tensors are written into the file one by one, without the need to
  dump all of them into the memory at once.
  """
  @spec write!(path :: Path.t(), %{String.t() => Nx.Tensor.t()}) :: :ok
  def write!(path, tensors) when is_map(tensors) do
    File.open!(path, [:write, :raw], fn file ->
      tensors = Enum.sort(tensors)

      {header_entries, _offset} =
        Enum.map_reduce(tensors, 0, fn {tensor_name, tensor}, offset ->
          tensor_header_entry(tensor_name, tensor, offset)
        end)

      :ok = :file.write(file, header_binary(header_entries))

      for {_tensor_name, tensor} <- tensors do
        :ok = :file.write(file, tensor_to_iodata(tensor))
      end
    end)

    :ok
  end

  cond do
    Code.ensure_loaded?(JSON) ->
      @json_module JSON

    Code.ensure_loaded?(Jason) ->
      @json_module Jason

    true ->
      raise "You need to include jason package in your dependencies to make safetensors work with your current Elixir (#{System.version()}) or upgrade to Elixir 1.18+"
  end

  defp tensor_header_entry(tensor_name, tensor, offset) do
    end_offset = offset + tensor_byte_size(tensor)

    header_entry = {
      tensor_name,
      %{
        dtype: tensor |> Nx.type() |> type_to_dtype(),
        shape: tensor |> Nx.shape() |> Tuple.to_list(),
        data_offsets: [offset, end_offset]
      }
    }

    {header_entry, end_offset}
  end

  defp header_binary(header_entries) do
    header_json =
      header_entries
      |> Map.new()
      |> @json_module.encode!()

    [<<byte_size(header_json)::unsigned-64-integer-little>>, header_json]
  end

  defp tensor_byte_size(tensor) do
    {_, elem_size} = Nx.type(tensor)
    elem_byte_size = div(elem_size, 8)
    Nx.size(tensor) * elem_byte_size
  end

  defp tensor_to_iodata(tensor) do
    {_, elem_size} = Nx.type(tensor)

    tensor
    |> Nx.to_binary()
    |> Shared.new_byte_order(elem_size, :little)
  end

  @doc """
  Serializes the given map of tensors to iodata.

  `iodata` is a list of binaries that can be written to any IO device,
  such as a file or a socket. You can ensure the result is a binary by
  calling `IO.iodata_to_binary/1`.
  """
  @spec dump(%{String.t() => Nx.Tensor.t()}) :: iodata()
  def dump(tensors) when is_map(tensors) do
    tensors = Enum.sort(tensors)

    {header_entries, {buffer, _offset}} =
      Enum.map_reduce(tensors, {[], 0}, fn {tensor_name, tensor}, {buffer, offset} ->
        {header_entry, end_offset} = tensor_header_entry(tensor_name, tensor, offset)
        binary = tensor_to_iodata(tensor)
        {header_entry, {[buffer, binary], end_offset}}
      end)

    [header_binary(header_entries), buffer]
  end

  @doc """
  Reads a serialized map of tensors from a file.

  Tensors are loaded into Nx one by one, without the need to load the
  entire file from disk into memory.

  ## Options

    * `:lazy` - when `true`, instead of returning tensors, the function
      returns lazy containers. Such a container can be converted to a
      tensor using `Nx.to_tensor/1` and it is only at that point that
      it is read from the file. Defaults to `false`

  """
  @spec read!(path :: Path.t(), keyword()) :: %{String.t() => Nx.LazyContainer.t()}
  def read!(path, opts \\ []) do
    opts = Keyword.validate!(opts, lazy: false)

    File.open!(path, [:read, :raw], fn file ->
      {:ok, <<header_size::unsigned-64-integer-little>>} = :file.read(file, 8)
      {:ok, header_json} = :file.read(file, header_size)

      header = decode_header!(header_json)

      for {tensor_name, tensor_info} <- header, into: %{} do
        %{"data_offsets" => [offset_start, offset_end]} = tensor_info

        {shape, type} = shape_and_type(tensor_info)

        byte_offset = header_size + 8 + offset_start
        byte_size = offset_end - offset_start

        value =
          if opts[:lazy] do
            %Safetensors.FileTensor{
              shape: shape,
              type: type,
              path: path,
              byte_offset: byte_offset,
              byte_size: byte_size
            }
          else
            {:ok, binary} = :file.pread(file, byte_offset, byte_size)
            Shared.build_tensor(binary, shape, type)
          end

        {tensor_name, value}
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
      {shape, type} = shape_and_type(tensor_info)

      tensor =
        buffer
        |> binary_slice(offset_start, offset_end - offset_start)
        |> Shared.build_tensor(shape, type)

      {tensor_name, tensor}
    end
  end

  defp decode_header!(header_json) do
    {_metadata, header} =
      header_json
      |> @json_module.decode!()
      |> Map.pop(@header_metadata_key)

    header
  end

  defp shape_and_type(%{"dtype" => dtype, "shape" => shape}) do
    {List.to_tuple(shape), dtype_to_type(dtype)}
  end

  defp type_to_dtype(type) do
    @type_to_dtype[type] || raise "unrecognized type #{inspect(type)}"
  end

  defp dtype_to_type(dtype) do
    @dtype_to_type[dtype] || raise "unrecognized dtype #{inspect(dtype)}"
  end
end
