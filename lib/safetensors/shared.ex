defmodule Safetensors.Shared do
  @moduledoc false

  @max_read_size 2_147_483_647

  @doc """
  Reads `byte_size` bytes from `file` at `byte_offset`.

  Falls back to reading in chunks where a single read of that size
  fails, which OTP versions without large read support do on macOS
  for 2 GiB or more, and on Windows for 4 GiB or more.
  """
  @spec pread!(:file.io_device(), non_neg_integer(), non_neg_integer()) :: binary()
  def pread!(file, byte_offset, byte_size) do
    case :file.pread(file, byte_offset, byte_size) do
      {:ok, binary} when byte_size(binary) == byte_size ->
        binary

      {:ok, binary} ->
        read = byte_size(binary)
        rest = read_chunks!(file, byte_offset, byte_size, read, @max_read_size)
        IO.iodata_to_binary([binary | rest])

      {:error, :einval} when byte_size > @max_read_size ->
        pread_chunks!(file, byte_offset, byte_size, @max_read_size)

      other ->
        fail!(other, byte_offset, byte_size, 0)
    end
  end

  @doc """
  Reads `byte_size` bytes from `file` at `byte_offset`, in chunks of
  at most `chunk_size` bytes, joined into one binary.
  """
  @spec pread_chunks!(:file.io_device(), non_neg_integer(), non_neg_integer(), pos_integer()) ::
          binary()
  def pread_chunks!(file, byte_offset, byte_size, chunk_size) do
    file
    |> read_chunks!(byte_offset, byte_size, 0, chunk_size)
    |> IO.iodata_to_binary()
  end

  defp read_chunks!(_file, _byte_offset, byte_size, byte_size, _chunk_size), do: []

  defp read_chunks!(file, byte_offset, byte_size, read, chunk_size) do
    offset = byte_offset + read
    size = min(chunk_size, byte_size - read)

    case :file.pread(file, offset, size) do
      {:ok, chunk} ->
        read = read + byte_size(chunk)
        [chunk | read_chunks!(file, byte_offset, byte_size, read, chunk_size)]

      other ->
        fail!(other, byte_offset, byte_size, read)
    end
  end

  defp fail!(:eof, byte_offset, byte_size, read) do
    raise "expected #{byte_size} bytes at offset #{byte_offset}, got #{read}"
  end

  defp fail!({:error, reason}, byte_offset, byte_size, _read) do
    message = :file.format_error(reason)
    raise "could not read #{byte_size} bytes at offset #{byte_offset}: #{message}"
  end

  @doc """
  Builds Nx tensor from the given safetensors binary.
  """
  @spec build_tensor(binary(), tuple(), Nx.Type.t()) :: Nx.Tensor.t()
  def build_tensor(binary, shape, type) do
    {_, elem_size} = type

    binary
    |> new_byte_order(elem_size, :little)
    |> IO.iodata_to_binary()
    |> Nx.from_binary(type, backend: Nx.BinaryBackend)
    |> Nx.reshape(shape)
    # Moved in its final shape, as a backend may not hold the flat tensor,
    # such as EMLX with more than 2^31 - 1 elements in one dimension
    |> Nx.backend_transfer(Nx.default_backend())
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
