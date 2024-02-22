defmodule Safetensors.FileTensor do
  @moduledoc false

  defstruct [:shape, :type, :path, :byte_offset, :byte_size]
end

defimpl Nx.LazyContainer, for: Safetensors.FileTensor do
  def traverse(lazy_tensor, acc, fun) do
    template = Nx.template(lazy_tensor.shape, lazy_tensor.type)

    load = fn ->
      File.open!(lazy_tensor.path, [:read, :raw], fn file ->
        {:ok, binary} = :file.pread(file, lazy_tensor.byte_offset, lazy_tensor.byte_size)
        Safetensors.Shared.build_tensor(binary, lazy_tensor.shape, lazy_tensor.type)
      end)
    end

    fun.(template, load, acc)
  end
end
