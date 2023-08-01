defmodule Safetensors do
  @moduledoc """
  Documentation for `Safetensors`.
  """

  # https://huggingface.co/docs/safetensors/index#format

  @header_metadata_key "__metadata__"

  @dtype_mapping %{
    "BF16" => :bf16,
    "F64" => :f64,
    "F32" => :f32,
    "F16" => :f16,
    "I64" => :s64,
    "I32" => :s32,
    "I16" => :s16,
    "I8" => :s8,
    "U64" => :u64,
    "U32" => :u32,
    "U16" => :u16,
    "U8" => :u8
    # "BOOL" => :u8
  }

  def load!(data) when is_binary(data) do
    <<
      header_size::unsigned-64-integer-little,
      header_json::binary-size(header_size),
      buffer::binary
    >> = data

    {_metadata, header} =
      header_json
      |> Jason.decode!()
      |> Map.pop(@header_metadata_key)

    header
    |> Enum.into(%{}, fn {tensor_name, tensor_info} ->
      %{
        "data_offsets" => [offset_start, offset_end],
        "dtype" => dtype,
        "shape" => shape
      } = tensor_info

      type = @dtype_mapping[dtype] || raise "unrecognized dtype #{dtype}"

      {
        tensor_name,
        buffer
        |> binary_part(offset_start, offset_end - offset_start)
        |> Nx.from_binary(type)
        |> Nx.reshape(List.to_tuple(shape))
      }
    end)
  end
end
