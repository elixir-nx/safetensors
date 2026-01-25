defmodule SafetensorsTest do
  use ExUnit.Case

  doctest Safetensors

  @tag :tmp_dir
  test "write", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "safetensor")

    data = %{test: Nx.tensor([[1, 2], [3, 4]], type: :s32)}
    Safetensors.write!(path, data)

    # source:
    # https://github.com/huggingface/safetensors/blob/1a65a3fdebcf280ef0ca32934901d3e2ad3b2c65/bindings/python/tests/test_simple.py#L22-L25
    # with the header padding removed and changed numbers
    assert File.read!(path) ==
             ~s(<\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00)
  end

  test "dump" do
    binary =
      %{test: Nx.tensor([[1, 2], [3, 4]], type: :s32)}
      |> Safetensors.dump()
      |> IO.iodata_to_binary()

    # source:
    # https://github.com/huggingface/safetensors/blob/1a65a3fdebcf280ef0ca32934901d3e2ad3b2c65/bindings/python/tests/test_simple.py#L22-L25
    # with the header padding removed and changed numbers
    assert binary ==
             ~s(<\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00)
  end

  @tag :tmp_dir
  test "read", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "safetensor")

    # source:
    # https://github.com/huggingface/safetensors/blob/1a65a3fdebcf280ef0ca32934901d3e2ad3b2c65/bindings/python/tests/test_simple.py#L35-L40
    File.write!(
      path,
      ~s(<\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)
    )

    assert Safetensors.read!(path) == %{"test" => Nx.tensor([[0, 0], [0, 0]], type: :s32)}
  end

  @tag :tmp_dir
  test "read lazy", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "safetensor")

    # source:
    # https://github.com/huggingface/safetensors/blob/1a65a3fdebcf280ef0ca32934901d3e2ad3b2c65/bindings/python/tests/test_simple.py#L35-L40
    File.write!(
      path,
      ~s(<\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)
    )

    assert %{"test" => %Safetensors.FileTensor{} = file_tensor} =
             Safetensors.read!(path, lazy: true)

    assert Nx.to_tensor(file_tensor) == Nx.tensor([[0, 0], [0, 0]], type: :s32)
  end

  test "load" do
    # source:
    # https://github.com/huggingface/safetensors/blob/1a65a3fdebcf280ef0ca32934901d3e2ad3b2c65/bindings/python/tests/test_simple.py#L35-L40
    serialized =
      ~s(<\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)

    assert Safetensors.load!(serialized) == %{"test" => Nx.tensor([[0, 0], [0, 0]], type: :s32)}
  end

  test "load with metadata" do
    # source:
    # https://github.com/huggingface/safetensors/blob/1a65a3fdebcf280ef0ca32934901d3e2ad3b2c65/bindings/python/tests/test_simple.py#L42-L50
    serialized =
      ~s(f\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{"framework":"pt"},"test1":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}       \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)

    assert Safetensors.load!(serialized) == %{"test1" => Nx.tensor([[0, 0], [0, 0]], type: :s32)}
  end

  @tag :tmp_dir
  test "write f8_e4m3fn", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "safetensor")

    data = %{test: Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f8_e4m3fn)}
    Safetensors.write!(path, data)

    assert File.read!(path) ==
             ~s(?\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[0,4]}}\x38\x40\x44\x48)
  end

  @tag :tmp_dir
  test "read f8_e4m3fn", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "safetensor")

    File.write!(
      path,
      ~s(?\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[0,4]}}\x38\x40\x44\x48)
    )

    assert Safetensors.read!(path) == %{
             "test" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f8_e4m3fn)
           }
  end

  @tag :tmp_dir
  test "write f8_e5m2", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "safetensor")

    data = %{test: Nx.tensor([[1.0, 2.0], [4.0, 8.0]], type: :f8)}
    Safetensors.write!(path, data)

    assert File.read!(path) ==
             ~s(?\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"F8_E5M2","shape":[2,2],"data_offsets":[0,4]}}\x3C\x40\x44\x48)
  end

  @tag :tmp_dir
  test "read f8_e5m2", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "safetensor")

    File.write!(
      path,
      ~s(?\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"F8_E5M2","shape":[2,2],"data_offsets":[0,4]}}\x3C\x40\x44\x48)
    )

    assert Safetensors.read!(path) == %{"test" => Nx.tensor([[1.0, 2.0], [4.0, 8.0]], type: :f8)}
  end
end
