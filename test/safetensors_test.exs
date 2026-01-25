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

  describe "fp8 support" do
    @tag :tmp_dir
    test "write and read fp8 E4M3FN tensors", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "fp8_e4m3fn")

      # Create E4M3FN tensor
      original = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f8_e4m3fn)
      Safetensors.write!(path, %{"weight" => original})

      # Read back
      loaded = Safetensors.read!(path)

      # Verify type is preserved
      assert Nx.type(loaded["weight"]) == {:f8_e4m3fn, 8}
      assert Nx.shape(loaded["weight"]) == {2, 2}

      # Note: Value accuracy testing is done in Nx core tests
      # SafeTensors tests focus on type preservation and serialization
    end

    @tag :tmp_dir
    test "write and read fp8 E5M2 tensors", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "fp8_e5m2")

      # Create E5M2 tensor
      original = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f8)
      Safetensors.write!(path, %{"weight" => original})

      # Read back
      loaded = Safetensors.read!(path)

      # Verify type is preserved
      assert Nx.type(loaded["weight"]) == {:f, 8}
      assert Nx.shape(loaded["weight"]) == {2, 2}
    end

    @tag :tmp_dir
    test "round-trip preserves fp8 types", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "fp8_mixed")

      # Create tensors with different types
      tensors = %{
        "e4m3_weight" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f8_e4m3fn),
        "e5m2_weight" => Nx.tensor([[5.0, 6.0], [7.0, 8.0]], type: :f8),
        "f16_weight" => Nx.tensor([[9.0, 10.0], [11.0, 12.0]], type: :f16)
      }

      Safetensors.write!(path, tensors)
      loaded = Safetensors.read!(path)

      # Verify all types are preserved
      assert Nx.type(loaded["e4m3_weight"]) == {:f8_e4m3fn, 8}
      assert Nx.type(loaded["e5m2_weight"]) == {:f, 8}
      assert Nx.type(loaded["f16_weight"]) == {:f, 16}
    end

    test "dump and load fp8 tensors" do
      tensors = %{
        "e4m3" => Nx.tensor([1.0, 2.0, 3.0], type: :f8_e4m3fn),
        "e5m2" => Nx.tensor([4.0, 5.0, 6.0], type: :f8)
      }

      # Dump to binary
      binary = tensors |> Safetensors.dump() |> IO.iodata_to_binary()

      # Load back
      loaded = Safetensors.load!(binary)

      # Verify types
      assert Nx.type(loaded["e4m3"]) == {:f8_e4m3fn, 8}
      assert Nx.type(loaded["e5m2"]) == {:f, 8}

      # Verify shapes
      assert Nx.shape(loaded["e4m3"]) == {3}
      assert Nx.shape(loaded["e5m2"]) == {3}
    end

    @tag :tmp_dir
    test "lazy load fp8 tensors", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "fp8_lazy")

      # Write fp8 tensor
      original = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f8_e4m3fn)
      Safetensors.write!(path, %{"weight" => original})

      # Read lazily
      %{"weight" => file_tensor} = Safetensors.read!(path, lazy: true)

      # Verify it's a FileTensor
      assert %Safetensors.FileTensor{} = file_tensor
      assert file_tensor.type == {:f8_e4m3fn, 8}
      assert file_tensor.shape == {2, 2}

      # Convert to tensor and verify type is preserved
      tensor = Nx.to_tensor(file_tensor)
      assert Nx.type(tensor) == {:f8_e4m3fn, 8}
    end

    @tag :tmp_dir
    test "fp8 tensor byte size calculation", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "fp8_size")

      # Create a large fp8 tensor
      tensor = Nx.iota({100, 100}, type: :f8_e4m3fn)
      Safetensors.write!(path, %{"large" => tensor})

      # Verify file size is correct (8 bytes header size + header + 10000 bytes data)
      file_size = File.stat!(path).size
      header_start = 8

      # Read header to get exact size
      <<header_size::unsigned-64-integer-little, _rest::binary>> = File.read!(path)

      # Data should be exactly 10000 bytes (100 * 100 * 1 byte per fp8)
      expected_data_size = 10000
      actual_data_size = file_size - header_start - header_size

      assert actual_data_size == expected_data_size
    end

    test "fp8 dtype strings in header" do
      # Create fp8 tensors
      tensors = %{
        "e4m3" => Nx.tensor([1.0], type: :f8_e4m3fn),
        "e5m2" => Nx.tensor([2.0], type: :f8)
      }

      # Dump to binary
      binary = tensors |> Safetensors.dump() |> IO.iodata_to_binary()

      # Extract and parse header
      <<header_size::unsigned-64-integer-little, header_json::binary-size(header_size),
        _data::binary>> = binary

      header = Jason.decode!(header_json)

      # Verify dtype strings
      assert header["e4m3"]["dtype"] == "F8_E4M3"
      assert header["e5m2"]["dtype"] == "F8_E5M2"
    end
  end
end
