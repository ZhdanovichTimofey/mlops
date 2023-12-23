from functools import lru_cache

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_onnx_model(img):
    triton_client = get_client()

    input_img = InferInput(
        name="INPUT", shape=img.shape, datatype=np_to_triton_dtype(img.dtype)
    )
    input_img.set_data_from_numpy(img, binary_data=True)

    infer_output = InferRequestedOutput("OUTPUT", binary_data=True)
    query_response = triton_client.infer(
        "simple-onnx-model", [input_img], outputs=[infer_output]
    )
    output = query_response.as_numpy("OUTPUT")[0]
    return output


def main():
    imgs = [
        np.zeros((1, 3, 32, 32), dtype=np.float32),
        np.ones((1, 3, 32, 32), dtype=np.float32),
    ]
    output = [
        np.array(
            [
                -0.62177646,
                -1.913877,
                1.070016,
                1.2920358,
                1.074248,
                0.36526975,
                0.13327089,
                -0.680465,
                -0.37174064,
                -1.8413801,
            ]
        ),
        np.array(
            [
                0.77934575,
                -2.9918816,
                0.9689988,
                1.3589456,
                -0.0732976,
                1.1159132,
                0.2385243,
                -1.3987451,
                0.1288021,
                -2.4693108,
            ]
        ),
    ]
    _output = call_triton_onnx_model(imgs[0])
    assert (output[0] - _output < 1e-7).all()
    _output = call_triton_onnx_model(imgs[1])
    assert (output[1] - _output < 1e-7).all()
    print("Success")


if __name__ == "__main__":
    main()
