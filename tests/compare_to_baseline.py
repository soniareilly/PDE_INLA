from pathlib import Path

import numpy as np


BASELINE = Path("baseline")
RESULTS = Path("results")


def compare_array(
    name: str,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> None:
    expected = np.load(BASELINE / f"{name}.npy")
    actual = np.load(RESULTS / f"{name}.npy")

    difference = np.max(np.abs(actual - expected))

    print(f"{name}:")
    print(f"  shape: {actual.shape}")
    print(f"  maximum absolute difference: {difference:.3e}")

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    )
    
    np.testing.assert_array_equal(actual, expected)

compare_array("lmbda_weak", rtol=1e-5, atol=1e-7)
compare_array("theta_MAP", rtol=1e-5, atol=1e-7)
compare_array("quad_points", rtol=1e-5, atol=1e-7)
compare_array("pi_theta_quad", rtol=1e-5, atol=1e-7)
compare_array("pi_qoi", rtol=1e-5, atol=1e-7)
compare_array("pi_qoi_th_MAP", rtol=1e-5, atol=1e-7)