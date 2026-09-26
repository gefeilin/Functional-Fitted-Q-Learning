"""Check the implemented curvature against analytic and adaptive integrals."""

import importlib.util
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Requires the training environment"
)
class CurvatureTests(unittest.TestCase):
    def test_known_cubic_and_linear_actions(self):
        import torch

        from functional_fitted_q.algorithms.theory_total_curvature import (
            total_curvature_matrix,
            torch_uncentered_total_curvature,
        )
        from functional_fitted_q.policies.bspline_policy import (
            BoundedCoefficientBSplinePolicy,
        )

        # With four clamped cubic B-splines, these are Bernstein coefficients
        # for u^3 and u. Their integrated squared curvatures are 12 and 0.
        policy = BoundedCoefficientBSplinePolicy(np.zeros((7, 4)))
        matrix, _ = total_curvature_matrix(policy, np.linspace(0, 1, 128))
        cubic = np.array([0.0, 0.0, 0.0, 1.0])
        linear = np.array([0.0, 1 / 3, 2 / 3, 1.0])
        self.assertAlmostEqual(cubic @ matrix @ cubic, 12.0, places=10)
        self.assertAlmostEqual(linear @ matrix @ linear, 0.0, places=10)

        # Identical actions across states still have nonzero curvature.
        # The leading axis indexes policies; only the state axis is averaged.
        coefficients = np.repeat(
            np.stack([cubic, 2 * cubic, linear])[:, None, :], 5, axis=1
        )
        energies = torch_uncentered_total_curvature(
            torch.as_tensor(coefficients), torch.as_tensor(matrix)
        )
        np.testing.assert_allclose(energies.numpy(), [12.0, 48.0, 0.0], atol=1e-10)

    def test_paper_spline_basis_matches_adaptive_integration(self):
        from functional_fitted_q.algorithms.theory_total_curvature import (
            total_curvature_matrix,
        )
        from functional_fitted_q.policies.bspline_policy import (
            BoundedCoefficientBSplinePolicy,
        )

        policy = BoundedCoefficientBSplinePolicy(np.zeros((7, 12)))
        matrix, _ = total_curvature_matrix(policy, np.linspace(0, 1, 128))
        for coefficients in np.random.default_rng(42).uniform(-2, 2, (4, 12)):
            # Integrate the scalar curve with QUADPACK independently of the
            # implementation's two-node matrix quadrature.
            second = BSpline(policy.knots, coefficients, 3).derivative(2)
            integral, _ = quad(
                lambda u: float(second(u)) ** 2,
                0,
                1,
                points=np.unique(policy.knots)[1:-1],
                epsabs=1e-8,
                epsrel=1e-12,
            )
            np.testing.assert_allclose(
                coefficients @ matrix @ coefficients, integral, rtol=1e-12
            )


if __name__ == "__main__":
    unittest.main()
