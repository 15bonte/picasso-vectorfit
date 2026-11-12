"""Tests for picasso.vectorfit_aberrations, the estimation of field-dependent
aberrations for the VectorFit fit.

The logic of the estimation (spot selection, initial values, the
symmetry bookkeeping of the global update, the likelihood derivatives) and
its wiring into ``localize``, the command line and the GUI are tested
without VectorFit. The tests that evaluate the model need VectorFit
bindings with ``Fitter.evaluate`` (vectorfit-py 0.4.0) and are skipped
without them.

:author: Thomas Bonte, 2026
"""

from __future__ import annotations

import argparse
import types

import numpy as np
import pandas as pd
import pytest

from picasso import localize, vectorfit
from picasso import vectorfit_aberrations as va


def _estimation_supported() -> bool:
    if not vectorfit.VECTORFIT_AVAILABLE:
        return False
    try:
        va.check_supported()
    except ImportError:
        return False
    return True


requires_evaluate = pytest.mark.skipif(
    not _estimation_supported(),
    reason="needs VectorFit bindings with Fitter.evaluate (vectorfit-py "
    ">= 0.4.0)",
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestParameters:
    def test_estimation_is_off_by_default(self):
        parameters = vectorfit.parameters_with_defaults(None)
        assert parameters["Estimate aberrations"] is False
        assert parameters["Aberration estimation spots"] == va.N_SPOTS

    def test_estimation_settings_pass(self):
        parameters = vectorfit.parameters_with_defaults(
            {
                "Estimate aberrations": True,
                "Aberration estimation spots": 500.0,
            }
        )
        assert parameters["Estimate aberrations"] is True
        assert parameters["Aberration estimation spots"] == 500
        assert isinstance(parameters["Aberration estimation spots"], int)

    @pytest.mark.parametrize("value", [1, "yes", None])
    def test_estimate_must_be_a_boolean(self, value):
        with pytest.raises(ValueError, match="Estimate aberrations"):
            vectorfit.parameters_with_defaults({"Estimate aberrations": value})

    @pytest.mark.parametrize("value", [10, 49, 100.5, True])
    def test_too_few_or_fractional_spots_are_rejected(self, value):
        with pytest.raises(ValueError, match="Aberration estimation spots"):
            vectorfit.parameters_with_defaults(
                {"Aberration estimation spots": value}
            )

    def test_one_label_per_legendre_coefficient(self):
        assert len(vectorfit.LEGENDRE_LABELS) == vectorfit._N_LEGENDRE
        assert len(set(vectorfit.LEGENDRE_LABELS)) == vectorfit._N_LEGENDRE
        # the primary aberrations are the constant coefficients
        for key, index in vectorfit._PRIMARY_ABERRATIONS.items():
            assert vectorfit.LEGENDRE_LABELS[index] == key


# ---------------------------------------------------------------------------
# NAT bookkeeping
# ---------------------------------------------------------------------------


class TestNatConstants:
    def test_symmetry_flips_the_even_aberrations(self):
        """Defocus, astigmatism and spherical aberration flip with z;
        coma does not (``params.symmetryvector``)."""
        expected = np.ones(va.N_GAMMAS)
        expected[[0, 1, 2, 3, 4, 5, 6, 7, 8, 12]] = -1
        np.testing.assert_array_equal(va.SYMMETRY, expected)

    def test_multiplex_factors(self):
        """Sum of the squared NAT weights (``params.multiplex_factors``)."""
        np.testing.assert_allclose(
            va.MULTIPLEX, [1, 1, 1, 2, 1, 1, 2, 2, 1.4, 1, 1, 2, 1]
        )

    def test_defocus_is_not_fitted(self):
        np.testing.assert_array_equal(va.FITTED_GAMMAS, np.arange(4, 13))

    def test_system_aberration_in_mlambda(self):
        gammas = np.zeros(va.N_GAMMAS)
        gammas[4] = 68.0
        assert va.system_aberration(gammas, 680.0) == pytest.approx(100.0)
        gammas[3] = 68.0  # multiplex factor 2
        assert va.system_aberration(gammas, 680.0) == pytest.approx(
            100.0 * np.sqrt(3)
        )


class TestReferenceGamma:
    def test_vertical_astigmatism_positive_by_default(self):
        """As VectorFit (``params.idx_pos = 5``)."""
        assert va.reference_gamma(np.zeros(va.N_GAMMAS)) == (4, 1.0)

    def test_coma_does_not_count(self):
        gammas = np.zeros(va.N_GAMMAS)
        gammas[9] = -50.0
        assert va.reference_gamma(gammas) == (4, 1.0)

    @pytest.mark.parametrize(
        "index, value", [(4, -80.0), (5, 60.0), (5, -60.0), (12, -30.0)]
    )
    def test_largest_even_aberration_keeps_its_sign(self, index, value):
        gammas = np.zeros(va.N_GAMMAS)
        gammas[index] = value
        gammas[4 if index != 4 else 5] = 0.5 * value  # smaller
        assert va.reference_gamma(gammas) == (index, np.sign(value))


class TestInitialGammas:
    def test_fixed_values_are_kept_and_the_rest_is_small(self):
        fixed = np.zeros(va.N_GAMMAS)
        fixed[[4, 9]] = [-80.0, 12.0]
        gammas = va.initial_gammas(
            fixed, 680.0, True, np.random.default_rng(0)
        )
        assert gammas[4] == -80.0
        assert gammas[9] == 12.0
        np.testing.assert_array_equal(gammas[:4], 0)
        others = np.setdiff1d(va.FITTED_GAMMAS, [4, 9])
        assert np.all(np.abs(gammas[others]) < 1.0)
        assert np.all(gammas[others] != 0)

    def test_random_part_has_the_initial_level(self):
        fixed = np.zeros(va.N_GAMMAS)
        fixed[5] = 20.0
        gammas = va.initial_gammas(
            fixed, 680.0, False, np.random.default_rng(1)
        )
        assert gammas[5] == 20.0
        random = gammas.copy()
        random[5] = 0
        assert va.system_aberration(random, 680.0) < va.INITIAL_LEVEL

    @pytest.mark.parametrize(
        "is_3d, level",
        [
            (True, va.INITIAL_ASTIGMATISM_3D),
            (False, va.INITIAL_ASTIGMATISM_2D),
        ],
    )
    def test_without_astigmatism_starts_from_some(self, is_3d, level):
        """Not from zero, where the estimation stalls."""
        fixed = np.zeros(va.N_GAMMAS)
        fixed[9] = 20.0  # coma alone
        gammas = va.initial_gammas(
            fixed, 680.0, is_3d, np.random.default_rng(0)
        )
        assert gammas[4] == pytest.approx(level * 680.0 * 1e-3)
        assert gammas[9] == 20.0

    def test_reproducible_with_a_seed(self):
        args = (np.zeros(va.N_GAMMAS), 680.0, False)
        np.testing.assert_array_equal(
            va.initial_gammas(*args, np.random.default_rng(3)),
            va.initial_gammas(*args, np.random.default_rng(3)),
        )


class TestSelectSpots:
    def test_all_spots_when_not_more(self):
        roi = np.ones((2, 30))
        np.testing.assert_array_equal(
            va.select_spots(roi, (64, 64), 30, np.random.default_rng(0)),
            np.arange(30),
        )

    def test_count_and_uniqueness(self):
        rng = np.random.default_rng(0)
        roi = rng.uniform(1, 512, (2, 5000))
        indices = va.select_spots(roi, (512, 512), 1000, rng)
        assert len(indices) == 1000
        assert len(np.unique(indices)) == 1000
        assert np.all(np.diff(indices) > 0)

    def test_spread_over_the_field_of_view(self):
        """A crowded corner does not dominate the selection: every cell of
        the 10 x 10 grid gets its share."""
        rng = np.random.default_rng(0)
        uniform = rng.uniform(1, 1000, (2, 2000))
        crowded = rng.uniform(1, 100, (2, 20000))  # one cell
        roi = np.concatenate([uniform, crowded], axis=1)
        indices = va.select_spots(roi, (1000, 1000), 1000, rng)

        def per_cell(points):
            cells = np.clip(np.ceil(points / 100), 1, 10) - 1
            return np.bincount(
                (cells[0] * 10 + cells[1]).astype(int), minlength=100
            )

        available = per_cell(roi)
        counts = per_cell(roi[:, indices])
        # 10 per cell (all of a cell's spots if fewer), the rest drawn at
        # random (mostly from the corner)
        assert np.all(counts >= np.minimum(available, 10))
        assert counts[0] < 0.9 * 1000

    def test_edge_spots_are_in_the_edge_cells(self):
        roi = np.array([[0.0, 1000.0, 500.0] * 40, [0.0, 1000.0, 1.0] * 40])
        indices = va.select_spots(
            roi, (1000, 1000), 100, np.random.default_rng(0)
        )
        assert len(indices) == 100


def test_normalized_fov_matches_vectorfit():
    """``(roixy*pixelsize + x)*2/(pixelsize*size) - 1``."""
    roi = np.array([[1.0, 64.0], [32.0, 1.0]])
    positions = np.array([[0.0, 0.0], [-50.0, 100.0]])
    fov = va.normalized_fov(roi, positions, 100.0, (64, 128))
    np.testing.assert_allclose(
        fov,
        [
            [2 * 100 / 6400 - 1, 2 * (6400 - 50) / 6400 - 1],
            [2 * 3200 / 12800 - 1, 2 * (100 + 100) / 12800 - 1],
        ],
    )


# ---------------------------------------------------------------------------
# Likelihood and global update
# ---------------------------------------------------------------------------


def test_likelihood_derivatives_match_finite_differences():
    """Gradient exact; the Hessian is the expected (Fisher) one, exact
    where the data equal the model."""
    rng = np.random.default_rng(0)
    basis = rng.uniform(0.5, 2.0, (3, 5, 5, 2))

    def model(p):
        return 10 + np.einsum("nijp,p->nij", basis, p) * 5, basis * 5

    p0 = np.array([1.0, 2.0])
    spots = rng.poisson(model(p0)[0]).astype(float)
    mu, dmu = model(p0)
    ll, grad, hess = va._log_likelihood_derivatives(spots, mu, dmu)
    for k in range(2):
        e = np.zeros(2)
        e[k] = 1e-5
        fd = (
            va._log_likelihood(spots, model(p0 + e)[0])
            - va._log_likelihood(spots, model(p0 - e)[0])
        ) / 2e-5
        np.testing.assert_allclose(grad[:, k], fd, rtol=1e-5)
    # spots == mu: Fisher Hessian equals the true one
    _, _, hess_exact = va._log_likelihood_derivatives(mu, mu, dmu)
    np.testing.assert_allclose(
        hess_exact, -np.einsum("nij,nijp,nijq->npq", 1 / mu, dmu, dmu)
    )
    assert np.all(np.linalg.eigvalsh(hess) < 0)


class _SymmetricModel:
    """A concave log-likelihood with the z-flip symmetry of the PSF: the
    merit of gammas g with the spots' z flipped equals that of
    ``SYMMETRY * g`` without; maximal at ``target`` (unflipped)."""

    def __init__(self, target):
        self.target = np.asarray(target, dtype=np.float64)
        self.calls = []

    def evaluate(self, gammas, flip_z):
        physical = gammas * va.SYMMETRY if flip_z else gammas
        self.calls.append((gammas.copy(), flip_z))
        diff = (physical - self.target)[va.FITTED_GAMMAS]
        merit = 1e4 - np.sum(diff**2)
        gradient = -2 * diff * (va.SYMMETRY[va.FITTED_GAMMAS] if flip_z else 1)
        hessian = -2 * np.eye(len(va.FITTED_GAMMAS))
        return merit, gradient, hessian


class TestGlobalUpdate:
    def test_converges_to_the_optimum(self):
        target = np.zeros(va.N_GAMMAS)
        target[va.FITTED_GAMMAS] = np.linspace(5, 40, 9)
        model = _SymmetricModel(target)
        gammas, damping, flip, iterations = va._global_update(
            model, np.full(va.N_GAMMAS, 1.0), 1e3, (4, 1.0), 1, 40, 1e-14
        )
        np.testing.assert_allclose(
            gammas[va.FITTED_GAMMAS], target[va.FITTED_GAMMAS], atol=1e-3
        )
        assert not flip
        assert damping < 1e3

    def test_crossing_zero_flips_gammas_and_z_together(self):
        """The optimum has the reference gamma negative, so it is reached
        on the mirrored branch: with the even gammas flipped and z
        flipped, which must go together."""
        target = np.zeros(va.N_GAMMAS)
        target[va.FITTED_GAMMAS] = np.linspace(5, 40, 9)
        target[4] = -30.0
        model = _SymmetricModel(target)
        start = np.zeros(va.N_GAMMAS)
        start[4] = 10.0
        gammas, _, flip, _ = va._global_update(
            model, start, 1.0, (4, 1.0), 1, 60, 1e-14
        )
        assert flip
        assert gammas[4] > 0  # the reference keeps its sign
        np.testing.assert_allclose(
            gammas[va.FITTED_GAMMAS],
            (va.SYMMETRY * target)[va.FITTED_GAMMAS],
            atol=1e-3,
        )
        # the returned pair is the optimum
        assert model.evaluate(gammas, flip)[0] == pytest.approx(1e4)

    def test_rejected_flipped_trial_leaves_z_unflipped(self):
        """A trial that flips but is rejected must not flip z (the bug of
        ``global_update.m``): the result stays consistent."""

        class Model(_SymmetricModel):
            def evaluate(self, gammas, flip_z):
                merit, gradient, hessian = super().evaluate(gammas, flip_z)
                if flip_z:
                    merit -= 1e6  # every flipped trial is rejected
                return merit, gradient, hessian

        target = np.zeros(va.N_GAMMAS)
        target[4] = -30.0
        target[9] = 15.0
        model = Model(target)
        start = np.zeros(va.N_GAMMAS)
        start[4] = 10.0
        gammas, _, flip, _ = va._global_update(
            model, start, 1e-3, (4, 1.0), 1, 30, 1e-14
        )
        assert any(call[1] for call in model.calls)  # flips were tried
        assert not flip
        assert gammas[4] > 0
        # the coma improved on the unflipped branch, and the returned pair
        # is the best one accepted
        assert 0 < gammas[9] <= 15.0
        accepted = [
            model.evaluate(g, f)[0] for g, f in model.calls[:-1] if not f
        ]
        assert model.evaluate(gammas, flip)[0] == pytest.approx(max(accepted))

    def test_step_flips_with_the_symmetry(self):
        gammas = np.zeros(va.N_GAMMAS)
        gammas[4] = 1.0
        gammas[9] = 3.0
        gradient = np.zeros(len(va.FITTED_GAMMAS))
        gradient[0] = -4.0  # pushes gamma 4 to -1
        hessian = -2 * np.eye(len(va.FITTED_GAMMAS))
        trial, flip = va._levenberg_marquardt_step(
            gammas, gradient, hessian, 0.0, (4, 1.0)
        )
        assert flip
        assert trial[4] == pytest.approx(1.0)
        assert trial[9] == pytest.approx(3.0)  # coma unchanged


def test_check_supported_needs_evaluate(monkeypatch):
    fake = types.SimpleNamespace(
        Fitter=type("Fitter", (), {}), __version__="0.3.2"
    )
    monkeypatch.setattr(vectorfit, "_import_vectorfit", lambda: fake)
    with pytest.raises(ImportError, match="0.4.0"):
        va.check_supported()


def test_too_few_spots_are_rejected(monkeypatch):
    monkeypatch.setattr(va, "check_supported", lambda: None)
    monkeypatch.setattr(
        vectorfit, "_import_vectorfit", lambda: types.SimpleNamespace()
    )
    with pytest.raises(ValueError, match="at least 50"):
        va.estimate_aberrations(
            np.zeros((10, 9, 9)),
            np.ones((2, 10)),
            parameters={},
            pixelsize=100.0,
            image_shape=(64, 64),
        )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def _identifications(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "x": rng.integers(5, 60, n).astype(float),
            "y": rng.integers(5, 60, n).astype(float),
            "net_gradient": np.ones(n),
        }
    )


class TestLocalizeWiring:
    def _run(self, monkeypatch, parameters, estimate=None):
        calls = {}
        gammas = np.arange(13.0)

        def fake_estimate(spots, roi, **kwargs):
            calls["estimate"] = (roi, kwargs)
            return estimate or {
                "gammas": gammas,
                "indices": np.arange(60),
                "iterations": 7,
                "converged": True,
            }

        def fake_fit(spots, identifications, **kwargs):
            calls["fit"] = kwargs
            return None  # as if aborted: nothing to convert

        monkeypatch.setattr(va, "estimate_aberrations", fake_estimate)
        monkeypatch.setattr(vectorfit, "fit_spots", fake_fit)
        fit_stats = {}
        identifications = _identifications(80)
        localize._fit2d_vectorfit(
            spots=np.ones((80, 9, 9)),
            identifications=identifications,
            em=False,
            parameters=vectorfit.parameters_with_defaults(parameters),
            pixelsize=110.0,
            image_shape=(64, 64),
            fit_stats=fit_stats,
        )
        return calls, fit_stats, identifications, gammas

    def test_estimate_is_used_by_the_fit_and_recorded(self, monkeypatch):
        calls, stats, identifications, gammas = self._run(
            monkeypatch,
            {"Estimate aberrations": True, "Aberration estimation spots": 60},
        )
        roi, kwargs = calls["estimate"]
        np.testing.assert_array_equal(
            roi, vectorfit.box_centers(identifications)
        )
        assert kwargs["n_spots"] == 60
        assert kwargs["image_shape"] == (64, 64)
        np.testing.assert_array_equal(calls["fit"]["gammas"], gammas)
        recorded = stats[vectorfit.ESTIMATED_ABERRATIONS_KEY]
        assert list(recorded) == list(vectorfit.LEGENDRE_LABELS)
        assert list(recorded.values()) == list(gammas)
        assert stats["VectorFit aberration estimation spots used"] == 60
        assert stats["VectorFit aberration estimation iterations"] == 7
        assert stats["VectorFit aberration estimation converged"] is True

    def test_no_estimate_by_default(self, monkeypatch):
        calls, stats, _, _ = self._run(monkeypatch, {})
        assert "estimate" not in calls
        assert calls["fit"]["gammas"] is None
        assert stats == {}

    def test_aborted_estimation_stops_before_the_fit(self, monkeypatch):
        monkeypatch.setattr(
            va, "estimate_aberrations", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            vectorfit,
            "fit_spots",
            lambda *args, **kwargs: pytest.fail("fitted after an abort"),
        )
        assert (
            localize._fit2d_vectorfit(
                spots=np.ones((80, 9, 9)),
                identifications=_identifications(80),
                em=False,
                parameters=vectorfit.parameters_with_defaults(
                    {"Estimate aberrations": True}
                ),
                pixelsize=110.0,
                image_shape=(64, 64),
            )
            is None
        )

    def test_unsupported_vectorfit_stops_before_cutting(
        self, monkeypatch, movie
    ):
        def unsupported():
            raise ImportError("needs vectorfit-py 0.4.0")

        monkeypatch.setattr(va, "check_supported", unsupported)
        monkeypatch.setattr(
            localize,
            "get_spots",
            lambda *args, **kwargs: pytest.fail("spots were cut"),
        )
        with pytest.raises(ImportError, match="0.4.0"):
            localize.fit(
                movie,
                camera_info={
                    "Baseline": 100,
                    "Sensitivity": 0.45,
                    "Gain": 1,
                    "Pixelsize": 130,
                },
                identifications=_identifications(3),
                box=9,
                fitting_method="vectorfit",
                vectorfit_parameters={"Estimate aberrations": True},
            )

    def test_fit_spots_rejects_malformed_gammas(self, monkeypatch):
        monkeypatch.setattr(
            vectorfit,
            "_fixed_aberrations",
            lambda *args: (np.zeros(13), np.zeros((6, 3))),
        )
        monkeypatch.setattr(vectorfit, "fit_params", lambda *a, **k: None)
        monkeypatch.setattr(
            vectorfit, "_import_vectorfit", lambda: types.SimpleNamespace()
        )
        with pytest.raises(ValueError, match="13 finite"):
            vectorfit.fit_spots(
                np.ones((2, 9, 9)),
                _identifications(2),
                parameters={},
                pixelsize=100.0,
                image_shape=(64, 64),
                gammas=np.zeros(5),
            )


class TestCommandLine:
    def _parse(self, *args):
        from picasso import __main__ as cli

        parser = argparse.ArgumentParser()
        cli._add_vectorfit_args(parser)
        return cli._vectorfit_parameters_from_args(parser.parse_args(args))

    def test_off_by_default(self):
        assert "Estimate aberrations" not in self._parse()

    def test_flags(self):
        parameters = self._parse(
            "--vf-estimate-aberrations", "--vf-estimation-spots", "400"
        )
        assert parameters["Estimate aberrations"] is True
        assert parameters["Aberration estimation spots"] == 400
        vectorfit.parameters_with_defaults(parameters)


@pytest.mark.gui
class TestGui:
    @pytest.fixture(autouse=True)
    def _installed(self, monkeypatch):
        monkeypatch.setattr(vectorfit, "VECTORFIT_AVAILABLE", True)

    def _dialog(self):
        from PyQt6 import QtWidgets
        from picasso.gui import localize as localize_gui

        class _StubWindow(QtWidgets.QMainWindow):
            movie = None

            def draw_frame(self):
                pass

        return localize_gui.ParametersDialog(_StubWindow())

    def test_round_trip_and_enabled_state(self, qt_offscreen):
        dialog = self._dialog()
        try:
            assert not dialog.vectorfit_estimate_checkbox.isChecked()
            assert not dialog.vectorfit_estimate_spots.isEnabled()
            fixed_tip = dialog.vectorfit_spinboxes[
                "Vertical astigmatism"
            ].toolTip()
            dialog.set_vectorfit_parameters(
                {
                    "Estimate aberrations": True,
                    "Aberration estimation spots": 2500,
                }
            )
            assert dialog.vectorfit_estimate_spots.isEnabled()
            parameters = dialog.vectorfit_parameters()
            assert parameters["Estimate aberrations"] is True
            assert parameters["Aberration estimation spots"] == 2500
            assert set(parameters) == set(vectorfit.DEFAULT_PARAMETERS)
            assert (
                dialog.vectorfit_spinboxes["Vertical astigmatism"].toolTip()
                != fixed_tip
            )
            assert (
                "Initial"
                in dialog.vectorfit_labels["Spherical aberration"].toolTip()
            )
            dialog.vectorfit_estimate_checkbox.setChecked(False)
            assert not dialog.vectorfit_estimate_spots.isEnabled()
        finally:
            dialog.deleteLater()


# ---------------------------------------------------------------------------
# With VectorFit
# ---------------------------------------------------------------------------


def _simulate(vectorfit_py, parameters, true, n, box, pixelsize, shape, rng):
    """Spots with the aberrations of the gammas ``true`` at their
    positions, by VectorFit's own model; returns spots, roi and theta."""
    parameters = vectorfit.parameters_with_defaults(parameters)
    common = dict(box=box, pixelsize=pixelsize, image_shape=shape)
    params = vectorfit.fit_params(
        parameters, model="xyz_aberrations", n_threads=4, **common
    )
    params_const = vectorfit.fit_params(
        parameters, model="xyz_constaberrations", **common
    )
    _, orders = vectorfit._fixed_aberrations(
        vectorfit_py, params_const, parameters
    )
    ident = pd.DataFrame(
        {
            "x": rng.integers(box, shape[1] - box, n).astype(float),
            "y": rng.integers(box, shape[0] - box, n).astype(float),
        }
    )
    roi = vectorfit.box_centers(ident)
    theta = np.zeros((n, 5 + len(orders)))
    theta[:, :2] = rng.uniform(-40, 40, (n, 2))
    theta[:, 2] = rng.uniform(-400, 400, n)
    theta[:, 3] = rng.uniform(3000, 5000, n)
    theta[:, 4] = 15.0
    fov = va.normalized_fov(roi, theta[:, :2], pixelsize, shape)
    theta[:, 5:] = va.zernike_jacobian(vectorfit_py, fov, orders) @ true
    fitter = vectorfit_py.Fitter(
        vectorfit_py.doubleArray4D(np.zeros((n, box, box, 1))),
        vectorfit_py.doubleArray2D(roi),
        params,
        "cpu",
    )
    fitter.set_theta(vectorfit_py.doubleArray2D(theta))
    fitter.evaluate()
    mu = np.array(fitter.mu())[..., 0]
    return rng.poisson(mu).astype(np.float32), roi, theta, orders, params


@requires_evaluate
class TestWithVectorFit:
    def test_zernike_derivatives_match_finite_differences(self):
        import vectorfit_py

        rng = np.random.default_rng(0)
        true = np.zeros(13)
        true[[4, 9]] = [60.0, 15.0]
        _, roi, theta, _, params = _simulate(
            vectorfit_py, {}, true, 3, 9, 100.0, (64, 64), rng
        )
        fitter = vectorfit_py.Fitter(
            vectorfit_py.doubleArray4D(np.zeros((3, 9, 9, 1))),
            vectorfit_py.doubleArray2D(roi),
            params,
            "cpu",
        )

        def mu_at(t):
            fitter.set_theta(vectorfit_py.doubleArray2D(t))
            fitter.evaluate()
            return np.array(fitter.mu())

        mu_at(theta)
        dmu = np.array(fitter.dmudtheta())[..., 0, :]
        for k in range(5, theta.shape[1]):
            t1, t2 = theta.copy(), theta.copy()
            t1[:, k] += 1e-3
            t2[:, k] -= 1e-3
            fd = (mu_at(t1) - mu_at(t2))[..., 0] / 2e-3
            np.testing.assert_allclose(
                fd, dmu[..., k], atol=1e-6 * np.abs(dmu[..., k]).max()
            )

    def test_set_theta_checks_the_shape(self):
        import vectorfit_py

        params = vectorfit.fit_params(
            {}, box=9, pixelsize=100.0, image_shape=(64, 64)
        )
        fitter = vectorfit_py.Fitter(
            vectorfit_py.doubleArray4D(np.ones((2, 9, 9, 1))),
            vectorfit_py.doubleArray2D(np.full((2, 2), 30.0)),
            params,
            "cpu",
        )
        with pytest.raises(Exception, match="shape"):
            fitter.set_theta(vectorfit_py.doubleArray2D(np.zeros((2, 3))))

    def test_theta_params_only_is_laid_out_correctly(self):
        """A column slice of theta: its rows must not be scrambled."""
        import vectorfit_py

        params = vectorfit.fit_params(
            {}, box=9, pixelsize=100.0, image_shape=(64, 64)
        )
        fitter = vectorfit_py.Fitter(
            vectorfit_py.doubleArray4D(np.ones((3, 9, 9, 1))),
            vectorfit_py.doubleArray2D(np.full((2, 3), 30.0)),
            params,
            "cpu",
        )
        theta = np.arange(3 * 10, dtype=np.float64).reshape(3, 10)
        fitter.set_theta(vectorfit_py.doubleArray2D(theta))
        np.testing.assert_array_equal(
            np.array(fitter.theta_params_only()), theta[:, :4]
        )

    @pytest.mark.slow
    def test_recovers_field_dependent_aberrations_3d(self):
        import vectorfit_py

        rng = np.random.default_rng(0)
        true = np.zeros(13)
        true[4:] = [70.0, 10.0, 15.0, -10.0, 8.0, 20.0, -15.0, 10.0, -15.0]
        parameters = {"Fit model": "3D", "Vertical astigmatism": 50.0}
        spots, roi, *_ = _simulate(
            vectorfit_py, parameters, true, 1500, 11, 100.0, (512, 512), rng
        )
        estimate = va.estimate_aberrations(
            spots,
            roi,
            parameters=parameters,
            pixelsize=100.0,
            image_shape=(512, 512),
            n_spots=600,
            n_threads=8,
        )
        # (the alternation may not meet its tolerance: each global update
        # draws another random subset, so the likelihood keeps jittering)
        np.testing.assert_allclose(
            estimate["gammas"][va.FITTED_GAMMAS],
            true[va.FITTED_GAMMAS],
            atol=2.0,
        )
        np.testing.assert_array_equal(estimate["gammas"][:4], 0)
        # far better than where it started
        assert estimate["log_likelihood"][-1] > estimate["log_likelihood"][0]

    @pytest.mark.slow
    def test_negative_astigmatism_keeps_its_sign(self):
        """The cylindrical lens can be the other way round: an initial
        negative astigmatism keeps the estimate on that branch."""
        import vectorfit_py

        rng = np.random.default_rng(1)
        true = np.zeros(13)
        true[[4, 9, 12]] = [-70.0, 15.0, 10.0]
        parameters = {"Fit model": "3D", "Vertical astigmatism": -40.0}
        spots, roi, *_ = _simulate(
            vectorfit_py, parameters, true, 600, 11, 100.0, (256, 256), rng
        )
        estimate = va.estimate_aberrations(
            spots,
            roi,
            parameters=parameters,
            pixelsize=100.0,
            image_shape=(256, 256),
            n_spots=400,
            n_threads=8,
        )
        assert estimate["gammas"][4] < 0
        np.testing.assert_allclose(
            estimate["gammas"][[4, 9, 12]], true[[4, 9, 12]], atol=3.0
        )

    def test_abort(self):
        import vectorfit_py

        rng = np.random.default_rng(0)
        spots, roi, *_ = _simulate(
            vectorfit_py,
            {},
            np.zeros(13),
            60,
            9,
            100.0,
            (64, 64),
            rng,
        )
        assert (
            va.estimate_aberrations(
                spots,
                roi,
                parameters={},
                pixelsize=100.0,
                image_shape=(64, 64),
                abort_callback=lambda: True,
            )
            is None
        )
