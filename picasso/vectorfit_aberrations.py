"""
picasso.vectorfit_aberrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimates the field-dependent aberrations of a VectorFit PSF model from
the data itself, as VectorFit's MATLAB pipeline does between segmentation
and localization (``SMLM_pipeline/estimate_aberrations.m``).

The aberrations are described by nodal aberration theory (NAT): the
Zernike coefficient of every primary aberration varies over the field of
view as a sum of products of Legendre polynomials in the normalized field
coordinates. Their 13 coefficients ("gammas", nm, VectorFit's
``NAT_order = 1``; see ``vectorfit.LEGENDRE_LABELS``) are fitted together
with the position, photons and background of a selection of spots, by
alternating between

- a local update: every spot is fitted (x, y, z, photons, background)
  with the aberrations the current gammas give at its position, by
  VectorFit's own fit, on the CPU or the GPU;
- a global update: the gammas are fitted by Levenberg-Marquardt with the
  local parameters fixed, maximizing the summed Poisson log-likelihood of
  a random subset of the spots. Its gradient and Hessian follow from
  VectorFit's derivatives of the PSF w.r.t. the Zernike coefficients
  (``Fitter.evaluate``), chained with the (linear) dependence of the
  Zernike coefficients on the gammas.

z is always fitted during the estimation, also for 2D data, since the
defocus of each spot must not be mistaken for an aberration. The four
defocus/field-curvature gammas are not fitted for that reason.

The PSF is unchanged when the sign of the aberrations of even azimuthal
order (astigmatism, spherical aberration) and of z flip together. That
ambiguity is broken by keeping one of them (the reference gamma, see
:func:`reference_gamma`) at a fixed sign, flipping the others and z of
every spot whenever an update crosses zero. Unlike the MATLAB code
(``global_update.m``, fixed alongside), the flip belongs to the trial
update and is applied to the spots only once that update is accepted.

References
----------
Droste, I. et al. "Calibration-free estimation of field-dependent
aberrations for single-molecule localization microscopy across large
fields of view." Optica 12, 1220-1229 (2025).
https://doi.org/10.1364/OPTICA.560721

:author: Thomas Bonte, 2026
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from . import lib, vectorfit

#: Defaults of the estimation, those of VectorFit's MATLAB pipeline
#: (``set_parameters_microtubules_3D.m``).
#:
#: - N_SPOTS: spots selected (uniformly over the field of view) for the
#:   estimation (``params.Ncfg``);
#: - N_SPOTS_GLOBAL: random subset of them used by each global update
#:   (``params.Ncfg_global``);
#: - MIN/MAX_TOTAL_ITERATIONS, TOLERANCE_TOTAL: the alternation between
#:   local and global updates stops once the summed log-likelihood
#:   changes by less than TOLERANCE_TOTAL (relative), after at least
#:   MIN_TOTAL_ITERATIONS;
#: - MIN/MAX_GLOBAL_ITERATIONS, TOLERANCE_GLOBAL: the same for the
#:   Levenberg-Marquardt iterations of one global update;
#: - LAMBDA_GLOBAL, LAMBDA_FACTOR: initial Levenberg-Marquardt damping of
#:   the global update and its factor;
#: - INITIAL_LEVEL: RMS level of the random initial gammas (mlambda);
#: - INITIAL_ASTIGMATISM_3D, INITIAL_ASTIGMATISM_2D: initial vertical
#:   astigmatism (mlambda) of a 3D / 2D estimation when no astigmatism or
#:   spherical aberration is set. VectorFit starts 2D data from (almost)
#:   zero, where the estimation stalls: the PSF is then symmetric about
#:   focus, so each spot's z takes an arbitrary sign and the spots' pulls
#:   on the astigmatism and spherical aberration cancel. Starting from
#:   either sign is equivalent (see ``SYMMETRY``).
N_SPOTS = 1000
N_SPOTS_GLOBAL = 200
MIN_TOTAL_ITERATIONS = 5
MAX_TOTAL_ITERATIONS = 20
TOLERANCE_TOTAL = 1e-6
MIN_GLOBAL_ITERATIONS = 1
MAX_GLOBAL_ITERATIONS = 20
TOLERANCE_GLOBAL = 5e-10
LAMBDA_GLOBAL = 1e3
LAMBDA_FACTOR = 10.0
INITIAL_LEVEL = 1.0
INITIAL_ASTIGMATISM_3D = 150.0
INITIAL_ASTIGMATISM_2D = 50.0

#: Fewest spots an estimation accepts: 9 gammas are fitted.
MIN_SPOTS = 50

#: Number of Legendre (NAT) coefficients, see ``vectorfit.LEGENDRE_LABELS``.
N_GAMMAS = vectorfit._N_LEGENDRE

#: The gammas fitted: all but the four of defocus / field curvature, which
#: the z of each spot absorbs.
FITTED_GAMMAS = np.arange(4, N_GAMMAS)

#: Azimuthal Zernike order of each gamma (of its first NAT term).
_AZIMUTHAL_ORDER = np.array([0, 0, 0, 0, 2, -2, 2, 2, 2, 1, -1, 1, 0])

#: Multiplying the gammas elementwise by this vector, while flipping the
#: z of every spot, leaves every PSF unchanged: the aberrations of even
#: azimuthal order change sign (``params.symmetryvector``).
SYMMETRY = np.where(_AZIMUTHAL_ORDER % 2 == 0, -1.0, 1.0)

#: Sum of the squared NAT weights of each gamma, so that the RMS
#: aberration over the field of view is ``sqrt(sum(MULTIPLEX *
#: gammas**2))`` (``params.multiplex_factors``).
MULTIPLEX = np.array(
    [1, 1, 1, 2, 1, 1, 2, 2, 1.4, 1, 1, 2, 1], dtype=np.float64
)

#: Constant gammas of even azimuthal order, the candidates of
#: :func:`reference_gamma`: vertical and oblique astigmatism, spherical
#: aberration.
_REFERENCE_CANDIDATES = (4, 5, 12)


def system_aberration(gammas: np.ndarray, wavelength: float) -> float:
    """RMS aberration over the field of view (mlambda) of ``gammas``
    (nm), see ``MULTIPLEX``."""
    gammas = np.asarray(gammas, dtype=np.float64)
    return float(np.sqrt(np.sum(MULTIPLEX * gammas**2)) * 1e3 / wavelength)


def reference_gamma(initial: np.ndarray) -> tuple[int, float]:
    """The gamma whose sign is kept fixed to break the z-flip symmetry
    (see ``SYMMETRY``), and that sign.

    VectorFit keeps the constant vertical astigmatism positive
    (``params.idx_pos``). Here the constant aberration of even order with
    the largest initial magnitude keeps its initial sign, so that the
    estimate stays on the side of the symmetry the user's setup is on
    (e.g. a negative astigmatism of the cylindrical lens); without one,
    the vertical astigmatism is kept positive, as in VectorFit.

    Parameters
    ----------
    initial : np.ndarray
        The 13 initial gammas (nm).

    Returns
    -------
    index : int
    sign : float
        +1.0 or -1.0.
    """
    initial = np.asarray(initial, dtype=np.float64)
    candidates = np.array(_REFERENCE_CANDIDATES)
    magnitudes = np.abs(initial[candidates])
    if not np.any(magnitudes > 0):
        return 4, 1.0
    index = int(candidates[np.argmax(magnitudes)])
    return index, 1.0 if initial[index] > 0 else -1.0


def initial_gammas(
    fixed: np.ndarray,
    wavelength: float,
    is_3d: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initial gammas of the estimation (``initialvalues_phasor_nat.m``):
    random fitted gammas at an RMS level of ``INITIAL_LEVEL`` mlambda,
    replaced by the non-zero ones of ``fixed`` (the aberrations the user
    set). Without any astigmatism or spherical aberration, the estimation
    starts from ``INITIAL_ASTIGMATISM_3D`` (as VectorFit does) or
    ``INITIAL_ASTIGMATISM_2D`` of vertical astigmatism.

    Parameters
    ----------
    fixed : np.ndarray
        13 gammas (nm); the non-zero ones are used as they are.
    wavelength : float
        Emission wavelength (nm).
    is_3d : bool
        Whether the data are 3D (astigmatic).
    rng : np.random.Generator

    Returns
    -------
    gammas : np.ndarray
        13 gammas (nm); the reference gamma (:func:`reference_gamma`) has
        its sign.
    """
    fixed = np.asarray(fixed, dtype=np.float64).copy()
    if not np.any(fixed[list(_REFERENCE_CANDIDATES)]):
        level = INITIAL_ASTIGMATISM_3D if is_3d else INITIAL_ASTIGMATISM_2D
        fixed[4] = level * wavelength * 1e-3
    gammas = np.zeros(N_GAMMAS)
    gammas[FITTED_GAMMAS] = rng.random(len(FITTED_GAMMAS)) - 0.5
    gammas *= INITIAL_LEVEL / system_aberration(gammas, wavelength)
    given = fixed != 0
    gammas[given] = fixed[given]
    index, sign = reference_gamma(fixed)
    gammas[index] = sign * abs(gammas[index])
    return gammas


def select_spots(
    roi: np.ndarray,
    image_shape: tuple[int, int],
    n_spots: int,
    rng: np.random.Generator,
    grid_size: int = 10,
) -> np.ndarray:
    """Select ``n_spots`` spots spread uniformly over the field of view
    (``select_spots.m``): the field is divided into ``grid_size`` x
    ``grid_size`` cells, an equal number of spots is drawn at random from
    each cell (all of a cell's spots if it has fewer), and the remaining
    ones at random from the spots not yet drawn.

    Parameters
    ----------
    roi : np.ndarray
        ``(2, n)`` 1-based box centers, in camera pixels, in the order of
        ``image_shape``.
    image_shape : tuple of int
        Size of the field of view (camera pixels).
    n_spots : int
        Spots to select; all of them if there are not more.
    rng : np.random.Generator
    grid_size : int, optional
        Default 10.

    Returns
    -------
    indices : np.ndarray
        Sorted indices of the selected spots.
    """
    n_total = roi.shape[1]
    if n_spots >= n_total:
        return np.arange(n_total)
    per_cell = n_spots // grid_size**2
    cells = []
    for axis in range(2):
        cell_size = image_shape[axis] / grid_size
        cell = np.ceil(roi[axis] / cell_size).astype(np.int64)
        cells.append(np.clip(cell, 1, grid_size) - 1)
    cell_index = cells[0] * grid_size + cells[1]
    selected = []
    if per_cell > 0:
        order = rng.permutation(n_total)
        # first per_cell spots of each cell, in random order
        sorted_cells = cell_index[order]
        by_cell = np.argsort(sorted_cells, kind="stable")
        sorted_cells = sorted_cells[by_cell]
        starts = np.searchsorted(sorted_cells, sorted_cells, side="left")
        rank = np.arange(n_total) - starts
        selected = order[by_cell[rank < per_cell]]
    remaining = np.setdiff1d(np.arange(n_total), selected)
    extra = rng.choice(remaining, n_spots - len(selected), replace=False)
    return np.sort(np.concatenate([selected, extra]).astype(np.int64))


def normalized_fov(
    roi: np.ndarray,
    positions: np.ndarray,
    pixelsize: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Normalized field coordinates in [-1, 1] of fitted spots, as
    VectorFit computes them (``Fitter::get_normalized_fov_coordinates``).

    Parameters
    ----------
    roi : np.ndarray
        ``(2, n)`` 1-based box centers (camera pixels).
    positions : np.ndarray
        ``(n, 2)`` fitted positions relative to the box centers (nm), in
        the order of ``roi``.
    pixelsize : float
        Camera pixel size (nm).
    image_shape : tuple of int
        Size of the field of view (camera pixels), in the order of
        ``roi``.

    Returns
    -------
    fov : np.ndarray
        ``(2, n)``.
    """
    size = np.asarray(image_shape, dtype=np.float64)[:, None] * pixelsize
    return (roi * pixelsize + positions.T) * 2.0 / size - 1.0


def zernike_jacobian(
    vectorfit_py, fov: np.ndarray, orders: np.ndarray
) -> np.ndarray:
    """Derivatives of the Zernike coefficients of each spot w.r.t. the
    gammas. The coefficients are linear in the gammas, so these are the
    coefficients VectorFit computes for each unit gamma.

    Parameters
    ----------
    vectorfit_py : module
    fov : np.ndarray
        ``(2, n)`` normalized field coordinates.
    orders : np.ndarray
        ``(n_zernike, 3)`` Zernike aberrations of the model.

    Returns
    -------
    jacobian : np.ndarray
        ``(n, n_zernike, N_GAMMAS)``.
    """
    xn = vectorfit_py.doubleArray1D(np.ascontiguousarray(fov[0]))
    yn = vectorfit_py.doubleArray1D(np.ascontiguousarray(fov[1]))
    aberrations = vectorfit._aberrations(vectorfit_py, orders)
    columns = []
    for j in range(N_GAMMAS):
        unit = np.zeros(N_GAMMAS)
        unit[j] = 1.0
        columns.append(
            np.array(
                vectorfit_py.compute_xy_gamma_legendre(
                    xn, yn, vectorfit_py.doubleArray1D(unit), aberrations
                )
            )
        )
    return np.stack(columns, axis=-1)


#: Floor of the Poisson rate in the log-likelihood (``likelihood_nat.m``).
_MU_FLOOR = 1e3 * np.finfo(np.float64).eps


def _log_likelihood(spots: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Poisson log-likelihood of each ``(n, box, box)`` spot, without the
    data-only term (``likelihood_nat.m`` without readout noise)."""
    mu = np.where(mu > 0, mu, _MU_FLOOR)
    return np.sum(spots * np.log(mu) - mu, axis=(1, 2))


def _log_likelihood_derivatives(
    spots: np.ndarray, mu: np.ndarray, dmu: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poisson log-likelihood of each spot and its gradient and Hessian
    w.r.t. the parameters of ``dmu`` (``likelihood_nat.m`` without readout
    noise; the Hessian is the usual Fisher approximation).

    Parameters
    ----------
    spots, mu : np.ndarray
        ``(n, box, box)``.
    dmu : np.ndarray
        ``(n, box, box, p)`` derivatives of ``mu``.

    Returns
    -------
    log_likelihood : np.ndarray
        ``(n,)``.
    gradient : np.ndarray
        ``(n, p)``.
    hessian : np.ndarray
        ``(n, p, p)``.
    """
    log_likelihood = _log_likelihood(spots, mu)
    mu = np.where(mu > 0, mu, _MU_FLOOR)
    weight = (spots - mu) / mu
    dweight = spots / mu**2
    gradient = np.einsum("nij,nijp->np", weight, dmu)
    hessian = -np.einsum("nij,nijp,nijq->npq", dweight, dmu, dmu)
    return log_likelihood, gradient, hessian


def _levenberg_marquardt_step(
    gammas: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    damping: float,
    reference: tuple[int, float],
) -> tuple[np.ndarray, bool]:
    """Trial update of the gammas (``thetaupdate_nat.m``), and whether it
    flipped the symmetric ones to keep the reference gamma's sign (which
    must go together with flipping z of every spot)."""
    matrix = hessian + damping * np.diag(np.diag(hessian))
    if 1.0 / np.linalg.cond(matrix) > np.finfo(np.float64).eps:
        step = -np.linalg.solve(matrix, gradient)
    else:
        step = -np.linalg.pinv(matrix) @ gradient
    trial = gammas.copy()
    trial[FITTED_GAMMAS] += step
    index, sign = reference
    flip = bool(sign * trial[index] < 0)
    if flip:
        trial *= SYMMETRY
    return trial, flip


class _GlobalModel:
    """The log-likelihood of a fixed set of spots, with their local
    parameters fixed, as a function of the gammas, with its gradient and
    Hessian w.r.t. the fitted gammas."""

    def __init__(
        self,
        vectorfit_py,
        params,
        spots: np.ndarray,
        roi: np.ndarray,
        theta_local: np.ndarray,
        orders: np.ndarray,
        pixelsize: float,
        image_shape: tuple[int, int],
    ):
        self.vectorfit_py = vectorfit_py
        self.spots = spots
        self.theta_local = theta_local
        self.fitter = vectorfit_py.Fitter(
            vectorfit_py.doubleArray4D(np.ascontiguousarray(spots[..., None])),
            vectorfit_py.doubleArray2D(np.ascontiguousarray(roi)),
            params,
            "cpu",
        )
        fov = normalized_fov(roi, theta_local[:, :2], pixelsize, image_shape)
        self.jacobian = zernike_jacobian(vectorfit_py, fov, orders)
        self.n_zernike = len(orders)

    def evaluate(
        self, gammas: np.ndarray, flip_z: bool
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Summed log-likelihood, gradient and Hessian at ``gammas``, with
        the z of the spots flipped if ``flip_z``."""
        theta_local = self.theta_local.copy()
        if flip_z:
            theta_local[:, 2] *= -1
        zernikes = self.jacobian @ gammas
        theta = np.ascontiguousarray(
            np.concatenate([theta_local, zernikes], axis=1)
        )
        self.fitter.set_theta(self.vectorfit_py.doubleArray2D(theta))
        self.fitter.evaluate()
        mu = np.array(self.fitter.mu())[..., 0]
        # dmu/dtheta: x, y, z, photons, background, then the Zernikes
        dmu = np.array(self.fitter.dmudtheta())[:, :, :, 0, 5:]
        log_likelihood, gradient, hessian = _log_likelihood_derivatives(
            self.spots, mu, dmu
        )
        jacobian = self.jacobian[:, :, FITTED_GAMMAS]
        gradient = np.einsum("nzg,nz->g", jacobian, gradient)
        hessian = np.einsum("nzg,nzw,nwh->gh", jacobian, hessian, jacobian)
        return float(np.sum(log_likelihood)), gradient, hessian


def _global_update(
    model: _GlobalModel,
    gammas: np.ndarray,
    damping: float,
    reference: tuple[int, float],
    min_iterations: int,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, float, bool, int]:
    """Fit the gammas to ``model`` by Levenberg-Marquardt
    (``global_update.m``).

    Returns
    -------
    gammas : np.ndarray
        The last accepted gammas.
    damping : float
        The damping to start the next global update with.
    flip_z : bool
        Whether the accepted gammas go with flipped z of every spot.
    iterations : int
    """
    accepted = None  # log-likelihood of the accepted gammas
    flip_z = False
    trial, trial_flip = gammas, False
    monitor = 2 * tolerance
    iterations = 0
    while iterations < max_iterations and (
        monitor > tolerance or iterations < min_iterations
    ):
        merit, gradient, hessian = model.evaluate(trial, flip_z ^ trial_flip)
        if accepted is not None and merit < accepted:
            damping *= LAMBDA_FACTOR  # rejected
        else:
            if accepted is not None:
                damping /= LAMBDA_FACTOR
            monitor = abs((merit - (accepted or 0.0)) / merit)
            accepted = merit
            gammas = trial
            flip_z ^= trial_flip
            best_gradient, best_hessian = gradient, hessian
        trial, trial_flip = _levenberg_marquardt_step(
            gammas, best_gradient, best_hessian, damping, reference
        )
        iterations += 1
    return gammas, damping, flip_z, iterations


def check_supported() -> None:
    """Raise an ImportError if VectorFit is missing or its bindings cannot
    evaluate the model at given parameters (``Fitter.evaluate``), which
    the estimation needs."""
    vectorfit_py = vectorfit._import_vectorfit()
    if not all(
        hasattr(vectorfit_py.Fitter, name)
        for name in ("evaluate", "set_theta")
    ) or not hasattr(vectorfit_py, "doubleView5D"):
        raise ImportError(
            "Estimating aberrations needs a VectorFit version with "
            "Fitter.evaluate and Fitter.set_theta (vectorfit-py 0.4.0 or "
            f"newer); the installed one is {vectorfit_py.__version__}."
        )


def estimate_aberrations(
    spots: lib.FloatArray3D,
    roi: np.ndarray,
    *,
    parameters: dict,
    pixelsize: float,
    image_shape: tuple[int, int],
    n_spots: int = N_SPOTS,
    n_spots_global: int = N_SPOTS_GLOBAL,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    min_total_iterations: int = MIN_TOTAL_ITERATIONS,
    max_total_iterations: int = MAX_TOTAL_ITERATIONS,
    tolerance_total: float = TOLERANCE_TOTAL,
    max_global_iterations: int = MAX_GLOBAL_ITERATIONS,
    tolerance_global: float = TOLERANCE_GLOBAL,
    seed: int | None = 0,
    n_threads: int | None = None,
    use_gpu: bool = False,
    progress: lib.Progress | Literal["console"] | None = None,
    abort_callback: Callable[[], bool] | None = None,
) -> dict | None:
    """Estimate the field-dependent aberrations (gammas) of the VectorFit
    model from the spots, see the module docstring.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n, box, box)`` photon counts; values below 1e-3 are clipped to
        it, as in ``vectorfit.fit_spots``.
    roi : np.ndarray
        ``(2, n)`` 1-based box centers (camera pixels), in VectorFit's
        order (the first axis of ``image_shape`` first).
    parameters : dict
        VectorFit settings, see ``vectorfit.DEFAULT_PARAMETERS``. The
        optics define the model; the primary aberrations are the initial
        values, and the one of the largest magnitude among the
        astigmatisms and the spherical aberration keeps its sign (see
        :func:`reference_gamma`).
    pixelsize : float
        Camera pixel size (nm).
    image_shape : tuple of int
        ``(height, width)`` of the movie frames (camera pixels).
    n_spots : int, optional
        Spots selected for the estimation, uniformly over the field of
        view (see :func:`select_spots`). Default ``N_SPOTS``.
    n_spots_global : int, optional
        Random subset of them that each global update fits the gammas to.
        Default ``N_SPOTS_GLOBAL``.
    tolerance, max_iterations : float, int or None, optional
        Convergence schedule of the local fits; None picks VectorFit's
        (``vectorfit.TOLERANCE``, ``vectorfit.MAX_ITERATIONS``).
    min_total_iterations, max_total_iterations, tolerance_total, \
            max_global_iterations, tolerance_global : optional
        Convergence schedule of the alternation and of the global updates,
        see ``N_SPOTS``.
    seed : int or None, optional
        Seed of the random spot selection and initial values; None for a
        different estimate on each call. Default 0.
    n_threads : int or None, optional
        CPU threads; None uses ``lib.n_workers()``.
    use_gpu : bool, optional
        Run the local fits on the GPU (the global updates always run on
        the CPU). Default False.
    progress : lib.Progress, "console" or None, optional
        Reports the phase "Estimating aberrations" over the (maximum
        number of) alternations.
    abort_callback : callable or None, optional
        Checked before every alternation; returning True stops the
        estimation.

    Returns
    -------
    estimate : dict or None
        None if aborted. Otherwise:

        - "gammas": the 13 estimated Legendre coefficients (nm), see
          ``vectorfit.LEGENDRE_LABELS``;
        - "initial_gammas": the ones the estimation started from;
        - "indices": the indices of the spots used;
        - "iterations": number of alternations;
        - "converged": whether the alternation converged before
          ``max_total_iterations``;
        - "log_likelihood": summed log-likelihood of the spots after
          each local update;
        - "gammas_history": ``(iterations, 13)`` gammas after each global
          update;
        - "flip_z": whether the spots' z ended flipped w.r.t. their first
          fit (informative only).

    Raises
    ------
    ValueError
        If there are fewer than ``MIN_SPOTS`` spots.
    ImportError
        If the VectorFit bindings lack ``Fitter.evaluate``.
    RuntimeError
        If ``use_gpu`` but VectorFit cannot fit on the GPU.
    """
    if use_gpu:
        vectorfit.check_gpu()
    check_supported()
    vectorfit_py = vectorfit._import_vectorfit()
    parameters = vectorfit.parameters_with_defaults(parameters)
    n_total = len(spots)
    if n_total < MIN_SPOTS:
        raise ValueError(
            f"Estimating aberrations needs at least {MIN_SPOTS} spots, got "
            f"{n_total}."
        )
    if n_spots < MIN_SPOTS:
        raise ValueError(
            f"Estimating aberrations needs at least {MIN_SPOTS} spots; "
            f"n_spots is {n_spots}."
        )
    if n_threads is None:
        n_threads = lib.n_workers()
    rng = np.random.default_rng(seed)
    box = spots.shape[-1]
    common = dict(
        box=box,
        pixelsize=pixelsize,
        image_shape=image_shape,
        tolerance=tolerance,
        max_iterations=max_iterations,
        n_threads=n_threads,
    )
    params_local = vectorfit.fit_params(
        parameters, model="xyz_constaberrations", **common
    )
    params_global = vectorfit.fit_params(
        parameters, model="xyz_aberrations", **common
    )
    fixed, orders = vectorfit._fixed_aberrations(
        vectorfit_py, params_local, parameters
    )
    is_3d = parameters["Fit model"] == "3D"
    gammas = initial_gammas(fixed, parameters["Wavelength"], is_3d, rng)
    initial = gammas.copy()
    reference = reference_gamma(gammas)

    indices = select_spots(roi, image_shape, n_spots, rng)
    selected = np.maximum(np.asarray(spots[indices], np.float64), 1e-3)
    roi_selected = np.ascontiguousarray(roi[:, indices], dtype=np.float64)
    n_selected = len(indices)
    n_global = min(n_spots_global, n_selected)
    aberrations = vectorfit._aberrations(vectorfit_py, orders)
    device = vectorfit.GPU_DEVICE if use_gpu else "cpu"

    local = vectorfit_py.Fitter(
        vectorfit_py.doubleArray4D(np.ascontiguousarray(selected[..., None])),
        vectorfit_py.doubleArray2D(roi_selected),
        params_local,
        device,
    )
    local.update_zernike_coefficients(
        vectorfit_py.doubleArray1D(gammas), aberrations
    )
    local.init()

    damping = LAMBDA_GLOBAL
    flip_pending = False
    flip_net = False
    log_likelihoods = []
    history = []
    converged = False
    with lib.tracked_progress(
        progress, "Estimating aberrations", "iteration"
    ) as progress:
        lib.start_phase(
            progress,
            "Estimating aberrations",
            max_total_iterations,
            unit="iteration",
        )
        for iteration in range(max_total_iterations):
            if callable(abort_callback) and abort_callback():
                return None
            # local update with the current gammas
            if flip_pending:
                local.flip_z()
            local.update_zernike_coefficients(
                vectorfit_py.doubleArray1D(gammas), aberrations
            )
            local.update()
            # x, y, z, photons, background (then the constant Zernikes)
            theta_local = np.array(local.theta())[:, :5]
            mu = np.array(local.mu())[..., 0]
            log_likelihoods.append(
                float(np.sum(_log_likelihood(selected, mu)))
            )
            outliers = np.asarray(local.get_outliers(), dtype=np.int64)

            # global update on a random subset of the spots
            subset = rng.choice(n_selected, n_global, replace=False)
            subset = np.sort(np.setdiff1d(subset, outliers))
            if len(subset) < 2 * len(FITTED_GAMMAS):
                # (almost) all outliers: no reliable global update
                history.append(gammas.copy())
                flip_pending = False
                progress.set_value(iteration + 1)
                continue
            model = _GlobalModel(
                vectorfit_py,
                params_global,
                selected[subset],
                roi_selected[:, subset],
                theta_local[subset],
                orders,
                pixelsize,
                image_shape,
            )
            gammas, damping, flip_pending, _ = _global_update(
                model,
                gammas,
                damping,
                reference,
                MIN_GLOBAL_ITERATIONS,
                max_global_iterations,
                tolerance_global,
            )
            flip_net ^= flip_pending
            history.append(gammas.copy())
            progress.set_value(iteration + 1)
            if len(log_likelihoods) >= 2:
                monitor = abs(
                    (log_likelihoods[-1] - log_likelihoods[-2])
                    / log_likelihoods[-1]
                )
                if (
                    iteration + 1 >= min_total_iterations
                    and monitor <= tolerance_total
                ):
                    converged = True
                    break
        progress.set_value(max_total_iterations)
    return {
        "gammas": gammas,
        "initial_gammas": initial,
        "indices": indices,
        "iterations": len(history),
        "converged": converged,
        "log_likelihood": np.array(log_likelihoods),
        "gammas_history": np.array(history),
        "flip_z": flip_net,
    }
