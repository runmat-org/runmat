//! Coupled pressure–velocity solve: Picard linearization of convection and viscosity, then
//! the whole momentum + continuity system for (u, p) at once by flexible GMRES from the shared
//! sparse layer, preconditioned by one SIMPLE sweep (momentum solves with component AMG, a
//! pressure Schur approximation — Rhie–Chow Laplacian plus the viscous term — by AMG, velocity
//! correction). The discrete equations are exactly the segregated
//! solver's: the same momentum operator, the same Rhie–Chow face flux, the same physical
//! acceptance; only the way the fixed point is reached differs.
use super::linear::{momentum_amg_options, within_admission, PreconditionerEnvelope};
use super::{
    geometry_ops::*, linear, momentum, pressure, recovery, CanonicalFluidError as Error,
    CanonicalFluidProblem, CanonicalFluidResult,
};
use crate::{
    algebra::{spmv, RealSparseMatrix},
    solve::shared_sparse::{ambient_plan, control, with_plan, CsrArrays},
};
use runmat_sparse_solvers::{
    csr::CsrRef,
    krylov::GmresWorkspace,
    krylov::PreconditionerSide,
    operator::FnOperator,
    precond::{Amg, AmgOptions},
    Preconditioner, SolverError, SolverResult,
};

/// Relative reduction of the coupled linear residual per Picard step (inexact Newton).
const LINEAR_REDUCTION: f64 = 1e-3;
/// Inner momentum and pressure solves of the preconditioner: loose, flexible.
const INNER_REDUCTION: f64 = 1e-2;
const INNER_ITERATIONS: usize = 40;

/// Coefficients frozen at the current iterate: the momentum operator and right-hand side
/// (convection from the current flux, deferred corrections from the current velocity), the
/// Rhie–Chow coefficients, and the residual at the iterate.
struct Frozen<'t> {
    matrix: RealSparseMatrix,
    /// The explicit momentum terms (non-orthogonal diffusion, higher-order advection,
    /// symmetry traction, boundary values) are linear in the velocity; the residual evaluates
    /// them at the trial velocity with the frozen flux, pressure and time history, so the
    /// linear step is exact in them instead of lagging them a Picard step. Lagged, graded or
    /// skewed cells (thin cells on a plane) contract by only the ratio of their explicit to
    /// implicit parts per step and stall.
    flux: Vec<f64>,
    time: Option<super::transient::TimeDerivative<'t>>,
    /// Velocity gradients of the frozen iterate: the viscosity of a shear-rate dependent law.
    viscosity_gradients: Vec<[V; 3]>,
    diagonal: Vec<f64>,
    rau: Vec<f64>,
    rf: Vec<f64>,
    /// Continuity rows are weighted by ρU so both residual blocks are forces.
    continuity_weight: f64,
    /// The momentum block J = A − ∂b/∂u assembled on the two-ring pattern: its action is the
    /// matrix-free one, at one sparse product per application.
    jacobian: super::jacobian::MomentumJacobian,
    /// The frozen iterate and the matrix-free momentum rows there: rows(u) =
    /// origin_rows + J (u − origin).
    origin: Vec<V>,
    origin_rows: Vec<f64>,
}

fn face_flux(
    p: &CanonicalFluidProblem<'_>,
    frozen: &Frozen<'_>,
    u: &[V],
    pressure_values: &[f64],
) -> Result<Vec<f64>, Error> {
    let gradient = pressure_gradient(p, pressure_values)?;
    let force = pressure_force(p, pressure_values, &gradient)?;
    face_flux_with(p, frozen, u, pressure_values, &gradient, &force)
}

/// [`face_flux`] with the pressure gradient and pressure force of `pressure_values` already
/// evaluated (the operator's residual needs both, and the preconditioner holds them fixed).
fn face_flux_with(
    p: &CanonicalFluidProblem<'_>,
    frozen: &Frozen<'_>,
    u: &[V],
    pressure_values: &[f64],
    gradient: &[V],
    force: &[V],
) -> Result<Vec<f64>, Error> {
    let term = (0..u.len())
        .map(|c| scale(force[c], frozen.rau[c] / p.topology.cells[c].volume_m3))
        .collect::<Vec<_>>();
    let qh = pressure::h_flux(p, u, &term)?;
    pressure::corrected_flux_with(p, &qh, &frozen.rf, pressure_values, gradient)
}

/// Weighted continuity rows `w·div q` of the face flux `q`, in face order.
fn continuity_rows(p: &CanonicalFluidProblem<'_>, frozen: &Frozen<'_>, q: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; p.topology.cells.len()];
    for (i, f) in p.topology.faces.iter().enumerate() {
        out[f.owner] += frozen.continuity_weight * q[i];
        if let Some(other) = f.neighbor {
            out[other] -= frozen.continuity_weight * q[i];
        }
    }
    out
}

/// Momentum rows of the linear step at velocity `u` and the frozen pressure: the assembled
/// operator plus the explicit terms evaluated at `u`, without the pressure force.
fn momentum_rows(
    p: &CanonicalFluidProblem<'_>,
    frozen: &Frozen<'_>,
    u: &[V],
) -> Result<Vec<f64>, Error> {
    let n = u.len();
    let rhs = momentum::rhs_with_time(
        p,
        u,
        &frozen.flux,
        momentum::MomentumMode::Coupled,
        frozen.time,
        &frozen.viscosity_gradients,
    )?;
    let mut out = vec![0.0; 3 * n];
    for k in 0..3 {
        let component = u.iter().map(|v| v[k]).collect::<Vec<_>>();
        let action =
            spmv(&frozen.matrix, &component).map_err(|e| Error::Operator(e.to_string()))?;
        for c in 0..n {
            out[k * n + c] = action[c] - rhs[c][k];
        }
    }
    Ok(out)
}

/// Momentum rows through the assembled block: origin_rows + J (u − origin).
fn assembled_rows(
    p: &CanonicalFluidProblem<'_>,
    frozen: &Frozen<'_>,
    u: &[V],
) -> Result<Vec<f64>, Error> {
    let n = u.len();
    let mut change = vec![0.0; 3 * n];
    for c in 0..n {
        for k in 0..3 {
            change[k * n + c] = u[c][k] - frozen.origin[c][k];
        }
    }
    let mut rows = frozen.origin_rows.clone();
    let mut action = vec![0.0; 3 * n];
    frozen
        .jacobian
        .view(&p.momentum_jacobian)
        .spmv(&change, &mut action);
    for (row, a) in rows.iter_mut().zip(&action) {
        *row += a;
    }
    Ok(rows)
}

/// Weighted residual [R_u (3n, force), w·div q (n)] at (u, p).
fn residual(
    p: &CanonicalFluidProblem<'_>,
    frozen: &Frozen<'_>,
    u: &[V],
    pressure_values: &[f64],
) -> Result<Vec<f64>, Error> {
    let n = u.len();
    let mut out = vec![0.0; 4 * n];
    let gradient = pressure_gradient(p, pressure_values)?;
    let force = pressure_force(p, pressure_values, &gradient)?;
    let rows = assembled_rows(p, frozen, u)?;
    for k in 0..3 {
        for c in 0..n {
            out[k * n + c] = rows[k * n + c] + force[c][k];
        }
    }
    let q = face_flux_with(p, frozen, u, pressure_values, &gradient, &force)?;
    out[3 * n..].copy_from_slice(&continuity_rows(p, frozen, &q));
    if out.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("nonfinite coupled residual"));
    }
    Ok(out)
}

fn split(x: &[f64], n: usize) -> (Vec<V>, Vec<f64>) {
    (
        (0..n).map(|c| [x[c], x[n + c], x[2 * n + c]]).collect(),
        x[3 * n..4 * n].to_vec(),
    )
}

/// Weight of the viscous term `diag(V_c / μ_eff,c)` in the sweep's Schur approximation.
///
/// For Stokes-like blocks, `S = D F⁻¹ G` is spectrally equivalent to `M_p / μ` with constants in
/// `[β², 1]`, where `β` is the discrete inf–sup constant (Cahouet–Chabard): the cellwise term
/// `V / μ_eff` is the upper end, exact only for cell-scale modes. Measured on the coarse channel
/// (12 302 cells), the Rayleigh quotients of the exact `S` against the unweighted sweep are 0.32
/// (lowest mode) and 0.84 (cell-scale mode). Weighting by `w` maps them to `{0.32, 0.84} / w`;
/// `w = √(0.32 · 0.84) ≈ 0.52` centers that interval on 1 in log scale, minimizing its condition
/// spread. We use 0.5: steady channel 699 → 239 FGMRES iterations, startup interval 454 → 288
/// (restart 128), with −dp/dx unchanged; the slab benchmarks are neutral (50 vs 51, 62 vs 56).
/// The weight changes only the preconditioner, never the discrete flux or the converged solution.
const VISCOUS_SCHUR_WEIGHT: f64 = 0.5;

/// The pressure operator of the SIMPLE sweep: an approximation of the Schur complement
/// `S = D F⁻¹ G + C` of the coupled step. It is the Rhie–Chow Laplacian `L(rf)` (SIMPLE's
/// `D diag(F)⁻¹ G` plus the Rhie–Chow dissipation `C`) plus the viscous term
/// `diag(V_c / μ_eff,c)`, `μ_eff,c = a_P,c / Σ_f α_f`, i.e. `rau_c Σ_f α_f`. On smooth pressure
/// modes `S` behaves like `M_p / μ`, which `L(rf)` alone misses (its eigenvalues there scale with
/// `h²`), so SIMPLE's outer iterations grow with the mesh; with the viscous term they stay nearly
/// constant (graded slab: 44, 51, 56 outer iterations at 384, 3 072, 10 368 cells, against 99, 165,
/// 275 without).
///
/// Also returns the outlet couplings `(owner, rf α)`: the viscous term breaks SIMPLE's cellwise
/// continuity exactness, and the sweep restores the global mass balance through them.
///
/// Where convection dominates the momentum block (`convective`) and the domain has an outlet,
/// the viscous term is left out: `S` then behaves like SIMPLE's `D diag(F)⁻¹ G`, and the term
/// overweights the pressure diagonal. On the uniform-inlet square duct at 0.2122 m/s (11 520
/// cells, skew 0.48) the coupled FGMRES needs 120 to 127 iterations per Picard step without it
/// and 300 to 670 with weights 0.5 and 1, for the same 10 Picard steps (47 s against 125 s).
/// A closed domain keeps the term, which is what makes its operator definite.
fn pressure_schur(
    p: &CanonicalFluidProblem<'_>,
    rf: &[f64],
    rau: &[f64],
    convective: bool,
) -> Result<(RealSparseMatrix, Vec<(usize, f64)>), Error> {
    let mut a = linear::assembler(p, "coupled-pressure-preconditioner")?;
    // Σ_f α_f per cell (the geometric Laplacian diagonal).
    let mut geometric = vec![0.0; p.topology.cells.len()];
    let mut outlets = Vec::new();
    for (i, f) in p.topology.faces.iter().enumerate() {
        let (alpha, _, _) = face_geometry(p, f)?;
        let d = rf[i] * alpha;
        geometric[f.owner] += alpha;
        if let Some(n) = f.neighbor {
            geometric[n] += alpha;
            a.add(f.owner, f.owner, d);
            a.add(f.owner, n, -d);
            a.add(n, n, d);
            a.add(n, f.owner, -d);
        } else if matches!(
            p.boundaries[i],
            super::boundary::FaceBoundary::PressureOutlet { .. }
        ) {
            a.add(f.owner, f.owner, d);
            outlets.push((f.owner, d));
        }
    }
    // The viscous Schur term (also makes closed domains definite), weighted by
    // VISCOUS_SCHUR_WEIGHT unless convection dominates an open domain.
    if !convective || outlets.is_empty() {
        for (c, g) in geometric.iter().enumerate() {
            a.add(c, c, VISCOUS_SCHUR_WEIGHT * rau[c] * g);
        }
    }
    let matrix = a.finish().map_err(|e| Error::Operator(e.to_string()))?;
    Ok((matrix, outlets))
}

struct SimplePreconditioner<'a, 'p> {
    p: &'a CanonicalFluidProblem<'p>,
    frozen: &'a Frozen<'a>,
    /// The assembled momentum block and its ILU factors.
    momentum: &'a dyn LinearOperatorView,
    momentum_preconditioner: &'a dyn Preconditioner<f64>,
    pressure: &'a dyn LinearOperatorView,
    pressure_preconditioner: &'a dyn Preconditioner<f64>,
    base_u: &'a [V],
    base_p: &'a [f64],
    base_residual: &'a [f64],
    /// Pressure gradient and pressure force of `base_p`, fixed per step: the sweep's continuity
    /// of the predictor reads the flux at (base_u + z_u*, base_p).
    base_gradient: Vec<V>,
    base_force: Vec<V>,
    /// Inner Krylov iterations of the momentum and pressure solves, over every application.
    momentum_iterations: std::sync::atomic::AtomicU64,
    pressure_iterations: std::sync::atomic::AtomicU64,
    applications: std::sync::atomic::AtomicU64,
    /// Outlet couplings `(owner, rf α)` of the pressure operator.
    outlets: &'a [(usize, f64)],
    /// Pressure force of a zero increment (the outlet reference constant), fixed per step.
    zero_force: Vec<[f64; 3]>,
    /// Inner-solve storage reused by every application.
    momentum_workspace: std::sync::Mutex<GmresWorkspace<f64>>,
    pressure_workspace: std::sync::Mutex<GmresWorkspace<f64>>,
}

/// The shared crate's operator trait object over a borrowed CSR view.
trait LinearOperatorView: runmat_sparse_solvers::LinearOperator<f64> {}
impl<T: runmat_sparse_solvers::LinearOperator<f64>> LinearOperatorView for T {}

/// One velocity component's preconditioner. Smoothed-aggregation AMG with its Chebyshev
/// smoother suits a diffusion-dominated block; a convection-dominated block has a strongly
/// complex spectrum, where the Chebyshev polynomial for a real interval amplifies error, so
/// ILU(0) takes its place (see [`convection_dominated`]).
enum MomentumFactors {
    Amg(Amg),
    Ilu(runmat_sparse_solvers::precond::Ilu0<f64>),
}

impl Preconditioner<f64> for MomentumFactors {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
        match self {
            Self::Amg(amg) => amg.apply(r, z),
            Self::Ilu(ilu) => ilu.apply(r, z),
        }
    }
    fn label(&self) -> &'static str {
        match self {
            Self::Amg(_) => "amg",
            Self::Ilu(_) => "ilu0",
        }
    }
}

/// Fraction of a momentum block's diagonal weight that is skew: Σ_{i≠j} |a_ij − a_ji| over
/// Σ_i 2 a_ii. Upwind convection of face Péclet number Pe = |ṁ|/D gives Pe / (2 + Pe) on a
/// uniform mesh; time and porous terms only add to the diagonal and lower it.
fn skew_fraction(a: &CsrRef<'_, f64>) -> f64 {
    let (row_ptr, col_idx, values) = (a.row_ptr(), a.col_idx(), a.values());
    let mut skew = 0.0;
    let mut diagonal = 0.0;
    for r in 0..a.nrows() {
        for at in row_ptr[r]..row_ptr[r + 1] {
            let c = col_idx[at];
            if c == r {
                diagonal += 2.0 * values[at].abs();
                continue;
            }
            let transpose = col_idx[row_ptr[c]..row_ptr[c + 1]]
                .binary_search(&r)
                .map_or(0.0, |k| values[row_ptr[c] + k]);
            skew += (values[at] - transpose).abs();
        }
    }
    if diagonal > 0.0 {
        skew / diagonal
    } else {
        0.0
    }
}

/// Skew fraction above which a momentum block counts as convection-dominated and takes ILU(0)
/// factors instead of AMG. Measured on the uniform-inlet square duct (11 520 cells, ISO VG 46)
/// from the second Picard step on: at 0.02 m/s (skew 0.08) and 0.06 m/s (skew 0.21) the AMG
/// step converges, and ILU(0) needs the same FGMRES iterations at 0.06 m/s; at 0.1 m/s and
/// above every inner AMG solve exhausts its iterations and the coupled FGMRES stagnates (at
/// 0.2122 m/s, relative residual 0.98 after 1 000 iterations), while ILU(0) converges in 10
/// Picard steps. The limit sits below the first failing case with margin.
const CONVECTION_SKEW_LIMIT: f64 = 0.15;

fn convection_dominated(skew: f64) -> bool {
    skew > CONVECTION_SKEW_LIMIT
}

/// Block-diagonal preconditioner: each velocity component by its own factors.
struct ComponentPreconditioner<F> {
    n: usize,
    factors: [F; 3],
}

impl<F: Preconditioner<f64>> Preconditioner<f64> for ComponentPreconditioner<F> {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
        for k in 0..3 {
            let range = k * self.n..(k + 1) * self.n;
            self.factors[k].apply(&r[range.clone()], &mut z[range])?;
        }
        Ok(())
    }
    fn label(&self) -> &'static str {
        "component-momentum"
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        false
    }
}

fn inner_solve(
    workspace: &std::sync::Mutex<GmresWorkspace<f64>>,
    operator: &dyn LinearOperatorView,
    preconditioner: &dyn Preconditioner<f64>,
    rhs: &[f64],
    counter: &std::sync::atomic::AtomicU64,
    iterations: usize,
    reduction: f64,
) -> SolverResult<Vec<f64>> {
    let mut x = vec![0.0; rhs.len()];
    if rhs.iter().all(|v| *v == 0.0) {
        return Ok(x);
    }
    if iterations == 0 {
        // One preconditioner application stands for the inner solve.
        preconditioner.apply(rhs, &mut x)?;
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(x);
    }
    let mut fresh = GmresWorkspace::default();
    let mut held = workspace.try_lock();
    let workspace = match held.as_deref_mut() {
        Ok(workspace) => workspace,
        Err(_) => &mut fresh,
    };
    let report = linear::restarted_gmres(
        workspace,
        operator,
        Some(preconditioner),
        rhs,
        &mut x,
        reduction,
        iterations,
        iterations,
        PreconditionerSide::Right,
    )?;
    counter.fetch_add(
        report.iterations as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    Ok(x)
}

impl Preconditioner<f64> for SimplePreconditioner<'_, '_> {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
        let n = self.base_u.len();
        let fail = |e: Error| SolverError::Breakdown(e.to_string());
        // Momentum predictor on the assembled momentum block J (the implicit operator and
        // the explicit terms' Jacobian): J z_u* = r_u by inner GMRES with its ILU factors.
        let solution = inner_solve(
            &self.momentum_workspace,
            self.momentum,
            self.momentum_preconditioner,
            &r[..3 * n],
            &self.momentum_iterations,
            INNER_ITERATIONS,
            INNER_REDUCTION,
        )?;
        self.applications
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let predicted = (0..n)
            .map(|c| [solution[c], solution[n + c], solution[2 * n + c]])
            .collect::<Vec<_>>();
        // Continuity of the predictor: the exact linear flux of z_u* (affine difference).
        let trial = (0..n)
            .map(|c| add(self.base_u[c], predicted[c]))
            .collect::<Vec<_>>();
        let shifted_flux = face_flux_with(
            self.p,
            self.frozen,
            &trial,
            self.base_p,
            &self.base_gradient,
            &self.base_force,
        )
        .map_err(fail)?;
        let shifted = continuity_rows(self.p, self.frozen, &shifted_flux);
        if shifted.iter().any(|v| !v.is_finite()) {
            return Err(fail(Error::Invalid("nonfinite coupled residual")));
        }
        let w = self.frozen.continuity_weight;
        // The Rhie–Chow Laplacian L approximates −∂(div q)/∂p: L z_p = D z_u* − r_c.
        let rhs = (0..n)
            .map(|c| (shifted[c] - self.base_residual[3 * n + c] - r[3 * n + c]) / w)
            .collect::<Vec<_>>();
        let mut correction = inner_solve(
            &self.pressure_workspace,
            self.pressure,
            self.pressure_preconditioner,
            &rhs,
            &self.pressure_iterations,
            INNER_ITERATIONS,
            INNER_REDUCTION,
        )?;
        // Global mass balance. With z_u = z_u* − G z_p / a_P the corrected continuity is
        // rhs − L z_p cell by cell, whose sum is Σ rhs − Σ_outlet rf α z_p (interior faces cancel).
        // The viscous Schur term leaves that sum nonzero; a uniform shift of z_p, which moves only
        // the outlet fluxes, makes it vanish so every application conserves mass globally.
        let outlet_weight: f64 = self.outlets.iter().map(|&(_, d)| d).sum();
        if outlet_weight > 0.0 {
            let outflow: f64 = self.outlets.iter().map(|&(c, d)| d * correction[c]).sum();
            let shift = (rhs.iter().sum::<f64>() - outflow) / outlet_weight;
            correction.iter_mut().for_each(|z| *z += shift);
        }
        // Velocity correction z_u = z_u* − G z_p / a_P.
        let force = pressure_force(
            self.p,
            &correction,
            &pressure_gradient(self.p, &correction).map_err(fail)?,
        )
        .map_err(fail)?;
        // The pressure force of an increment has no boundary constants except the outlet
        // reference, which pressure_force evaluates at the absolute value: remove it.
        let zero = &self.zero_force;
        for c in 0..n {
            for k in 0..3 {
                z[k * n + c] =
                    predicted[c][k] - (force[c][k] - zero[c][k]) / self.frozen.diagonal[c];
            }
            z[3 * n + c] = correction[c];
        }
        Ok(())
    }
    fn label(&self) -> &'static str {
        "simple-block"
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        false
    }
}

/// Freeze the coefficients at (u, p, q): the assembled implicit operator, the momentum
/// Jacobian, the Rhie–Chow coefficients and the origin of the affine momentum rows.
fn freeze<'t>(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_values: &[f64],
    q: &[f64],
    time: Option<super::transient::TimeDerivative<'t>>,
) -> Result<Frozen<'t>, Error> {
    let n = u.len();
    let m = momentum::assemble_with_time(
        p,
        u,
        pressure_values,
        q,
        momentum::MomentumMode::Coupled,
        time,
    )?;
    let rau = (0..n)
        .map(|c| p.topology.cells[c].volume_m3 * p.policy.momentum_relaxation / m.diagonal[c])
        .collect::<Vec<_>>();
    let rf = pressure::coefficients(p, &rau)?;
    let speed = u
        .iter()
        .map(|v| norm(*v))
        .chain(p.boundaries.iter().filter_map(|b| match b {
            super::boundary::FaceBoundary::Velocity { value, .. } => Some(norm(*value)),
            _ => None,
        }))
        .fold(0.0f64, f64::max);
    let gradients = velocity_gradients(p, u)?;
    let jacobian = super::jacobian::assemble(
        p,
        &m.matrix,
        q,
        momentum::MomentumMode::Coupled,
        time,
        &gradients,
    )?;
    let mut frozen = Frozen {
        matrix: m.matrix,
        flux: q.to_vec(),
        time,
        diagonal: m.diagonal,
        rau,
        rf,
        continuity_weight: p.density * speed.max(f64::MIN_POSITIVE.sqrt()),
        viscosity_gradients: gradients,
        jacobian,
        origin: u.to_vec(),
        origin_rows: Vec::new(),
    };
    frozen.origin_rows = momentum_rows(p, &frozen, u)?;
    Ok(frozen)
}

/// Momentum rows of the assembled block and of the matrix-free right-hand side at
/// `origin + change`, with coefficients frozen at (origin, pressure, q).
#[cfg(test)]
pub(super) fn assembled_and_matrix_free_rows(
    p: &CanonicalFluidProblem<'_>,
    origin: &[V],
    pressure_values: &[f64],
    q: &[f64],
    change: &[V],
) -> Result<(Vec<f64>, Vec<f64>), Error> {
    let frozen = freeze(p, origin, pressure_values, q, None)?;
    let trial = origin
        .iter()
        .zip(change)
        .map(|(a, b)| add(*a, *b))
        .collect::<Vec<_>>();
    Ok((
        assembled_rows(p, &frozen, &trial)?,
        momentum_rows(p, &frozen, &trial)?,
    ))
}

pub(super) fn solve_with_time(
    p: &CanonicalFluidProblem<'_>,
    initial: Option<super::transient::CanonicalFluidFields<'_>>,
    time: Option<super::transient::TimeDerivative<'_>>,
    numerical_input_identity: runmat_execution::Digest,
) -> Result<CanonicalFluidResult, Error> {
    let n = p.topology.cells.len();
    let (mut u, mut pressure_values, mut q) = match initial {
        Some(fields) => (
            fields.velocity_m_per_s.canonical_values().to_vec(),
            fields.pressure_fluctuation_pa.canonical_values().to_vec(),
            fields.face_volume_flux_m3_per_s.canonical_values().to_vec(),
        ),
        None => {
            let u = vec![[0.0; 3]; n];
            let q = pressure::h_flux(p, &u, &vec![[0.0; 3]; n])?;
            (u, vec![0.0; n], q)
        }
    };
    let mut linear_iterations = 0u64;
    let mut last = [f64::INFINITY; 5];
    let mut worst = String::from("not evaluated");
    let mut first_worst: Option<f64> = None;
    for iteration in 1..=p.policy.maximum_outer_iterations {
        if crate::is_fea_cancelled() {
            return Err(Error::Cancelled);
        }
        let step_clock = runmat_time::Instant::now();
        let frozen = freeze(p, &u, &pressure_values, &q, time)?;
        let base_residual = residual(p, &frozen, &u, &pressure_values)?;
        let base_gradient = pressure_gradient(p, &pressure_values)?;
        let base_force = pressure_force(p, &pressure_values, &base_gradient)?;
        let blocks =
            super::jacobian::component_operators(p, &frozen.matrix, &frozen.viscosity_gradients)?;
        let skew = skew_fraction(&blocks[0].view());
        let convective = convection_dominated(skew);
        let (laplacian, outlets) = pressure_schur(p, &frozen.rf, &frozen.rau, convective)?;
        let pressure_arrays = CsrArrays::of(&laplacian);
        let planned = PreconditionerEnvelope::for_counts(n as u64, p.reservation.sparse_entries)
            .ok_or(Error::Resource("planned preconditioner envelope"))?;
        let mut setup_seconds = 0.0;
        let mut inner = [0u64; 3];
        let solved = with_plan(ambient_plan(), || {
            let momentum_view = frozen.jacobian.view(&p.momentum_jacobian);
            let pressure_view = pressure_arrays.view();
            // Each velocity component is preconditioned by its implicit operator's factors:
            // smoothed-aggregation AMG (a weak strength threshold keeps the thin-cell couplings
            // of graded meshes in the aggregates) or, where convection dominates, ILU(0). The
            // inner solve applies the full block J.
            let component = |k: usize| {
                if convective {
                    return Ok::<_, SolverError>(MomentumFactors::Ilu(
                        runmat_sparse_solvers::precond::Ilu0::new(blocks[k].view(), false)?,
                    ));
                }
                let amg = Amg::new(blocks[k].view(), momentum_amg_options(), control())?;
                within_admission(&amg, &planned)?;
                Ok(MomentumFactors::Amg(amg))
            };
            let momentum_preconditioner = ComponentPreconditioner {
                n,
                factors: [component(0)?, component(1)?, component(2)?],
            };
            let amg = Amg::new(pressure_view, AmgOptions::default(), control())?;
            within_admission(&amg, &planned)?;
            let zero_force = pressure_force(
                p,
                &vec![0.0; n],
                &pressure_gradient(p, &vec![0.0; n])
                    .map_err(|e| SolverError::Breakdown(e.to_string()))?,
            )
            .map_err(|e| SolverError::Breakdown(e.to_string()))?;
            let preconditioner = SimplePreconditioner {
                zero_force,
                base_gradient,
                base_force,
                momentum_iterations: Default::default(),
                pressure_iterations: Default::default(),
                applications: Default::default(),
                momentum_workspace: Default::default(),
                pressure_workspace: Default::default(),
                p,
                frozen: &frozen,
                momentum: &momentum_view,
                momentum_preconditioner: &momentum_preconditioner,
                pressure: &pressure_view,
                pressure_preconditioner: &amg,
                base_u: &u,
                base_p: &pressure_values,
                base_residual: &base_residual,
                outlets: &outlets,
            };
            let operator = FnOperator {
                nrows: 4 * n,
                ncols: 4 * n,
                apply: |x: &[f64], y: &mut [f64]| {
                    let (du, dp) = split(x, n);
                    let trial_u = (0..n).map(|c| add(u[c], du[c])).collect::<Vec<_>>();
                    let trial_p = (0..n)
                        .map(|c| pressure_values[c] + dp[c])
                        .collect::<Vec<_>>();
                    let shifted = residual(p, &frozen, &trial_u, &trial_p)
                        .map_err(|e| SolverError::Breakdown(e.to_string()))?;
                    for (y, (s, b)) in y.iter_mut().zip(shifted.iter().zip(&base_residual)) {
                        *y = s - b;
                    }
                    Ok(())
                },
            };
            let rhs = base_residual.iter().map(|v| -v).collect::<Vec<_>>();
            let mut step = vec![0.0; 4 * n];
            setup_seconds = step_clock.elapsed().as_secs_f64();
            let report = linear::restarted_gmres(
                &mut GmresWorkspace::default(),
                &operator,
                Some(&preconditioner),
                &rhs,
                &mut step,
                LINEAR_REDUCTION.max(p.policy.linear_relative_tolerance),
                p.policy.maximum_linear_iterations,
                p.policy.gmres_restart,
                PreconditionerSide::Flexible,
            )?;
            let load =
                |c: &std::sync::atomic::AtomicU64| c.load(std::sync::atomic::Ordering::Relaxed);
            inner = [
                load(&preconditioner.applications),
                load(&preconditioner.momentum_iterations),
                load(&preconditioner.pressure_iterations),
            ];
            Ok::<_, SolverError>((step, report))
        });
        let (step, report) = solved.map_err(|error| match error {
            SolverError::Cancelled => Error::Cancelled,
            SolverError::OverBudget { .. } => {
                Error::Resource("preconditioner exceeds original admission")
            }
            other => Error::Operator(other.to_string()),
        })?;
        tracing::debug!(
            iteration,
            iterations = report.iterations,
            relative_residual = report.relative_residual,
            restart = p.policy.gmres_restart,
            cells = n,
            skew,
            convective,
            preconditioner_applications = inner[0],
            momentum_inner_iterations = inner[1],
            pressure_inner_iterations = inner[2],
            setup_seconds,
            krylov_seconds = step_clock.elapsed().as_secs_f64() - setup_seconds,
            "fluid coupled FGMRES step"
        );
        linear_iterations += report.iterations as u64;
        let (du, mut dp) = split(&step, n);
        // A closed component's pressure is defined up to a constant, which the Krylov step may
        // carry at any magnitude: remove it from the increment before it reaches the iterate.
        p.pressure_reference_plan.normalize(p, &mut dp)?;
        for c in 0..n {
            u[c] = add(u[c], du[c]);
            pressure_values[c] += dp[c];
        }
        p.pressure_reference_plan
            .normalize(p, &mut pressure_values)?;
        if u.iter()
            .flatten()
            .chain(&pressure_values)
            .any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("nonfinite coupled iterate"));
        }
        q = face_flux(p, &frozen, &u, &pressure_values)?;
        if p.topology.faces.iter().enumerate().any(|(i, f)| {
            f.neighbor.is_none()
                && matches!(
                    p.boundaries[i],
                    super::boundary::FaceBoundary::PressureOutlet { .. }
                )
                && q[i] < 0.0
        }) {
            continue;
        }
        let residual = recovery::evaluate_with_time(p, &u, &pressure_values, &q, time)?;
        last = [
            residual.momentum_relative,
            residual.continuity_relative,
            residual.flux_constitutive_relative,
            residual.global_momentum_relative,
            residual
                .cell_momentum_relative
                .iter()
                .copied()
                .fold(0.0, f64::max),
        ];
        tracing::debug!(
            iteration,
            momentum = last[0],
            continuity = last[1],
            flux = last[2],
            global_momentum = last[3],
            maximum_cell_momentum = last[4],
            seconds = step_clock.elapsed().as_secs_f64(),
            "fluid coupled Picard step"
        );
        let (worst_relative, description) = recovery::describe_worst_cell(p, &residual);
        let first = *first_worst.get_or_insert(worst_relative);
        worst = format!("{description}; it was {first:e} at the first evaluated step");
        if residual.accepted(p.policy.physical_relative_tolerance) {
            let pressure_pa = pressure_values
                .iter()
                .zip(p.pressure_reference_pa.canonical_values())
                .map(|(value, reference)| {
                    let absolute = value + reference;
                    if absolute.is_finite() {
                        Ok(absolute)
                    } else {
                        Err(Error::Invalid("nonfinite absolute pressure projection"))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(CanonicalFluidResult {
                numerical_input_identity,
                authored_authority_identity: p.authored_authority_identity,
                velocity_m_per_s: runmat_types::physical::Quantity::new(u),
                pressure_pa: runmat_types::physical::Quantity::new(pressure_pa),
                pressure_fluctuation_pa: runmat_types::physical::Quantity::new(pressure_values),
                pressure_reference_pa: p.pressure_reference_pa.clone(),
                face_volume_flux_m3_per_s: runmat_types::physical::Quantity::new(q),
                outer_iterations: iteration,
                linear_iterations,
                residual,
            });
        }
    }
    Err(Error::IterationExhausted {
        outer_iterations: p.policy.maximum_outer_iterations,
        linear_iterations,
        momentum_relative: last[0],
        continuity_relative: last[1],
        flux_constitutive_relative: last[2],
        global_momentum_relative: last[3],
        maximum_cell_momentum_relative: last[4],
        worst_cell: worst,
    })
}

/// Rayleigh quotients `zᵀ S z` of the exact Schur complement of the coupled linear step and
/// `zᵀ S̃ z` of the sweep's pressure operator, for a pressure increment `z` at the frozen
/// iterate (u, p, q). `S z` is the weighted continuity change of the coupled increment
/// `(−J⁻¹ G z, z)` with `J` solved to 1e-10 or better; the second value is the same for the
/// sweep's own velocity correction `−G z / a_P` (SIMPLE's operator).
#[cfg(test)]
pub(super) fn schur_rayleigh_quotients(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_values: &[f64],
    q: &[f64],
    z: &[f64],
) -> Result<(f64, f64, f64), Error> {
    let n = u.len();
    let frozen = freeze(p, u, pressure_values, q, None)?;
    let base_residual = residual(p, &frozen, u, pressure_values)?;
    let (laplacian, _) = pressure_schur(p, &frozen.rf, &frozen.rau, false)?;
    let blocks =
        super::jacobian::component_operators(p, &frozen.matrix, &frozen.viscosity_gradients)?;
    let operator_error = |e: SolverError| Error::Operator(e.to_string());
    let zero = vec![0.0; n];
    let zero_force = pressure_force(p, &zero, &pressure_gradient(p, &zero)?)?;
    let force = pressure_force(p, z, &pressure_gradient(p, z)?)?;
    let mut rhs = vec![0.0; 3 * n];
    for c in 0..n {
        for k in 0..3 {
            rhs[k * n + c] = -(force[c][k] - zero_force[c][k]);
        }
    }
    let mut velocity = vec![0.0; 3 * n];
    let report = with_plan(ambient_plan(), || {
        let amg = |k: usize| Amg::new(blocks[k].view(), momentum_amg_options(), control());
        let preconditioner = ComponentPreconditioner {
            n,
            factors: [amg(0)?, amg(1)?, amg(2)?],
        };
        let view = frozen.jacobian.view(&p.momentum_jacobian);
        linear::restarted_gmres(
            &mut GmresWorkspace::default(),
            &view,
            Some(&preconditioner),
            &rhs,
            &mut velocity,
            1e-12,
            2000,
            200,
            PreconditionerSide::Right,
        )
    })
    .map_err(operator_error)?;
    assert!(
        report.relative_residual <= 1e-10,
        "momentum solve {report:?}"
    );
    let trial_p = (0..n)
        .map(|c| pressure_values[c] + z[c])
        .collect::<Vec<_>>();
    let w = frozen.continuity_weight;
    let quotient = |change: &dyn Fn(usize) -> V| -> Result<f64, Error> {
        let trial_u = (0..n).map(|c| add(u[c], change(c))).collect::<Vec<_>>();
        let shifted = residual(p, &frozen, &trial_u, &trial_p)?;
        Ok((0..n)
            .map(|c| z[c] * (shifted[3 * n + c] - base_residual[3 * n + c]) / w)
            .sum::<f64>())
    };
    let exact = quotient(&|c| [velocity[c], velocity[n + c], velocity[2 * n + c]])?;
    // The sweep's own correction z_u = −(G z)/a_P: the operator its pressure step stands for.
    let simple = quotient(&|c| std::array::from_fn(|k| rhs[k * n + c] / frozen.diagonal[c]))?;
    let action = spmv(&laplacian, z).map_err(|e| Error::Operator(e.to_string()))?;
    let approximate = z.iter().zip(&action).map(|(a, b)| a * b).sum::<f64>();
    Ok((exact, simple, approximate))
}
