//! Coupled pressure–velocity solve: Picard linearization of convection and viscosity, then
//! the whole momentum + continuity system for (u, p) at once by flexible GMRES from the shared
//! sparse layer, preconditioned by one SIMPLE sweep (momentum ILU solves, Rhie–Chow pressure
//! Laplacian by AMG, velocity correction). The discrete equations are exactly the segregated
//! solver's: the same momentum operator, the same Rhie–Chow face flux, the same physical
//! acceptance; only the way the fixed point is reached differs.
use super::{
    geometry_ops::*, linear, momentum, pressure, recovery, CanonicalFluidError as Error,
    CanonicalFluidProblem, CanonicalFluidResult,
};
use crate::{
    algebra::{spmv, RealSparseMatrix},
    solve::shared_sparse::{ambient_plan, control, with_plan, CsrArrays},
};
use runmat_sparse_solvers::{
    krylov::PreconditionerSide,
    operator::FnOperator,
    precond::{Amg, AmgOptions, Ilu0},
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
    let force = pressure_force(p, pressure_values, &pressure_gradient(p, pressure_values)?)?;
    let term = (0..u.len())
        .map(|c| scale(force[c], frozen.rau[c] / p.topology.cells[c].volume_m3))
        .collect::<Vec<_>>();
    let qh = pressure::h_flux(p, u, &term)?;
    pressure::corrected_flux(p, &qh, &frozen.rf, pressure_values)
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
        let action = spmv(&frozen.matrix, &component).map_err(|e| Error::Operator(e.to_string()))?;
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
    let force = pressure_force(p, pressure_values, &pressure_gradient(p, pressure_values)?)?;
    let rows = assembled_rows(p, frozen, u)?;
    for k in 0..3 {
        for c in 0..n {
            out[k * n + c] = rows[k * n + c] + force[c][k];
        }
    }
    let q = face_flux(p, frozen, u, pressure_values)?;
    for (i, f) in p.topology.faces.iter().enumerate() {
        out[3 * n + f.owner] += frozen.continuity_weight * q[i];
        if let Some(other) = f.neighbor {
            out[3 * n + other] -= frozen.continuity_weight * q[i];
        }
    }
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

/// Rhie–Chow pressure Laplacian Σ_f rf α (p_N − p_P), outlet faces on the diagonal; a closed
/// domain gets a vanishing shift so the preconditioner stays definite.
fn pressure_laplacian(p: &CanonicalFluidProblem<'_>, rf: &[f64]) -> Result<RealSparseMatrix, Error> {
    let mut a = linear::assembler(p, "coupled-pressure-preconditioner")?;
    let mut diagonal = vec![0.0; p.topology.cells.len()];
    let mut open = false;
    for (i, f) in p.topology.faces.iter().enumerate() {
        let (alpha, _, _) = face_geometry(p, f)?;
        let d = rf[i] * alpha;
        if let Some(n) = f.neighbor {
            a.add(f.owner, f.owner, d);
            a.add(f.owner, n, -d);
            a.add(n, n, d);
            a.add(n, f.owner, -d);
            diagonal[f.owner] += d;
            diagonal[n] += d;
        } else if matches!(
            p.boundaries[i],
            super::boundary::FaceBoundary::PressureOutlet { .. }
        ) {
            a.add(f.owner, f.owner, d);
            open = true;
        }
    }
    if !open {
        let mean = diagonal.iter().sum::<f64>() / diagonal.len().max(1) as f64;
        for c in 0..diagonal.len() {
            a.add(c, c, 1e-10 * mean);
        }
    }
    a.finish().map_err(|e| Error::Operator(e.to_string()))
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
}

/// The shared crate's operator trait object over a borrowed CSR view.
trait LinearOperatorView: runmat_sparse_solvers::LinearOperator<f64> {}
impl<T: runmat_sparse_solvers::LinearOperator<f64>> LinearOperatorView for T {}

/// Block-diagonal preconditioner: each velocity component by its own ILU factors.
struct ComponentIlu<F> {
    n: usize,
    factors: [F; 3],
}

impl<F: Preconditioner<f64>> Preconditioner<f64> for ComponentIlu<F> {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
        for k in 0..3 {
            let range = k * self.n..(k + 1) * self.n;
            self.factors[k].apply(&r[range.clone()], &mut z[range])?;
        }
        Ok(())
    }
    fn label(&self) -> &'static str {
        "component-ilu"
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        false
    }
}

fn inner_solve(
    operator: &dyn LinearOperatorView,
    preconditioner: &dyn Preconditioner<f64>,
    rhs: &[f64],
) -> SolverResult<Vec<f64>> {
    let iterations = INNER_ITERATIONS;
    let mut x = vec![0.0; rhs.len()];
    if rhs.iter().all(|v| *v == 0.0) {
        return Ok(x);
    }
    linear::restarted_gmres(
        operator,
        Some(preconditioner),
        rhs,
        &mut x,
        INNER_REDUCTION,
        iterations,
        iterations,
        PreconditionerSide::Right,
    )?;
    Ok(x)
}

impl Preconditioner<f64> for SimplePreconditioner<'_, '_> {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
        let n = self.base_u.len();
        let fail = |e: Error| SolverError::Breakdown(e.to_string());
        // Momentum predictor on the assembled momentum block J (the implicit operator and
        // the explicit terms' Jacobian): J z_u* = r_u by inner GMRES with its ILU factors.
        let solution = inner_solve(self.momentum, self.momentum_preconditioner, &r[..3 * n])?;
        let predicted = (0..n)
            .map(|c| [solution[c], solution[n + c], solution[2 * n + c]])
            .collect::<Vec<_>>();
        // Continuity of the predictor: the exact linear flux of z_u* (affine difference).
        let trial = (0..n)
            .map(|c| add(self.base_u[c], predicted[c]))
            .collect::<Vec<_>>();
        let shifted = residual(self.p, self.frozen, &trial, self.base_p).map_err(fail)?;
        let w = self.frozen.continuity_weight;
        // The Rhie–Chow Laplacian L approximates −∂(div q)/∂p: L z_p = D z_u* − r_c.
        let rhs = (0..n)
            .map(|c| (shifted[3 * n + c] - self.base_residual[3 * n + c] - r[3 * n + c]) / w)
            .collect::<Vec<_>>();
        let correction = inner_solve(self.pressure, self.pressure_preconditioner, &rhs)?;
        // Velocity correction z_u = z_u* − G z_p / a_P.
        let force = pressure_force(
            self.p,
            &correction,
            &pressure_gradient(self.p, &correction).map_err(fail)?,
        )
        .map_err(fail)?;
        // The pressure force of an increment has no boundary constants except the outlet
        // reference, which pressure_force evaluates at the absolute value: remove it.
        let zero = pressure_force(
            self.p,
            &vec![0.0; n],
            &pressure_gradient(self.p, &vec![0.0; n]).map_err(fail)?,
        )
        .map_err(fail)?;
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
            fields.velocity_m_per_s.to_vec(),
            fields.pressure_fluctuation_pa.to_vec(),
            fields.face_volume_flux_m3_per_s.to_vec(),
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
        let frozen = freeze(p, &u, &pressure_values, &q, time)?;
        let base_residual = residual(p, &frozen, &u, &pressure_values)?;
        let laplacian = pressure_laplacian(p, &frozen.rf)?;
        let pressure_arrays = CsrArrays::of(&laplacian);
        let blocks = super::jacobian::component_operators(
            p,
            &frozen.matrix,
            &frozen.viscosity_gradients,
        )?;
        let solved = with_plan(ambient_plan(), || {
            let momentum_view = frozen.jacobian.view(&p.momentum_jacobian);
            let pressure_view = pressure_arrays.view();
            // Each velocity component is preconditioned by the ILU factors of its implicit
            // operator; the inner solve applies the full block J.
            let ilu = ComponentIlu {
                n,
                factors: [
                    Ilu0::new(blocks[0].view(), false)?,
                    Ilu0::new(blocks[1].view(), false)?,
                    Ilu0::new(blocks[2].view(), false)?,
                ],
            };
            let amg = Amg::new(pressure_view, AmgOptions::default(), control())?;
            let preconditioner = SimplePreconditioner {
                p,
                frozen: &frozen,
                momentum: &momentum_view,
                momentum_preconditioner: &ilu,
                pressure: &pressure_view,
                pressure_preconditioner: &amg,
                base_u: &u,
                base_p: &pressure_values,
                base_residual: &base_residual,
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
            let report = linear::restarted_gmres(
                &operator,
                Some(&preconditioner),
                &rhs,
                &mut step,
                LINEAR_REDUCTION.max(p.policy.linear_relative_tolerance),
                p.policy.maximum_linear_iterations,
                p.policy.gmres_restart,
                PreconditionerSide::Flexible,
            )?;
            Ok::<_, SolverError>((step, report))
        });
        let (step, report) = solved.map_err(|error| match error {
            SolverError::Cancelled => Error::Cancelled,
            other => Error::Operator(other.to_string()),
        })?;
        linear_iterations += report.iterations as u64;
        let (du, mut dp) = split(&step, n);
        // A closed component's pressure is defined up to a constant, which the Krylov step may
        // carry at any magnitude: remove it from the increment before it reaches the iterate.
        p.pressure_reference_plan.normalize(p, &mut dp)?;
        for c in 0..n {
            u[c] = add(u[c], du[c]);
            pressure_values[c] += dp[c];
        }
        p.pressure_reference_plan.normalize(p, &mut pressure_values)?;
        if u.iter().flatten().chain(&pressure_values).any(|v| !v.is_finite()) {
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
        let (worst_relative, description) = recovery::describe_worst_cell(p, &residual);
        let first = *first_worst.get_or_insert(worst_relative);
        worst = format!("{description}; it was {first:e} at the first evaluated step");
        if residual.accepted(p.policy.physical_relative_tolerance) {
            let pressure_pa = pressure_values
                .iter()
                .zip(&p.pressure_reference_pa)
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
                velocity_m_per_s: u,
                pressure_pa,
                pressure_fluctuation_pa: pressure_values,
                pressure_reference_pa: p.pressure_reference_pa.clone(),
                face_volume_flux_m3_per_s: q,
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

