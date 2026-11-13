use super::{CanonicalFluidError as Error, CanonicalFluidProblem};
use crate::{
    algebra::{
        spmv, ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties,
        OperatorRole, RealSparseAssembler, RealSparseMatrix, SparsePattern,
    },
    solve::shared_sparse::{ambient_plan, control, with_plan, CsrArrays},
};
use runmat_sparse_solvers::{
    krylov::{gmres, GmresOptions, KrylovOptions, PreconditionerSide, StopFlag},
    precond::{Amg, AmgOptions, Ilu0},
    Preconditioner, SolverError,
};

pub(super) fn cell_operator_pattern(
    topology: &crate::topology::FiniteVolumeTopology,
) -> Result<SparsePattern, Error> {
    let n = topology.cells.len();
    let mut offsets = Vec::with_capacity(n + 1);
    let mut columns = Vec::with_capacity(n * 5);
    offsets.push(0);
    let mut row = Vec::new();
    for (c, cell) in topology.cells.iter().enumerate() {
        row.clear();
        row.push(c);
        for incidence in cell.faces {
            let face = &topology.faces[incidence.face_index];
            if face.owner != c {
                row.push(face.owner);
            } else if let Some(other) = face.neighbor {
                row.push(other);
            }
        }
        row.sort_unstable();
        row.dedup();
        columns.extend(row.iter().map(|&v| v as u64));
        offsets.push(columns.len() as u64);
    }
    SparsePattern::try_new(n as u64, n as u64, offsets, columns)
        .map_err(|e| Error::Operator(e.to_string()))
}

pub(super) fn assembler(
    p: &CanonicalFluidProblem<'_>,
    label: &str,
) -> Result<RealSparseAssembler, Error> {
    let pattern = p.operator_pattern.clone();
    let mut assembler = RealSparseAssembler::with_pattern(
        pattern,
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    )
    .map_err(|e| Error::Operator(e.to_string()))?;
    assembler.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Fluid,
        contribution_id: ContributionId(label.into()),
    });
    Ok(assembler)
}

/// The two cell-centred linear systems of the canonical flow solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LinearSystem {
    /// Momentum predictor of the segregated solve: convection-diffusion, preconditioned by
    /// ILU(0).
    Momentum,
    /// Pressure correction: a Laplacian-like system, preconditioned by smoothed-aggregation AMG.
    PressureCorrection,
}

/// Conservative preconditioner reservation for one cell system (`n` rows, `nnz` entries):
/// ILU(0) on the pattern, or an AMG hierarchy sized as operator complexity 3 plus
/// prolongation/restriction of `nnz` entries each, smoother vectors and a sparse coarse factor
/// of at most `AmgOptions::coarse_size` rows. A built hierarchy is admitted by the bytes it
/// actually holds ([`Amg::resident_bytes`]), not by how those bytes split between levels.
pub(super) struct PreconditionerEnvelope {
    pub resident_bytes_upper_bound: u64,
    pub factor_scalar_operations: u64,
    pub application_scalar_operations: u64,
}

/// AMG for a velocity component's implicit operator. A weak strength threshold keeps the
/// couplings across thin (graded) cells in the aggregates, which ILU(0) and the default
/// threshold leave unresolved.
pub(super) fn momentum_amg_options() -> AmgOptions {
    AmgOptions {
        strength_threshold: 0.1,
        ..AmgOptions::default()
    }
}

/// An AMG hierarchy must stay inside the admitted per-system preconditioner envelope.
pub(super) fn within_admission(
    amg: &Amg,
    planned: &PreconditionerEnvelope,
) -> Result<(), SolverError> {
    let required = amg.resident_bytes();
    tracing::debug!(
        required,
        envelope = planned.resident_bytes_upper_bound,
        "fluid AMG hierarchy resident bytes"
    );
    if required > planned.resident_bytes_upper_bound {
        return Err(SolverError::OverBudget {
            required,
            budget: planned.resident_bytes_upper_bound,
        });
    }
    Ok(())
}

impl PreconditionerEnvelope {
    pub(super) fn for_counts(n: u64, nnz: u64) -> Option<Self> {
        let coarse = AmgOptions::default().coarse_size as u64;
        let coarse_bytes = runmat_sparse_solvers::estimate::direct_reservation_bytes(
            usize::try_from(coarse).ok()?,
            usize::try_from(coarse.checked_mul(27)?).ok()?,
            true,
        );
        let hierarchy = nnz
            .checked_mul(3)?
            .checked_add(nnz.checked_mul(2)?)?
            .checked_mul(24)?
            .checked_add(n.checked_mul(3 * 8 * 8)?)?;
        let ilu = nnz.checked_mul(24)?.checked_add(n.checked_mul(24)?)?;
        Some(Self {
            resident_bytes_upper_bound: hierarchy.max(ilu).checked_add(coarse_bytes)?,
            factor_scalar_operations: nnz.checked_mul(400)?,
            application_scalar_operations: nnz.checked_mul(40)?,
        })
    }
}

pub(super) fn solve(
    p: &CanonicalFluidProblem<'_>,
    a: &RealSparseMatrix,
    b: &[f64],
    current: &[f64],
    system: LinearSystem,
) -> Result<(Vec<f64>, u64), Error> {
    if current.len() != b.len()
        || a.rows() != b.len()
        || a.columns() != current.len()
        || current.iter().chain(b).any(|v| !v.is_finite())
    {
        return Err(Error::Invalid("linear correction inventory"));
    }
    // Solve A δ = b − A x, then retain x + δ. The relative Krylov target
    // applies to the current defect, not the much larger relaxed RHS. Starting
    // each outer solve from zero can discard precisely the low-Re correction
    // needed by the independently evaluated original force-balance gate.
    let action = spmv(a, current).map_err(|e| Error::Operator(e.to_string()))?;
    let defect = b
        .iter()
        .zip(&action)
        .map(|(b, a)| b - a)
        .collect::<Vec<_>>();
    // Near outer convergence the defect approaches the rounding level of the assembled
    // system; a purely relative target on it would ask Krylov for digits below that level.
    // The target is the larger of the relative defect reduction and a rounding floor of the
    // original right-hand side and operator action.
    let rounding_floor = 64.0
        * f64::EPSILON
        * b.iter()
            .zip(&action)
            .fold(0.0f64, |sum, (b, a)| sum.hypot(b.abs() + a.abs()));
    drop(action);
    let defect_norm = defect.iter().fold(0.0f64, |sum, v| sum.hypot(*v));
    if defect_norm <= rounding_floor {
        return Ok((current.to_vec(), 0));
    }
    let relative_tolerance = p
        .policy
        .linear_relative_tolerance
        .max(rounding_floor / defect_norm);
    let planned = PreconditionerEnvelope::for_counts(
        p.topology.cells.len() as u64,
        p.reservation.sparse_entries,
    )
    .ok_or(Error::Resource("planned preconditioner envelope"))?;
    let arrays = CsrArrays::of(a);
    let view = arrays.view();
    let maximum_iterations = p.policy.maximum_linear_iterations;
    let restart = p.policy.gmres_restart;
    let solved = with_plan(ambient_plan(), || {
        let preconditioner: Box<dyn Preconditioner<f64>> = match system {
            LinearSystem::Momentum => Box::new(Ilu0::new(view, false)?),
            LinearSystem::PressureCorrection => {
                let amg = Amg::new(view, AmgOptions::default(), control())?;
                within_admission(&amg, &planned)?;
                Box::new(amg)
            }
        };
        let mut correction = vec![0.0; b.len()];
        let report = gmres(
            &view,
            Some(preconditioner.as_ref()),
            &defect,
            &mut correction,
            &KrylovOptions {
                rtol: relative_tolerance,
                max_iterations: maximum_iterations,
                ..KrylovOptions::default()
            },
            &GmresOptions {
                restart,
                side: PreconditionerSide::Right,
                ..GmresOptions::default()
            },
            control(),
        )?;
        Ok::<_, SolverError>((correction, report))
    });
    let (correction, report) = solved.map_err(|error| match error {
        SolverError::Cancelled => Error::Cancelled,
        SolverError::OverBudget { .. } => {
            Error::Resource("preconditioner exceeds original admission")
        }
        other => Error::Operator(other.to_string()),
    })?;
    if !report.converged() {
        return Err(Error::LinearNotConverged {
            iterations: u32::try_from(report.iterations).unwrap_or(u32::MAX),
            relative_residual: Some(report.relative_residual),
            breakdown: report.flag == StopFlag::Breakdown,
        });
    }
    let x = current
        .iter()
        .zip(correction)
        .map(|(x, d)| x + d)
        .collect::<Vec<_>>();
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("nonfinite linear correction"));
    }
    Ok((x, report.iterations as u64))
}

/// The CFD entry point to the shared restarted GMRES: every fluid Krylov solve goes through
/// here, so a change of the shared solver interface is made in one place.
#[allow(clippy::too_many_arguments)]
pub(super) fn restarted_gmres(
    workspace: &mut runmat_sparse_solvers::krylov::GmresWorkspace<f64>,
    operator: &dyn runmat_sparse_solvers::LinearOperator<f64>,
    preconditioner: Option<&dyn Preconditioner<f64>>,
    rhs: &[f64],
    x: &mut [f64],
    relative_tolerance: f64,
    maximum_iterations: usize,
    restart: usize,
    side: PreconditionerSide,
) -> Result<runmat_sparse_solvers::krylov::KrylovReport, SolverError> {
    runmat_sparse_solvers::krylov::gmres_with_workspace(
        workspace,
        operator,
        preconditioner,
        rhs,
        x,
        &KrylovOptions {
            rtol: relative_tolerance,
            max_iterations: maximum_iterations,
            return_best_iterate: true,
            ..KrylovOptions::default()
        },
        &GmresOptions {
            restart,
            side,
            ..GmresOptions::default()
        },
        control(),
    )
}
