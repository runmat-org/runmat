use super::{CanonicalFluidError as Error, CanonicalFluidProblem};
use crate::{
    algebra::{
        spmv, ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties,
        OperatorRole, RealSparseAssembler, RealSparseMatrix, SparsePattern,
    },
    solve::krylov::{
        restarted_gmres,
        right_ilu::{SparseRightIlu, SparseRightIluEnvelope, SparseRightIluError},
        GmresPolicy,
    },
};

pub(super) fn assembler(
    p: &CanonicalFluidProblem<'_>,
    label: &str,
) -> Result<RealSparseAssembler, Error> {
    let n = p.topology.cells.len();
    let mut offsets = Vec::with_capacity(n + 1);
    let mut columns = Vec::with_capacity(n * 5);
    offsets.push(0);
    for (c, cell) in p.topology.cells.iter().enumerate() {
        let mut row = vec![c];
        for incidence in cell.faces {
            let face = &p.topology.faces[incidence.face_index];
            if face.owner != c {
                row.push(face.owner);
            } else if let Some(other) = face.neighbor {
                row.push(other);
            }
        }
        row.sort_unstable();
        row.dedup();
        columns.extend(row.into_iter().map(|v| v as u64));
        offsets.push(columns.len() as u64);
    }
    let pattern = SparsePattern::try_new(n as u64, n as u64, offsets, columns)
        .map_err(|e| Error::Operator(e.to_string()))?;
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

pub(super) fn solve(
    p: &CanonicalFluidProblem<'_>,
    a: &RealSparseMatrix,
    b: &[f64],
    current: &[f64],
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
    let actual = SparseRightIluEnvelope::derive(a)
        .ok_or(Error::Resource("actual ILU component envelope"))?;
    let planned = SparseRightIluEnvelope::for_counts(
        p.topology.cells.len() as u64,
        p.reservation.sparse_entries,
        5,
    )
    .ok_or(Error::Resource("planned ILU component envelope"))?;
    if actual.resident_bytes_upper_bound > planned.resident_bytes_upper_bound
        || actual.factor_scalar_operations > planned.factor_scalar_operations
        || actual.application_scalar_operations > planned.application_scalar_operations
    {
        return Err(Error::Resource("ILU component exceeds original admission"));
    }
    let preconditioner = SparseRightIlu::factor(a, crate::is_fea_cancelled).map_err(ilu_error)?;
    let result = restarted_gmres(
        &defect,
        &vec![0.0; b.len()],
        GmresPolicy {
            relative_tolerance,
            restart: p.policy.gmres_restart,
            maximum_iterations: p.policy.maximum_linear_iterations,
        },
        |v| {
            let x = preconditioner
                .apply(v, crate::is_fea_cancelled)
                .map_err(ilu_error)?;
            spmv(a, &x).map_err(|e| Error::Operator(e.to_string()))
        },
        |x, y| {
            let sum = crate::algebra::dot_canonical(x, y);
            if sum.is_finite() {
                Ok(sum)
            } else {
                Err(Error::Invalid("nonfinite Krylov dot"))
            }
        },
        crate::is_fea_cancelled,
    )?
    .ok_or(Error::Invalid("GMRES returned no correction inventory"))?;
    if result.cancelled {
        return Err(Error::Cancelled);
    }
    if std::env::var_os("QFLUID_DEBUG_LINEAR").is_some() {
        eprintln!(
            "qfluid-linear n={} defect={:e} floor={:e} tol={:e} its={} rel={:?} conv={}",
            b.len(),
            defect_norm,
            rounding_floor,
            relative_tolerance,
            result.iterations,
            result.relative_residual,
            result.converged
        );
    }
    if !result.converged || result.breakdown {
        return Err(Error::LinearNotConverged {
            iterations: result.iterations,
            relative_residual: result.relative_residual,
            breakdown: result.breakdown,
        });
    }
    let correction = preconditioner
        .apply(&result.solution, crate::is_fea_cancelled)
        .map_err(ilu_error)?;
    let x = current
        .iter()
        .zip(correction)
        .map(|(x, d)| x + d)
        .collect::<Vec<_>>();
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("nonfinite linear correction"));
    }
    Ok((x, u64::from(result.iterations)))
}

fn ilu_error(error: SparseRightIluError) -> Error {
    match error {
        SparseRightIluError::Cancelled => Error::Cancelled,
        other => Error::Operator(other.to_string()),
    }
}
