use super::{CanonicalFluidError as Error, CanonicalFluidProblem};
use crate::{
    algebra::{
        spmv, ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties,
        OperatorRole, RealSparseAssembler, RealSparseMatrix, SparsePattern,
    },
    solve::krylov::{restarted_gmres, right_diagonal::DiagonalRightPreconditioner, GmresPolicy},
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

pub(super) fn diagonal(a: &RealSparseMatrix) -> Result<Vec<f64>, Error> {
    (0..a.rows())
        .map(|r| {
            let start = a.pattern().row_offsets()[r] as usize;
            let end = a.pattern().row_offsets()[r + 1] as usize;
            let k = a.pattern().column_indices()[start..end]
                .binary_search(&(r as u64))
                .map_err(|_| Error::Invalid("missing momentum diagonal"))?;
            let value = a.values()[start + k];
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::Invalid("nonpositive finite-volume diagonal"));
            }
            Ok(value)
        })
        .collect()
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
    drop(action);
    let d = diagonal(a)?;
    let preconditioner =
        DiagonalRightPreconditioner::new(&d).map_err(|e| Error::Operator(e.to_string()))?;
    let result = restarted_gmres(
        &defect,
        &vec![0.0; b.len()],
        GmresPolicy {
            relative_tolerance: p.policy.linear_relative_tolerance,
            restart: p.policy.gmres_restart,
            maximum_iterations: p.policy.maximum_linear_iterations,
        },
        |v| {
            let x = preconditioner
                .apply(v)
                .map_err(|e| Error::Operator(e.to_string()))?;
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
        || false,
    )?
    .ok_or(Error::NotConverged)?;
    if !result.converged || result.cancelled || result.breakdown {
        return Err(Error::NotConverged);
    }
    let correction = preconditioner
        .apply(&result.solution)
        .map_err(|e| Error::Operator(e.to_string()))?;
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
