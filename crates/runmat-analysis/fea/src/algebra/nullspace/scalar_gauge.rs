//! Constant-mode gauge lowering. Original equations/RHS are never repaired.
use super::{connected_components, NullspaceEvidenceError as Error, RealSparseMatrix};

pub(crate) fn prepare_scalar_component_gauges(
    matrix: &RealSparseMatrix,
    rhs: &[f64],
    roots: &[usize],
    assembly_roundoff: &[f64],
) -> Result<(RealSparseMatrix, Vec<f64>), Error> {
    let n = matrix.rows();
    if matrix.columns() != n
        || assembly_roundoff.len() != n
        || assembly_roundoff.iter().any(|x| !x.is_finite() || *x < 0.0)
        || rhs.len() != n
        || rhs.iter().any(|x| !x.is_finite())
        || roots.iter().any(|r| *r >= n)
    {
        return Err(Error::DimensionMismatch);
    }
    let mut neighbors = vec![Vec::new(); n];
    let mut row_sum = vec![0.0; n];
    let mut row_scale = vec![0.0; n];
    for r in 0..n {
        for entry in matrix.pattern().row_offsets()[r] as usize
            ..matrix.pattern().row_offsets()[r + 1] as usize
        {
            let c = matrix.pattern().column_indices()[entry] as usize;
            let value = matrix.values()[entry];
            if r != c && value != 0.0 {
                if value > 0.0 {
                    return Err(Error::IncompatibleOperator);
                }
                let start = matrix.pattern().row_offsets()[c] as usize;
                let end = matrix.pattern().row_offsets()[c + 1] as usize;
                let j = matrix.pattern().column_indices()[start..end]
                    .binary_search(&(r as u64))
                    .map_err(|_| Error::IncompatibleOperator)?;
                if matrix.values()[start + j].to_bits() != value.to_bits() {
                    return Err(Error::IncompatibleOperator);
                }
                neighbors[r].push(c);
            }
            row_sum[r] += value;
            row_scale[r] += value.abs();
        }
    }
    let mut pins = vec![false; n];
    for &r in roots {
        if std::mem::replace(&mut pins[r], true) {
            return Err(Error::InvalidEvidence);
        }
    }
    for component in connected_components(&neighbors)? {
        let count = component.iter().filter(|c| pins[**c]).count();
        if count > 1 {
            return Err(Error::InvalidEvidence);
        }
        if count == 0 {
            continue;
        }
        let mut sum = 0.0;
        let mut scale = 0.0;
        for &c in &component {
            if !row_sum[c].is_finite()
                || !row_scale[c].is_finite()
                || row_sum[c].abs() > 128.0 * f64::EPSILON * row_scale[c]
            {
                return Err(Error::NullModeViolation);
            }
            sum += rhs[c];
            scale += rhs[c].abs();
        }
        // Fredholm compatibility is checked on every original row. Never subtract
        // the incompatible mean source or hide it by replacing the anchor row.
        let source_roundoff = component.iter().map(|c| assembly_roundoff[*c]).sum::<f64>();
        if !sum.is_finite()
            || !scale.is_finite()
            || !source_roundoff.is_finite()
            || sum.abs() > 128.0 * f64::EPSILON * scale + source_roundoff
        {
            return Err(Error::IncompatibleConstantModeSource);
        }
    }
    let prepared = matrix
        .with_identity_constraints(&pins)
        .map_err(|_| Error::DimensionMismatch)?;
    let mut prepared_rhs = rhs.to_vec();
    for &root in roots {
        prepared_rhs[root] = 0.0;
    }
    Ok((prepared, prepared_rhs))
}

#[cfg(test)]
mod tests;
