//! Exact sparse real representation of the assembled complex pressure operator.
use super::*;
use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler, RealSparseMatrix};

pub(super) fn materialize(
    assembly: &AcousticAssembly,
) -> Result<RealSparseMatrix, AcousticHarmonicError> {
    let n = assembly.projection.nodes.len();
    let mut result = RealSparseAssembler::square(
        2 * n,
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    );
    for (matrix, scale, imaginary) in [
        (&assembly.stiffness, 1.0, false),
        (
            &assembly.mass,
            -assembly.derivative_rate.canonical_value()
                * assembly.derivative_rate.canonical_value(),
            false,
        ),
        (
            &assembly.impedance,
            assembly.derivative_rate.canonical_value(),
            true,
        ),
    ] {
        matrix.with_canonical_storage(|matrix| -> Result<(), AcousticHarmonicError> {
            for contribution in matrix.contributions() {
                result.add_contribution(contribution.clone());
            }
            for row in 0..n {
                if crate::is_fea_cancelled() {
                    return Err(AcousticHarmonicError::Cancelled);
                }
                for entry in matrix.pattern().row_range(row) {
                    let column = matrix.pattern().column(entry);
                    let value = scale * matrix.values()[entry];
                    if imaginary {
                        result.add(row, n + column, -value);
                        result.add(n + row, column, value);
                    } else {
                        result.add(row, column, value);
                        result.add(n + row, n + column, value);
                    }
                }
            }
            Ok(())
        })?;
    }
    result
        .finish()
        .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))
}
