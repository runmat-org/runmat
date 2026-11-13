//! Structural recipes use the existing native substitution kernel, not LU.
use super::*;
use crate::builtins::math::linalg::ops::{
    linear_operator::ProductOrientation,
    triangular_operator::{
        self, BorrowedTriangleGeometry, Triangle, TriangularOperatorError, TriangularScalar,
        TriangularSolvePolicy,
    },
};
use crate::context::host_allocation::HostTypedWorkspace;

pub(super) async fn solve<F: DenseScalar + TriangularScalar>(
    spec: &DecompositionSpec,
    geometry: FactorRhsGeometry,
    adjoint: bool,
    scale: (f64, f64),
    runtime: &RuntimeContext,
    mut read: impl FnMut(usize) -> Option<F>,
    mut sink: impl FnMut(usize, F) -> Result<(), HostAllocationError>,
) -> BuiltinResult<RetainedSolveStatus> {
    // The public driver already proved the immutable physical recipe once
    // under this invocation, before admitting any channel's private output.
    let (n, _) = matrix_dims(&spec.matrix);
    if n.checked_mul(geometry.columns) != Some(geometry.len) {
        return Err(invalid("decomposition: structural RHS extent mismatch"));
    }
    let bytes =
        HostTypedWorkspace::<F>::quoted_bytes(n).map_err(|e| native::allocation(e, runtime))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(bytes, 0).map_err(|e| native::allocation(e, runtime))?,
    )
    .map_err(|e| native::allocation(e, runtime))?;
    let mut work = OperatorWork::new(runtime).map_err(|e| work_error(e, runtime))?;
    work.flush().await.map_err(|e| work_error(e, runtime))?;
    let mut managed = if let Some(r) = &receipt {
        Some(
            r.temporary_typed::<F>(runtime, n)
                .map_err(|e| native::allocation(e, runtime))?,
        )
    } else {
        None
    };
    let mut ordinary = Vec::new();
    if managed.is_none() {
        ordinary.try_reserve_exact(n).map_err(|_| {
            native::allocation(
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
                runtime,
            )
        })?;
        for _ in 0..n {
            ordinary.push(F::default());
            work.tick().await.map_err(|e| work_error(e, runtime))?;
        }
    }
    let rhs = if let Some(v) = &mut managed {
        v.as_mut_slice()
    } else {
        ordinary.as_mut_slice()
    };
    let triangle = if spec.triangle == Some(TriangularFlag::Lower) {
        Triangle::Lower
    } else {
        Triangle::Upper
    };
    let source = if let Value::SparseTensor(t) = &spec.matrix {
        Some((t.col_ptrs.as_ref(), t.row_indices.as_ref()))
    } else {
        None
    };
    let geometry_source = BorrowedTriangleGeometry {
        order: n,
        triangle,
        unit_diagonal: false,
        diagonal_only: spec.decomp_type == DecompositionType::Diagonal,
        csc: source,
    };
    let coefficient = |index| {
        let (real, imag) = native::coefficient(&spec.matrix, index).ok()?;
        let coefficient = F::from_parts(
            native::checked_double(real).ok()?,
            native::checked_double(imag).ok()?,
        )?;
        // The existing indexed substitution conjugates a source coefficient
        // for adjoint. Conjugating the scalar here makes the realized entry
        // scale*conj(source), never conj(scale*source).
        coefficient.scaled_complex(if adjoint { complex_conj(scale) } else { scale })
    };
    let mut singular = false;
    for column in 0..geometry.columns {
        singular |= triangular_operator::substitute_indexed(
            geometry_source,
            if adjoint {
                ProductOrientation::Adjoint
            } else {
                ProductOrientation::Normal
            },
            TriangularSolvePolicy::PublicDecompositionIeee,
            coefficient,
            |row| read(column * n + row),
            rhs,
            &mut work,
        )
        .await
        .map_err(|e| error(e, runtime))?;
        for (row, &value) in rhs.iter().enumerate() {
            sink(column * n + row, value).map_err(|e| native::allocation(e, runtime))?;
            work.tick().await.map_err(|e| work_error(e, runtime))?;
        }
    }
    work.flush().await.map_err(|e| work_error(e, runtime))?;
    Ok(RetainedSolveStatus {
        singular,
        deficient_rank: None,
    })
}
fn error(error: TriangularOperatorError, runtime: &RuntimeContext) -> RuntimeError {
    use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
    native::terminal(match error {
        TriangularOperatorError::Admission(e)
        | TriangularOperatorError::Operator(SparseOperatorError::Admission(e)) => {
            native::allocation(e, runtime)
        }
        TriangularOperatorError::Execution(e) => e.into_runtime_error_in_context(Some(runtime)),
        other => build_runtime_error(format!("decomposition: {other}"))
            .with_builtin(NAME)
            .with_identifier(
                ERROR_INVALID_INPUT
                    .identifier
                    .unwrap_or("RunMat:decomposition:InvalidInput"),
            )
            .with_source(other)
            .build(),
    })
}
