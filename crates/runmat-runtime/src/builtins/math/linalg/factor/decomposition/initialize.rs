//! One selected constructor/import factorization, never a method-time fallback.
use super::*;
use crate::builtins::math::linalg::ops::{
    dense_factor::{factor_dense, DenseFactorOptions},
    sparse_factor::{factor_sparse, SelectedTriangle},
};
use crate::context::RuntimeContext;
use runmat_value::NumericFactorKind;

pub(super) async fn initialize(
    mut spec: DecompositionSpec,
    runtime: &RuntimeContext,
) -> BuiltinResult<DecompositionSpec> {
    ensure_nonfloating_object_enabled(&spec)?;
    if spec.triangle.is_some()
        && !matches!(
            spec.decomp_type,
            DecompositionType::Triangular | DecompositionType::Chol
        )
    {
        return Err(invalid(
            "decomposition: triangularFlag is supported only with types 'triangular' and 'chol'",
        ));
    }
    let (m, n) = matrix_dims(&spec.matrix);
    let facts = native::inspect_original(
        &spec.matrix,
        spec.decomp_type == DecompositionType::Chol && spec.triangle.is_none(),
        runtime,
    )
    .await?;
    spec.source_is_real = Some(if spec.triangle.is_some() {
        native::selected_real(&spec.matrix, spec.triangle, runtime).await?
    } else {
        facts.real
    });
    if spec.decomp_type == DecompositionType::Auto {
        spec.decomp_type = if m != n {
            DecompositionType::Qr
        } else if facts.diagonal {
            DecompositionType::Diagonal
        } else if facts.upper || facts.lower {
            DecompositionType::Triangular
        } else {
            DecompositionType::Lu
        };
    }
    if matches!(
        spec.decomp_type,
        DecompositionType::Lu
            | DecompositionType::Chol
            | DecompositionType::Triangular
            | DecompositionType::Diagonal
    ) && m != n
    {
        return Err(invalid(
            "decomposition: selected factor type requires a square matrix",
        ));
    }
    match spec.decomp_type {
        DecompositionType::Diagonal => {
            if !facts.diagonal {
                return Err(invalid(
                    "decomposition: type 'diagonal' requires a diagonal matrix",
                ));
            }
            spec.factor_data = None;
            return Ok(spec);
        }
        DecompositionType::Triangular => {
            if spec.triangle.is_none() && !facts.upper && !facts.lower {
                return Err(invalid(
                    "decomposition: type 'triangular' requires a triangular matrix",
                ));
            }
            if spec.triangle.is_none() {
                spec.triangle = Some(if facts.lower {
                    TriangularFlag::Lower
                } else {
                    TriangularFlag::Upper
                });
            }
            spec.factor_data = None;
            return Ok(spec);
        }
        DecompositionType::Chol => {
            if crate::builtins::common::validation::is_typed_complex_integer(&spec.matrix) {
                return Err(invalid("decomposition: type 'chol' requires a positive definite coefficient matrix; paired complex integer storage is not admitted"));
            }
            if spec.triangle.is_none() && !facts.hermitian {
                return Err(invalid(
                    "decomposition: type 'chol' requires a symmetric or Hermitian matrix",
                ));
            }
        }
        _ => {}
    }
    let kind = match spec.decomp_type {
        DecompositionType::Lu => NumericFactorKind::Lu,
        DecompositionType::Chol => NumericFactorKind::Cholesky,
        DecompositionType::Qr => NumericFactorKind::Qr,
        DecompositionType::Cod => NumericFactorKind::Cod,
        _ => unreachable!("selected factored type"),
    };
    let matrix =
        native::prepare_factor_matrix(&spec.matrix, spec.decomp_type, spec.triangle, runtime)
            .await?;
    spec.factor_data = Some(if let Value::SparseTensor(matrix) = &matrix {
        factor_sparse(
            matrix,
            kind,
            spec.rank_tolerance,
            spec.triangle != Some(TriangularFlag::Upper),
            runtime,
        )
        .await
        .map_err(|e| native::retained_error(e, runtime))?
    } else {
        factor_dense(
            &matrix,
            DenseFactorOptions {
                kind,
                rank_tolerance: spec.rank_tolerance,
                triangle: spec.triangle.map(|t| match t {
                    TriangularFlag::Upper => SelectedTriangle::Upper,
                    TriangularFlag::Lower => SelectedTriangle::Lower,
                }),
            },
            runtime,
        )
        .await
        .map_err(|e| native::dense_error(e, runtime))?
    });
    Ok(spec)
}

/// MAT's supported property recipe is reconstructed exactly once under the
/// importing invocation owner. Existing validated DATA is never refactored.
pub(crate) async fn restore_decomposition_factor_object(
    object: ObjectInstance,
    runtime: &RuntimeContext,
) -> BuiltinResult<ObjectInstance> {
    if !object.is_class(DECOMPOSITION_CLASS_IDENTITY) {
        return Err(invalid("decomposition import requires its canonical class"));
    }
    ensure_decomposition_class_registered();
    if object.numeric_factor_data().is_some() {
        validate_factor_object_binding(&object, runtime).await?;
        return Ok(object);
    }
    let spec = object_to_spec(&Value::Object(object))?;
    let spec = initialize(spec, runtime).await.map_err(native::terminal)?;
    let root = publication::publish(&spec, runtime).map_err(native::terminal)?;
    let Value::Object(object) = root.value() else {
        unreachable!("canonical object publisher")
    };
    Ok(object.clone())
}
