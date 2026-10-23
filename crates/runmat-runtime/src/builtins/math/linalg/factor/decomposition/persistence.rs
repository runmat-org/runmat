//! One canonical recipe/DATA classification authority for persistence owners.
use super::*;
use crate::context::RuntimeContext;
use runmat_value::{ExactHostSparseClass, NumericFactorCandidate, NumericFactorKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecompositionPersistenceKind {
    Factor(NumericFactorKind),
    Diagonal,
    Triangular,
}

/// O(1) identity, parsed recipe and factor-header coherence. Content validation
/// of a structural recipe belongs to the async owner below, before commit.
pub(crate) fn decomposition_persistence_kind(
    object: &ObjectInstance,
) -> BuiltinResult<DecompositionPersistenceKind> {
    if !object.is_class(DECOMPOSITION_CLASS_IDENTITY) {
        return Err(invalid(
            "numeric factor requires the canonical decomposition class",
        ));
    }
    let spec = object_to_spec(&Value::Object(object.clone()))?;
    let (m, n) = matrix_dims(&spec.matrix);
    let class = expected_factor_class(&spec.matrix)?;
    let shape = match &spec.matrix {
        Value::Tensor(t) => Some(t.shape()),
        Value::ComplexTensor(t) => Some(t.shape()),
        Value::LogicalArray(t) => Some(t.shape.as_ref()),
        _ => None,
    };
    if let Some(shape) = shape {
        validate_matrix_shape(shape)?;
    }
    if spec.triangle.is_some()
        && !matches!(
            spec.decomp_type,
            DecompositionType::Chol | DecompositionType::Triangular
        )
    {
        return Err(invalid(
            "decomposition triangle is incompatible with its Type",
        ));
    }
    let kind = match spec.decomp_type {
        DecompositionType::Lu => DecompositionPersistenceKind::Factor(NumericFactorKind::Lu),
        DecompositionType::Chol => {
            DecompositionPersistenceKind::Factor(NumericFactorKind::Cholesky)
        }
        DecompositionType::Qr => DecompositionPersistenceKind::Factor(NumericFactorKind::Qr),
        DecompositionType::Cod => DecompositionPersistenceKind::Factor(NumericFactorKind::Cod),
        DecompositionType::Diagonal => DecompositionPersistenceKind::Diagonal,
        DecompositionType::Triangular => DecompositionPersistenceKind::Triangular,
        DecompositionType::Auto => {
            return Err(invalid(
                "persisted decomposition must have a selected factor type",
            ))
        }
    };
    match (kind, spec.factor_data.as_ref()) {
        (DecompositionPersistenceKind::Factor(kind), Some(data))
            if data.kind() == kind
                && data.dimensions() == (m, n)
                && data.storage_class() == class =>
        {
            let dense = matches!(
                data.parts(),
                NumericFactorCandidate::DenseLu { .. }
                    | NumericFactorCandidate::DenseCholesky { .. }
                    | NumericFactorCandidate::DenseQr(_)
                    | NumericFactorCandidate::DenseCod { .. }
            );
            if !matches!(data.parts(), NumericFactorCandidate::Empty { .. })
                && dense == matches!(&spec.matrix, Value::SparseTensor(_))
            {
                return Err(invalid(
                    "retained factor storage does not match coefficient storage",
                ));
            }
        }
        (
            DecompositionPersistenceKind::Diagonal | DecompositionPersistenceKind::Triangular,
            None,
        ) if m == n => {}
        _ => {
            return Err(invalid(
                "decomposition recipe and retained factor DATA do not agree",
            ))
        }
    }
    if spec.is_sparse != matches!(&spec.matrix, Value::SparseTensor(_)) {
        return Err(invalid(
            "decomposition IsSparse does not match its coefficient matrix",
        ));
    }
    // MatrixSize is the public Double projection, including rounded empty
    // axes above 2^53. Exact recipe shape and sealed DATA dimensions above,
    // never this observational projection, prove physical solve geometry.
    let (rows, columns) = if spec.is_conjugate_transposed {
        (n, m)
    } else {
        (m, n)
    };
    let size = object
        .properties()
        .get(MATRIX_SIZE_FIELD)
        .and_then(|v| {
            if let Value::Tensor(t) = v {
                Some(t)
            } else {
                None
            }
        })
        .ok_or_else(|| invalid("decomposition MatrixSize must be a double row vector"))?;
    if size.shape() != [1, 2]
        || size
            .as_f64_slice()
            .is_none_or(|v| v != [rows as f64, columns as f64])
    {
        return Err(invalid(
            "decomposition MatrixSize does not match its oriented coefficient geometry",
        ));
    }
    if object
        .properties()
        .get(DATATYPE_FIELD)
        .map(text_from_value)
        .transpose()?
        .flatten()
        .as_deref()
        != Some(matrix_datatype(&spec.matrix))
    {
        return Err(invalid(
            "decomposition Datatype does not match its original coefficient class",
        ));
    }
    Ok(kind)
}

pub(crate) async fn validate_factor_object_binding(
    object: &ObjectInstance,
    runtime: &RuntimeContext,
) -> BuiltinResult<DecompositionPersistenceKind> {
    let kind = decomposition_persistence_kind(object)?;
    let spec = object_to_spec(&Value::Object(object.clone()))?;
    let facts = super::native::inspect_original(&spec.matrix, false, runtime).await?;
    let real = if spec.triangle.is_some() {
        super::native::selected_real(&spec.matrix, spec.triangle, runtime).await?
    } else {
        facts.real
    };
    if spec.source_is_real != Some(real)
        || object
            .properties()
            .get(IS_REAL_FIELD)
            .map(|v| bool_scalar(v, IS_REAL_FIELD))
            .transpose()?
            != Some(real && spec.scale.1 == 0.0)
    {
        return Err(invalid(
            "decomposition IsReal does not match its selected original coefficients and scale",
        ));
    }
    if matches!(
        kind,
        DecompositionPersistenceKind::Diagonal | DecompositionPersistenceKind::Triangular
    ) {
        let valid = match kind {
            DecompositionPersistenceKind::Diagonal => facts.diagonal,
            DecompositionPersistenceKind::Triangular => {
                spec.triangle.is_some() || facts.upper || facts.lower
            }
            _ => unreachable!(),
        };
        if !valid {
            return Err(invalid(
                "structural decomposition recipe does not match its original coefficients",
            ));
        }
    } else {
        super::native::check_factor_recipe(&spec.matrix, spec.triangle, runtime).await?;
    }
    Ok(kind)
}

pub(super) fn expected_factor_class(matrix: &Value) -> BuiltinResult<ExactHostSparseClass> {
    let native = |dtype| {
        if matches!(dtype, NumericDType::F32 | NumericDType::F64) {
            dtype
        } else {
            NumericDType::F64
        }
    };
    match matrix {
        Value::Tensor(t) => Ok(ExactHostSparseClass::Numeric(native(t.numeric_dtype()))),
        Value::ComplexTensor(t) => Ok(ExactHostSparseClass::Complex(native(t.numeric_dtype()))),
        Value::LogicalArray(_) => Ok(ExactHostSparseClass::Numeric(NumericDType::F64)),
        Value::SparseTensor(t) => Ok(if t.as_complex_f32_slice().is_some() {
            ExactHostSparseClass::Complex(NumericDType::F32)
        } else if t.as_complex_f64_slice().is_some() {
            ExactHostSparseClass::Complex(NumericDType::F64)
        } else {
            ExactHostSparseClass::Numeric(native(t.numeric_dtype().unwrap_or(NumericDType::F64)))
        }),
        _ => Err(invalid(
            "decomposition recipe requires a normalized numeric matrix",
        )),
    }
}
