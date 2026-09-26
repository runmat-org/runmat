//! Class and underlying-type predicates.

use super::{
    handle_integer_type, handle_is_logical, validate_resident_metadata, value_has_logical_class,
    value_has_native_integer_class, NumericDType, RuntimeError, Value,
};
use crate::builtins::introspection::class::class_identity_for_value;
use runmat_types::{standard, ClassIdentity};

pub fn value_is_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Complex(_, _)
        | Value::Tensor(_)
        | Value::ComplexTensor(_) => true,
        Value::SparseTensor(tensor) => !tensor.is_logical(),
        Value::GpuTensor(handle) => !handle_is_logical(handle),
        _ => false,
    }
}

pub fn value_is_float(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Complex(_, _) => true,
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_none(),
        Value::Tensor(t) => matches!(t.numeric_dtype(), NumericDType::F64 | NumericDType::F32),
        Value::SparseTensor(tensor) => !tensor.is_logical() && tensor.integer_storage().is_none(),
        Value::GpuTensor(handle) => {
            !handle_is_logical(handle) && handle_integer_type(handle).is_none()
        }
        _ => false,
    }
}

pub fn value_is_numeric_or_logical(value: &Value) -> bool {
    value_is_numeric(value) || value_has_logical_class(value)
}

/// Category queries are semantic predicates, not a second table of builtin classes.
/// Concrete class spellings resolve through the shared class identity authority.
enum ClassQuery {
    Numeric,
    Float,
    Integer,
    Sparse,
    Identity(ClassIdentity),
}

impl ClassQuery {
    fn parse(name: &str) -> Option<Self> {
        let name = name.trim().to_ascii_lowercase();
        Some(match name.as_str() {
            "numeric" => Self::Numeric,
            "float" => Self::Float,
            "integer" => Self::Integer,
            "sparse" => Self::Sparse,
            _ => Self::Identity(ClassIdentity::new(name).ok()?),
        })
    }

    fn matches(&self, value: &Value) -> bool {
        // Explicit wrappers retain their class, even for category queries.
        if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
        {
            return matches!(self, Self::Identity(identity)
                if identity.display_name().eq_ignore_ascii_case(standard::GPU_ARRAY.display_name()));
        }
        match self {
            Self::Numeric => value_is_numeric(value),
            Self::Float => value_is_float(value),
            Self::Integer => value_has_native_integer_class(value),
            Self::Sparse => matches!(value, Value::SparseTensor(_)),
            Self::Identity(requested) => {
                // Preserve the existing case-insensitive custom-class equality.
                // This validator does not introduce isa-style inheritance lookup.
                let actual = class_identity_for_value(value);
                ClassIdentity::new(actual.display_name().to_ascii_lowercase())
                    .is_ok_and(|actual| actual == *requested)
            }
        }
    }
}

pub fn value_matches_class(value: &Value, class_name: &str) -> bool {
    ClassQuery::parse(class_name).is_some_and(|query| query.matches(value))
}

pub fn must_be_a(value: &Value, class_names: Vec<String>) -> Result<bool, RuntimeError> {
    Ok(class_names
        .iter()
        .filter_map(|name| ClassQuery::parse(name))
        .any(|query| query.matches(value)))
}

pub fn value_underlying_type_matches(
    value: &Value,
    class_names: Vec<String>,
) -> Result<bool, RuntimeError> {
    validate_resident_metadata(value)?;
    let actual = match value {
        Value::GpuTensor(handle) => {
            crate::builtins::common::gpu_helpers::expected_gpu_class_identity(
                runmat_accelerate_api::handle_precision(handle),
                handle_integer_type(handle),
                handle_is_logical(handle),
            )
        }
        _ => Some(class_identity_for_value(value)),
    };
    Ok(actual.is_some_and(|actual| {
        class_names
            .iter()
            // underlying-type queries remain case-sensitive, unlike mustBeA.
            .filter_map(|name| ClassIdentity::new(name.trim()).ok())
            .any(|requested| requested == actual)
    }))
}
