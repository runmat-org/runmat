//! Structural queries for declared object-backed array representations.

use runmat_types::{ClassArrayPayload, ClassArrayShape, MemberName};
use runmat_value::{ObjectInstance, Value};

use crate::RuntimeError;

pub(super) fn dimensions<'a>(
    object: &'a ObjectInstance,
    declaration: &ClassArrayShape,
) -> Result<super::ValueDimensions<'a>, RuntimeError> {
    let (primary, others, kind) = match declaration {
        ClassArrayShape::Property { property, payload } => (property, &[][..], *payload),
        ClassArrayShape::ConsistentProperties {
            primary,
            others,
            payload,
        } => (primary, others.as_slice(), *payload),
    };
    let dimensions = property_dimensions(object, primary, kind)?;
    for property in others {
        let sibling = property_dimensions(object, property, kind)?;
        if sibling != dimensions {
            return Err(crate::build_runtime_error(format!(
                "{}: array payload properties {} and {} have inconsistent dimensions",
                object.class_name(),
                primary.0,
                property.0
            ))
            .build());
        }
    }
    Ok(dimensions)
}

fn property_dimensions<'a>(
    object: &'a ObjectInstance,
    property: &MemberName,
    kind: ClassArrayPayload,
) -> Result<super::ValueDimensions<'a>, RuntimeError> {
    let payload = object.properties().get(&property.0).ok_or_else(|| {
        crate::build_runtime_error(format!(
            "{}: array payload property {} is missing",
            object.class_name(),
            property.0
        ))
        .build()
    })?;
    if !super::array_payload_storage_matches(payload, kind) {
        return Err(crate::build_runtime_error(format!(
            "{}: array payload property {} has invalid storage",
            object.class_name(),
            property.0
        ))
        .build());
    }
    // This reads geometry, not numerical contents. Numerical admission and
    // conversions remain the responsibility of the consuming domain.
    super::value_dimensions_borrowed(payload)
}

pub(crate) fn array_payload_storage_matches(value: &Value, kind: ClassArrayPayload) -> bool {
    let real = matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
        || matches!(value, Value::GpuTensor(handle) if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real && !handle.view().is_logical());
    match kind {
        ClassArrayPayload::RealNumeric => real,
        ClassArrayPayload::NumericOrLogical => {
            real || matches!(
                value,
                Value::Complex(_, _)
                    | Value::ComplexTensor(_)
                    | Value::Bool(_)
                    | Value::LogicalArray(_)
                    | Value::SparseTensor(_)
                    | Value::GpuTensor(_)
            )
        }
    }
}
