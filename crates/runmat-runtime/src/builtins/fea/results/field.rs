use crate::analysis::AnalysisFieldDescriptor;
use crate::builtins::fea::contracts::descriptors::ERROR_INTERNAL;
use crate::builtins::fea::contracts::identities::{
    FEA_FIELD_CLASS, FEA_PAYLOAD_JSON_PROPERTY, FIELD_NAME,
};
use crate::builtins::fea::errors::{builtin_error, builtin_error_with_source};
use crate::builtins::fea::integer_serialization::serializable_to_value_preserving_integers;
use crate::builtins::fea::registry::ensure_fea_classes_registered;
use crate::BuiltinResult;
use runmat_analysis_core::AnalysisField;
use runmat_analysis_core::AnalysisFieldValues;
use runmat_value::IntValue;
use runmat_value::IntegerStorage;
use runmat_value::ObjectInstance;
use runmat_value::Tensor;
use runmat_value::Value;

pub(in crate::builtins::fea) fn field_to_object(
    field: &AnalysisField,
    descriptor: &AnalysisFieldDescriptor,
) -> BuiltinResult<ObjectInstance> {
    ensure_fea_classes_registered();
    let mut object = ObjectInstance::new(FEA_FIELD_CLASS.to_string());
    object.properties.insert(
        "field_id".to_string(),
        Value::String(field.field_id.clone()),
    );
    object
        .properties
        .insert("id".to_string(), Value::String(field.field_id.clone()));
    object.properties.insert(
        "shape".to_string(),
        usize_slice_tensor(&field.shape, 1, field.shape.len())?,
    );
    object
        .properties
        .insert("values".to_string(), field_values_value(field)?);
    object.properties.insert(
        "unit".to_string(),
        Value::String(descriptor.unit.clone().unwrap_or_default()),
    );
    object.properties.insert(
        "location".to_string(),
        Value::String(format!("{:?}", descriptor.location).to_ascii_lowercase()),
    );
    object.properties.insert(
        "kind".to_string(),
        Value::String(format!("{:?}", descriptor.kind).to_ascii_lowercase()),
    );
    object.properties.insert(
        "family".to_string(),
        Value::String(descriptor.family.clone()),
    );
    object.properties.insert(
        "quantity".to_string(),
        Value::String(descriptor.quantity.clone()),
    );
    object.properties.insert(
        "topology_id".to_string(),
        descriptor
            .topology_id
            .as_ref()
            .map(|value| Value::String(value.clone()))
            .unwrap_or_else(empty_double_value),
    );
    object.properties.insert(
        "element_kind".to_string(),
        descriptor
            .element_kind
            .as_ref()
            .map(|value| Value::String(value.clone()))
            .unwrap_or_else(empty_double_value),
    );
    object.properties.insert(
        "component_count".to_string(),
        descriptor
            .component_count
            .map(|value| Value::Int(IntValue::U64(value as u64)))
            .unwrap_or_else(empty_double_value),
    );
    object.properties.insert(
        "element_count".to_string(),
        Value::Int(IntValue::U64(descriptor.element_count as u64)),
    );
    object.properties.insert(
        "entity_count".to_string(),
        Value::Int(IntValue::U64(descriptor.entity_count as u64)),
    );
    object.properties.insert(
        "value_count".to_string(),
        Value::Int(IntValue::U64(descriptor.value_count as u64)),
    );
    object.properties.insert(
        "storage".to_string(),
        Value::String(format!("{:?}", descriptor.storage).to_ascii_lowercase()),
    );
    object.properties.insert(
        "descriptor".to_string(),
        serializable_to_value_preserving_integers(
            FIELD_NAME,
            &ERROR_INTERNAL,
            descriptor,
            &[],
            &["shape", "element_count", "component_count", "size_bytes"],
        )?,
    );
    let json = serde_json::to_string(field).map_err(|err| {
        builtin_error_with_source(FIELD_NAME, &ERROR_INTERNAL, err.to_string(), err)
    })?;
    object
        .properties
        .insert(FEA_PAYLOAD_JSON_PROPERTY.to_string(), Value::String(json));
    Ok(object)
}

pub(in crate::builtins::fea) fn field_values_value(field: &AnalysisField) -> BuiltinResult<Value> {
    match &field.values {
        AnalysisFieldValues::HostF64(values) => Tensor::new(values.clone(), field.shape.clone())
            .map(Value::Tensor)
            .map_err(|err| {
                builtin_error(
                    FIELD_NAME,
                    &ERROR_INTERNAL,
                    format!("fea.field: failed to build values tensor: {err}"),
                )
            }),
        AnalysisFieldValues::DeviceRef(device) => serializable_to_value_preserving_integers(
            FIELD_NAME,
            &ERROR_INTERNAL,
            device,
            &[],
            &["element_count"],
        ),
    }
}

pub(in crate::builtins::fea) fn usize_slice_tensor(
    values: &[usize],
    rows: usize,
    cols: usize,
) -> BuiltinResult<Value> {
    let values = values
        .iter()
        .map(|value| u64::try_from(*value))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            builtin_error(
                FIELD_NAME,
                &ERROR_INTERNAL,
                "FEA field shape exceeds uint64",
            )
        })?;
    Tensor::new_integer(IntegerStorage::U64(values), vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| {
            builtin_error(
                FIELD_NAME,
                &ERROR_INTERNAL,
                format!("fea.field: failed to build metadata tensor: {err}"),
            )
        })
}

pub(in crate::builtins::fea) fn empty_double_value() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor shape is valid"))
}

pub(in crate::builtins::fea) fn find_field<I>(fields: I, requested: &str) -> Option<AnalysisField>
where
    I: IntoIterator<Item = AnalysisField>,
{
    let mut suffix_matches = Vec::new();
    for field in fields {
        if field.field_id == requested {
            return Some(field);
        }
        if field_id_matches(&field.field_id, requested) {
            suffix_matches.push(field);
        }
    }
    if suffix_matches.len() == 1 {
        suffix_matches.pop()
    } else {
        None
    }
}

pub(in crate::builtins::fea) fn find_descriptor<'a, I>(
    descriptors: I,
    requested: &str,
) -> Option<&'a AnalysisFieldDescriptor>
where
    I: IntoIterator<Item = &'a AnalysisFieldDescriptor>,
{
    let mut suffix_matches = Vec::new();
    for descriptor in descriptors {
        if descriptor.field_id == requested {
            return Some(descriptor);
        }
        if field_id_matches(&descriptor.field_id, requested) {
            suffix_matches.push(descriptor);
        }
    }
    if suffix_matches.len() == 1 {
        suffix_matches.pop()
    } else {
        None
    }
}

pub(in crate::builtins::fea) fn field_id_matches(candidate: &str, requested: &str) -> bool {
    candidate == requested
        || candidate
            .strip_suffix(requested)
            .is_some_and(|prefix| prefix.ends_with('.'))
        || candidate
            .rsplit_once('.')
            .is_some_and(|(_, tail)| tail == requested)
}
