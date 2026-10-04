//! Durable aggregate admission at the canonical Value encoder boundary.

use super::{encode, field_path, index_path, shape, CodecPurpose};
use crate::execution::value_codec::ValueCodecError;
use runmat_execution::value::{
    DurableDynamicProperty, DurableObjectArray, DurablePlainObject, DurableStructArray,
    DurableStructArrayField, StructField,
};
use runmat_value::{ObjectArray, ObjectInstance, ObjectPersistencePayloadKind, StructArray, Value};

pub(super) fn struct_array(
    value: &StructArray,
    path: &str,
    purpose: CodecPurpose,
) -> Result<DurableStructArray, ValueCodecError> {
    let mut fields = value
        .field_names()
        .map(|name| {
            Ok(DurableStructArrayField {
                name: name.clone(),
                values: value
                    .field_values(name)
                    .expect("field name from same array")
                    .iter()
                    .enumerate()
                    .map(|(index, item)| {
                        encode(
                            item,
                            &format!("{}[{index}]", field_path(path, name)),
                            purpose,
                        )
                    })
                    .collect::<Result<_, _>>()?,
            })
        })
        .collect::<Result<Vec<_>, ValueCodecError>>()?;
    fields.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(DurableStructArray {
        shape: shape(value.shape(), path)?,
        fields,
    })
}

pub(super) fn object_array(
    value: &ObjectArray,
    path: &str,
    purpose: CodecPurpose,
) -> Result<DurableObjectArray, ValueCodecError> {
    let elements = value
        .data()
        .iter()
        .enumerate()
        .map(|(index, item)| match item {
            Value::Object(object) => plain_object(object, &index_path(path, index), purpose),
            _ => Err(ValueCodecError::unsupported(
                &index_path(path, index),
                "handle objects are not durable",
            )),
        })
        .collect::<Result<_, _>>()?;
    Ok(DurableObjectArray {
        class_identity: value.class_name().clone(),
        shape: shape(value.shape(), path)?,
        elements,
    })
}

pub(super) fn plain_object(
    value: &ObjectInstance,
    path: &str,
    purpose: CodecPurpose,
) -> Result<DurablePlainObject, ValueCodecError> {
    if value.persistence_payload_kind() != ObjectPersistencePayloadKind::PlainValue {
        return Err(ValueCodecError::unsupported(
            path,
            "native object authority is not durable",
        ));
    }
    let mut properties = value
        .properties
        .iter()
        .map(|(name, item)| {
            Ok(StructField {
                name: name.clone(),
                value: encode(item, &field_path(path, name), purpose)?,
            })
        })
        .collect::<Result<Vec<_>, ValueCodecError>>()?;
    properties.sort_by(|a, b| a.name.cmp(&b.name));
    let mut dynamic_properties = value
        .dynamic_property_definitions()
        .map(|(name, definition)| {
            if definition.metadata_handle.is_some() {
                return Err(ValueCodecError::unsupported(
                    &field_path(path, name),
                    "GC-owned dynamic property metadata is not durable",
                ));
            }
            Ok(DurableDynamicProperty {
                name: name.clone(),
                defining_class: definition.defining_class.clone(),
                get_access: definition.get_access,
                set_access: definition.set_access,
                dependent: definition.dependent,
                hidden: definition.hidden,
                transient: definition.transient,
                non_copyable: definition.non_copyable,
                abort_set: definition.abort_set,
                set_observable: definition.set_observable,
                get_observable: definition.get_observable,
                description: definition.description.clone(),
            })
        })
        .collect::<Result<Vec<_>, ValueCodecError>>()?;
    dynamic_properties.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(DurablePlainObject {
        class_identity: value.class_name.clone(),
        properties,
        dynamic_properties,
    })
}
