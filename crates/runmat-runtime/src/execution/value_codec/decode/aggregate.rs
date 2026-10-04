//! Reconstruction of validated durable preference aggregates.

use super::{decode, usize_shape};
use crate::execution::value_codec::path::field_path;
use crate::execution::value_codec::ValueCodecError;
use runmat_execution::value::{DurableObjectArray, DurablePlainObject, DurableStructArray};
use runmat_value::{DynamicPropertyDef, ObjectArray, ObjectInstance, StructArray, Value};

pub(super) fn struct_array(
    value: &DurableStructArray,
    path: &str,
) -> Result<Value, ValueCodecError> {
    let names = value
        .fields
        .iter()
        .map(|field| field.name.clone())
        .collect();
    let values = value
        .fields
        .iter()
        .flat_map(|field| {
            field
                .values
                .iter()
                .enumerate()
                .map(move |(index, item)| (field.name.as_str(), index, item))
        })
        .map(|(name, index, item)| decode(item, &format!("{}[{index}]", field_path(path, name))))
        .collect::<Result<Vec<_>, _>>()?;
    StructArray::normalize_field_major(names, values, usize_shape(&value.shape, path)?)
        .map_err(|error| ValueCodecError::invalid(path, error))
}

pub(super) fn object_array(
    value: &DurableObjectArray,
    path: &str,
) -> Result<Value, ValueCodecError> {
    let objects = value
        .elements
        .iter()
        .enumerate()
        .map(|(index, item)| plain_object(item, &format!("{path}[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value::ObjectArray(
        ObjectArray::from_objects(
            value.class_identity.clone(),
            objects,
            usize_shape(&value.shape, path)?,
        )
        .map_err(|error| ValueCodecError::invalid(path, error))?,
    ))
}

pub(super) fn plain_object(
    value: &DurablePlainObject,
    path: &str,
) -> Result<ObjectInstance, ValueCodecError> {
    let mut object = ObjectInstance::new(value.class_identity.clone());
    for field in &value.properties {
        object.properties.insert(
            field.name.clone(),
            decode(&field.value, &field_path(path, &field.name))?,
        );
    }
    for definition in &value.dynamic_properties {
        let property = DynamicPropertyDef {
            name: definition.name.clone(),
            defining_class: definition.defining_class.clone(),
            metadata_handle: None,
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
        };
        object
            .dynamic_properties
            .get_or_insert_with(Default::default)
            .insert(definition.name.clone(), property);
    }
    Ok(object)
}
