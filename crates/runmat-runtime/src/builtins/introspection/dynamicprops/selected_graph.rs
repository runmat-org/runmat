//! Exact selected metadata/receiver pairing for the Graph codec.

use runmat_value::{DynamicPropertyDef, HandleRef, ObjectInstance, Value};

use super::{metadata::dynamic_def_to_metadata_object, DYNAMIC_PROPERTY_CLASS, TARGET_FIELD};

pub(crate) struct SelectedMetadataOwner {
    pub receiver: HandleRef,
    pub property: String,
}

/// A metadata handle is selected by its reciprocal GC identity in the owning
/// dynamic definition, never by metadata class/name spelling alone.
pub(crate) fn checked_metadata_owner(
    handle: &HandleRef,
) -> Result<Option<SelectedMetadataOwner>, String> {
    if !handle.class_name.is(DYNAMIC_PROPERTY_CLASS) {
        return Ok(None);
    }
    if !handle.valid || !crate::is_handle_valid(handle) {
        return Err("dynamic metadata handle is invalid".into());
    }
    let value = runmat_gc::gc_clone_value(&handle.target)
        .map_err(|_| "dynamic metadata target is unavailable")?;
    let Value::Object(metadata) = value else {
        return Err("dynamic metadata target is not an object".into());
    };
    if !metadata.class_name().is(DYNAMIC_PROPERTY_CLASS) {
        return Err("dynamic metadata class differs from its handle".into());
    }
    let Some(Value::String(runmat_value::StringElement::Text(property))) =
        metadata.property("Name")
    else {
        return Err("dynamic metadata has no canonical property name".into());
    };
    let Some(Value::HandleObject(receiver)) = metadata.property(TARGET_FIELD) else {
        return Err("dynamic metadata has no receiver handle".into());
    };
    if !crate::is_handle_valid(receiver) {
        return Err("dynamic metadata receiver is invalid".into());
    }
    let definition = runmat_gc::gc_with_value(&receiver.target, |stored| {
        let Value::Object(object) = stored else {
            return None;
        };
        if object.class_name() != &receiver.class_name
            || object.has_native_payload()
            || object.selected_class().is_none()
        {
            return None;
        }
        object
            .dynamic_property(property)
            .filter(|definition| definition.metadata_handle == Some(handle.target))
            .cloned()
    })
    .map_err(|_| "dynamic metadata receiver is unavailable")?
    .ok_or("dynamic metadata is not owned by its selected receiver")?;
    if definition.name != *property {
        return Err("dynamic metadata property differs from its definition".into());
    }
    let canonical = dynamic_def_to_metadata_object(
        &definition,
        Some(Value::HandleObject(receiver.clone())),
        DYNAMIC_PROPERTY_CLASS,
    );
    if metadata != canonical {
        return Err("dynamic metadata differs from its owning definition".into());
    }
    Ok(Some(SelectedMetadataOwner {
        receiver: receiver.clone(),
        property: property.as_str().to_owned(),
    }))
}

/// Reconstruct the same canonical metadata object that `addprop` issues.
pub(crate) fn selected_metadata_object(
    definition: &DynamicPropertyDef,
    receiver: HandleRef,
) -> ObjectInstance {
    dynamic_def_to_metadata_object(
        definition,
        Some(Value::HandleObject(receiver)),
        DYNAMIC_PROPERTY_CLASS,
    )
}
