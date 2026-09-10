use runmat_types::ClassIdentity;
use runmat_value::{HandleRef, Listener, ObjectInstance, Value};
use std::collections::{BTreeSet, HashSet};

pub(super) fn object(object: &ObjectInstance) -> Vec<String> {
    let mut names = class_properties(&object.class_name);
    names.extend(object.properties.keys().cloned());
    names.into_iter().collect()
}

pub(super) fn handle(handle: &HandleRef) -> crate::BuiltinResult<Vec<String>> {
    let mut names = class_properties(&handle.class_name);
    if crate::is_handle_valid(handle) {
        let target = runmat_gc::gc_clone_value(&handle.target).map_err(|error| {
            super::super::error::invalid_target(format!(
                "fieldnames: invalid handle target: {error}"
            ))
        })?;
        extend_target_names(&mut names, &target)?;
    }
    Ok(names.into_iter().collect())
}

pub(super) fn listener(_listener: &Listener) -> Vec<String> {
    ["callback", "enabled", "event_name", "id", "target", "valid"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

fn extend_target_names(names: &mut BTreeSet<String>, target: &Value) -> crate::BuiltinResult<()> {
    match target {
        Value::Struct(structure) => names.extend(super::structures::scalar(structure)),
        Value::StructArray(array) => names.extend(super::structures::array(array)),
        Value::Object(object) => names.extend(self::object(object)),
        Value::Listener(listener) => names.extend(self::listener(listener)),
        Value::HandleObject(handle) => names.extend(class_properties(&handle.class_name)),
        _ => {}
    }
    Ok(())
}

fn class_properties(class_name: &ClassIdentity) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut current = Some(class_name.clone());
    let mut visited = HashSet::new();
    while let Some(name) = current {
        if !visited.insert(name.clone()) {
            break;
        }
        let Some(class) = crate::class_registry::get_class(&name) else {
            break;
        };
        names.extend(
            class
                .properties
                .iter()
                .filter(|(_, property)| !property.is_static)
                .map(|(name, _)| name.to_string()),
        );
        current = class.parent.clone();
    }
    names
}
