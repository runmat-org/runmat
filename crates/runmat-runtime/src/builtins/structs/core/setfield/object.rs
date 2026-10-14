mod property;

use super::assignment::assign_into_value;
use super::assignment::root;
use super::errors::BUILTIN_NAME;
use super::selector::assign_with_selector;
use crate::builtins::structs::core::field_path::FieldStep;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_builtins::SETFIELD_OBJECT_FAMILY_EXTENSION;
use runmat_value::{ObjectInstance, Value};

pub(super) async fn assign_into_object(
    mut object: ObjectInstance,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let _aggregate_root =
        RootedValueLease::for_traced(&object).map_err(|error| error.into_runtime_error())?;
    let _rhs_root =
        RootedValueLease::for_traced(&rhs).map_err(|error| error.into_runtime_error())?;
    ensure_object_family_enabled()?;
    let (first, rest) = steps
        .split_first()
        .expect("assignment path is validated before traversal");
    if rest.is_empty() {
        let updated = if let Some(selector) = &first.index {
            let current = property::read_property(&object, &first.name).await?;
            assign_with_selector(current.value().clone(), selector, rest, rhs).await?
        } else {
            root(rhs)?
        };
        let setter_result =
            property::write_property(&mut object, &first.name, updated.value().clone()).await?;
        let output = root(Value::Object(object));
        drop(setter_result);
        return output;
    }
    let current = property::read_property(&object, &first.name).await?;
    let updated = if let Some(selector) = &first.index {
        assign_with_selector(current.value().clone(), selector, rest, rhs).await?
    } else {
        assign_into_value(current.value().clone(), rest, rhs).await?
    };
    let setter_result =
        property::write_property(&mut object, &first.name, updated.value().clone()).await?;
    let output = root(Value::Object(object));
    drop(setter_result);
    output
}

pub(super) fn ensure_object_family_enabled() -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &SETFIELD_OBJECT_FAMILY_EXTENSION,
        BUILTIN_NAME,
    )
}
