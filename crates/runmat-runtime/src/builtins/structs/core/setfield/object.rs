mod property;

use super::assignment::assign_into_value;
use super::errors::BUILTIN_NAME;
use super::selector::assign_with_selector;
use crate::builtins::structs::core::field_path::FieldStep;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_builtins::SETFIELD_OBJECT_FAMILY_EXTENSION;
use runmat_value::{ObjectInstance, Value};

pub(super) async fn assign_into_object(
    runtime: Option<&crate::context::RuntimeContext>,
    mut object: ObjectInstance,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let _aggregate_root =
        super::super::construction::input(runtime, &Value::Object(object.clone()))?;
    let _rhs_root = super::super::construction::input(runtime, &rhs)?;
    ensure_object_family_enabled()?;
    if let Some(prepared) = super::super::field_path::native::prepare(
        runtime,
        &object,
        crate::object::protocol::ObjectProtocol::Subsasgn,
        &steps.first().expect("validated assignment path").name,
    )? {
        return prepared.assign(object, steps, rhs).await;
    }
    let (first, rest) = steps
        .split_first()
        .expect("assignment path is validated before traversal");
    if rest.is_empty() {
        let updated = if let Some(selector) = &first.index {
            let current = property::read_property(runtime, &object, &first.name).await?;
            assign_with_selector(runtime, current.value().clone(), selector, rest, rhs).await?
        } else {
            super::super::construction::root(runtime, rhs)?
        };
        let setter_result =
            property::write_property(runtime, &mut object, &first.name, updated.value().clone())
                .await?;
        let output = super::super::construction::root(runtime, Value::Object(object));
        drop(setter_result);
        return output;
    }
    let current = property::read_property(runtime, &object, &first.name).await?;
    let updated = if let Some(selector) = &first.index {
        assign_with_selector(runtime, current.value().clone(), selector, rest, rhs).await?
    } else {
        assign_into_value(runtime, current.value().clone(), rest, rhs).await?
    };
    let setter_result =
        property::write_property(runtime, &mut object, &first.name, updated.value().clone())
            .await?;
    let output = super::super::construction::root(runtime, Value::Object(object));
    drop(setter_result);
    output
}

pub(super) fn ensure_object_family_enabled() -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &SETFIELD_OBJECT_FAMILY_EXTENSION,
        BUILTIN_NAME,
    )
}
