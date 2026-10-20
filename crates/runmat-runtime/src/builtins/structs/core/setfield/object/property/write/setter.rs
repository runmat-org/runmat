use super::errors;
use crate::execution::RootedValue;
use crate::sequence::ResolveValueSequence;
use crate::{call_builtin_async, object_property_setter_name, BuiltinResult};
use runmat_builtins::SETFIELD_ERROR_OBJECT_PROPERTY;
use runmat_value::{ObjectInstance, Value};

pub(super) async fn invoke(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &ObjectInstance,
    name: &str,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    match runtime {
        Some(runtime) => {
            let outputs =
                crate::object::dispatch::call_object_property_setter_with_outputs_in_context(
                    runtime,
                    Value::Object(obj.clone()),
                    name,
                    rhs,
                    1,
                )
                .await?;
            outputs.try_map_owned(
                |sequence| {
                    let values = sequence.resolve(
                        runmat_types::SequenceUse::RequireSingle,
                        crate::sequence::SequenceResolutionContext::default(),
                    )?;
                    crate::sequence::into_first_output(values)?.ok_or_else(|| {
                        errors::with_message(
                            "dependent setter returned no value",
                            &SETFIELD_ERROR_OBJECT_PROPERTY,
                        )
                    })
                },
                |error| error.into_runtime_error(),
            )
        }
        None => {
            let setter = object_property_setter_name(name);
            call_builtin_async(&setter, &[Value::Object(obj.clone()), rhs]).await
        }
    }
}
