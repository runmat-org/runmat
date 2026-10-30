use super::{callback::CallbackFailure, output};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use crate::execution::{construct_struct_from_fields_in_context, RootedValue};
use runmat_value::{ExactOrderedMap, Utf8Text, Value};

pub(super) fn value(
    error: &CallbackFailure,
    index: usize,
    field: &str,
    runtime: Option<&RuntimeContext>,
) -> crate::BuiltinResult<RootedValue> {
    let values = [
        text(error.identifier().unwrap_or(""), runtime)?,
        text(error.message().as_ref(), runtime)?,
        Value::Num((index + 1) as f64),
        text(field, runtime)?,
    ];
    let names = ["identifier", "message", "index", "field"];
    if let Some(runtime) = runtime {
        construct_struct_from_fields_in_context(runtime, names.into_iter().zip(values.iter()))
    } else {
        output::ordinary_policy()?;
        output::root(
            Value::Struct(runmat_value::StructValue::from_fields_unbounded(
                names
                    .into_iter()
                    .zip(values)
                    .map(|(name, value)| (name.to_owned(), value)),
            )),
            None,
        )
    }
}
fn text(value: &str, runtime: Option<&RuntimeContext>) -> crate::BuiltinResult<Value> {
    let (bytes, claims) = ExactOrderedMap::<Value>::key_quote(value)
        .map_err(|e| output::allocation(runtime, e.into()))?;
    let bytes = u64::try_from(bytes)
        .map_err(|_| output::allocation(runtime, HostAllocationError::Overflow))?;
    let bytes = HostAllocationPlan::retained_exact_output_bytes(bytes, claims)
        .map_err(|e| output::allocation(runtime, e))?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(|e| output::allocation(runtime, e))?;
    let receipt =
        HostAllocationReceipt::admit(runtime, plan).map_err(|e| output::allocation(runtime, e))?;
    if let Some(receipt) = receipt {
        let runtime = runtime.expect("managed receipt has selected context");
        let permit = receipt
            .output_permit(runtime)
            .map_err(|e| output::allocation(Some(runtime), e))?;
        let text =
            Utf8Text::try_copy_str_exact_with(value, &permit, |_| receipt.checkpoint(runtime));
        receipt
            .checkpoint(runtime)
            .map_err(|e| output::allocation(Some(runtime), e))?;
        let text = text.map_err(|e| {
            output::allocation(
                Some(runtime),
                match e {
                    runmat_value::ExactArrayInitializationError::Allocation(e) => e.into(),
                    runmat_value::ExactArrayInitializationError::Element(e) => e,
                },
            )
        })?;
        Ok(Value::String(text.into()))
    } else {
        Ok(Value::String(value.into()))
    }
}
