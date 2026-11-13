#[path = "mode_object/admission.rs"]
mod admission;
#[path = "mode_object/field.rs"]
mod field;

use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use admission::{allocation, exact_receipt, initialization};
use runmat_types::StaticClassIdentity;
use runmat_value::{ExactHostTensorAllocation, GraphicsReceiverRef, NumericDType, Utf8Text, Value};

pub(crate) enum FieldValue<'a> {
    Text(&'a str),
    EmptyF64Tensor(&'a [usize]),
    Value(Option<Value>),
}

pub(crate) struct Field<'a> {
    pub(crate) name: &'static str,
    pub(crate) value: FieldValue<'a>,
}

pub(crate) fn prepare_fields(
    runtime: Option<&RuntimeContext>,
    fields: &mut [Field<'_>],
) -> BuiltinResult<()> {
    let Some(runtime) = runtime else {
        for field in fields {
            let value = match &field.value {
                FieldValue::Text(value) => Value::string((*value).to_owned()),
                FieldValue::EmptyF64Tensor(shape) => Value::Tensor(
                    runmat_value::Tensor::new(Vec::new(), shape.to_vec())
                        .expect("empty mode-object tensor shape is valid"),
                ),
                FieldValue::Value(_) => continue,
            };
            field.value = FieldValue::Value(Some(value));
        }
        return Ok(());
    };
    let mut physical = 0usize;
    let mut claims = 0usize;
    for field in fields.iter() {
        if let FieldValue::Text(value) = &field.value {
            physical =
                physical
                    .checked_add(Utf8Text::requested_bytes_for_capacity(value.len()).map_err(
                        |source| allocation(runtime, HostAllocationError::Numeric(source)),
                    )?)
                    .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
            claims = claims
                .checked_add(usize::from(!value.is_empty()))
                .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        } else if let FieldValue::EmptyF64Tensor(shape) = &field.value {
            physical = physical
                .checked_add(
                    ExactHostTensorAllocation::requested_bytes(NumericDType::F64, shape).map_err(
                        |source| allocation(runtime, HostAllocationError::Numeric(source)),
                    )?,
                )
                .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
            claims = claims
                .checked_add(ExactHostTensorAllocation::retained_claim_count())
                .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        }
    }
    let receipt = exact_receipt(runtime, physical, claims)?;
    let Some(receipt) = receipt else {
        for field in fields {
            let value = match &field.value {
                FieldValue::Text(value) => Value::string((*value).to_owned()),
                FieldValue::EmptyF64Tensor(shape) => Value::Tensor(
                    runmat_value::Tensor::new(Vec::new(), shape.to_vec())
                        .expect("empty mode-object tensor shape is valid"),
                ),
                FieldValue::Value(_) => continue,
            };
            field.value = FieldValue::Value(Some(value));
        }
        return Ok(());
    };
    let permit = receipt
        .output_permit(runtime)
        .map_err(|source| allocation(runtime, source))?;
    for field in fields {
        let value = match &field.value {
            FieldValue::Text(value) => Value::String(
                Utf8Text::try_copy_str_exact_with(value, &permit, |_| receipt.checkpoint(runtime))
                    .map_err(|source| initialization(runtime, source))?
                    .into(),
            ),
            FieldValue::EmptyF64Tensor(shape) => Value::Tensor(
                ExactHostTensorAllocation::new(NumericDType::F64, shape, &permit)
                    .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?
                    .into_tensor()
                    .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?,
            ),
            FieldValue::Value(_) => continue,
        };
        field.value = FieldValue::Value(Some(value));
    }
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation(runtime, source))
}

pub(crate) fn construct(
    runtime: Option<&RuntimeContext>,
    class: StaticClassIdentity,
    receiver: GraphicsReceiverRef,
    fields: &mut [Field<'_>],
) -> BuiltinResult<crate::execution::RootedValue> {
    prepare_fields(runtime, fields)?;
    crate::execution::object::construct_object_from_fields_with_native(
        runtime,
        class.display_name(),
        fields.iter().map(|field| {
            let FieldValue::Value(Some(value)) = &field.value else {
                unreachable!("mode object field preparation is complete")
            };
            (field.name, value)
        }),
        Some(crate::execution::object::ObjectNativeInitialization::GraphicsReceiver(receiver)),
    )
}
