//! Context-bound metadata objects retain the original declaration owners.
use super::metadata::{dynamic_error_with_source, empty_default_value};
use super::{TARGET_FIELD, VALID_FIELD};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_types::MemberAccess;
use runmat_value::{DynamicPropertyDef, NumericDType, Utf8Text, Value};

fn access_name(access: MemberAccess) -> &'static str {
    match access {
        MemberAccess::Public => "public",
        MemberAccess::Private => "private",
        MemberAccess::Protected => "protected",
    }
}
fn allocation(message: &'static str, source: HostAllocationError) -> crate::RuntimeError {
    dynamic_error_with_source(
        "addprop",
        &runmat_builtins::ADDPROP_ERROR_ALLOCATION,
        message,
        source,
    )
}

pub(super) fn metadata_object(
    runtime: Option<&RuntimeContext>,
    definition: &DynamicPropertyDef,
    target: Option<Value>,
    class: runmat_types::StaticClassIdentity,
) -> BuiltinResult<RootedValue> {
    let names = [
        access_name(definition.get_access),
        access_name(definition.set_access),
    ];
    let mut bytes = 0usize;
    let mut claims = 0usize;
    let overflow = || {
        allocation(
            "dynamic metadata geometry overflow",
            HostAllocationError::Overflow,
        )
    };
    for value in [&definition.name, &definition.description] {
        if !value.has_retained_claim() && value.capacity() != 0 {
            let unclaimed = Utf8Text::requested_bytes_for_capacity(value.capacity())
                .map_err(|e| allocation("dynamic text geometry is invalid", e.into()))?;
            bytes = bytes.checked_add(unclaimed).ok_or_else(overflow)?;
            claims += 1;
        }
    }
    if !definition.defining_class.has_retained_claim() {
        bytes = bytes
            .checked_add(
                definition
                    .defining_class
                    .requested_bytes()
                    .map_err(|e| allocation("dynamic class geometry is invalid", e.into()))?,
            )
            .ok_or_else(overflow)?;
        claims += 1;
    }
    for name in names {
        bytes = bytes
            .checked_add(
                Utf8Text::requested_bytes_for_capacity(name.len())
                    .map_err(|e| allocation("dynamic access text geometry is invalid", e.into()))?,
            )
            .ok_or_else(overflow)?;
        claims += 1;
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| overflow())?,
        claims,
    )
    .map_err(|e| allocation("dynamic text quote failed", e))?;
    let receipt = HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, retained)
            .map_err(|e| allocation("dynamic text plan failed", e))?,
    )
    .map_err(|e| allocation("dynamic text admission failed", e))?;
    let [get, set] = if let Some(receipt) = &receipt {
        let runtime = runtime.expect("managed metadata has context");
        let permit = receipt
            .output_permit(runtime)
            .map_err(|e| allocation("dynamic text permit failed", e))?;
        definition.clone().try_admit_exact(&permit).map_err(|e| {
            allocation(
                "dynamic declaration metadata admission failed",
                e.source().into(),
            )
        })?;
        let copy = |text: &str| {
            Utf8Text::try_copy_str_exact_with(text, &permit, |_| receipt.checkpoint(runtime))
                .map_err(|e| match e {
                    runmat_value::ExactArrayInitializationError::Allocation(e) => {
                        allocation("dynamic access text allocation failed", e.into())
                    }
                    runmat_value::ExactArrayInitializationError::Element(e) => {
                        allocation("dynamic access text copy was cancelled", e)
                    }
                })
        };
        [copy(names[0])?, copy(names[1])?]
    } else {
        [names[0].into(), names[1].into()]
    };
    let fields = [
        (
            "Name",
            Value::String(runmat_value::StringElement::Text(definition.name.clone())),
        ),
        (
            "DefiningClass",
            Value::ClassRef(definition.defining_class.clone()),
        ),
        (
            "GetAccess",
            Value::String(runmat_value::StringElement::Text(get)),
        ),
        (
            "SetAccess",
            Value::String(runmat_value::StringElement::Text(set)),
        ),
        ("Dependent", Value::Bool(definition.dependent)),
        ("Hidden", Value::Bool(definition.hidden)),
        ("Transient", Value::Bool(definition.transient)),
        ("NonCopyable", Value::Bool(definition.non_copyable)),
        ("AbortSet", Value::Bool(definition.abort_set)),
        ("SetObservable", Value::Bool(definition.set_observable)),
        ("GetObservable", Value::Bool(definition.get_observable)),
        (
            "Description",
            Value::String(runmat_value::StringElement::Text(
                definition.description.clone(),
            )),
        ),
        (VALID_FIELD, Value::Bool(true)),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        class.display_name(),
        fields
            .iter()
            .map(|(name, value)| (*name, value))
            .chain(target.as_ref().map(|value| (TARGET_FIELD, value))),
    )?;
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(runtime.expect("managed metadata runtime"))
            .map_err(|e| allocation("dynamic metadata publication was cancelled", e))?;
    }
    Ok(object)
}

pub(super) fn empty_default(runtime: Option<&RuntimeContext>) -> BuiltinResult<RootedValue> {
    let retained = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[0, 0])
        .map_err(|e| allocation("dynamic default geometry is invalid", e))?;
    let receipt = HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, retained)
            .map_err(|e| allocation("dynamic default plan failed", e))?,
    )
    .map_err(|e| allocation("dynamic default admission failed", e))?;
    let value = match &receipt {
        Some(receipt) => {
            let runtime = runtime.expect("managed default has context");
            let value = receipt
                .exact_output_tensor(runtime, NumericDType::F64, &[0, 0])
                .map_err(|e| allocation("dynamic default allocation failed", e))?
                .into_tensor()
                .map_err(|e| allocation("dynamic default publication failed", e.into()))?;
            receipt
                .checkpoint(runtime)
                .map_err(|e| allocation("dynamic default was cancelled", e))?;
            Value::Tensor(value)
        }
        None => empty_default_value(),
    };
    RootedValue::in_optional_context(value, runtime)
}
