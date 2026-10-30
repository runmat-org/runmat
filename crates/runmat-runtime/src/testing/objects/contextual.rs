//! Contextual publication of the same test-framework object schemas.
use super::*;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease, RootedValues};
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{ExactArrayInitializationError, ExactArrayStorage, SharedShape, StringElement};

pub fn object_from_fields_in_context<'a>(
    runtime: &RuntimeContext,
    class: &str,
    fields: impl Iterator<Item = (&'a str, &'a Value)> + Clone,
) -> BuiltinResult<RootedValue> {
    crate::execution::object::construct_object_from_borrowed_fields(Some(runtime), class, fields)
}

/// The source and updates remain independently rooted until the successor's
/// registration is published. Class, native, dynamic and selected owners are
/// retained by the canonical replacement implementation.
pub fn replace_object_properties_in_context<'a>(
    runtime: &RuntimeContext,
    source: RootedValue,
    updates: impl Iterator<Item = (&'a str, &'a Value)> + Clone,
) -> BuiltinResult<RootedValue> {
    let _updates = RootedValueLease::for_values_iterator_in_context(
        updates.clone().map(|(_, value)| value),
        runtime,
    )?;
    let Value::Object(object) = source.value() else {
        return Err(invalid("test projection source is not an object"));
    };
    let mut successor = object.clone();
    crate::execution::object::replacement::replace_properties(
        Some(runtime),
        &mut successor,
        updates,
        |message, cause| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Testing:Allocation",
                message,
                cause,
            )
        },
        invalid,
    )?;
    RootedValue::in_context(Value::Object(successor), runtime)
}

pub fn test_suite_object_in_context(
    runtime: &RuntimeContext,
    test: &TestDescriptor,
) -> BuiltinResult<RootedValue> {
    let procedure = test
        .procedure
        .display_name
        .rsplit('/')
        .next()
        .unwrap_or(test.procedure.display_name.as_str());
    let fields = [
        ("Name", text(runtime, &test.display_name)?),
        ("ProcedureName", text(runtime, procedure)?),
        (
            "TestFile",
            text(runtime, &test.procedure.source.relative_path)?,
        ),
        ("Tags", string_row(runtime, &test.tags)?),
        ("__runmat_test_id", text(runtime, test.id.as_str())?),
    ];
    object_from_fields_in_context(
        runtime,
        TEST_SUITE_CLASS.display_name(),
        fields.iter().map(|(key, value)| (*key, value)),
    )
}

pub fn test_result_object_in_context(
    runtime: &RuntimeContext,
    name: &str,
    result: &TestResult,
) -> BuiltinResult<RootedValue> {
    let details = result
        .attempts
        .last()
        .map(|attempt| attempt.output.as_str())
        .unwrap_or_default();
    let fields = [
        ("Name", text(runtime, name)?),
        ("Passed", Value::Bool(result.state.is_success())),
        ("Failed", Value::Bool(result.state.failed)),
        ("Incomplete", Value::Bool(result.state.incomplete)),
        ("Flaky", Value::Bool(result.flaky)),
        ("Details", text(runtime, details)?),
        ("__runmat_test_id", text(runtime, result.test_id.as_str())?),
    ];
    object_from_fields_in_context(
        runtime,
        TEST_RESULT_CLASS.display_name(),
        fields.iter().map(|(key, value)| (*key, value)),
    )
}

pub fn object_array_or_scalar_in_context(
    runtime: &RuntimeContext,
    class: runmat_types::StaticClassIdentity,
    values: RootedValues,
) -> BuiltinResult<RootedValue> {
    if values.len() == 1 {
        return RootedValue::in_context(values[0].clone(), runtime);
    }
    use crate::execution::object_array::{Class, ObjectArrayConstruction};
    let mut output = ObjectArrayConstruction::new(
        Some(runtime),
        Class::Name(class.display_name()),
        values.len(),
        &[1, values.len()],
        std::iter::empty(),
        true,
    )?;
    for value in values.iter() {
        output.push(value.clone())?;
    }
    output.finish()
}

fn text(runtime: &RuntimeContext, source: &str) -> BuiltinResult<Value> {
    crate::call::argument_text::copy_str(source, Some(runtime))
        .map(|text| Value::String(text.into()))
}

fn string_row(runtime: &RuntimeContext, source: &[String]) -> BuiltinResult<Value> {
    let allocation = |cause| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:Testing:Allocation",
            "test tag storage admission failed",
            cause,
        )
    };
    let bytes = ExactArrayStorage::<StringElement>::requested_bytes(source.len())
        .and_then(|bytes| {
            SharedShape::requested_bytes_for_rank(2).and_then(|shape| {
                bytes
                    .checked_add(shape)
                    .ok_or(runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
            })
        })
        .map_err(|cause| allocation(HostAllocationError::Numeric(cause)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| allocation(HostAllocationError::Overflow))?,
        usize::from(!source.is_empty()) + 1,
    )
    .map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(allocation)?,
    )
    .map_err(allocation)?;
    let array = if let Some(receipt) = &receipt {
        let permit = receipt.output_permit(runtime).map_err(allocation)?;
        let data = ExactArrayStorage::try_initialize(source.len(), &permit, |index| {
            receipt.checkpoint(runtime).map_err(allocation)?;
            crate::call::argument_text::copy_str(&source[index], Some(runtime))
                .map(StringElement::Text)
        })
        .map_err(|cause| match cause {
            ExactArrayInitializationError::Allocation(cause) => {
                allocation(HostAllocationError::Numeric(cause))
            }
            ExactArrayInitializationError::Element(cause) => cause,
        })?;
        let shape = SharedShape::new_admitted_from_slice(&[1, source.len()], &permit)
            .map_err(|cause| allocation(HostAllocationError::Numeric(cause)))?;
        receipt.checkpoint(runtime).map_err(allocation)?;
        runmat_value::StringArray::from_admitted(data, shape).map_err(invalid)?
    } else {
        // An actual selected unbounded context was checked before allocating.
        runmat_value::StringArray::new(source.to_vec(), vec![1, source.len()]).map_err(invalid)?
    };
    Ok(Value::StringArray(array))
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    crate::runtime_error::semantic_error("Testing:Projection", message)
}
