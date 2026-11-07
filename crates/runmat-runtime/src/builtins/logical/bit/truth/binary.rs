use runmat_builtins::LogicalBinaryOperator;
use runmat_value::Value;

use crate::builtins::common::binary::BinaryInputPlan;
use crate::builtins::common::broadcast::broadcast_shapes;
use crate::execution::{RootedValue, RootedValueLease, RootedValuesBuilder};
use crate::BuiltinResult;

use super::{contract, evaluate, operand, provider};

pub(crate) async fn execute(
    context: Option<&crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
    operation: LogicalBinaryOperator,
) -> BuiltinResult<RootedValue> {
    let contract = contract::binary(operation);
    let _inputs = RootedValueLease::for_values([&lhs, &rhs].into_iter())
        .map_err(|error| error.into_runtime_error())?;
    match crate::builtins::table::plan_binary(lhs, rhs)
        .map_err(|error| operand::error_with_detail(contract.name, contract.mismatch, error))?
    {
        BinaryInputPlan::Values(values) => {
            value(context, values.0, values.1, operation, &contract).await
        }
        BinaryInputPlan::Structured(plan) => {
            let crate::builtins::table::TabularBinaryPlan {
                source,
                variables,
                right_values,
            } = plan;
            right_values
                .with_owned_output_storage_async(|right_values| async move {
                    let right_values = right_values.try_into_owned().map_err(|_| {
                        crate::sequence::sequence_error_to_runtime(
                            runmat_value::ValueSequenceError::SharedOutputStorage,
                        )
                    })?;
                    let mut names = Vec::with_capacity(variables.len());
                    let mut output = RootedValuesBuilder::with_capacity(variables.len());
                    for ((name, lhs), rhs) in variables.into_iter().zip(right_values) {
                        let result = value(context, lhs, rhs, operation, &contract)
                            .await
                            .map_err(|error| {
                                operand::error_with_detail(
                                    contract.name,
                                    contract.invalid,
                                    format!("table variable {name}: {}", error.message()),
                                )
                            })?;
                        output.push_owned(result);
                        names.push(name);
                    }
                    let output = output
                        .finish()
                        .map_err(|error| error.into_runtime_error())?;
                    crate::builtins::table::finish_binary_owned(&source, names, output).map_err(
                        |error| operand::error_with_detail(contract.name, contract.invalid, error),
                    )
                })
                .await
        }
    }
}

async fn value(
    context: Option<&crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
    operation: LogicalBinaryOperator,
    contract: &contract::BinaryContract,
) -> BuiltinResult<RootedValue> {
    enforce_extensions(&lhs, &rhs, contract)?;
    let output_source = provider::select_binary_source([&lhs, &rhs], contract.name)?;
    let shape = broadcast_shapes(
        contract.name,
        operand::admitted_shape(contract.name, &lhs, contract.invalid)?,
        operand::admitted_shape(contract.name, &rhs, contract.invalid)?,
    )
    .map_err(|error| operand::error_with_detail(contract.name, contract.mismatch, error))?;
    if matches!(lhs, Value::Tensor(_)) || matches!(rhs, Value::Tensor(_)) {
        return super::preparation::binary(
            context,
            contract.entry,
            lhs,
            rhs,
            |lhs, rhs| async move {
                let output =
                    admitted_value(context, lhs, rhs, operation, contract, output_source, shape)
                        .await?;
                RootedValue::new(output).map_err(|error| error.into_runtime_error())
            },
        )
        .await;
    }
    let output =
        admitted_value(context, lhs, rhs, operation, contract, output_source, shape).await?;
    RootedValue::new(output).map_err(|error| error.into_runtime_error())
}

async fn admitted_value(
    context: Option<&crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
    operation: LogicalBinaryOperator,
    contract: &contract::BinaryContract,
    output_source: Option<runmat_accelerate_api::GpuTensorHandle>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    if let (Value::GpuTensor(left), Value::GpuTensor(right)) = (&lhs, &rhs) {
        if let Some(value) = provider::binary_hook(left, right, contract.name, operation)? {
            return Ok(value);
        }
    }
    let left = operand::from_value(contract.name, lhs, contract.invalid).await?;
    let right = operand::from_value(contract.name, rhs, contract.invalid).await?;
    let data = evaluate::binary(&left, &right, &shape, operation);
    let value = operand::into_value(contract.name, data, shape.into(), contract.invalid)?;
    if let Some(context) = context {
        if let Some(route) = crate::builtins::math::host_route::selected(
            context,
            crate::context::VmFusionHostInstruction::ElementwiseBinary,
        ) {
            route.confirm_host_branch(
                context,
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
            );
        }
    }
    provider::restore_explicit(value, output_source.as_ref(), contract.name)
}

fn enforce_extensions(
    lhs: &Value,
    rhs: &Value,
    contract: &contract::BinaryContract,
) -> BuiltinResult<()> {
    if [lhs, rhs]
        .into_iter()
        .any(|value| operand::is_complex(value))
    {
        if let Some(extension) = contract.complex_extension {
            crate::compatibility::ensure_builtin_extension_enabled(extension, contract.name)?;
        }
    }
    if [lhs, rhs]
        .into_iter()
        .any(|value| matches!(value, Value::CharArray(_)))
    {
        if let Some(extension) = contract.character_extension {
            crate::compatibility::ensure_builtin_extension_enabled(extension, contract.name)?;
        }
    }
    Ok(())
}
