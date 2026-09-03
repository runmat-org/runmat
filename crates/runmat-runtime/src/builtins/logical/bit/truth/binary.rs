use runmat_builtins::LogicalBinaryOperator;
use runmat_value::Value;

use crate::builtins::common::binary::BinaryInputPlan;
use crate::builtins::common::broadcast::broadcast_shapes;
use crate::BuiltinResult;

use super::{contract, evaluate, operand, provider};

pub(crate) async fn execute(
    lhs: Value,
    rhs: Value,
    operation: LogicalBinaryOperator,
) -> BuiltinResult<Value> {
    let contract = contract::binary(operation);
    match crate::builtins::table::plan_binary(lhs, rhs)
        .map_err(|error| operand::error_with_detail(contract.name, contract.mismatch, error))?
    {
        BinaryInputPlan::Values(values) => value(values.0, values.1, operation, &contract).await,
        BinaryInputPlan::Structured(plan) => {
            let (source, variables) = plan.variables();
            let mut output = Vec::with_capacity(variables.len());
            for (name, lhs, rhs) in variables {
                let result = value(lhs, rhs, operation, &contract)
                    .await
                    .map_err(|error| {
                        operand::error_with_detail(
                            contract.name,
                            contract.invalid,
                            format!("table variable {name}: {}", error.message()),
                        )
                    })?;
                output.push((name, result));
            }
            crate::builtins::table::finish_binary(&source, output)
                .map_err(|error| operand::error_with_detail(contract.name, contract.invalid, error))
        }
    }
}

async fn value(
    lhs: Value,
    rhs: Value,
    operation: LogicalBinaryOperator,
    contract: &contract::BinaryContract,
) -> BuiltinResult<Value> {
    enforce_extensions(&lhs, &rhs, contract)?;
    let output_source = provider::select_binary_source([&lhs, &rhs], contract.name)?;
    if let (Value::GpuTensor(left), Value::GpuTensor(right)) = (&lhs, &rhs) {
        if let Some(value) = provider::binary_hook(left, right, contract.name, operation)? {
            return Ok(value);
        }
    }
    let left = operand::from_value(contract.name, lhs, contract.invalid).await?;
    let right = operand::from_value(contract.name, rhs, contract.invalid).await?;
    let shape = broadcast_shapes(contract.name, &left.shape, &right.shape)
        .map_err(|error| operand::error_with_detail(contract.name, contract.mismatch, error))?;
    let data = evaluate::binary(&left, &right, &shape, operation);
    let value = operand::into_value(contract.name, data, shape, contract.invalid)?;
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
