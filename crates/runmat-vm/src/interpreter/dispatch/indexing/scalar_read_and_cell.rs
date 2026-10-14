use super::{
    execute_brace_operation, execute_brace_operation_from_stack, paren_index_value, pop_index_base,
    pop_index_values, BraceIndexOperation, BraceIndexOutcome, BraceOutcomeExpectation,
    BraceStackRequest, RuntimeError, Value,
};

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
    function_registry: &crate::bytecode::FunctionRegistry,
    _pc: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
    sequence_register: &mut super::super::SequenceState,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::Index(num_indices) => {
            let mut raw_indices = Vec::with_capacity(*num_indices);
            for _ in 0..*num_indices {
                raw_indices.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            raw_indices.reverse();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            paren_index_value(runtime, base, raw_indices, 1, function_registry)
                .await?
                .with_owned(|value| stack.push(value));
            Ok(true)
        }
        crate::bytecode::Instr::IndexCell { num_indices } => {
            execute_brace_operation_from_stack(
                stack,
                vars,
                BraceStackRequest {
                    num_indices: *num_indices,
                    operation: BraceIndexOperation::ReadSingle,
                    expectation: BraceOutcomeExpectation::SingleValue {
                        invalid_message: "IndexCell expected a single value outcome",
                    },
                },
                runtime,
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::IndexCellExpand {
            num_indices,
            out_count,
        } => {
            execute_brace_operation_from_stack(
                stack,
                vars,
                BraceStackRequest {
                    num_indices: *num_indices,
                    operation: BraceIndexOperation::Expand {
                        out_count: *out_count,
                    },
                    expectation: BraceOutcomeExpectation::ExpandedValues {
                        invalid_message: "IndexCellExpand expected an expanded value list",
                    },
                },
                runtime,
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::IndexCellList { num_indices } => {
            let raw_indices = pop_index_values(stack, *num_indices)?;
            let base = pop_index_base(stack)?;
            let BraceIndexOutcome::Expanded(values) =
                execute_brace_operation(base, &raw_indices, BraceIndexOperation::List, runtime)
                    .await?
            else {
                return Err(crate::interpreter::errors::mex(
                    "InvalidBraceIndexOutcome",
                    "IndexCellList expected a transient value sequence",
                ));
            };
            super::super::object::write_rooted_sequence_register(stack, sequence_register, values)?;
            Ok(true)
        }
        crate::bytecode::Instr::StoreIndexCell { num_indices }
        | crate::bytecode::Instr::StoreIndexCellDelete { num_indices } => {
            let delete = matches!(instr, crate::bytecode::Instr::StoreIndexCellDelete { .. });
            if delete {
                return Err(crate::interpreter::errors::mex(
                    "UnsupportedCellBraceDeletion",
                    "Cell brace assignment does not support deletion",
                ));
            }
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            execute_brace_operation_from_stack(
                stack,
                vars,
                BraceStackRequest {
                    num_indices: *num_indices,
                    operation: BraceIndexOperation::Store { rhs },
                    expectation: BraceOutcomeExpectation::SingleValue {
                        invalid_message: "StoreIndexCell expected a single base value",
                    },
                },
                runtime,
            )
            .await?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
