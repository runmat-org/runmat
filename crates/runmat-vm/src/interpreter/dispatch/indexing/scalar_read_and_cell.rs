use super::{
    execute_brace_operation, execute_brace_operation_from_stack, pop_index_base, pop_index_values,
    BraceIndexOperation, BraceIndexOutcome, BraceOutcomeExpectation, BraceStackRequest,
    RuntimeError,
};
use crate::interpreter::vars::InterpreterVars;

#[path = "scalar_read_and_cell/parentheses.rs"]
mod parentheses;

pub(super) fn dispatch<'a>(
    instr: &'a crate::bytecode::Instr,
    stack: &'a mut crate::interpreter::stack::InterpreterStack,
    vars: &'a mut InterpreterVars,
    function_registry: &'a crate::bytecode::FunctionRegistry,
    _pc: usize,
    runtime: &'a runmat_runtime::context::RuntimeContext,
    sequence_register: &'a mut super::super::SequenceState,
) -> impl std::future::Future<Output = Result<bool, RuntimeError>> + 'a {
    match instr {
        crate::bytecode::Instr::Index(num_indices) => futures::future::Either::Left(
            parentheses::read(*num_indices, stack, function_registry, runtime),
        ),
        _ => futures::future::Either::Right(cell(instr, stack, vars, runtime, sequence_register)),
    }
}

async fn cell(
    instr: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &mut InterpreterVars,
    runtime: &runmat_runtime::context::RuntimeContext,
    sequence_register: &mut super::super::SequenceState,
) -> Result<bool, RuntimeError> {
    match instr {
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
