use super::*;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::StoreIndexCell {
            num_indices,
            end_offsets,
            end_exprs,
        }
        | crate::bytecode::Instr::StoreIndexCellDelete {
            num_indices,
            end_offsets,
            end_exprs,
        } => {
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
                *num_indices,
                end_offsets,
                end_exprs,
                vars,
                BraceIndexOperation::Store { rhs },
                BraceOutcomeExpectation::SingleValue {
                    invalid_message: "StoreIndexCell expected a single base value",
                },
            )
            .await?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
