use super::*;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::IndexCell {
            num_indices,
            end_offsets,
            end_exprs,
        } => {
            execute_brace_operation_from_stack(
                stack,
                *num_indices,
                end_offsets,
                end_exprs,
                vars,
                BraceIndexOperation::ReadSingle,
                BraceOutcomeExpectation::SingleValue {
                    invalid_message: "IndexCell expected a single value outcome",
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::IndexCellExpand {
            num_indices,
            out_count,
            end_offsets,
            end_exprs,
        } => {
            execute_brace_operation_from_stack(
                stack,
                *num_indices,
                end_offsets,
                end_exprs,
                vars,
                BraceIndexOperation::Expand {
                    out_count: *out_count,
                },
                BraceOutcomeExpectation::ExpandedValues {
                    invalid_message: "IndexCellExpand expected an expanded value list",
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::IndexCellList {
            num_indices,
            end_offsets,
            end_exprs,
        } => {
            execute_brace_operation_from_stack(
                stack,
                *num_indices,
                end_offsets,
                end_exprs,
                vars,
                BraceIndexOperation::List,
                BraceOutcomeExpectation::SingleValue {
                    invalid_message: "IndexCellList expected a single list value",
                },
            )
            .await?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
