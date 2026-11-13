use crate::bytecode::{Bytecode, EmitLabel, Instr};
use crate::interpreter::errors::mex;
use runmat_hir::FunctionId;
use runmat_runtime::RuntimeError;

use super::DynamicEvalCompileFailure;

/// One source-derived terminal expression retained for a dynamic caller.
/// The display bit belongs to the same compiler fact as the slot: it records
/// whether the original source statement was unsuppressed before its output
/// opcode was converted into an internal capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamicEvalResult {
    pub slot: usize,
    pub source_display: bool,
}

/// The internal numeric-text source is one bracketed expression statement.
/// Await lowering can put its output before a trailing control-flow return, so
/// it cannot use eval's terminal-expression capture rule.
#[derive(Clone, Copy)]
pub(super) enum DynamicEvalResultCapture {
    TerminalExpression(DynamicTerminalOutput),
    ExpressionStatement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedDynamicEvalResultCapture {
    TerminalExpression {
        source_display: bool,
        output: DynamicTerminalOutput,
    },
    ExpressionStatement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DynamicTerminalOutput {
    /// Preserve the source statement's ordinary inline output behavior.
    Preserve,
    /// Suppress inline output and emit the retained result through evalc's
    /// capture-aware provider after earlier source effects have replayed.
    Capture,
    /// Suppress inline output because the retained value is returned as an
    /// additional evalc output rather than displayed.
    Suppress,
}

pub(super) fn prepare_dynamic_result_capture(
    capture: DynamicEvalResultCapture,
    hir: &runmat_hir::HirAssembly,
    entrypoint: FunctionId,
) -> Result<PreparedDynamicEvalResultCapture, DynamicEvalCompileFailure> {
    match capture {
        DynamicEvalResultCapture::TerminalExpression(output) => {
            let function = hir
                .functions
                .iter()
                .find(|function| function.id == entrypoint)
                .ok_or_else(|| {
                    DynamicEvalCompileFailure::from(mex(
                        "DynamicWorkspaceCompile",
                        "eval entrypoint has no HIR body",
                    ))
                })?;
            let source_display = matches!(
                function
                    .body
                    .statements
                    .last()
                    .map(|statement| &statement.kind),
                Some(runmat_hir::HirStmtKind::ExprStmt(expression, false))
                    if expression.emits_statement_output()
            );
            Ok(PreparedDynamicEvalResultCapture::TerminalExpression {
                source_display,
                output,
            })
        }
        DynamicEvalResultCapture::ExpressionStatement => {
            Ok(PreparedDynamicEvalResultCapture::ExpressionStatement)
        }
    }
}

pub(super) fn retain_dynamic_result_slot(
    bytecode: &mut Bytecode,
    requested_outputs: usize,
    workspace_extent: usize,
    capture: PreparedDynamicEvalResultCapture,
) -> Result<Option<DynamicEvalResult>, RuntimeError> {
    if requested_outputs == 0 {
        return Ok(None);
    }
    let expression_pop = match capture {
        PreparedDynamicEvalResultCapture::TerminalExpression { .. } => {
            matches!(bytecode.instructions.last(), Some(Instr::Pop))
                .then_some(bytecode.instructions.len().saturating_sub(1))
        }
        PreparedDynamicEvalResultCapture::ExpressionStatement => {
            let mut positions =
                bytecode
                    .instructions
                    .windows(2)
                    .enumerate()
                    .filter_map(|(pc, pair)| {
                        matches!(pair, [Instr::EmitStackTop { .. }, Instr::Pop]).then_some(pc + 1)
                    });
            let Some(position) = positions.next() else {
                return Ok(None);
            };
            positions.next().is_none().then_some(position)
        }
    };
    let Some(expression_pop) = expression_pop else {
        return Ok(None);
    };
    let temp_slot = bytecode.var_count.max(workspace_extent);
    match capture {
        PreparedDynamicEvalResultCapture::TerminalExpression {
            source_display,
            output,
        } => {
            if source_display && output != DynamicTerminalOutput::Preserve {
                let Some(Instr::EmitStackTop { label }) = expression_pop
                    .checked_sub(1)
                    .and_then(|pc| bytecode.instructions.get_mut(pc))
                else {
                    return Err(mex(
                        "DynamicWorkspaceCompile",
                        "unsuppressed eval result has no statement output",
                    ));
                };
                *label = EmitLabel::InternalCapture;
            }
            bytecode.instructions[expression_pop] = Instr::StoreVar(temp_slot);
        }
        PreparedDynamicEvalResultCapture::ExpressionStatement => {
            // Keep the instruction count and all resume-point PCs unchanged.
            // The preceding typed statement-output opcode proves this Pop
            // consumes the expression being converted. Its capture-only label
            // prevents internal conversion from publishing an `ans` echo or
            // gathering a resident result solely for display.
            let Instr::EmitStackTop { label } = &mut bytecode.instructions[expression_pop - 1]
            else {
                unreachable!("expression capture site must follow a statement output");
            };
            *label = EmitLabel::InternalCapture;
            bytecode.instructions[expression_pop] = Instr::StoreVar(temp_slot);
        }
    }
    bytecode.var_count = temp_slot
        .checked_add(1)
        .ok_or_else(super::dynamic_workspace_slot_overflow)?;
    let source_display = matches!(
        capture,
        PreparedDynamicEvalResultCapture::TerminalExpression {
            source_display: true,
            ..
        }
    );
    Ok(Some(DynamicEvalResult {
        slot: temp_slot,
        source_display,
    }))
}
