use std::collections::HashMap;

use runmat_hir::{ExprId, HirError, HirStmt, HirStmtKind, OutputTarget};

use crate::{MirOperand, MirStmt};

use super::MirLoweringContext;

#[path = "await_preparation_expression.rs"]
pub(super) mod expression;
#[path = "await_preparation_place.rs"]
mod place;
use expression::prepare_expr_prefix;
use place::{prepare_place_prefix, prepare_sequence_prefix};

pub(super) fn prepare_statement_prefix(
    ctx: &MirLoweringContext,
    statement: &HirStmt,
    awaited: ExprId,
    output: &mut Vec<MirStmt>,
    replacements: &mut HashMap<ExprId, MirOperand>,
) -> Result<(), HirError> {
    match &statement.kind {
        HirStmtKind::Assign(place, value, _) => {
            if prepare_place_prefix(ctx, place, awaited, output, replacements)? {
                return Ok(());
            }
            prepare_expr_prefix(ctx, value, awaited, output, replacements)?;
        }
        HirStmtKind::MultiAssign(targets, value, _) => {
            for target in &targets.targets {
                let found = match target {
                    OutputTarget::Place(place) => {
                        prepare_place_prefix(ctx, place, awaited, output, replacements)?
                    }
                    OutputTarget::Sequence(target) => {
                        prepare_sequence_prefix(ctx, target, awaited, output, replacements)?
                    }
                    OutputTarget::Discard => false,
                };
                if found {
                    return Ok(());
                }
            }
            prepare_expr_prefix(ctx, value, awaited, output, replacements)?;
        }
        HirStmtKind::SequenceAssign { target, value, .. } => {
            if prepare_sequence_prefix(ctx, target, awaited, output, replacements)? {
                return Ok(());
            }
            prepare_expr_prefix(ctx, value, awaited, output, replacements)?;
        }
        HirStmtKind::ExprStmt(value, _)
        | HirStmtKind::While { cond: value, .. }
        | HirStmtKind::For { range: value, .. }
        | HirStmtKind::Switch { expr: value, .. } => {
            prepare_expr_prefix(ctx, value, awaited, output, replacements)?;
        }
        HirStmtKind::If { cond, .. } => {
            prepare_expr_prefix(ctx, cond, awaited, output, replacements)?;
        }
        HirStmtKind::ParFor {
            range,
            maximum_workers,
            ..
        } => {
            if prepare_expr_prefix(ctx, range, awaited, output, replacements)? {
                return Ok(());
            }
            if let Some(maximum_workers) = maximum_workers {
                prepare_expr_prefix(ctx, maximum_workers, awaited, output, replacements)?;
            }
        }
        HirStmtKind::Spmd { header, .. } => {
            use runmat_hir::parallel::SpmdHeader;
            let values = match header {
                SpmdHeader::Default => Vec::new(),
                SpmdHeader::One(first) => vec![first],
                SpmdHeader::Two(first, second) => vec![first, second],
                SpmdHeader::Three(first, second, third) => vec![first, second, third],
            };
            for value in values {
                if prepare_expr_prefix(ctx, value, awaited, output, replacements)? {
                    return Ok(());
                }
            }
        }
        HirStmtKind::TryCatch { .. }
        | HirStmtKind::Global(_)
        | HirStmtKind::Persistent(_)
        | HirStmtKind::Break
        | HirStmtKind::Continue
        | HirStmtKind::Return
        | HirStmtKind::Import(_) => {}
    }
    Ok(())
}
