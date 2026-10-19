use crate::{MirOperand, MirPlace, MirRvalue, MirStmt, MirStmtKind};
use runmat_hir::Span;

use super::super::MirLoweringContext;

pub(super) fn is_bound_operand(ctx: &MirLoweringContext, operand: &MirOperand) -> bool {
    matches!(operand, MirOperand::Local(local) if ctx.binding_for_local(*local).is_some())
}

/// A bound local left in an enclosing rvalue is loaded only when that rvalue
/// executes. Materialize it at its source position if a later operand has
/// preparation statements which may run before the enclosing rvalue.
pub(super) fn capture_binding_before_later_preparation(
    ctx: &MirLoweringContext,
    operand: &mut MirOperand,
    span: Span,
    temps: &mut Vec<MirStmt>,
) {
    let MirOperand::Local(local) = operand else {
        return;
    };
    if ctx.binding_for_local(*local).is_none() {
        return;
    }
    let snapshot = ctx.fresh_temp(span);
    temps.push(MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(snapshot),
            value: MirRvalue::Use(MirOperand::Local(*local)),
        },
        span,
    });
    *operand = MirOperand::Local(snapshot);
}

pub(super) fn append_after_prior<'a>(
    ctx: &MirLoweringContext,
    prior: impl IntoIterator<Item = (&'a mut MirOperand, Span)>,
    mut preparation: Vec<MirStmt>,
    temps: &mut Vec<MirStmt>,
) {
    if preparation.is_empty() {
        return;
    }
    for (operand, span) in prior {
        capture_binding_before_later_preparation(ctx, operand, span, temps);
    }
    temps.append(&mut preparation);
}
