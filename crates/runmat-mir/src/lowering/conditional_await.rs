use runmat_hir::{ExprId, HirExpr, HirExprKind, HirStmt, OperatorKind};
use std::collections::HashSet;

use super::evaluation_order::{expression_children, statement_expressions};

pub(super) struct ConditionalAwaitGuard<'a> {
    pub expression: &'a HirExpr,
    pub left: &'a HirExpr,
    pub evaluate_when_true: bool,
    pub skipped_value: bool,
}

pub(super) fn conditional_await_guard<'a>(
    statement: &'a HirStmt,
    awaited: ExprId,
    ignored: &HashSet<ExprId>,
) -> Option<ConditionalAwaitGuard<'a>> {
    statement_expressions(statement)
        .into_iter()
        .find_map(|expression| conditional_await_guard_in_expr(expression, awaited, ignored))
}

pub(super) fn conditional_await_guard_in_expr<'a>(
    expression: &'a HirExpr,
    awaited: ExprId,
    ignored: &HashSet<ExprId>,
) -> Option<ConditionalAwaitGuard<'a>> {
    if let HirExprKind::Binary(left, operator, right) = &expression.kind {
        if contains(right, awaited) && !ignored.contains(&expression.id) {
            return match operator {
                OperatorKind::ShortCircuitAnd => Some(ConditionalAwaitGuard {
                    expression,
                    left,
                    evaluate_when_true: true,
                    skipped_value: false,
                }),
                OperatorKind::ShortCircuitOr => Some(ConditionalAwaitGuard {
                    expression,
                    left,
                    evaluate_when_true: false,
                    skipped_value: true,
                }),
                _ => expression_children(expression)
                    .into_iter()
                    .find_map(|child| conditional_await_guard_in_expr(child, awaited, ignored)),
            };
        }
    }
    expression_children(expression)
        .into_iter()
        .find_map(|child| conditional_await_guard_in_expr(child, awaited, ignored))
}

fn contains(expression: &HirExpr, sought: ExprId) -> bool {
    expression.id == sought
        || expression_children(expression)
            .into_iter()
            .any(|child| contains(child, sought))
}
