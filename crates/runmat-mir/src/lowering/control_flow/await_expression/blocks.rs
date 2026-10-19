use std::collections::HashMap;

use runmat_hir::{ExprId, HirError, HirExpr, Span};

use crate::{
    BasicBlock, BasicBlockId, MirConstant, MirLocalId, MirOperand, MirPlace, MirRvalue, MirStmt,
    MirStmtKind, MirTerminator, MirTerminatorKind,
};

use crate::lowering::{
    expr::lower_expr_with_replacements, stmt::effect_stmts_for_rvalue, MirLoweringContext,
};

use super::ExpressionTarget;

pub(super) fn finish_expression(
    ctx: &MirLoweringContext,
    expression: &HirExpr,
    target: ExpressionTarget,
    replacements: &HashMap<ExprId, MirOperand>,
    mut statements: Vec<MirStmt>,
) -> Result<BasicBlock, HirError> {
    let value = lower_expr_with_replacements(ctx, expression, &mut statements, replacements)?;
    statements.extend(effect_stmts_for_rvalue(&value, target.span));
    statements.push(MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(target.destination),
            value,
        },
        span: target.span,
    });
    Ok(BasicBlock {
        id: target.block,
        statements,
        terminator: MirTerminator {
            kind: MirTerminatorKind::Goto(target.continuation),
            span: target.span,
        },
    })
}

pub(super) fn skipped_guard(
    id: BasicBlockId,
    value: bool,
    destination: MirLocalId,
    continuation: BasicBlockId,
    span: Span,
) -> BasicBlock {
    BasicBlock {
        id,
        statements: vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(destination),
                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Bool(value))),
            },
            span,
        }],
        terminator: MirTerminator {
            kind: MirTerminatorKind::Goto(continuation),
            span,
        },
    }
}

pub(super) fn branch(
    id: BasicBlockId,
    statements: Vec<MirStmt>,
    cond: MirOperand,
    then_block: BasicBlockId,
    else_block: BasicBlockId,
    span: Span,
) -> BasicBlock {
    BasicBlock {
        id,
        statements,
        terminator: MirTerminator {
            kind: MirTerminatorKind::Branch {
                cond,
                then_block,
                else_block,
            },
            span,
        },
    }
}
