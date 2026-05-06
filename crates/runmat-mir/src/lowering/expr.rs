use crate::{MirCall, MirConstant, MirIndexing, MirOperand, MirRvalue};
use runmat_hir::{HirExpr, HirExprKind, SemanticError};

use super::MirLoweringContext;

pub(crate) fn lower_expr(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
) -> Result<MirRvalue, SemanticError> {
    Ok(match &expr.kind {
        HirExprKind::Number(value) => {
            MirRvalue::Use(MirOperand::Constant(MirConstant::Number(value.clone())))
        }
        HirExprKind::String(value) => {
            MirRvalue::Use(MirOperand::Constant(MirConstant::String(value.clone())))
        }
        HirExprKind::Constant(name) => {
            MirRvalue::Use(MirOperand::Constant(MirConstant::Symbol(name.clone())))
        }
        HirExprKind::Binding(binding) => {
            MirRvalue::Use(MirOperand::Local(ctx.local_for_binding(*binding)?))
        }
        HirExprKind::Unary(op, inner) => MirRvalue::Unary(op.clone(), lower_operand(ctx, inner)?),
        HirExprKind::Binary(left, op, right) => MirRvalue::Binary(
            lower_operand(ctx, left)?,
            op.clone(),
            lower_operand(ctx, right)?,
        ),
        HirExprKind::Call(call) => MirRvalue::Call(MirCall {
            callee: call.callee.clone(),
            args: call
                .args
                .iter()
                .map(|arg| lower_operand(ctx, arg))
                .collect::<Result<_, _>>()?,
            syntax: call.syntax.clone(),
            requested_outputs: call.requested_outputs.clone(),
        }),
        HirExprKind::Index(base, indexing) => MirRvalue::Index {
            base: lower_operand(ctx, base)?,
            indexing: MirIndexing {
                kind: indexing.kind.clone(),
                components: indexing.components.clone(),
                result_context: indexing.result_context.clone(),
            },
        },
        HirExprKind::Spawn(inner) => MirRvalue::Spawn(lower_operand(ctx, inner)?),
        HirExprKind::FunctionHandle(target) => {
            MirRvalue::Use(MirOperand::FunctionHandle(target.clone()))
        }
        HirExprKind::AnonymousFunction(function) => MirRvalue::Future(*function),
        _ => {
            return Err(SemanticError::new(
                "MIR lowering for expression is not implemented yet",
            ))
        }
    })
}

pub(crate) fn lower_operand(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
) -> Result<MirOperand, SemanticError> {
    match &expr.kind {
        HirExprKind::Number(value) => Ok(MirOperand::Constant(MirConstant::Number(value.clone()))),
        HirExprKind::String(value) => Ok(MirOperand::Constant(MirConstant::String(value.clone()))),
        HirExprKind::Constant(name) => Ok(MirOperand::Constant(MirConstant::Symbol(name.clone()))),
        HirExprKind::Binding(binding) => Ok(MirOperand::Local(ctx.local_for_binding(*binding)?)),
        HirExprKind::FunctionHandle(target) => Ok(MirOperand::FunctionHandle(target.clone())),
        HirExprKind::AnonymousFunction(function) => Ok(MirOperand::Constant(MirConstant::Symbol(
            runmat_hir::SymbolName(format!("anonymous#{}", function.0)),
        ))),
        _ => Err(SemanticError::new(
            "complex expression operands require MIR temporaries",
        )),
    }
}
