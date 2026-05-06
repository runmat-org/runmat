use crate::{
    MirAggregateKind, MirCall, MirCallArg, MirConstant, MirIndexComponent, MirIndexing, MirOperand,
    MirPlace, MirRvalue, MirStmt, MirStmtKind,
};
use runmat_hir::{
    CommandArgument, HirCommandCall, HirExpr, HirExprKind, IndexComponent, IndexResultContext,
    IndexingSemantics, RequestedOutputCount, SemanticError, StringLiteral,
};

use super::MirLoweringContext;

pub(crate) fn lower_expr(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
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
        HirExprKind::Unary(op, inner) => {
            MirRvalue::Unary(op.clone(), lower_operand(ctx, inner, temps)?)
        }
        HirExprKind::Binary(left, op, right) => MirRvalue::Binary(
            lower_operand(ctx, left, temps)?,
            op.clone(),
            lower_operand(ctx, right, temps)?,
        ),
        HirExprKind::Range(start, step, end) => MirRvalue::Range {
            start: lower_operand(ctx, start, temps)?,
            step: step
                .as_ref()
                .map(|step| lower_operand(ctx, step, temps))
                .transpose()?,
            end: lower_operand(ctx, end, temps)?,
        },
        HirExprKind::Tensor(rows) => MirRvalue::Aggregate {
            kind: MirAggregateKind::Tensor,
            rows: rows.len(),
            cols: aggregate_col_count(rows),
            elements: lower_aggregate_elements(ctx, rows, temps)?,
        },
        HirExprKind::Cell(rows) => MirRvalue::Aggregate {
            kind: MirAggregateKind::Cell,
            rows: rows.len(),
            cols: aggregate_col_count(rows),
            elements: lower_aggregate_elements(ctx, rows, temps)?,
        },
        HirExprKind::Call(call) => MirRvalue::Call(MirCall {
            callee: call.callee.clone(),
            args: call
                .args
                .iter()
                .map(|arg| lower_call_arg(ctx, arg, temps))
                .collect::<Result<_, _>>()?,
            syntax: call.syntax.clone(),
            requested_outputs: call.requested_outputs.clone(),
        }),
        HirExprKind::CommandCall(call) => lower_command_call(call),
        HirExprKind::Index(base, indexing) => MirRvalue::Index {
            base: lower_operand(ctx, base, temps)?,
            indexing: lower_indexing(ctx, indexing, temps)?,
        },
        HirExprKind::Spawn(inner) => MirRvalue::Spawn(lower_operand(ctx, inner, temps)?),
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

fn lower_call_arg(
    ctx: &MirLoweringContext,
    arg: &HirExpr,
    temps: &mut Vec<MirStmt>,
) -> Result<MirCallArg, SemanticError> {
    let operand = lower_operand(ctx, arg, temps)?;
    if matches!(
        &arg.kind,
        HirExprKind::Index(_, indexing)
            if matches!(indexing.result_context, IndexResultContext::FunctionArgumentExpansion)
    ) {
        Ok(MirCallArg::Expansion(operand))
    } else {
        Ok(MirCallArg::Single(operand))
    }
}

fn lower_aggregate_elements(
    ctx: &MirLoweringContext,
    rows: &[Vec<HirExpr>],
    temps: &mut Vec<MirStmt>,
) -> Result<Vec<MirOperand>, SemanticError> {
    rows.iter()
        .flat_map(|row| row.iter())
        .map(|element| lower_operand(ctx, element, temps))
        .collect()
}

fn aggregate_col_count(rows: &[Vec<HirExpr>]) -> usize {
    rows.iter().map(Vec::len).max().unwrap_or(0)
}

pub(crate) fn lower_indexing(
    ctx: &MirLoweringContext,
    indexing: &IndexingSemantics,
    temps: &mut Vec<MirStmt>,
) -> Result<MirIndexing, SemanticError> {
    Ok(MirIndexing {
        kind: indexing.kind.clone(),
        components: indexing
            .components
            .iter()
            .map(|component| lower_index_component(ctx, component, temps))
            .collect::<Result<_, _>>()?,
        result_context: indexing.result_context.clone(),
    })
}

fn lower_index_component(
    ctx: &MirLoweringContext,
    component: &IndexComponent,
    temps: &mut Vec<MirStmt>,
) -> Result<MirIndexComponent, SemanticError> {
    Ok(match component {
        IndexComponent::Colon => MirIndexComponent::Colon,
        IndexComponent::End { dim, offset } => MirIndexComponent::End {
            dim: *dim,
            offset: *offset,
        },
        IndexComponent::Expr(expr) => MirIndexComponent::Expr(lower_operand(ctx, expr, temps)?),
        IndexComponent::Logical(expr) => {
            MirIndexComponent::Logical(lower_operand(ctx, expr, temps)?)
        }
    })
}

fn lower_command_call(call: &HirCommandCall) -> MirRvalue {
    MirRvalue::Call(MirCall {
        callee: call.command.clone(),
        args: call
            .args
            .iter()
            .map(|arg| MirCallArg::Single(MirOperand::Constant(command_arg_constant(arg))))
            .collect(),
        syntax: runmat_hir::CallSyntax::Command,
        requested_outputs: RequestedOutputCount::Zero,
    })
}

fn command_arg_constant(arg: &CommandArgument) -> MirConstant {
    match arg {
        CommandArgument::Word(word) => MirConstant::String(StringLiteral(word.0.clone())),
        CommandArgument::StringLiteral(value) => {
            MirConstant::String(StringLiteral(value.0.trim_matches('"').to_string()))
        }
        CommandArgument::OptionToken(option) => {
            MirConstant::String(StringLiteral(option.0.clone()))
        }
    }
}

pub(crate) fn lower_operand(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
) -> Result<MirOperand, SemanticError> {
    if let Some(operand) = lower_simple_operand(ctx, expr)? {
        return Ok(operand);
    }

    let value = lower_expr(ctx, expr, temps)?;
    let local = ctx.fresh_temp(expr.span, Some(expr.id));
    temps.push(MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(local),
            value,
        },
        span: expr.span,
    });
    Ok(MirOperand::Local(local))
}

pub(crate) fn lower_simple_operand(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
) -> Result<Option<MirOperand>, SemanticError> {
    Ok(Some(match &expr.kind {
        HirExprKind::Number(value) => MirOperand::Constant(MirConstant::Number(value.clone())),
        HirExprKind::String(value) => MirOperand::Constant(MirConstant::String(value.clone())),
        HirExprKind::Constant(name) => MirOperand::Constant(MirConstant::Symbol(name.clone())),
        HirExprKind::Binding(binding) => MirOperand::Local(ctx.local_for_binding(*binding)?),
        HirExprKind::FunctionHandle(target) => MirOperand::FunctionHandle(target.clone()),
        _ => return Ok(None),
    }))
}
