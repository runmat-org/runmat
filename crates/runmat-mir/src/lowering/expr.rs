use crate::{
    MirAggregateKind, MirCall, MirCallArg, MirConstant, MirIndexComponent, MirIndexing, MirOperand,
    MirPlace, MirRvalue, MirStmt, MirStmtKind,
};
use runmat_builtins::{BuiltinAsyncBehavior, BuiltinSemantics};
use runmat_hir::{
    CommandArgument, ExprId, HirCallableRef, HirCommandCall, HirExpr, HirExprKind, IndexComponent,
    IndexKind, IndexResultContext, IndexingSemantics, RequestedOutputCount, SemanticError,
    StringLiteral,
};
use std::collections::HashMap;

use super::MirLoweringContext;

pub(crate) fn lower_expr_with_replacements(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirRvalue, SemanticError> {
    if let Some(operand) = await_replacements.get(&expr.id) {
        return Ok(MirRvalue::Use(operand.clone()));
    }

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
        HirExprKind::Unary(op, inner) => MirRvalue::Unary(
            op.clone(),
            lower_operand_with_replacements(ctx, inner, temps, await_replacements)?,
        ),
        HirExprKind::Binary(left, op, right) => MirRvalue::Binary(
            lower_operand_with_replacements(ctx, left, temps, await_replacements)?,
            op.clone(),
            lower_operand_with_replacements(ctx, right, temps, await_replacements)?,
        ),
        HirExprKind::Range(start, step, end) => MirRvalue::Range {
            start: lower_operand_with_replacements(ctx, start, temps, await_replacements)?,
            step: step
                .as_ref()
                .map(|step| lower_operand_with_replacements(ctx, step, temps, await_replacements))
                .transpose()?,
            end: lower_operand_with_replacements(ctx, end, temps, await_replacements)?,
        },
        HirExprKind::Tensor(rows) => MirRvalue::Aggregate {
            kind: MirAggregateKind::Tensor,
            rows: rows.len(),
            cols: aggregate_col_count(rows),
            elements: lower_aggregate_elements(ctx, rows, temps, await_replacements)?,
        },
        HirExprKind::Cell(rows) => MirRvalue::Aggregate {
            kind: MirAggregateKind::Cell,
            rows: rows.len(),
            cols: aggregate_col_count(rows),
            elements: lower_aggregate_elements(ctx, rows, temps, await_replacements)?,
        },
        HirExprKind::Call(call) => {
            let args = call
                .args
                .iter()
                .map(|arg| lower_call_arg(ctx, arg, temps, await_replacements))
                .collect::<Result<_, _>>()?;
            if let HirCallableRef::Function(function) = call.callee {
                if ctx.is_async_function(function) {
                    MirRvalue::Future {
                        function,
                        args,
                        syntax: call.syntax.clone(),
                        requested_outputs: call.requested_outputs.clone(),
                    }
                } else {
                    call_rvalue(call, args)
                }
            } else {
                call_rvalue(call, args)
            }
        }
        HirExprKind::CommandCall(call) => lower_command_call(call),
        HirExprKind::Index(base, indexing) => MirRvalue::Index {
            base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
            indexing: lower_indexing_with_replacements(ctx, indexing, temps, await_replacements)?,
        },
        HirExprKind::Member(base, member) => MirRvalue::Member {
            base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
            member: member.clone(),
        },
        HirExprKind::MemberDynamic(base, member) => MirRvalue::DynamicMember {
            base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
            member: lower_operand_with_replacements(ctx, member, temps, await_replacements)?,
        },
        HirExprKind::MetaClass(name) => MirRvalue::MetaClass(name.clone()),
        HirExprKind::Colon => MirRvalue::Colon,
        HirExprKind::End => MirRvalue::End,
        HirExprKind::Spawn(inner) => MirRvalue::Spawn(lower_operand_with_replacements(
            ctx,
            inner,
            temps,
            await_replacements,
        )?),
        HirExprKind::FunctionHandle(target) => {
            MirRvalue::Use(MirOperand::FunctionHandle(target.clone()))
        }
        HirExprKind::AnonymousFunction(function) => MirRvalue::Use(MirOperand::FunctionHandle(
            runmat_hir::FunctionHandleTarget::Anonymous(*function),
        )),
        HirExprKind::Await(_) => {
            return Err(SemanticError::new(
                "await expression was not lowered through an await terminator",
            ))
        }
    })
}

fn lower_call_arg(
    ctx: &MirLoweringContext,
    arg: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirCallArg, SemanticError> {
    if matches!(
        &arg.kind,
        HirExprKind::Index(_, indexing)
            if matches!(indexing.result_context, IndexResultContext::FunctionArgumentExpansion)
    ) {
        let HirExprKind::Index(base, indexing) = &arg.kind else {
            unreachable!()
        };
        if indexing.kind != IndexKind::Brace {
            return Err(SemanticError::new(
                "comma-list expansion requires cell/content indexing",
            ));
        }
        let base = lower_operand_with_replacements(ctx, base, temps, await_replacements)?;
        let mut indices = Vec::new();
        let mut expand_all = false;
        for component in &indexing.components {
            match component {
                IndexComponent::Colon => expand_all = true,
                IndexComponent::Expr(expr) => indices.push(lower_operand_with_replacements(
                    ctx,
                    expr,
                    temps,
                    await_replacements,
                )?),
                _ => {
                    return Err(SemanticError::new(
                        "MIR lowering for this expansion index is not implemented yet",
                    ))
                }
            }
        }
        Ok(MirCallArg::Expansion {
            base,
            indices,
            expand_all,
        })
    } else {
        let operand = lower_operand_with_replacements(ctx, arg, temps, await_replacements)?;
        Ok(MirCallArg::Single(operand))
    }
}

fn lower_aggregate_elements(
    ctx: &MirLoweringContext,
    rows: &[Vec<HirExpr>],
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<Vec<MirOperand>, SemanticError> {
    rows.iter()
        .flat_map(|row| row.iter())
        .map(|element| lower_operand_with_replacements(ctx, element, temps, await_replacements))
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
    lower_indexing_with_replacements(ctx, indexing, temps, &HashMap::new())
}

pub(crate) fn lower_indexing_with_replacements(
    ctx: &MirLoweringContext,
    indexing: &IndexingSemantics,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirIndexing, SemanticError> {
    Ok(MirIndexing {
        kind: indexing.kind.clone(),
        components: indexing
            .components
            .iter()
            .map(|component| lower_index_component(ctx, component, temps, await_replacements))
            .collect::<Result<_, _>>()?,
        result_context: indexing.result_context.clone(),
    })
}

fn lower_index_component(
    ctx: &MirLoweringContext,
    component: &IndexComponent,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirIndexComponent, SemanticError> {
    Ok(match component {
        IndexComponent::Colon => MirIndexComponent::Colon,
        IndexComponent::End { dim, offset } => MirIndexComponent::End {
            dim: *dim,
            offset: *offset,
        },
        IndexComponent::Expr(expr) => MirIndexComponent::Expr(lower_operand_with_replacements(
            ctx,
            expr,
            temps,
            await_replacements,
        )?),
        IndexComponent::Logical(expr) => MirIndexComponent::Logical(
            lower_operand_with_replacements(ctx, expr, temps, await_replacements)?,
        ),
    })
}

fn lower_command_call(call: &HirCommandCall) -> MirRvalue {
    let semantics = call_semantics(&call.command);
    MirRvalue::Call(MirCall {
        callee: call.command.clone(),
        args: call
            .args
            .iter()
            .map(|arg| MirCallArg::Single(MirOperand::Constant(command_arg_constant(arg))))
            .collect(),
        syntax: runmat_hir::CallSyntax::Command,
        requested_outputs: RequestedOutputCount::Zero,
        async_behavior: map_async_behavior(semantics.async_behavior),
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    })
}

fn call_rvalue(call: &runmat_hir::HirCall, args: Vec<MirCallArg>) -> MirRvalue {
    let semantics = call_semantics(&call.callee);
    MirRvalue::Call(MirCall {
        callee: call.callee.clone(),
        args,
        syntax: call.syntax.clone(),
        requested_outputs: call.requested_outputs.clone(),
        async_behavior: map_async_behavior(semantics.async_behavior),
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    })
}

fn call_semantics(callee: &HirCallableRef) -> BuiltinSemantics {
    match callee {
        HirCallableRef::Builtin(id) => runmat_builtins::builtin_function_by_name(&id.0)
            .map(|builtin| builtin.semantics())
            .or_else(|| runmat_builtins::builtin_semantics_for_name(&id.0))
            .unwrap_or_else(BuiltinSemantics::unknown),
        HirCallableRef::Unresolved(_) => BuiltinSemantics::unknown(),
        HirCallableRef::DynamicExpr(_) | HirCallableRef::Imported(_) => BuiltinSemantics::unknown(),
        HirCallableRef::Function(_) | HirCallableRef::ClassConstructor(_) => BuiltinSemantics {
            compatibility: runmat_builtins::BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            effects: runmat_builtins::BuiltinEffects::none(),
            workspace_effect: None,
            environment_effect: None,
            purity: runmat_builtins::BuiltinPurity::Impure,
            semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
        },
    }
}

fn map_async_behavior(behavior: BuiltinAsyncBehavior) -> crate::AsyncBehaviorFact {
    match behavior {
        BuiltinAsyncBehavior::NeverSuspends => crate::AsyncBehaviorFact::NeverSuspends,
        BuiltinAsyncBehavior::MaySuspend => crate::AsyncBehaviorFact::MaySuspend,
        BuiltinAsyncBehavior::RequiresAsyncRuntime => {
            crate::AsyncBehaviorFact::RequiresAsyncRuntime
        }
    }
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
    lower_operand_with_replacements(ctx, expr, temps, &HashMap::new())
}

pub(crate) fn lower_operand_with_replacements(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirOperand, SemanticError> {
    if let Some(operand) = await_replacements.get(&expr.id) {
        return Ok(operand.clone());
    }
    if let Some(operand) = lower_simple_operand(ctx, expr)? {
        return Ok(operand);
    }

    let value = lower_expr_with_replacements(ctx, expr, temps, await_replacements)?;
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
