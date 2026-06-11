use crate::{
    MirAggregateKind, MirCall, MirCallArg, MirCallee, MirConstant, MirIndexComponent, MirIndexPlan,
    MirIndexing, MirOperand, MirPlace, MirRvalue, MirShortCircuitOp, MirStmt, MirStmtKind,
};
use runmat_builtins::{BuiltinAsyncBehavior, BuiltinSemantics};
use runmat_hir::{
    CallableIdentity, CommandArgument, ExprId, HirCallableRef, HirCommandCall, HirError, HirExpr,
    HirExprKind, IndexComponent, IndexKind, IndexResultContext, IndexingSemantics, OperatorKind,
    RequestedOutputCount, StringLiteral,
};
use std::collections::HashMap;

use super::MirLoweringContext;

pub(crate) fn lower_expr_with_replacements(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirRvalue, HirError> {
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
        HirExprKind::Binary(left, op, right) => match op {
            OperatorKind::ShortCircuitAnd | OperatorKind::ShortCircuitOr => {
                let left = lower_operand_with_replacements(ctx, left, temps, await_replacements)?;
                let mut right_temps = Vec::new();
                let right = lower_operand_with_replacements(
                    ctx,
                    right,
                    &mut right_temps,
                    await_replacements,
                )?;
                MirRvalue::ShortCircuit {
                    left,
                    op: if matches!(op, OperatorKind::ShortCircuitAnd) {
                        MirShortCircuitOp::And
                    } else {
                        MirShortCircuitOp::Or
                    },
                    right_temps,
                    right,
                }
            }
            _ => MirRvalue::Binary(
                lower_operand_with_replacements(ctx, left, temps, await_replacements)?,
                op.clone(),
                lower_operand_with_replacements(ctx, right, temps, await_replacements)?,
            ),
        },
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
        HirExprKind::StructLiteral(fields) => MirRvalue::StructLiteral {
            fields: fields
                .iter()
                .map(|(name, expr)| {
                    Ok((
                        name.clone(),
                        lower_operand_with_replacements(ctx, expr, temps, await_replacements)?,
                    ))
                })
                .collect::<Result<_, HirError>>()?,
        },
        HirExprKind::ObjectLiteral { class_name, fields } => MirRvalue::ObjectLiteral {
            class_name: class_name.clone(),
            fields: fields
                .iter()
                .map(|(name, expr)| {
                    Ok((
                        name.clone(),
                        lower_operand_with_replacements(ctx, expr, temps, await_replacements)?,
                    ))
                })
                .collect::<Result<_, HirError>>()?,
        },
        HirExprKind::Call(call) => {
            let mut arg_spans: Vec<runmat_hir::Span> =
                call.args.iter().map(|arg| arg.span).collect();
            let mut args: Vec<MirCallArg> = call
                .args
                .iter()
                .map(|arg| lower_call_arg(ctx, arg, temps, await_replacements))
                .collect::<Result<_, _>>()?;
            if let HirCallableRef::DynamicExpr(callee) = &call.callee {
                dynamic_call_rvalue(
                    call,
                    lower_operand_with_replacements(ctx, callee, temps, await_replacements)?,
                    args,
                    arg_spans,
                )?
            } else if call.callee.is_feval_builtin_like() {
                if args.is_empty() {
                    return Err(HirError::new("feval: missing function argument"));
                }
                let MirCallArg::Single(callee) = args.remove(0) else {
                    return Err(HirError::new(
                        "feval: function argument cannot be a comma-list expansion",
                    ));
                };
                arg_spans.remove(0);
                dynamic_call_rvalue(call, callee, args, arg_spans)?
            } else if let HirCallableRef::Function(function) = call.callee {
                if ctx.is_async_function(function) {
                    MirRvalue::Future {
                        function,
                        args,
                        syntax: call.syntax.clone(),
                        requested_outputs: call.requested_outputs.clone(),
                    }
                } else {
                    call_rvalue(call, args, arg_spans)?
                }
            } else {
                call_rvalue(call, args, arg_spans)?
            }
        }
        HirExprKind::CommandCall(call) => lower_command_call(call)?,
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
            MirRvalue::Use(MirOperand::FunctionHandle(target.identity()))
        }
        HirExprKind::AnonymousFunction(function) => MirRvalue::Use(MirOperand::FunctionHandle(
            CallableIdentity::AnonymousFunction(*function),
        )),
        HirExprKind::Await(_) => {
            return Err(HirError::new(
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
) -> Result<MirCallArg, HirError> {
    if let HirExprKind::Call(call) = &arg.kind {
        let requested_count = requested_output_count_for_arg_expansion(&call.requested_outputs);
        if requested_count > 1 {
            let operand = lower_operand_with_replacements(ctx, arg, temps, await_replacements)?;
            return Ok(MirCallArg::Expansion {
                base: operand,
                indices: Vec::new(),
                expand_all: true,
            });
        }
    }

    if matches!(
        &arg.kind,
        HirExprKind::Index(_, indexing)
            if matches!(indexing.result_context, IndexResultContext::FunctionArgumentExpansion)
    ) {
        let HirExprKind::Index(base, indexing) = &arg.kind else {
            unreachable!()
        };
        if indexing.kind != IndexKind::Brace {
            return Err(HirError::new(
                "comma-list expansion requires cell/content indexing",
            ));
        }
        let base = lower_operand_with_replacements(ctx, base, temps, await_replacements)?;
        let mut indices = Vec::new();
        let mut saw_colon = false;
        let mut saw_non_colon = false;
        for component in &indexing.components {
            match component {
                IndexComponent::Colon => {
                    saw_colon = true;
                    indices.push(MirOperand::Constant(MirConstant::String(StringLiteral(
                        ":".to_string(),
                    ))));
                }
                IndexComponent::Expr(expr) => {
                    saw_non_colon = true;
                    indices.push(lower_operand_with_replacements(
                        ctx,
                        expr,
                        temps,
                        await_replacements,
                    )?);
                }
                IndexComponent::Logical(expr) => {
                    saw_non_colon = true;
                    indices.push(lower_operand_with_replacements(
                        ctx,
                        expr,
                        temps,
                        await_replacements,
                    )?);
                }
                IndexComponent::End { offset, .. } => {
                    saw_non_colon = true;
                    if *offset == 0 {
                        let local = ctx.fresh_temp(arg.span);
                        temps.push(MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(local),
                                value: MirRvalue::End,
                            },
                            span: arg.span,
                        });
                        indices.push(MirOperand::Local(local));
                    } else {
                        let end_local = ctx.fresh_temp(arg.span);
                        temps.push(MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(end_local),
                                value: MirRvalue::End,
                            },
                            span: arg.span,
                        });
                        let local = ctx.fresh_temp(arg.span);
                        let op = if offset.is_negative() {
                            OperatorKind::Subtract
                        } else {
                            OperatorKind::Add
                        };
                        temps.push(MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(local),
                                value: MirRvalue::Binary(
                                    MirOperand::Local(end_local),
                                    op,
                                    MirOperand::Constant(MirConstant::Number(
                                        offset.unsigned_abs().to_string(),
                                    )),
                                ),
                            },
                            span: arg.span,
                        });
                        indices.push(MirOperand::Local(local));
                    }
                }
            }
        }
        let expand_all = saw_colon && !saw_non_colon;
        if expand_all {
            indices.clear();
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
) -> Result<Vec<MirOperand>, HirError> {
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
) -> Result<MirIndexing, HirError> {
    lower_indexing_with_replacements(ctx, indexing, temps, &HashMap::new())
}

pub(crate) fn lower_indexing_with_replacements(
    ctx: &MirLoweringContext,
    indexing: &IndexingSemantics,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirIndexing, HirError> {
    Ok(MirIndexing {
        kind: indexing.kind.clone(),
        plan: classify_mir_index_plan(indexing),
        components: indexing
            .components
            .iter()
            .enumerate()
            .map(|(dim, component)| {
                lower_index_component(ctx, dim, component, temps, await_replacements)
            })
            .collect::<Result<_, _>>()?,
        result_context: indexing.result_context.clone(),
        cell_expand_all: indexing.kind == IndexKind::Brace
            && indexing
                .components
                .iter()
                .all(|component| matches!(component, IndexComponent::Colon)),
    })
}

fn classify_mir_index_plan(indexing: &IndexingSemantics) -> MirIndexPlan {
    match indexing.kind {
        IndexKind::Brace => MirIndexPlan::Cell,
        IndexKind::Paren => {
            if indexing
                .components
                .iter()
                .any(index_component_needs_slice_expr)
            {
                MirIndexPlan::SliceExpr
            } else if indexing
                .components
                .iter()
                .all(index_component_is_definitely_scalar)
            {
                MirIndexPlan::Scalar
            } else {
                MirIndexPlan::Slice
            }
        }
    }
}

fn index_component_needs_slice_expr(component: &IndexComponent) -> bool {
    match component {
        IndexComponent::Colon => false,
        IndexComponent::End { offset, .. } => *offset != 0,
        IndexComponent::Expr(expr) | IndexComponent::Logical(expr) => {
            hir_expr_needs_slice_expr(expr)
        }
    }
}

fn hir_expr_needs_slice_expr(expr: &HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::End => true,
        // Preserve range selectors as structured slice metadata in MIR/bytecode,
        // rather than reclassifying them at runtime from temporary tensors.
        HirExprKind::Range(_, _, _) => true,
        HirExprKind::Unary(_, inner) => hir_expr_needs_slice_expr(inner),
        HirExprKind::Binary(left, _, right) => {
            hir_expr_needs_slice_expr(left) || hir_expr_needs_slice_expr(right)
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => rows
            .iter()
            .flat_map(|row| row.iter())
            .any(hir_expr_needs_slice_expr),
        HirExprKind::Call(call) => call.args.iter().any(hir_expr_needs_slice_expr),
        HirExprKind::CommandCall(_) => false,
        HirExprKind::Index(base, indexing) => {
            hir_expr_needs_slice_expr(base)
                || indexing
                    .components
                    .iter()
                    .any(index_component_needs_slice_expr)
        }
        HirExprKind::Member(base, _) => hir_expr_needs_slice_expr(base),
        HirExprKind::MemberDynamic(base, member) => {
            hir_expr_needs_slice_expr(base) || hir_expr_needs_slice_expr(member)
        }
        HirExprKind::Spawn(inner) => hir_expr_needs_slice_expr(inner),
        HirExprKind::Await(inner) => hir_expr_needs_slice_expr(inner),
        _ => false,
    }
}

fn index_component_is_definitely_scalar(component: &IndexComponent) -> bool {
    matches!(component, IndexComponent::Expr(expr) if hir_expr_is_definitely_scalar_index(expr))
}

fn hir_expr_is_definitely_scalar_index(expr: &HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::Number(_) => true,
        HirExprKind::Constant(name)
            if name.0.eq_ignore_ascii_case("true") || name.0.eq_ignore_ascii_case("false") =>
        {
            true
        }
        _ => false,
    }
}

fn lower_index_component(
    ctx: &MirLoweringContext,
    dim: usize,
    component: &IndexComponent,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirIndexComponent, HirError> {
    Ok(match component {
        IndexComponent::Colon => MirIndexComponent::Colon,
        IndexComponent::End { dim, offset } => MirIndexComponent::End {
            dim: *dim,
            offset: *offset,
        },
        IndexComponent::Expr(expr) => match expr.kind {
            HirExprKind::Colon => MirIndexComponent::Colon,
            HirExprKind::End => MirIndexComponent::End {
                dim: Some(dim),
                offset: 0,
            },
            _ => MirIndexComponent::Expr(lower_operand_with_replacements(
                ctx,
                expr,
                temps,
                await_replacements,
            )?),
        },
        IndexComponent::Logical(expr) => MirIndexComponent::Expr(lower_operand_with_replacements(
            ctx,
            expr,
            temps,
            await_replacements,
        )?),
    })
}

fn lower_command_call(call: &HirCommandCall) -> Result<MirRvalue, HirError> {
    let Some(identity) = call.command.identity() else {
        return Err(HirError::new(
            "command call requires a statically classified callee identity",
        ));
    };
    let callee = MirCallee::Static(identity);
    let semantics = call_semantics(&callee);
    let fallback_policy = call_fallback_policy(&callee, &runmat_hir::CallSyntax::Command);
    Ok(MirRvalue::Call(MirCall {
        callee,
        args: call
            .args
            .iter()
            .map(|arg| MirCallArg::Single(MirOperand::Constant(command_arg_constant(arg))))
            .collect(),
        arg_spans: Vec::new(),
        syntax: runmat_hir::CallSyntax::Command,
        requested_outputs: RequestedOutputCount::Zero,
        fallback_policy,
        workspace_first_name: None,
        async_behavior: map_async_behavior(semantics.async_behavior),
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    }))
}

fn call_rvalue(
    call: &runmat_hir::HirCall,
    args: Vec<MirCallArg>,
    arg_spans: Vec<runmat_hir::Span>,
) -> Result<MirRvalue, HirError> {
    let callee = match &call.callee {
        HirCallableRef::SuperConstructor {
            current_class,
            super_class,
        } => MirCallee::SuperConstructor {
            current_class: current_class.0.clone(),
            super_class: super_class
                .0
                .iter()
                .map(|segment| segment.0.as_str())
                .collect::<Vec<_>>()
                .join("."),
        },
        HirCallableRef::SuperMethod {
            current_class,
            super_class,
            method,
        } => MirCallee::SuperMethod {
            current_class: current_class.0.clone(),
            super_class: super_class
                .0
                .iter()
                .map(|segment| segment.0.as_str())
                .collect::<Vec<_>>()
                .join("."),
            method: method.0.clone(),
        },
        _ => {
            let Some(identity) = call.callee.identity() else {
                return Err(HirError::new(
                    "call requires either a static callable identity or a dynamic callee expression",
                ));
            };
            MirCallee::Static(identity)
        }
    };
    let semantics = call_semantics(&callee);
    let fallback_policy = call_fallback_policy(&callee, &call.syntax);
    Ok(MirRvalue::Call(MirCall {
        callee,
        args,
        arg_spans,
        syntax: call.syntax.clone(),
        requested_outputs: call.requested_outputs.clone(),
        fallback_policy,
        workspace_first_name: call.workspace_first_name.clone(),
        async_behavior: map_async_behavior(semantics.async_behavior),
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    }))
}

fn dynamic_call_rvalue(
    call: &runmat_hir::HirCall,
    callee: MirOperand,
    args: Vec<MirCallArg>,
    arg_spans: Vec<runmat_hir::Span>,
) -> Result<MirRvalue, HirError> {
    let callee = MirCallee::Dynamic(callee);
    let semantics = call_semantics(&callee);
    let fallback_policy = call_fallback_policy(&callee, &call.syntax);
    Ok(MirRvalue::Call(MirCall {
        callee,
        args,
        arg_spans,
        syntax: call.syntax.clone(),
        requested_outputs: call.requested_outputs.clone(),
        fallback_policy,
        workspace_first_name: call.workspace_first_name.clone(),
        async_behavior: map_async_behavior(semantics.async_behavior),
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    }))
}

fn requested_output_count_for_arg_expansion(requested: &RequestedOutputCount) -> usize {
    requested.fixed_count()
}

fn call_semantics(callee: &MirCallee) -> BuiltinSemantics {
    match callee {
        MirCallee::Static(CallableIdentity::Builtin(id)) => {
            runmat_builtins::builtin_function_by_name(&id.0)
                .map(|builtin| builtin.semantics())
                .or_else(|| runmat_builtins::builtin_semantics_for_name(&id.0))
                .unwrap_or_else(BuiltinSemantics::unknown)
        }
        MirCallee::Static(
            CallableIdentity::ExternalName(_)
            | CallableIdentity::DynamicName(_)
            | CallableIdentity::Imported(_)
            | CallableIdentity::Method(_)
            | CallableIdentity::AnonymousFunction(_),
        )
        | MirCallee::SuperConstructor { .. }
        | MirCallee::SuperMethod { .. }
        | MirCallee::Dynamic(_) => BuiltinSemantics::unknown(),
        MirCallee::Static(CallableIdentity::BoundFunction(_)) => BuiltinSemantics {
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

fn call_fallback_policy(
    callee: &MirCallee,
    syntax: &runmat_hir::CallSyntax,
) -> runmat_hir::CallableFallbackPolicy {
    if matches!(
        syntax,
        runmat_hir::CallSyntax::Method | runmat_hir::CallSyntax::DottedInvoke
    ) && !matches!(
        callee,
        MirCallee::Static(runmat_hir::CallableIdentity::BoundFunction(_))
            | MirCallee::SuperMethod { .. }
    ) {
        return runmat_hir::CallableFallbackPolicy::ObjectDispatch;
    }
    match callee {
        MirCallee::Static(runmat_hir::CallableIdentity::BoundFunction(_))
        | MirCallee::Static(runmat_hir::CallableIdentity::Builtin(_))
        | MirCallee::SuperConstructor { .. }
        | MirCallee::SuperMethod { .. } => runmat_hir::CallableFallbackPolicy::None,
        MirCallee::Static(runmat_hir::CallableIdentity::ExternalName(_)) => {
            runmat_hir::CallableFallbackPolicy::ExternalBoundary
        }
        MirCallee::Static(
            runmat_hir::CallableIdentity::DynamicName(_)
            | runmat_hir::CallableIdentity::Imported(_)
            | runmat_hir::CallableIdentity::Method(_)
            | runmat_hir::CallableIdentity::AnonymousFunction(_),
        )
        | MirCallee::Dynamic(_) => runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
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
    }
}

pub(crate) fn lower_operand(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
) -> Result<MirOperand, HirError> {
    lower_operand_with_replacements(ctx, expr, temps, &HashMap::new())
}

pub(crate) fn lower_operand_with_replacements(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirOperand, HirError> {
    if let Some(operand) = await_replacements.get(&expr.id) {
        return Ok(operand.clone());
    }
    if let Some(operand) = lower_simple_operand(ctx, expr)? {
        return Ok(operand);
    }

    let value = lower_expr_with_replacements(ctx, expr, temps, await_replacements)?;
    let local = ctx.fresh_temp(expr.span);
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
) -> Result<Option<MirOperand>, HirError> {
    Ok(Some(match &expr.kind {
        HirExprKind::Number(value) => MirOperand::Constant(MirConstant::Number(value.clone())),
        HirExprKind::String(value) => MirOperand::Constant(MirConstant::String(value.clone())),
        HirExprKind::Constant(name) => MirOperand::Constant(MirConstant::Symbol(name.clone())),
        HirExprKind::Binding(binding) => MirOperand::Local(ctx.local_for_binding(*binding)?),
        HirExprKind::FunctionHandle(target) => MirOperand::FunctionHandle(target.identity()),
        _ => return Ok(None),
    }))
}
