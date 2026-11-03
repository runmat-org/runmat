use crate::{
    MirOutputTarget, MirOutputTargetList, MirPlace, MirPlaceMutation, MirRvalue,
    MirStatementOutput, MirStatementOutputLabel, MirStatementOutputSource, MirStmt, MirStmtKind,
};
use runmat_hir::{
    AssignmentShapePolicy, EnvironmentEffect, HirError, HirExpr, HirExprKind, HirSequenceTarget,
    HirStmt, HirStmtKind, OutputTarget, PlaceMutationKind, RequestedOutputCount, Span,
    WorkspaceEffect,
};

use super::{
    expr::lower_expr_with_replacements,
    place::{lower_expr_place_with_replacements, lower_place},
    MirLoweringContext,
};

pub(crate) fn lower_stmt_with_replacements(
    ctx: &MirLoweringContext,
    stmt: &HirStmt,
    await_replacements: &crate::lowering::AwaitReplacements,
    prepared_assignment: Option<&crate::MirPreparedAssignment>,
) -> Result<Vec<MirStmt>, HirError> {
    Ok(match &stmt.kind {
        HirStmtKind::Assign(place, expr, suppressed) => {
            let mut stmts = Vec::new();
            let retained = prepared_assignment.filter(|prepared| prepared.source_statement == stmt.id);
            let (place, preparation) = if let Some(prepared) = retained {
                (prepared.mutation.place.clone(), Some(prepared.clone()))
            } else if let Some((prepared, phases)) = super::fixed_target::lower_completed(ctx, stmt, await_replacements)? {
                stmts.extend(phases);
                (prepared.mutation.place.clone(), Some(prepared))
            } else {
                let mut target_steps = Vec::new();
                let place = lower_place_with_replacements(ctx, place, &mut target_steps, await_replacements)?;
                stmts.extend(target_steps);
                (place, None)
            };
            let value = lower_expr_with_replacements(ctx, expr, &mut stmts, await_replacements)?;
            stmts.extend(effect_stmts_for_rvalue(&value, stmt.span));
            let deletion = is_empty_array_deletion_place(&place, expr);
            if preparation.is_none() && (!matches!(place, crate::MirPlace::Local(_)) || deletion) {
                stmts.push(MirStmt {
                    kind: MirStmtKind::PlaceMutation(place_mutation(place.clone(), deletion)),
                    span: stmt.span,
                });
            }
            stmts.push(MirStmt {
                kind: match preparation {
                    Some(preparation) => MirStmtKind::PreparedAssign { preparation, value },
                    None => MirStmtKind::Assign { place: place.clone(), value },
                },
                span: stmt.span,
            });
            if !*suppressed {
                if let Some(output) = output_for_place(ctx, &place) {
                    stmts.push(MirStmt {
                        kind: MirStmtKind::StatementOutput(output),
                        span: stmt.span,
                    });
                }
            }
            stmts
        }
        HirStmtKind::MultiAssign(targets, expr, suppressed) => {
            let mut stmts = Vec::new();
            let lowered_targets = lower_output_targets(ctx, &targets.targets, &mut stmts)?;
            let value = lower_expr_with_replacements(ctx, expr, &mut stmts, await_replacements)?;
            stmts.extend(effect_stmts_for_rvalue(&value, stmt.span));
            let lowered_target_list = MirOutputTargetList {
                requested_outputs: targets.requested_outputs,
                targets: lowered_targets,
            };
            let requested_outputs = if lowered_target_list
                .targets
                .iter()
                .any(|target| matches!(target, MirOutputTarget::Sequence(_)))
            {
                if lowered_target_list.requested_outputs
                    != RequestedOutputCount::DestinationSequenceCardinality
                {
                    return Err(HirError::new(
                        "MIR sequence-target assignment is missing destination cardinality",
                    ));
                }
                None
            } else {
                Some(
                    lowered_target_list
                        .validate_fixed_arity("MIR multi-assign")
                        .map_err(HirError::new)?,
                )
            };
            if let MirRvalue::Call(call) = &value {
                if let Some(requested_outputs) = requested_outputs {
                    let call_outputs = fixed_requested_output_count(
                        &call.requested_outputs,
                        "MIR call requested output count",
                    )?;
                    if call_outputs != requested_outputs {
                    return Err(HirError::new(format!(
                        "MIR call requested outputs ({call_outputs}) must match multi-assign targets ({requested_outputs})"
                    )));
                    }
                }
            }
            stmts.push(MirStmt {
                kind: MirStmtKind::MultiAssign {
                    targets: lowered_target_list.clone(),
                    value,
                },
                span: stmt.span,
            });
            if !*suppressed {
                for target in &lowered_target_list.targets {
                    let place = match target {
                        MirOutputTarget::Place(place) => Some(place),
                        MirOutputTarget::Sequence(target) => Some(target.base()),
                        MirOutputTarget::Discard => None,
                    };
                    if let Some(output) = place.and_then(|place| output_for_place(ctx, place)) {
                        stmts.push(MirStmt {
                            kind: MirStmtKind::StatementOutput(output),
                            span: stmt.span,
                        });
                    }
                }
            }
            stmts
        }
        HirStmtKind::SequenceAssign {
            target,
            value,
            suppressed,
        } => {
            let mut stmts = Vec::new();
            let target = lower_sequence_target(ctx, target, &mut stmts, await_replacements)?;
            let value = lower_expr_with_replacements(ctx, value, &mut stmts, await_replacements)?;
            stmts.extend(effect_stmts_for_rvalue(&value, stmt.span));
            stmts.push(MirStmt {
                kind: MirStmtKind::MultiAssign {
                    targets: MirOutputTargetList {
                        targets: vec![MirOutputTarget::Sequence(target.clone())],
                        requested_outputs: RequestedOutputCount::DestinationSequenceCardinality,
                    },
                    value,
                },
                span: stmt.span,
            });
            if !*suppressed {
                if let Some(output) = output_for_place(ctx, target.base()) {
                    stmts.push(MirStmt {
                        kind: MirStmtKind::StatementOutput(output),
                        span: stmt.span,
                    });
                }
            }
            stmts
        }
        HirStmtKind::ExprStmt(expr, suppressed) => {
            let mut stmts = Vec::new();
            let mut value = lower_expr_with_replacements(ctx, expr, &mut stmts, await_replacements)?;
            stmts.extend(effect_stmts_for_rvalue(&value, stmt.span));
            if !*suppressed
                && expr.emits_statement_output()
                && !matches!(expr.kind, HirExprKind::Binding(_))
            {
                if let Some(ans) = ctx.statement_ans_local()? {
                    // Evaluate once, then publish through the ordinary binding
                    // assignment used by every executor. The temporary is also
                    // the exact source of statement output, never a console read.
                    let result = ctx.fresh_temp(stmt.span);
                    stmts.push(MirStmt {
                        kind: MirStmtKind::Assign {
                            place: MirPlace::Local(result),
                            value,
                        },
                        span: stmt.span,
                    });
                    stmts.push(MirStmt {
                        kind: MirStmtKind::Assign {
                            place: MirPlace::Local(ans),
                            value: MirRvalue::Use(crate::MirOperand::Local(result)),
                        },
                        span: stmt.span,
                    });
                    value = MirRvalue::Use(crate::MirOperand::Local(result));
                }
            }
            stmts.push(MirStmt {
                kind: MirStmtKind::Expr(value),
                span: stmt.span,
            });
            if !*suppressed && expr.emits_statement_output() {
                let label = match &expr.kind {
                    HirExprKind::Binding(binding) => MirStatementOutputLabel::Binding(*binding),
                    _ => MirStatementOutputLabel::Ans,
                };
                stmts.push(MirStmt {
                    kind: MirStmtKind::StatementOutput(MirStatementOutput {
                        source: MirStatementOutputSource::Expression,
                        label,
                    }),
                    span: stmt.span,
                });
            }
            stmts
        }
        HirStmtKind::Global(bindings) => vec![MirStmt {
            kind: MirStmtKind::WorkspaceEffect {
                effect: WorkspaceEffect::MutatesGlobal,
                bindings: bindings
                    .iter()
                    .map(|binding| ctx.local_for_binding(*binding))
                    .collect::<Result<_, _>>()?,
            },
            span: stmt.span,
        }],
        HirStmtKind::Persistent(bindings) => vec![MirStmt {
            kind: MirStmtKind::WorkspaceEffect {
                effect: WorkspaceEffect::MutatesPersistent,
                bindings: bindings
                    .iter()
                    .map(|binding| ctx.local_for_binding(*binding))
                    .collect::<Result<_, _>>()?,
            },
            span: stmt.span,
        }],
        HirStmtKind::Return | HirStmtKind::Import(_) => Vec::new(),
        _ => {
            return Err(HirError::new(format!(
                "unexpected statement in stmt-level MIR lowering; control-flow lowering should have handled: {:?}",
                stmt.kind
            )))
        }
    })
}

pub(super) fn assignment_target_requires_preparation(target: &runmat_hir::HirPlace) -> bool {
    fn indexed(indexing: &runmat_hir::IndexingSemantics) -> bool {
        indexing.components.iter().any(|component| {
            matches!(component,
            runmat_hir::IndexComponent::Expr(value) if value.expands_call_argument_sequence())
        })
    }
    fn expression(value: &HirExpr) -> bool {
        match &value.kind {
            HirExprKind::Index(base, indexing) => indexed(indexing) || expression(base),
            HirExprKind::Member { base, .. } | HirExprKind::MemberDynamic { base, .. } => {
                expression(base)
            }
            _ => false,
        }
    }
    let expands = match target {
        runmat_hir::HirPlace::Index(base, indexing)
        | runmat_hir::HirPlace::IndexCell(base, indexing) => indexed(indexing) || expression(base),
        runmat_hir::HirPlace::Member(base, _) | runmat_hir::HirPlace::MemberDynamic(base, _) => {
            expression(base)
        }
        runmat_hir::HirPlace::Binding(_) => false,
    };
    fn depth(value: &HirExpr) -> usize {
        match &value.kind {
            HirExprKind::Index(base, _)
            | HirExprKind::Member { base, .. }
            | HirExprKind::MemberDynamic { base, .. } => depth(base) + 1,
            _ => 0,
        }
    }
    let prefix_depth = match target {
        runmat_hir::HirPlace::Index(base, _)
        | runmat_hir::HirPlace::IndexCell(base, _)
        | runmat_hir::HirPlace::Member(base, _)
        | runmat_hir::HirPlace::MemberDynamic(base, _) => depth(base),
        runmat_hir::HirPlace::Binding(_) => 0,
    };
    expands || prefix_depth != 0
}

pub(super) fn prepare_expanding_assignment(
    ctx: &MirLoweringContext,
    statement: &HirStmt,
    replacements: &crate::lowering::AwaitReplacements,
) -> Result<Option<(crate::MirPreparedAssignment, Vec<MirStmt>)>, HirError> {
    super::fixed_target::lower_completed(ctx, statement, replacements)
}

fn output_for_place(ctx: &MirLoweringContext, place: &MirPlace) -> Option<MirStatementOutput> {
    let root = match place {
        MirPlace::Local(local) => *local,
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            return output_for_place(ctx, base)
        }
        MirPlace::Binding(_) => return None,
    };
    Some(MirStatementOutput {
        source: MirStatementOutputSource::Local(root),
        label: MirStatementOutputLabel::Binding(ctx.binding_for_local(root)?),
    })
}

fn lower_place_with_replacements(
    ctx: &MirLoweringContext,
    place: &runmat_hir::HirPlace,
    stmts: &mut Vec<MirStmt>,
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<crate::MirPlace, HirError> {
    use runmat_hir::HirPlace;
    Ok(match place {
        HirPlace::Binding(binding) => crate::MirPlace::Local(ctx.local_for_binding(*binding)?),
        HirPlace::Member(base, member) => crate::MirPlace::Member(
            Box::new(lower_expr_place_with_replacements(
                ctx,
                base,
                stmts,
                await_replacements,
            )?),
            member.clone(),
        ),
        HirPlace::MemberDynamic(base, member) => crate::MirPlace::DynamicMember(
            Box::new(lower_expr_place_with_replacements(
                ctx,
                base,
                stmts,
                await_replacements,
            )?),
            super::expr::lower_operand_with_replacements(ctx, member, stmts, await_replacements)?,
        ),
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
            crate::MirPlace::Index(
                Box::new(lower_expr_place_with_replacements(
                    ctx,
                    base,
                    stmts,
                    await_replacements,
                )?),
                super::expr::lower_indexing_with_replacements(
                    ctx,
                    indexing,
                    stmts,
                    await_replacements,
                )?,
            )
        }
    })
}

fn lower_sequence_target(
    ctx: &MirLoweringContext,
    target: &HirSequenceTarget,
    stmts: &mut Vec<MirStmt>,
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<crate::MirSequenceTarget, HirError> {
    use super::expr::lower_operand_with_replacements;
    use super::place::lower_expr_place_with_replacements;
    Ok(match target {
        HirSequenceTarget::Member { base, member } => crate::MirSequenceTarget::Member {
            base: lower_expr_place_with_replacements(ctx, base, stmts, await_replacements)?,
            member: member.clone(),
        },
        HirSequenceTarget::DynamicMember { base, member } => {
            crate::MirSequenceTarget::DynamicMember {
                base: lower_expr_place_with_replacements(ctx, base, stmts, await_replacements)?,
                member: lower_operand_with_replacements(ctx, member, stmts, await_replacements)?,
            }
        }
        HirSequenceTarget::CellContents { base, indexing } => {
            crate::MirSequenceTarget::CellContents {
                base: lower_expr_place_with_replacements(ctx, base, stmts, await_replacements)?,
                indexing: super::expr::lower_indexing_with_replacements(
                    ctx,
                    indexing,
                    stmts,
                    await_replacements,
                )?,
            }
        }
    })
}

pub(super) fn effect_stmts_for_rvalue(value: &MirRvalue, span: Span) -> Vec<MirStmt> {
    let MirRvalue::Call(call) = value else {
        return Vec::new();
    };
    let mut stmts = Vec::new();
    if let Some(effect) = call.workspace_effect.map(workspace_effect_from_builtin) {
        stmts.push(MirStmt {
            kind: MirStmtKind::WorkspaceEffect {
                effect,
                bindings: Vec::new(),
            },
            span,
        });
    }
    if let Some(effect) = call.environment_effect.map(environment_effect_from_builtin) {
        stmts.push(MirStmt {
            kind: MirStmtKind::EnvironmentEffect(effect),
            span,
        });
    }
    stmts
}

fn workspace_effect_from_builtin(
    effect: runmat_builtins::BuiltinWorkspaceEffect,
) -> WorkspaceEffect {
    match effect {
        runmat_builtins::BuiltinWorkspaceEffect::ReadsWorkspace => WorkspaceEffect::ReadsWorkspace,
        runmat_builtins::BuiltinWorkspaceEffect::CreatesBinding => WorkspaceEffect::CreatesBinding,
        runmat_builtins::BuiltinWorkspaceEffect::ClearsBinding => WorkspaceEffect::ClearsBinding,
        runmat_builtins::BuiltinWorkspaceEffect::ClearsFunctionCache => {
            WorkspaceEffect::ClearsFunctionCache
        }
        runmat_builtins::BuiltinWorkspaceEffect::LoadsExternalBindings => {
            WorkspaceEffect::LoadsExternalBindings
        }
        runmat_builtins::BuiltinWorkspaceEffect::DynamicEvalCurrentWorkspace
        | runmat_builtins::BuiltinWorkspaceEffect::DynamicEval => WorkspaceEffect::DynamicEval,
    }
}

fn environment_effect_from_builtin(
    effect: runmat_builtins::BuiltinEnvironmentEffect,
) -> EnvironmentEffect {
    match effect {
        runmat_builtins::BuiltinEnvironmentEffect::PathMutation => EnvironmentEffect::PathMutation,
        runmat_builtins::BuiltinEnvironmentEffect::WorkingDirectoryMutation => {
            EnvironmentEffect::WorkingDirectoryMutation
        }
        runmat_builtins::BuiltinEnvironmentEffect::FunctionCacheInvalidation => {
            EnvironmentEffect::FunctionCacheInvalidation
        }
        runmat_builtins::BuiltinEnvironmentEffect::DynamicLookupInvalidation => {
            EnvironmentEffect::DynamicLookupInvalidation
        }
    }
}

pub(super) fn place_mutation(place: crate::MirPlace, deletion: bool) -> MirPlaceMutation {
    MirPlaceMutation {
        kind: mutation_kind_for_place(&place, deletion),
        creation_policy: place.assignment_creation_policy(deletion),
        shape_policy: AssignmentShapePolicy::MatlabCompatible,
        place,
    }
}

fn mutation_kind_for_place(place: &crate::MirPlace, deletion: bool) -> PlaceMutationKind {
    if deletion {
        return PlaceMutationKind::Delete;
    }
    match place {
        crate::MirPlace::Local(_) | crate::MirPlace::Binding(_) => PlaceMutationKind::BindOrAssign,
        crate::MirPlace::Index(_, indexing) if indexing.kind == runmat_hir::IndexKind::Brace => {
            PlaceMutationKind::CellAssign
        }
        crate::MirPlace::Index(_, _) => PlaceMutationKind::IndexedAssign,
        crate::MirPlace::Member(_, _) | crate::MirPlace::DynamicMember(_, _) => {
            PlaceMutationKind::MemberAssign
        }
    }
}

fn is_empty_array_expr(expr: &HirExpr) -> bool {
    matches!(&expr.kind, HirExprKind::Tensor(rows) if rows.is_empty() || rows.iter().all(Vec::is_empty))
}

pub(super) fn is_empty_array_deletion_place(place: &crate::MirPlace, expr: &HirExpr) -> bool {
    if !is_empty_array_expr(expr) {
        return false;
    }
    matches!(
        place,
        crate::MirPlace::Index(
            _,
            crate::MirIndexing {
                kind: runmat_hir::IndexKind::Paren,
                ..
            }
        )
    )
}

fn lower_output_targets(
    ctx: &MirLoweringContext,
    targets: &[OutputTarget],
    stmts: &mut Vec<MirStmt>,
) -> Result<Vec<MirOutputTarget>, HirError> {
    targets
        .iter()
        .map(|target| lower_output_target(ctx, target, stmts))
        .collect()
}

fn lower_output_target(
    ctx: &MirLoweringContext,
    target: &OutputTarget,
    stmts: &mut Vec<MirStmt>,
) -> Result<MirOutputTarget, HirError> {
    Ok(match target {
        OutputTarget::Place(place) => {
            let mut temps = Vec::new();
            let place = lower_place(ctx, place, &mut temps)?;
            stmts.extend(temps);
            MirOutputTarget::Place(place)
        }
        OutputTarget::Sequence(target) => MirOutputTarget::Sequence(lower_sequence_target(
            ctx,
            target,
            stmts,
            &crate::lowering::AwaitReplacements::default(),
        )?),
        OutputTarget::Discard => MirOutputTarget::Discard,
    })
}

fn fixed_requested_output_count(
    requested_outputs: &RequestedOutputCount,
    context: &str,
) -> Result<usize, HirError> {
    requested_outputs
        .known_count()
        .ok_or_else(|| HirError::new(format!("{context} is runtime-determined")))
}
