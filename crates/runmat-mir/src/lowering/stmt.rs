use crate::{
    MirOutputTarget, MirOutputTargetList, MirPlaceMutation, MirRvalue, MirStmt, MirStmtKind,
};
use runmat_hir::{
    AssignmentCreationPolicy, AssignmentShapePolicy, EnvironmentEffect, ExprId, HirExpr,
    HirExprKind, HirStmt, HirStmtKind, OutputTarget, PlaceMutationKind, SemanticError, Span,
    WorkspaceEffect,
};
use std::collections::HashMap;

use super::{expr::lower_expr_with_replacements, place::lower_place, MirLoweringContext};

pub(crate) fn lower_stmt_with_replacements(
    ctx: &MirLoweringContext,
    stmt: &HirStmt,
    await_replacements: &HashMap<ExprId, crate::MirOperand>,
) -> Result<Vec<MirStmt>, SemanticError> {
    Ok(match &stmt.kind {
        HirStmtKind::Assign(place, expr, _) => {
            let mut stmts = Vec::new();
            let value = lower_expr_with_replacements(ctx, expr, &mut stmts, await_replacements)?;
            stmts.extend(effect_stmts_for_rvalue(&value, stmt.span));
            let place = lower_place(ctx, place, &mut stmts)?;
            if !matches!(place, crate::MirPlace::Local(_)) || is_empty_array_expr(expr) {
                stmts.push(MirStmt {
                    kind: MirStmtKind::PlaceMutation(place_mutation(
                        place.clone(),
                        is_empty_array_expr(expr),
                    )),
                    span: stmt.span,
                });
            }
            stmts.push(MirStmt {
                kind: MirStmtKind::Assign { place, value },
                span: stmt.span,
            });
            stmts
        }
        HirStmtKind::MultiAssign(targets, expr, _) => {
            let mut stmts = Vec::new();
            let value = lower_expr_with_replacements(ctx, expr, &mut stmts, await_replacements)?;
            stmts.extend(effect_stmts_for_rvalue(&value, stmt.span));
            let lowered_targets = lower_output_targets(ctx, &targets.targets, &mut stmts)?;
            stmts.push(MirStmt {
                kind: MirStmtKind::MultiAssign {
                    targets: MirOutputTargetList {
                        requested_outputs: targets.requested_outputs.clone(),
                        targets: lowered_targets,
                    },
                    value,
                },
                span: stmt.span,
            });
            stmts
        }
        HirStmtKind::ExprStmt(expr, _) => {
            let mut stmts = Vec::new();
            let value = lower_expr_with_replacements(ctx, expr, &mut stmts, await_replacements)?;
            stmts.extend(effect_stmts_for_rvalue(&value, stmt.span));
            stmts.push(MirStmt {
                kind: MirStmtKind::Expr(value),
                span: stmt.span,
            });
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
            return Err(SemanticError::new(
                "MIR lowering for statement is not implemented yet",
            ))
        }
    })
}

fn effect_stmts_for_rvalue(value: &MirRvalue, span: Span) -> Vec<MirStmt> {
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
        runmat_builtins::BuiltinWorkspaceEffect::DynamicEval => WorkspaceEffect::DynamicEval,
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

fn place_mutation(place: crate::MirPlace, deletion: bool) -> MirPlaceMutation {
    MirPlaceMutation {
        kind: mutation_kind_for_place(&place, deletion),
        creation_policy: creation_policy_for_place(&place, deletion),
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
        crate::MirPlace::Index(_, _) => PlaceMutationKind::IndexedAssign,
        crate::MirPlace::Member(_, _) | crate::MirPlace::DynamicMember(_, _) => {
            PlaceMutationKind::MemberAssign
        }
    }
}

fn creation_policy_for_place(place: &crate::MirPlace, deletion: bool) -> AssignmentCreationPolicy {
    if deletion {
        return AssignmentCreationPolicy::ExistingOnly;
    }
    match place {
        crate::MirPlace::Local(_) | crate::MirPlace::Binding(_) => {
            AssignmentCreationPolicy::CreateBinding
        }
        crate::MirPlace::Index(_, _) => AssignmentCreationPolicy::CreateArrayByIndex,
        crate::MirPlace::Member(_, _) | crate::MirPlace::DynamicMember(_, _) => {
            AssignmentCreationPolicy::CreateStructFieldPath
        }
    }
}

fn is_empty_array_expr(expr: &HirExpr) -> bool {
    matches!(&expr.kind, HirExprKind::Tensor(rows) if rows.is_empty() || rows.iter().all(Vec::is_empty))
}

fn lower_output_targets(
    ctx: &MirLoweringContext,
    targets: &[OutputTarget],
    stmts: &mut Vec<MirStmt>,
) -> Result<Vec<MirOutputTarget>, SemanticError> {
    targets
        .iter()
        .map(|target| lower_output_target(ctx, target, stmts))
        .collect()
}

fn lower_output_target(
    ctx: &MirLoweringContext,
    target: &OutputTarget,
    stmts: &mut Vec<MirStmt>,
) -> Result<MirOutputTarget, SemanticError> {
    Ok(match target {
        OutputTarget::Place(place) => {
            let mut temps = Vec::new();
            let place = lower_place(ctx, place, &mut temps)?;
            stmts.extend(temps);
            MirOutputTarget::Place(place)
        }
        OutputTarget::Discard => MirOutputTarget::Discard,
    })
}
