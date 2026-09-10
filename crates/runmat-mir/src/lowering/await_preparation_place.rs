use std::collections::HashMap;

use runmat_hir::{
    ExprId, HirError, HirExpr, HirExprKind, HirPlace, HirSequenceTarget, IndexComponent,
};

use crate::{MirOperand, MirStmt};

use super::super::MirLoweringContext;
use super::expression::{cache_expr, prepare_expr_prefix};

pub(super) fn prepare_sequence_prefix(
    ctx: &MirLoweringContext,
    target: &HirSequenceTarget,
    awaited: ExprId,
    output: &mut Vec<MirStmt>,
    replacements: &mut HashMap<ExprId, MirOperand>,
) -> Result<bool, HirError> {
    match target {
        HirSequenceTarget::Member { base, .. } => {
            prepare_place_expr_prefix(ctx, base, awaited, output, replacements)
        }
        HirSequenceTarget::DynamicMember { base, member } => {
            prepare_ordered(ctx, base, Some(member), awaited, output, replacements)
        }
        HirSequenceTarget::CellContents { base, indexing } => {
            prepare_indexed(ctx, base, indexing, awaited, output, replacements)
        }
    }
}

pub(super) fn prepare_place_prefix(
    ctx: &MirLoweringContext,
    place: &HirPlace,
    awaited: ExprId,
    output: &mut Vec<MirStmt>,
    replacements: &mut HashMap<ExprId, MirOperand>,
) -> Result<bool, HirError> {
    match place {
        HirPlace::Binding(_) => Ok(false),
        HirPlace::Member(base, _) => {
            prepare_place_expr_prefix(ctx, base, awaited, output, replacements)
        }
        HirPlace::MemberDynamic(base, member) => {
            prepare_ordered(ctx, base, Some(member), awaited, output, replacements)
        }
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
            prepare_indexed(ctx, base, indexing, awaited, output, replacements)
        }
    }
}

fn prepare_indexed(
    ctx: &MirLoweringContext,
    base: &HirExpr,
    indexing: &runmat_hir::IndexingSemantics,
    awaited: ExprId,
    output: &mut Vec<MirStmt>,
    replacements: &mut HashMap<ExprId, MirOperand>,
) -> Result<bool, HirError> {
    if prepare_place_expr_prefix(ctx, base, awaited, output, replacements)? {
        return Ok(true);
    }
    for component in &indexing.components {
        if let IndexComponent::Expr(value) | IndexComponent::Logical(value) = component {
            if prepare_expr_prefix(ctx, value, awaited, output, replacements)? {
                return Ok(true);
            }
            cache_expr(ctx, value, output, replacements)?;
        }
    }
    Ok(false)
}

fn prepare_ordered(
    ctx: &MirLoweringContext,
    base: &HirExpr,
    member: Option<&HirExpr>,
    awaited: ExprId,
    output: &mut Vec<MirStmt>,
    replacements: &mut HashMap<ExprId, MirOperand>,
) -> Result<bool, HirError> {
    if prepare_place_expr_prefix(ctx, base, awaited, output, replacements)? {
        return Ok(true);
    }
    if let Some(member) = member {
        if prepare_expr_prefix(ctx, member, awaited, output, replacements)? {
            return Ok(true);
        }
        cache_expr(ctx, member, output, replacements)?;
    }
    Ok(false)
}

fn prepare_place_expr_prefix(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    awaited: ExprId,
    output: &mut Vec<MirStmt>,
    replacements: &mut HashMap<ExprId, MirOperand>,
) -> Result<bool, HirError> {
    match &expr.kind {
        HirExprKind::Binding(_) => Ok(false),
        HirExprKind::Member { base, .. } => {
            prepare_place_expr_prefix(ctx, base, awaited, output, replacements)
        }
        HirExprKind::MemberDynamic { base, member, .. } => {
            prepare_ordered(ctx, base, Some(member), awaited, output, replacements)
        }
        HirExprKind::Index(base, indexing) => {
            prepare_indexed(ctx, base, indexing, awaited, output, replacements)
        }
        _ => prepare_expr_prefix(ctx, expr, awaited, output, replacements),
    }
}
