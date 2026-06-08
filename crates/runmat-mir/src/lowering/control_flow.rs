use crate::{BasicBlock, BasicBlockId, MirOperand, MirPlace, MirTerminator, MirTerminatorKind};
use runmat_hir::{
    ExprId, HirBlock, HirError, HirExpr, HirExprKind, HirStmt, HirStmtKind, Span, StmtId,
};
use std::collections::HashMap;

use super::{
    expr::{lower_expr_with_replacements, lower_operand_with_replacements},
    place::lower_place,
    stmt::lower_stmt_with_replacements,
    MirLoweringContext,
};

#[derive(Debug, Default)]
pub(crate) struct ControlFlowBuilder {
    next_block: usize,
    blocks: Vec<BasicBlock>,
}

#[derive(Clone, Copy)]
struct BlockLoweringEnv<'a> {
    ctx: &'a MirLoweringContext,
    body: &'a HirBlock,
    return_terminator: &'a MirTerminator,
    loop_targets: Option<(BasicBlockId, BasicBlockId)>,
    await_replacements: &'a HashMap<ExprId, MirOperand>,
}

impl ControlFlowBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn fresh_block(&mut self) -> BasicBlockId {
        let id = BasicBlockId(self.next_block);
        self.next_block += 1;
        id
    }

    pub(crate) fn lower_function_body(
        mut self,
        ctx: &MirLoweringContext,
        body: &HirBlock,
        return_terminator: MirTerminator,
    ) -> Result<Vec<BasicBlock>, HirError> {
        let base_env = BlockLoweringEnv {
            ctx,
            body,
            return_terminator: &return_terminator,
            loop_targets: None,
            await_replacements: &HashMap::new(),
        };
        let entry = self.fresh_block();
        let entry = self.lower_block_from(entry, 0, return_terminator.clone(), base_env)?;
        self.blocks.push(entry);
        self.blocks.sort_by_key(|block| block.id.0);
        Ok(self.blocks)
    }

    fn lower_block_from(
        &mut self,
        id: BasicBlockId,
        start: usize,
        final_terminator: MirTerminator,
        env: BlockLoweringEnv<'_>,
    ) -> Result<BasicBlock, HirError> {
        let BlockLoweringEnv {
            ctx,
            body,
            return_terminator,
            loop_targets,
            await_replacements,
        } = env;
        let mut statements = Vec::new();
        for (idx, stmt) in body.statements.iter().enumerate().skip(start) {
            if let Some(await_expr) = first_unlowered_await_in_stmt(stmt, await_replacements) {
                let HirExprKind::Await(future_expr) = &await_expr.kind else {
                    unreachable!();
                };
                let future = lower_operand_with_replacements(
                    ctx,
                    future_expr,
                    &mut statements,
                    await_replacements,
                )?;
                let await_result = top_level_await_result(ctx, stmt, await_expr, &mut statements)?;
                let (result, resume_start, resume_replacements) = match await_result {
                    TopLevelAwaitResult::ExpressionStatement => (None, idx + 1, None),
                    TopLevelAwaitResult::Assignment(place) => (Some(place), idx + 1, None),
                    TopLevelAwaitResult::Nested => {
                        let local = ctx.fresh_temp(await_expr.span);
                        let mut replacements = await_replacements.clone();
                        replacements.insert(await_expr.id, MirOperand::Local(local));
                        (Some(MirPlace::Local(local)), idx, Some(replacements))
                    }
                };
                let resume = self.lower_continuation_target(
                    resume_start,
                    final_terminator,
                    BlockLoweringEnv {
                        ctx,
                        body,
                        return_terminator,
                        loop_targets,
                        await_replacements: resume_replacements
                            .as_ref()
                            .unwrap_or(await_replacements),
                    },
                )?;
                return Ok(BasicBlock {
                    id,
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Await {
                            future,
                            result,
                            resume,
                        },
                        span: stmt.span,
                    },
                });
            }
            if let HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } = &stmt.kind
            {
                let then_id = self.fresh_block();
                let else_id = self.fresh_block();
                let merge_id = self.lower_continuation_target(
                    idx + 1,
                    final_terminator,
                    BlockLoweringEnv {
                        ctx,
                        body,
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                let merge_terminator = MirTerminator {
                    kind: MirTerminatorKind::Goto(merge_id),
                    span: stmt.span,
                };
                let then_block = self.lower_block_from(
                    then_id,
                    0,
                    merge_terminator.clone(),
                    BlockLoweringEnv {
                        ctx,
                        body: then_body,
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                let nested_elseif_else =
                    lower_elseif_blocks(elseif_blocks, else_body.as_ref(), stmt.id, stmt.span);
                let empty_else = HirBlock { statements: vec![] };
                let else_body = nested_elseif_else.as_ref().or(else_body.as_ref());
                let else_block = self.lower_block_from(
                    else_id,
                    0,
                    merge_terminator,
                    BlockLoweringEnv {
                        ctx,
                        body: else_body.unwrap_or(&empty_else),
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                self.blocks.push(then_block);
                self.blocks.push(else_block);
                let cond = lower_operand_with_replacements(
                    ctx,
                    cond,
                    &mut statements,
                    await_replacements,
                )?;
                return Ok(BasicBlock {
                    id,
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Branch {
                            cond,
                            then_block: then_id,
                            else_block: else_id,
                        },
                        span: stmt.span,
                    },
                });
            }
            if let HirStmtKind::While {
                cond,
                body: loop_body,
            } = &stmt.kind
            {
                let header_id = if statements.is_empty() {
                    id
                } else {
                    self.fresh_block()
                };
                let loop_body_id = self.fresh_block();
                let exit_id = self.lower_continuation_target(
                    idx + 1,
                    final_terminator,
                    BlockLoweringEnv {
                        ctx,
                        body,
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                let body_block = self.lower_block_from(
                    loop_body_id,
                    0,
                    MirTerminator {
                        kind: MirTerminatorKind::Goto(header_id),
                        span: stmt.span,
                    },
                    BlockLoweringEnv {
                        ctx,
                        body: loop_body,
                        return_terminator,
                        loop_targets: Some((id, exit_id)),
                        await_replacements,
                    },
                )?;
                self.blocks.push(body_block);
                let mut header_statements = Vec::new();
                let cond = lower_operand_with_replacements(
                    ctx,
                    cond,
                    &mut header_statements,
                    await_replacements,
                )?;
                let header_block = BasicBlock {
                    id: header_id,
                    statements: header_statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Branch {
                            cond,
                            then_block: loop_body_id,
                            else_block: exit_id,
                        },
                        span: stmt.span,
                    },
                };
                if header_id != id {
                    self.blocks.push(header_block);
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(header_id),
                            span: stmt.span,
                        },
                    });
                }
                return Ok(header_block);
            }
            if let HirStmtKind::For {
                binding,
                range,
                body: loop_body,
            } = &stmt.kind
            {
                let iterable =
                    lower_expr_with_replacements(ctx, range, &mut statements, await_replacements)?;
                let header_id = if statements.is_empty() {
                    id
                } else {
                    self.fresh_block()
                };
                let body_id = self.fresh_block();
                let exit_id = self.lower_continuation_target(
                    idx + 1,
                    final_terminator,
                    BlockLoweringEnv {
                        ctx,
                        body,
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                let body_block = self.lower_block_from(
                    body_id,
                    0,
                    MirTerminator {
                        kind: MirTerminatorKind::Goto(header_id),
                        span: stmt.span,
                    },
                    BlockLoweringEnv {
                        ctx,
                        body: loop_body,
                        return_terminator,
                        loop_targets: Some((id, exit_id)),
                        await_replacements,
                    },
                )?;
                self.blocks.push(body_block);
                let header_block = BasicBlock {
                    id: header_id,
                    statements: Vec::new(),
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::For {
                            binding: ctx.local_for_binding(*binding)?,
                            iterable,
                            body_block: body_id,
                            exit_block: exit_id,
                        },
                        span: stmt.span,
                    },
                };
                if header_id != id {
                    self.blocks.push(header_block);
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(header_id),
                            span: stmt.span,
                        },
                    });
                }
                return Ok(header_block);
            }
            if let HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
            } = &stmt.kind
            {
                let merge_id = self.lower_continuation_target(
                    idx + 1,
                    final_terminator,
                    BlockLoweringEnv {
                        ctx,
                        body,
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                let merge_terminator = MirTerminator {
                    kind: MirTerminatorKind::Goto(merge_id),
                    span: stmt.span,
                };
                let mut lowered_cases = Vec::new();
                for (case_expr, case_body) in cases {
                    let case_id = self.fresh_block();
                    let case_block = self.lower_block_from(
                        case_id,
                        0,
                        merge_terminator.clone(),
                        BlockLoweringEnv {
                            ctx,
                            body: case_body,
                            return_terminator,
                            loop_targets,
                            await_replacements,
                        },
                    )?;
                    self.blocks.push(case_block);
                    lowered_cases.push((
                        lower_operand_with_replacements(
                            ctx,
                            case_expr,
                            &mut statements,
                            await_replacements,
                        )?,
                        case_id,
                    ));
                }
                let otherwise_id = self.fresh_block();
                let empty_otherwise = HirBlock { statements: vec![] };
                let otherwise_block = self.lower_block_from(
                    otherwise_id,
                    0,
                    merge_terminator,
                    BlockLoweringEnv {
                        ctx,
                        body: otherwise.as_ref().unwrap_or(&empty_otherwise),
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                self.blocks.push(otherwise_block);
                let discr = lower_operand_with_replacements(
                    ctx,
                    expr,
                    &mut statements,
                    await_replacements,
                )?;
                return Ok(BasicBlock {
                    id,
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Switch {
                            discr,
                            cases: lowered_cases,
                            otherwise: otherwise_id,
                        },
                        span: stmt.span,
                    },
                });
            }
            if let HirStmtKind::TryCatch {
                try_body,
                catch_binding,
                catch_body,
                ..
            } = &stmt.kind
            {
                let try_id = self.fresh_block();
                let catch_id = self.fresh_block();
                let merge_id = self.lower_continuation_target(
                    idx + 1,
                    final_terminator,
                    BlockLoweringEnv {
                        ctx,
                        body,
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                let merge_terminator = MirTerminator {
                    kind: MirTerminatorKind::Goto(merge_id),
                    span: stmt.span,
                };
                let try_block = self.lower_block_from(
                    try_id,
                    0,
                    merge_terminator.clone(),
                    BlockLoweringEnv {
                        ctx,
                        body: try_body,
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                let catch_block = self.lower_block_from(
                    catch_id,
                    0,
                    merge_terminator,
                    BlockLoweringEnv {
                        ctx,
                        body: catch_body,
                        return_terminator,
                        loop_targets,
                        await_replacements,
                    },
                )?;
                self.blocks.push(try_block);
                self.blocks.push(catch_block);
                return Ok(BasicBlock {
                    id,
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::TryCatch {
                            try_block: try_id,
                            catch_block: catch_id,
                            catch_binding: catch_binding
                                .map(|binding| ctx.local_for_binding(binding))
                                .transpose()?,
                        },
                        span: stmt.span,
                    },
                });
            }
            if matches!(stmt.kind, HirStmtKind::Break) {
                let Some((_, break_target)) = loop_targets else {
                    return Err(HirError::new("break outside loop"));
                };
                return Ok(BasicBlock {
                    id,
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Goto(break_target),
                        span: stmt.span,
                    },
                });
            }
            if matches!(stmt.kind, HirStmtKind::Continue) {
                let Some((continue_target, _)) = loop_targets else {
                    return Err(HirError::new("continue outside loop"));
                };
                return Ok(BasicBlock {
                    id,
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Goto(continue_target),
                        span: stmt.span,
                    },
                });
            }
            if let HirStmtKind::ExprStmt(expr, _) = &stmt.kind {
                if let HirExprKind::Await(future) = &expr.kind {
                    let resume = self.lower_continuation_target(
                        idx + 1,
                        final_terminator,
                        BlockLoweringEnv {
                            ctx,
                            body,
                            return_terminator,
                            loop_targets,
                            await_replacements,
                        },
                    )?;
                    let future = lower_operand_with_replacements(
                        ctx,
                        future,
                        &mut statements,
                        await_replacements,
                    )?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Await {
                                future,
                                result: None,
                                resume,
                            },
                            span: stmt.span,
                        },
                    });
                }
            }
            if let HirStmtKind::Assign(place, expr, _) = &stmt.kind {
                if let HirExprKind::Await(future) = &expr.kind {
                    let resume = self.lower_continuation_target(
                        idx + 1,
                        final_terminator,
                        BlockLoweringEnv {
                            ctx,
                            body,
                            return_terminator,
                            loop_targets,
                            await_replacements,
                        },
                    )?;
                    let future = lower_operand_with_replacements(
                        ctx,
                        future,
                        &mut statements,
                        await_replacements,
                    )?;
                    let result = lower_place(ctx, place, &mut statements)?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Await {
                                future,
                                result: Some(result),
                                resume,
                            },
                            span: stmt.span,
                        },
                    });
                }
            }
            if matches!(stmt.kind, HirStmtKind::Return) {
                return Ok(BasicBlock {
                    id,
                    statements,
                    terminator: MirTerminator {
                        kind: return_terminator.kind.clone(),
                        span: stmt.span,
                    },
                });
            }
            statements.extend(lower_stmt_with_replacements(ctx, stmt, await_replacements)?);
        }
        Ok(BasicBlock {
            id,
            statements,
            terminator: final_terminator,
        })
    }

    fn lower_continuation_target(
        &mut self,
        start: usize,
        final_terminator: MirTerminator,
        env: BlockLoweringEnv<'_>,
    ) -> Result<BasicBlockId, HirError> {
        let id = self.fresh_block();
        let block = self.lower_block_from(id, start, final_terminator, env)?;
        self.blocks.push(block);
        Ok(id)
    }
}

enum TopLevelAwaitResult {
    ExpressionStatement,
    Assignment(MirPlace),
    Nested,
}

fn top_level_await_result(
    ctx: &MirLoweringContext,
    stmt: &HirStmt,
    await_expr: &HirExpr,
    statements: &mut Vec<crate::MirStmt>,
) -> Result<TopLevelAwaitResult, HirError> {
    match &stmt.kind {
        HirStmtKind::ExprStmt(expr, _) if expr.id == await_expr.id => {
            Ok(TopLevelAwaitResult::ExpressionStatement)
        }
        HirStmtKind::Assign(place, expr, _) if expr.id == await_expr.id => Ok(
            TopLevelAwaitResult::Assignment(lower_place(ctx, place, statements)?),
        ),
        _ => Ok(TopLevelAwaitResult::Nested),
    }
}

fn first_unlowered_await_in_stmt<'a>(
    stmt: &'a HirStmt,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Option<&'a HirExpr> {
    match &stmt.kind {
        HirStmtKind::ExprStmt(expr, _) | HirStmtKind::Assign(_, expr, _) => {
            first_unlowered_await(expr, await_replacements)
        }
        HirStmtKind::MultiAssign(_, expr, _) => first_unlowered_await(expr, await_replacements),
        HirStmtKind::If {
            cond,
            elseif_blocks,
            ..
        } => first_unlowered_await(cond, await_replacements).or_else(|| {
            elseif_blocks
                .iter()
                .find_map(|(cond, _)| first_unlowered_await(cond, await_replacements))
        }),
        HirStmtKind::While { cond, .. } => first_unlowered_await(cond, await_replacements),
        HirStmtKind::For { range, .. } => first_unlowered_await(range, await_replacements),
        HirStmtKind::Switch { expr, cases, .. } => first_unlowered_await(expr, await_replacements)
            .or_else(|| {
                cases
                    .iter()
                    .find_map(|(case, _)| first_unlowered_await(case, await_replacements))
            }),
        HirStmtKind::TryCatch { .. }
        | HirStmtKind::Global(_)
        | HirStmtKind::Persistent(_)
        | HirStmtKind::Break
        | HirStmtKind::Continue
        | HirStmtKind::Return
        | HirStmtKind::Import(_) => None,
    }
}

fn first_unlowered_await<'a>(
    expr: &'a HirExpr,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Option<&'a HirExpr> {
    if await_replacements.contains_key(&expr.id) {
        return None;
    }
    match &expr.kind {
        HirExprKind::Await(_) => Some(expr),
        HirExprKind::Unary(_, inner) => first_unlowered_await(inner, await_replacements),
        HirExprKind::Binary(left, _, right) => first_unlowered_await(left, await_replacements)
            .or_else(|| first_unlowered_await(right, await_replacements)),
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => rows
            .iter()
            .flat_map(|row| row.iter())
            .find_map(|expr| first_unlowered_await(expr, await_replacements)),
        HirExprKind::StructLiteral(fields) => fields
            .iter()
            .find_map(|(_, value)| first_unlowered_await(value, await_replacements)),
        HirExprKind::ObjectLiteral { fields, .. } => fields
            .iter()
            .find_map(|(_, value)| first_unlowered_await(value, await_replacements)),
        HirExprKind::Range(start, step, end) => first_unlowered_await(start, await_replacements)
            .or_else(|| {
                step.as_ref()
                    .and_then(|step| first_unlowered_await(step, await_replacements))
            })
            .or_else(|| first_unlowered_await(end, await_replacements)),
        HirExprKind::Index(base, indexing) => first_unlowered_await(base, await_replacements)
            .or_else(|| first_unlowered_await_in_indexing(indexing, await_replacements)),
        HirExprKind::Member(base, _) => first_unlowered_await(base, await_replacements),
        HirExprKind::MemberDynamic(base, member) => first_unlowered_await(base, await_replacements)
            .or_else(|| first_unlowered_await(member, await_replacements)),
        HirExprKind::Call(call) => call
            .args
            .iter()
            .find_map(|arg| first_unlowered_await(arg, await_replacements)),
        HirExprKind::Spawn(inner) => first_unlowered_await(inner, await_replacements),
        HirExprKind::Number(_)
        | HirExprKind::String(_)
        | HirExprKind::Constant(_)
        | HirExprKind::Binding(_)
        | HirExprKind::Colon
        | HirExprKind::End
        | HirExprKind::CommandCall(_)
        | HirExprKind::FunctionHandle(_)
        | HirExprKind::AnonymousFunction(_)
        | HirExprKind::MetaClass(_) => None,
    }
}

fn first_unlowered_await_in_indexing<'a>(
    indexing: &'a runmat_hir::IndexingSemantics,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Option<&'a HirExpr> {
    indexing
        .components
        .iter()
        .find_map(|component| match component {
            runmat_hir::IndexComponent::Expr(expr) | runmat_hir::IndexComponent::Logical(expr) => {
                first_unlowered_await(expr, await_replacements)
            }
            runmat_hir::IndexComponent::Colon | runmat_hir::IndexComponent::End { .. } => None,
        })
}

fn lower_elseif_blocks(
    elseif_blocks: &[(HirExpr, HirBlock)],
    else_body: Option<&HirBlock>,
    stmt_id: StmtId,
    span: Span,
) -> Option<HirBlock> {
    let ((cond, then_body), rest) = elseif_blocks.split_first()?;
    let nested_else =
        lower_elseif_blocks(rest, else_body, stmt_id, span).or_else(|| else_body.cloned());
    Some(HirBlock {
        statements: vec![HirStmt {
            id: stmt_id,
            kind: HirStmtKind::If {
                cond: cond.clone(),
                then_body: then_body.clone(),
                elseif_blocks: Vec::new(),
                else_body: nested_else,
            },
            span,
        }],
    })
}
