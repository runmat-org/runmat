use crate::{BasicBlock, BasicBlockId, MirSourceRecord, MirTerminator, MirTerminatorKind};
use runmat_hir::{
    HirBlock, HirExpr, HirExprKind, HirStmt, HirStmtKind, SemanticError, Span, StmtId,
};

use super::{
    expr::{lower_expr, lower_operand},
    place::lower_place,
    stmt::lower_stmt,
    MirLoweringContext,
};

#[derive(Debug, Default)]
pub(crate) struct ControlFlowBuilder {
    next_block: usize,
    blocks: Vec<BasicBlock>,
    source_records: Vec<MirSourceRecord>,
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
    ) -> Result<(Vec<BasicBlock>, Vec<MirSourceRecord>), SemanticError> {
        let entry = self.fresh_block();
        let entry = self.lower_block_from(
            ctx,
            entry,
            body,
            0,
            return_terminator.clone(),
            &return_terminator,
            None,
        )?;
        self.blocks.push(entry);
        self.blocks.sort_by_key(|block| block.id.0);
        Ok((self.blocks, self.source_records))
    }

    fn lower_block_from(
        &mut self,
        ctx: &MirLoweringContext,
        id: BasicBlockId,
        body: &HirBlock,
        start: usize,
        final_terminator: MirTerminator,
        return_terminator: &MirTerminator,
        loop_targets: Option<(BasicBlockId, BasicBlockId)>,
    ) -> Result<BasicBlock, SemanticError> {
        let mut statements = Vec::new();
        for (idx, stmt) in body.statements.iter().enumerate().skip(start) {
            self.source_records.push(MirSourceRecord {
                block: id,
                stmt: Some(stmt.id),
                expr: None,
                span: stmt.span,
            });
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
                    ctx,
                    body,
                    idx + 1,
                    final_terminator,
                    return_terminator,
                    loop_targets,
                )?;
                let merge_terminator = MirTerminator {
                    kind: MirTerminatorKind::Goto(merge_id),
                    span: stmt.span,
                };
                let then_block = self.lower_block_from(
                    ctx,
                    then_id,
                    then_body,
                    0,
                    merge_terminator.clone(),
                    return_terminator,
                    loop_targets,
                )?;
                let nested_elseif_else =
                    lower_elseif_blocks(elseif_blocks, else_body.as_ref(), stmt.id, stmt.span);
                let empty_else = HirBlock { statements: vec![] };
                let else_body = nested_elseif_else.as_ref().or(else_body.as_ref());
                let else_block = self.lower_block_from(
                    ctx,
                    else_id,
                    else_body.unwrap_or(&empty_else),
                    0,
                    merge_terminator,
                    return_terminator,
                    loop_targets,
                )?;
                self.blocks.push(then_block);
                self.blocks.push(else_block);
                let cond = lower_operand(ctx, cond, &mut statements)?;
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
                let loop_body_id = self.fresh_block();
                let exit_id = self.lower_continuation_target(
                    ctx,
                    body,
                    idx + 1,
                    final_terminator,
                    return_terminator,
                    loop_targets,
                )?;
                let body_block = self.lower_block_from(
                    ctx,
                    loop_body_id,
                    loop_body,
                    0,
                    MirTerminator {
                        kind: MirTerminatorKind::Goto(id),
                        span: stmt.span,
                    },
                    return_terminator,
                    Some((id, exit_id)),
                )?;
                self.blocks.push(body_block);
                let cond = lower_operand(ctx, cond, &mut statements)?;
                return Ok(BasicBlock {
                    id,
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Branch {
                            cond,
                            then_block: loop_body_id,
                            else_block: exit_id,
                        },
                        span: stmt.span,
                    },
                });
            }
            if let HirStmtKind::For {
                binding,
                range,
                body,
                semantics,
            } = &stmt.kind
            {
                let body_id = self.fresh_block();
                let exit_id = self.lower_continuation_target(
                    ctx,
                    body,
                    idx + 1,
                    final_terminator,
                    return_terminator,
                    loop_targets,
                )?;
                let body_block = self.lower_block_from(
                    ctx,
                    body_id,
                    body,
                    0,
                    MirTerminator {
                        kind: MirTerminatorKind::Goto(id),
                        span: stmt.span,
                    },
                    return_terminator,
                    Some((id, exit_id)),
                )?;
                self.blocks.push(body_block);
                let iterable = lower_expr(ctx, range, &mut statements)?;
                return Ok(BasicBlock {
                    id,
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::For {
                            binding: ctx.local_for_binding(*binding)?,
                            iterable,
                            semantics: semantics.clone(),
                            body_block: body_id,
                            exit_block: exit_id,
                        },
                        span: stmt.span,
                    },
                });
            }
            if let HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
            } = &stmt.kind
            {
                let merge_id = self.lower_continuation_target(
                    ctx,
                    body,
                    idx + 1,
                    final_terminator,
                    return_terminator,
                    loop_targets,
                )?;
                let merge_terminator = MirTerminator {
                    kind: MirTerminatorKind::Goto(merge_id),
                    span: stmt.span,
                };
                let mut lowered_cases = Vec::new();
                for (case_expr, case_body) in cases {
                    let case_id = self.fresh_block();
                    let case_block = self.lower_block_from(
                        ctx,
                        case_id,
                        case_body,
                        0,
                        merge_terminator.clone(),
                        return_terminator,
                        loop_targets,
                    )?;
                    self.blocks.push(case_block);
                    lowered_cases.push((lower_operand(ctx, case_expr, &mut statements)?, case_id));
                }
                let otherwise_id = self.fresh_block();
                let empty_otherwise = HirBlock { statements: vec![] };
                let otherwise_block = self.lower_block_from(
                    ctx,
                    otherwise_id,
                    otherwise.as_ref().unwrap_or(&empty_otherwise),
                    0,
                    merge_terminator,
                    return_terminator,
                    loop_targets,
                )?;
                self.blocks.push(otherwise_block);
                let discr = lower_operand(ctx, expr, &mut statements)?;
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
                catch_body,
                ..
            } = &stmt.kind
            {
                let try_id = self.fresh_block();
                let catch_id = self.fresh_block();
                let merge_id = self.lower_continuation_target(
                    ctx,
                    body,
                    idx + 1,
                    final_terminator,
                    return_terminator,
                    loop_targets,
                )?;
                let merge_terminator = MirTerminator {
                    kind: MirTerminatorKind::Goto(merge_id),
                    span: stmt.span,
                };
                let try_block = self.lower_block_from(
                    ctx,
                    try_id,
                    try_body,
                    0,
                    merge_terminator.clone(),
                    return_terminator,
                    loop_targets,
                )?;
                let catch_block = self.lower_block_from(
                    ctx,
                    catch_id,
                    catch_body,
                    0,
                    merge_terminator,
                    return_terminator,
                    loop_targets,
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
                        },
                        span: stmt.span,
                    },
                });
            }
            if matches!(stmt.kind, HirStmtKind::Break) {
                let Some((_, break_target)) = loop_targets else {
                    return Err(SemanticError::new("break outside loop"));
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
                    return Err(SemanticError::new("continue outside loop"));
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
                        ctx,
                        body,
                        idx + 1,
                        final_terminator,
                        return_terminator,
                        loop_targets,
                    )?;
                    let future = lower_operand(ctx, future, &mut statements)?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Await {
                                future,
                                result: None,
                                resume,
                                cleanup: None,
                            },
                            span: stmt.span,
                        },
                    });
                }
            }
            if let HirStmtKind::Assign(place, expr, _) = &stmt.kind {
                if let HirExprKind::Await(future) = &expr.kind {
                    let resume = self.lower_continuation_target(
                        ctx,
                        body,
                        idx + 1,
                        final_terminator,
                        return_terminator,
                        loop_targets,
                    )?;
                    let future = lower_operand(ctx, future, &mut statements)?;
                    let result = lower_place(ctx, place, &mut statements)?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Await {
                                future,
                                result: Some(result),
                                resume,
                                cleanup: None,
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
            statements.extend(lower_stmt(ctx, stmt)?);
        }
        Ok(BasicBlock {
            id,
            statements,
            terminator: final_terminator,
        })
    }

    fn lower_continuation_target(
        &mut self,
        ctx: &MirLoweringContext,
        body: &HirBlock,
        start: usize,
        final_terminator: MirTerminator,
        return_terminator: &MirTerminator,
        loop_targets: Option<(BasicBlockId, BasicBlockId)>,
    ) -> Result<BasicBlockId, SemanticError> {
        let id = self.fresh_block();
        let block = self.lower_block_from(
            ctx,
            id,
            body,
            start,
            final_terminator,
            return_terminator,
            loop_targets,
        )?;
        self.blocks.push(block);
        Ok(id)
    }
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
