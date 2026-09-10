use crate::{
    BasicBlock, BasicBlockId, MirConstant, MirOperand, MirPlace, MirTerminator, MirTerminatorKind,
};
use runmat_hir::{
    ExprId, HirBlock, HirError, HirExpr, HirExprKind, HirStmt, HirStmtKind, Span, StmtId,
};
use std::collections::{HashMap, HashSet};

use super::{
    conditional_await::conditional_await_guard,
    expr::{lower_expr_with_replacements, lower_operand_with_replacements},
    place::lower_place,
    stmt::lower_stmt_with_replacements,
    MirLoweringContext,
};

#[path = "control_flow/await_expression.rs"]
mod await_expression;

#[derive(Debug, Default)]
pub(crate) struct ControlFlowBuilder {
    next_block: usize,
    blocks: Vec<BasicBlock>,
    active_conditional_awaits: HashSet<ExprId>,
    repeating_while_headers: HashMap<StmtId, BasicBlockId>,
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
                if matches!(stmt.kind, HirStmtKind::While { .. })
                    && !statements.is_empty()
                    && !self.repeating_while_headers.contains_key(&stmt.id)
                {
                    let header = self.fresh_block();
                    self.repeating_while_headers.insert(stmt.id, header);
                    let header_block = self.lower_block_from(
                        header,
                        idx,
                        final_terminator,
                        BlockLoweringEnv {
                            ctx,
                            body,
                            return_terminator,
                            loop_targets,
                            await_replacements,
                        },
                    );
                    self.repeating_while_headers.remove(&stmt.id);
                    self.blocks.push(header_block?);
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(header),
                            span: stmt.span,
                        },
                    });
                }
                if let Some(guard) =
                    conditional_await_guard(stmt, await_expr.id, &self.active_conditional_awaits)
                {
                    let prior_repeat = if matches!(stmt.kind, HirStmtKind::While { .. }) {
                        self.repeating_while_headers.insert(stmt.id, id)
                    } else {
                        None
                    };
                    let mut prepared = await_replacements.clone();
                    super::await_preparation::prepare_statement_prefix(
                        ctx,
                        stmt,
                        guard.expression.id,
                        &mut statements,
                        &mut prepared,
                    )?;
                    let condition = lower_operand_with_replacements(
                        ctx,
                        guard.left,
                        &mut statements,
                        &prepared,
                    )?;
                    prepared.insert(guard.left.id, condition.clone());

                    let result = ctx.fresh_temp(guard.expression.span);
                    let mut joined = prepared.clone();
                    joined.insert(guard.expression.id, MirOperand::Local(result));
                    let joined_id = self.lower_continuation_target(
                        idx,
                        final_terminator.clone(),
                        BlockLoweringEnv {
                            ctx,
                            body,
                            return_terminator,
                            loop_targets,
                            await_replacements: &joined,
                        },
                    )?;

                    let skipped_id = self.fresh_block();
                    self.blocks.push(BasicBlock {
                        id: skipped_id,
                        statements: vec![crate::MirStmt {
                            kind: crate::MirStmtKind::Assign {
                                place: MirPlace::Local(result),
                                value: crate::MirRvalue::Use(MirOperand::Constant(
                                    MirConstant::Bool(guard.skipped_value),
                                )),
                            },
                            span: guard.expression.span,
                        }],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(joined_id),
                            span: guard.expression.span,
                        },
                    });
                    self.active_conditional_awaits.insert(guard.expression.id);
                    let evaluated_id = self.fresh_block();
                    let evaluated = self.lower_expression_into(
                        ctx,
                        guard.expression,
                        await_expression::ExpressionTarget {
                            block: evaluated_id,
                            destination: result,
                            continuation: joined_id,
                            span: stmt.span,
                        },
                        &prepared,
                    );
                    self.active_conditional_awaits.remove(&guard.expression.id);
                    self.blocks.push(evaluated?);
                    restore_repeat_header(&mut self.repeating_while_headers, stmt.id, prior_repeat);
                    let (then_block, else_block) = if guard.evaluate_when_true {
                        (evaluated_id, skipped_id)
                    } else {
                        (skipped_id, evaluated_id)
                    };
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Branch {
                                cond: condition,
                                then_block,
                                else_block,
                            },
                            span: stmt.span,
                        },
                    });
                }
                let HirExprKind::Await(future_expr) = &await_expr.kind else {
                    unreachable!();
                };
                let mut prepared_replacements = await_replacements.clone();
                let is_direct_assignment_rhs = matches!(
                    &stmt.kind,
                    HirStmtKind::Assign(_, value, _) if value.id == await_expr.id
                );
                if !is_direct_assignment_rhs {
                    super::await_preparation::prepare_statement_prefix(
                        ctx,
                        stmt,
                        await_expr.id,
                        &mut statements,
                        &mut prepared_replacements,
                    )?;
                }
                let future = lower_operand_with_replacements(
                    ctx,
                    future_expr,
                    &mut statements,
                    &prepared_replacements,
                )?;
                let await_result = top_level_await_result(ctx, stmt, await_expr, &mut statements)?;
                let (result, resume_start, resume_replacements) = match await_result {
                    TopLevelAwaitResult::ExpressionStatement => (None, idx + 1, None),
                    TopLevelAwaitResult::Assignment(place) => (Some(place), idx + 1, None),
                    TopLevelAwaitResult::Nested => {
                        let local = ctx.fresh_temp(await_expr.span);
                        let mut replacements = prepared_replacements;
                        replacements.insert(await_expr.id, MirOperand::Local(local));
                        (Some(MirPlace::Local(local)), idx, Some(replacements))
                    }
                };
                let prior_repeat = if matches!(stmt.kind, HirStmtKind::While { .. }) {
                    self.repeating_while_headers.insert(stmt.id, id)
                } else {
                    None
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
                );
                restore_repeat_header(&mut self.repeating_while_headers, stmt.id, prior_repeat);
                let resume = resume?;
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
                let repeat_header = self
                    .repeating_while_headers
                    .get(&stmt.id)
                    .copied()
                    .unwrap_or(header_id);
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
                        kind: MirTerminatorKind::Goto(repeat_header),
                        span: stmt.span,
                    },
                    BlockLoweringEnv {
                        ctx,
                        body: loop_body,
                        return_terminator,
                        loop_targets: Some((repeat_header, exit_id)),
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
            if let HirStmtKind::ParFor {
                region,
                binding,
                range,
                maximum_workers,
                body: loop_body,
            } = &stmt.kind
            {
                let iterable =
                    lower_expr_with_replacements(ctx, range, &mut statements, await_replacements)?;
                let maximum_workers = maximum_workers
                    .as_ref()
                    .map(|value| {
                        lower_expr_with_replacements(
                            ctx,
                            value,
                            &mut statements,
                            await_replacements,
                        )
                    })
                    .transpose()?
                    .map(Box::new);
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
                        loop_targets: Some((header_id, exit_id)),
                        await_replacements,
                    },
                )?;
                self.blocks.push(body_block);
                let header_block = BasicBlock {
                    id: header_id,
                    statements: Vec::new(),
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::ParFor {
                            region: *region,
                            binding: ctx.local_for_binding(*binding)?,
                            iterable,
                            maximum_workers,
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
            if let HirStmtKind::Spmd {
                region,
                header,
                body: spmd_body,
            } = &stmt.kind
            {
                let mut lower_header_value = |value| {
                    lower_expr_with_replacements(ctx, value, &mut statements, await_replacements)
                };
                let header = match header {
                    runmat_hir::parallel::SpmdHeader::Default => {
                        crate::parallel::MirSpmdHeader::Default
                    }
                    runmat_hir::parallel::SpmdHeader::One(value) => {
                        crate::parallel::MirSpmdHeader::One(lower_header_value(value)?)
                    }
                    runmat_hir::parallel::SpmdHeader::Two(first, second) => {
                        crate::parallel::MirSpmdHeader::Two(
                            lower_header_value(first)?,
                            lower_header_value(second)?,
                        )
                    }
                    runmat_hir::parallel::SpmdHeader::Three(first, second, third) => {
                        crate::parallel::MirSpmdHeader::Three(
                            lower_header_value(first)?,
                            lower_header_value(second)?,
                            lower_header_value(third)?,
                        )
                    }
                };
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
                let body_block = ctx.with_spmd_region(*region, || {
                    self.lower_block_from(
                        body_id,
                        0,
                        MirTerminator {
                            kind: MirTerminatorKind::Goto(exit_id),
                            span: stmt.span,
                        },
                        BlockLoweringEnv {
                            ctx,
                            body: spmd_body,
                            return_terminator,
                            loop_targets: None,
                            await_replacements,
                        },
                    )
                })?;
                self.blocks.push(body_block);
                let header_block = BasicBlock {
                    id: header_id,
                    statements: Vec::new(),
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Spmd {
                            region: *region,
                            header: Box::new(header),
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

fn restore_repeat_header(
    headers: &mut HashMap<StmtId, BasicBlockId>,
    statement: StmtId,
    prior: Option<BasicBlockId>,
) {
    if let Some(prior) = prior {
        headers.insert(statement, prior);
    } else {
        headers.remove(&statement);
    }
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
    let direct = super::evaluation_order::statement_expressions(stmt)
        .into_iter()
        .find_map(|expression| first_unlowered_await(expression, await_replacements));
    direct.or_else(|| match &stmt.kind {
        // Switch-case expressions retain the existing eager contract. Else-if
        // conditions deliberately do not appear here: their synthesized nested
        // `if` owns evaluation after preceding conditions fail.
        HirStmtKind::Switch { cases, .. } => cases
            .iter()
            .find_map(|(case, _)| first_unlowered_await(case, await_replacements)),
        _ => None,
    })
}

fn first_unlowered_await<'a>(
    expr: &'a HirExpr,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Option<&'a HirExpr> {
    if await_replacements.contains_key(&expr.id) {
        return None;
    }
    if matches!(expr.kind, HirExprKind::Await(_)) {
        return Some(expr);
    }
    super::evaluation_order::expression_children(expr)
        .into_iter()
        .find_map(|child| first_unlowered_await(child, await_replacements))
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
