use runmat_hir::{HirError, HirExpr, HirExprKind, Span};

use crate::{
    BasicBlock, BasicBlockId, MirFixedSelectorDestination, MirOperand, MirPlace, MirTerminator,
    MirTerminatorKind,
};

use super::{first_unlowered_await, ControlFlowBuilder};
use crate::lowering::{
    await_preparation::expression::{operand_in_context, prepare_expr_prefix_in_context},
    conditional_await::conditional_await_guard_in_expr,
    MirLoweringContext,
};

#[path = "await_expression/blocks.rs"]
mod blocks;

#[derive(Clone, Copy)]
pub(super) struct ExpressionTarget<'a> {
    pub block: BasicBlockId,
    pub destination: MirFixedSelectorDestination,
    pub selector: Option<crate::lowering::fixed_target::SelectorRegionContext<'a>>,
    pub continuation: BasicBlockId,
    pub span: Span,
}

impl ControlFlowBuilder {
    pub(super) fn lower_expression_into(
        &mut self,
        ctx: &MirLoweringContext,
        expression: &HirExpr,
        target: ExpressionTarget<'_>,
        replacements: &crate::lowering::AwaitReplacements,
    ) -> Result<BasicBlock, HirError> {
        self.catch_scopes
            .insert(target.block, self.active_catches.clone());
        let mut statements = Vec::new();
        let Some(await_expr) = first_unlowered_await(expression, replacements) else {
            return blocks::finish_expression(ctx, expression, target, replacements, statements);
        };
        if let Some(guard) = conditional_await_guard_in_expr(
            expression,
            await_expr.id,
            &self.active_conditional_awaits,
        ) {
            let mut prepared = replacements.clone();
            prepare_expr_prefix_in_context(
                ctx,
                expression,
                guard.expression.id,
                &mut statements,
                &mut prepared,
                target.selector,
            )?;
            let condition =
                operand_in_context(ctx, guard.left, &mut statements, &prepared, target.selector)?;
            prepared.insert(guard.left.id, condition.clone());
            let guarded_result = ctx.fresh_temp(guard.expression.span);
            let mut joined = prepared.clone();
            joined.insert(guard.expression.id, MirOperand::Local(guarded_result));
            let joined_id = self.fresh_block();
            let joined_block = self.lower_expression_into(
                ctx,
                expression,
                ExpressionTarget {
                    block: joined_id,
                    ..target
                },
                &joined,
            )?;
            self.blocks.push(joined_block);
            let skipped_id = self.fresh_block();
            self.catch_scopes
                .insert(skipped_id, self.active_catches.clone());
            self.blocks.push(blocks::skipped_guard(
                skipped_id,
                guard.skipped_value,
                guarded_result,
                joined_id,
                guard.expression.span,
                target.selector,
            ));
            self.active_conditional_awaits.insert(guard.expression.id);
            let evaluated_id = self.fresh_block();
            let evaluated = self.lower_expression_into(
                ctx,
                guard.expression,
                ExpressionTarget {
                    block: evaluated_id,
                    destination: MirFixedSelectorDestination::Scalar(guarded_result),
                    selector: target.selector,
                    continuation: joined_id,
                    span: target.span,
                },
                &prepared,
            );
            self.active_conditional_awaits.remove(&guard.expression.id);
            self.blocks.push(evaluated?);
            let (then_block, else_block) = if guard.evaluate_when_true {
                (evaluated_id, skipped_id)
            } else {
                (skipped_id, evaluated_id)
            };
            return Ok(blocks::branch(
                target.block,
                statements,
                condition,
                then_block,
                else_block,
                target.span,
            ));
        }
        let HirExprKind::Await(future_expression) = &await_expr.kind else {
            unreachable!()
        };
        let mut prepared = replacements.clone();
        prepare_expr_prefix_in_context(
            ctx,
            expression,
            await_expr.id,
            &mut statements,
            &mut prepared,
            target.selector,
        )?;
        let future = operand_in_context(
            ctx,
            future_expression,
            &mut statements,
            &prepared,
            target.selector,
        )?;
        let awaited_result = ctx.fresh_temp(await_expr.span);
        prepared.insert(await_expr.id, MirOperand::Local(awaited_result));
        let resume = self.fresh_block();
        let resume_block = self.lower_expression_into(
            ctx,
            expression,
            ExpressionTarget {
                block: resume,
                ..target
            },
            &prepared,
        )?;
        self.blocks.push(resume_block);
        Ok(BasicBlock {
            id: target.block,
            statements,
            terminator: MirTerminator {
                kind: MirTerminatorKind::Await {
                    future,
                    result: Some(MirPlace::Local(awaited_result)),
                    resume,
                },
                span: target.span,
            },
        })
    }
}
