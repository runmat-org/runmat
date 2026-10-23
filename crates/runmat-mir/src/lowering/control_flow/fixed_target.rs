//! Split target evaluation through the ordinary expression/Await CFG owner.
//! The descriptor reserves destinations; only the phase regions evaluate them.

use runmat_hir::{HirError, HirStmt};

use crate::lowering::fixed_target::{
    self, FixedTargetEvent, SelectorRegionContext, SelectorSource,
};
use crate::{
    BasicBlock, BasicBlockId, MirFixedPreparationPhase, MirFixedSelectorDestination, MirOperand,
    MirTerminator, MirTerminatorKind,
};

use super::{await_expression::ExpressionTarget, BlockLoweringEnv, ControlFlowBuilder};

impl ControlFlowBuilder {
    pub(super) fn lower_suspended_fixed_target(
        &mut self,
        statement_index: usize,
        statement: &HirStmt,
        final_terminator: MirTerminator,
        env: BlockLoweringEnv<'_>,
    ) -> Result<BasicBlockId, HirError> {
        let plan = fixed_target::plan(env.ctx, statement)?.ok_or_else(|| {
            HirError::new("suspended fixed target requires an expanded assignment")
        })?;
        // Each event sees only earlier source receipts. The final replacement
        // set is used solely to skip the already-completed LHS on RHS resume.
        let mut completed = env.await_replacements.clone();
        let mut prefixes = Vec::with_capacity(plan.events.len());
        for event in &plan.events {
            prefixes.push(completed.clone());
            match event {
                FixedTargetEvent::Receiver {
                    expression,
                    destination,
                } => {
                    completed.insert(expression.id, MirOperand::Local(*destination));
                }
                FixedTargetEvent::Selector {
                    source: SelectorSource::Expression(expression),
                    destination,
                    ..
                } => match destination {
                    MirFixedSelectorDestination::Scalar(local) => {
                        completed.insert(expression.id, MirOperand::Local(*local));
                    }
                    MirFixedSelectorDestination::Sequence(sequence) => {
                        completed.insert_sequence(expression.id, *sequence);
                    }
                },
                _ => {}
            }
        }
        let mut next = self.lower_continuation_target(
            statement_index,
            final_terminator,
            BlockLoweringEnv {
                fixed_preparation: Some(&plan.owner),
                await_replacements: &completed,
                ..env
            },
        )?;
        let capture_before = plan
            .events
            .iter()
            .position(|event| !matches!(event, FixedTargetEvent::Receiver { .. }))
            .ok_or_else(|| HirError::new("fixed target has no destination path"))?;
        for (index, (event, replacements)) in plan.events.iter().zip(&prefixes).enumerate().rev() {
            let id = self.fresh_block();
            self.catch_scopes.insert(id, self.active_catches.clone());
            let mut statements = Vec::new();
            let block = match event {
                FixedTargetEvent::Receiver {
                    expression,
                    destination,
                } => self.lower_expression_into(
                    env.ctx,
                    expression,
                    ExpressionTarget {
                        block: id,
                        destination: MirFixedSelectorDestination::Scalar(*destination),
                        selector: None,
                        continuation: next,
                        span: expression.span,
                    },
                    replacements,
                )?,
                FixedTargetEvent::Selector {
                    path_step,
                    component,
                    source: SelectorSource::Expression(expression),
                    destination,
                } => self.lower_expression_into(
                    env.ctx,
                    expression,
                    ExpressionTarget {
                        block: id,
                        destination: *destination,
                        selector: Some(SelectorRegionContext {
                            assignment: &plan.owner,
                            path_step: *path_step,
                            component: *component,
                            span: statement.span,
                        }),
                        continuation: next,
                        span: expression.span,
                    },
                    replacements,
                )?,
                _ => {
                    let phase = match event {
                        FixedTargetEvent::BeginIndex { path_step } => {
                            MirFixedPreparationPhase::BeginIndex {
                                path_step: *path_step,
                            }
                        }
                        FixedTargetEvent::FinishStep { path_step } => {
                            MirFixedPreparationPhase::FinishStep {
                                path_step: *path_step,
                            }
                        }
                        FixedTargetEvent::Selector {
                            path_step,
                            component,
                            source: SelectorSource::End { offset, span },
                            destination,
                        } => MirFixedPreparationPhase::EvaluateSelector {
                            path_step: *path_step,
                            component: *component,
                            region: Box::new(
                                crate::lowering::expr::contextual_end_argument_region(
                                    env.ctx, *offset, *span,
                                ),
                            ),
                            destination: *destination,
                        },
                        _ => unreachable!("expression events handled above"),
                    };
                    fixed_target::push_phase(&mut statements, &plan.owner, phase, statement.span);
                    BasicBlock {
                        id,
                        statements,
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(next),
                            span: statement.span,
                        },
                    }
                }
            };
            self.blocks.push(block);
            next = id;
            if index == capture_before {
                let capture = self.fresh_block();
                self.catch_scopes
                    .insert(capture, self.active_catches.clone());
                let mut statements = Vec::new();
                fixed_target::push_phase(
                    &mut statements,
                    &plan.owner,
                    MirFixedPreparationPhase::CaptureReceiver,
                    statement.span,
                );
                self.blocks.push(BasicBlock {
                    id: capture,
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Goto(next),
                        span: statement.span,
                    },
                });
                next = capture;
            }
        }
        Ok(next)
    }
}
