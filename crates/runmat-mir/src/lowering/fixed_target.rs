//! Source-ordered construction of the existing fixed destination phases.
//! Reservations are compiler locals/sequence slots, not a runtime registry.

use super::{AwaitReplacements, MirLoweringContext};

#[derive(Clone, Copy)]
pub(crate) struct SelectorRegionContext<'a> {
    pub assignment: &'a MirPreparedAssignment,
    pub path_step: usize,
    pub component: Option<usize>,
    pub span: Span,
}

impl SelectorRegionContext<'_> {
    pub fn publish(
        self,
        output: &mut Vec<crate::MirStmt>,
        region: MirCallArgumentRegion,
        destination: MirFixedSelectorDestination,
    ) {
        push_phase(
            output,
            self.assignment,
            crate::MirFixedPreparationPhase::EvaluateSelector {
                path_step: self.path_step,
                component: self.component,
                region: Box::new(region),
                destination,
            },
            self.span,
        );
    }
}
use crate::{
    MirCallArg, MirCallArgumentRegion, MirFixedSelectorDestination, MirIndexComponent, MirIndexing,
    MirLocalId, MirOperand, MirPlace, MirPreparedAssignment,
};
use runmat_hir::{
    HirError, HirExpr, HirExprKind, HirPlace, HirStmt, HirStmtKind, IndexComponent,
    IndexingSemantics, Span,
};

pub(super) enum SelectorSource<'a> {
    Expression(&'a HirExpr),
    End { offset: isize, span: Span },
}

pub(super) enum FixedTargetEvent<'a> {
    /// A nonbinding receiver expression is evaluated before receiver capture.
    Receiver {
        expression: &'a HirExpr,
        destination: MirLocalId,
    },
    Selector {
        path_step: usize,
        component: Option<usize>,
        source: SelectorSource<'a>,
        destination: MirFixedSelectorDestination,
    },
    FinishStep {
        path_step: usize,
    },
    BeginIndex {
        path_step: usize,
    },
}

pub(super) struct FixedTargetPlan<'a> {
    pub owner: MirPreparedAssignment,
    pub events: Vec<FixedTargetEvent<'a>>,
}

pub(super) fn plan<'a>(
    ctx: &MirLoweringContext,
    statement: &'a HirStmt,
) -> Result<Option<FixedTargetPlan<'a>>, HirError> {
    let HirStmtKind::Assign(target, rhs, _) = &statement.kind else {
        return Ok(None);
    };
    if !super::stmt::assignment_target_requires_preparation(target) {
        return Ok(None);
    }
    let mut builder = TargetBuilder {
        ctx,
        span: statement.span,
        events: Vec::new(),
        steps: 0,
    };
    let place = builder.place(target)?;
    let delete = super::stmt::is_empty_array_deletion_place(&place, rhs);
    let owner = MirPreparedAssignment {
        source_statement: statement.id,
        mutation: super::stmt::place_mutation(place, delete),
        // Selector work is in typed per-component regions, not eager lifted
        // work that could cross an earlier path callback or Await.
        steps: Vec::new(),
    };
    owner.validate_phased().map_err(HirError::new)?;
    Ok(Some(FixedTargetPlan {
        owner,
        events: builder.events,
    }))
}

struct TargetBuilder<'ctx, 'hir> {
    ctx: &'ctx MirLoweringContext,
    span: Span,
    events: Vec<FixedTargetEvent<'hir>>,
    steps: usize,
}

impl<'ctx, 'hir> TargetBuilder<'ctx, 'hir> {
    fn place(&mut self, place: &'hir HirPlace) -> Result<MirPlace, HirError> {
        match place {
            HirPlace::Binding(binding) => {
                Ok(MirPlace::Local(self.ctx.local_for_binding(*binding)?))
            }
            HirPlace::Member(base, member) => {
                let base = self.receiver(base)?;
                let place = MirPlace::Member(Box::new(base), member.clone());
                self.finish();
                Ok(place)
            }
            HirPlace::MemberDynamic(base, member) => {
                let base = self.receiver(base)?;
                let member = self.scalar(member, None);
                let place = MirPlace::DynamicMember(Box::new(base), member);
                self.finish();
                Ok(place)
            }
            HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
                let base = self.receiver(base)?;
                let indexing = self.indexing(indexing)?;
                let place = MirPlace::Index(Box::new(base), indexing);
                self.finish();
                Ok(place)
            }
        }
    }

    fn receiver(&mut self, expression: &'hir HirExpr) -> Result<MirPlace, HirError> {
        match &expression.kind {
            HirExprKind::Binding(binding) => {
                Ok(MirPlace::Local(self.ctx.local_for_binding(*binding)?))
            }
            HirExprKind::Member { base, member, .. } => {
                let base = self.receiver(base)?;
                let place = MirPlace::Member(Box::new(base), member.clone());
                self.finish();
                Ok(place)
            }
            HirExprKind::MemberDynamic { base, member, .. } => {
                let base = self.receiver(base)?;
                let member = self.scalar(member, None);
                let place = MirPlace::DynamicMember(Box::new(base), member);
                self.finish();
                Ok(place)
            }
            HirExprKind::Index(base, indexing) => {
                let base = self.receiver(base)?;
                let indexing = self.indexing(indexing)?;
                let place = MirPlace::Index(Box::new(base), indexing);
                self.finish();
                Ok(place)
            }
            _ => {
                let destination = self.ctx.fresh_temp(expression.span);
                self.events.push(FixedTargetEvent::Receiver {
                    expression,
                    destination,
                });
                Ok(MirPlace::Local(destination))
            }
        }
    }

    fn scalar(&mut self, expression: &'hir HirExpr, component: Option<usize>) -> MirOperand {
        let local = self.ctx.fresh_temp(expression.span);
        self.events.push(FixedTargetEvent::Selector {
            path_step: self.steps,
            component,
            source: SelectorSource::Expression(expression),
            destination: MirFixedSelectorDestination::Scalar(local),
        });
        MirOperand::Local(local)
    }

    fn indexing(&mut self, indexing: &'hir IndexingSemantics) -> Result<MirIndexing, HirError> {
        let expands = indexing.components.iter().any(|component| matches!(component, IndexComponent::Expr(expression) if expression.expands_call_argument_sequence()));
        let outer_end = indexing.components.iter().any(|component| match component {
            IndexComponent::End { .. } => true,
            IndexComponent::Expr(expression) | IndexComponent::Logical(expression) => {
                !expression.expands_call_argument_sequence() && contains_outer_end(expression)
            }
            IndexComponent::Colon => false,
        });
        if expands && outer_end {
            return Err(HirError::new(
                "contextual end cannot share an index list with variable-width argument expansion",
            ));
        }
        self.events.push(FixedTargetEvent::BeginIndex {
            path_step: self.steps,
        });
        let mut components = Vec::with_capacity(indexing.components.len());
        for (component, source) in indexing.components.iter().enumerate() {
            components.push(match source {
                IndexComponent::Colon => MirIndexComponent::Colon,
                IndexComponent::End { offset, .. } => {
                    let local = self.ctx.fresh_temp(self.span);
                    self.events.push(FixedTargetEvent::Selector {
                        path_step: self.steps,
                        component: Some(component),
                        source: SelectorSource::End {
                            offset: *offset,
                            span: self.span,
                        },
                        destination: MirFixedSelectorDestination::Scalar(local),
                    });
                    MirIndexComponent::Expr(MirOperand::Local(local))
                }
                IndexComponent::Expr(expression) if expression.expands_call_argument_sequence() => {
                    let sequence = self.ctx.fresh_sequence_local();
                    self.events.push(FixedTargetEvent::Selector {
                        path_step: self.steps,
                        component: Some(component),
                        source: SelectorSource::Expression(expression),
                        destination: MirFixedSelectorDestination::Sequence(sequence),
                    });
                    MirIndexComponent::Expansion(MirCallArgumentRegion {
                        steps: Vec::new(),
                        argument: MirCallArg::CapturedSequence(sequence),
                        span: expression.span,
                    })
                }
                IndexComponent::Expr(expression) | IndexComponent::Logical(expression) => {
                    MirIndexComponent::Expr(self.scalar(expression, Some(component)))
                }
            });
        }
        Ok(MirIndexing {
            argument_spans: super::expr::index_argument_spans(indexing),
            kind: indexing.kind,
            plan: super::expr::classify_mir_index_plan(indexing),
            components,
            result_context: indexing.result_context,
            cell_expand_all: indexing.kind == runmat_hir::IndexKind::Brace
                && indexing
                    .components
                    .iter()
                    .all(|component| matches!(component, IndexComponent::Colon)),
        })
    }

    fn finish(&mut self) {
        self.events.push(FixedTargetEvent::FinishStep {
            path_step: self.steps,
        });
        self.steps += 1;
    }
}

fn contains_outer_end(expression: &HirExpr) -> bool {
    match &expression.kind {
        HirExprKind::End => true,
        // Nested indexing supplies its own component context. Its receiver
        // expression can still depend on the surrounding context.
        HirExprKind::Index(base, _) => contains_outer_end(base),
        HirExprKind::SubscriptChain(chain) => contains_outer_end(&chain.root),
        _ => super::evaluation_order::expression_children(expression)
            .into_iter()
            .any(contains_outer_end),
    }
}

/// Lower a completed scalar/CSL selector chunk with the existing argument
/// region owner; Await CFG construction supplies completed replacements.
pub(super) fn expression_region(
    ctx: &MirLoweringContext,
    expression: &HirExpr,
    replacements: &AwaitReplacements,
    destination: MirFixedSelectorDestination,
) -> Result<MirCallArgumentRegion, HirError> {
    let mut steps = Vec::new();
    let argument = match destination {
        MirFixedSelectorDestination::Scalar(_) => {
            MirCallArg::Single(super::expr::lower_operand_with_replacements(
                ctx,
                expression,
                &mut steps,
                replacements,
            )?)
        }
        MirFixedSelectorDestination::Sequence(_) => {
            super::expr::lower_call_arg(ctx, expression, &mut steps, replacements)?
        }
    };
    Ok(MirCallArgumentRegion {
        steps: crate::MirExpressionRegion::steps_from_lowered(steps)?,
        argument,
        span: expression.span,
    })
}

pub(super) fn lower_completed(
    ctx: &MirLoweringContext,
    statement: &HirStmt,
    replacements: &AwaitReplacements,
) -> Result<Option<(MirPreparedAssignment, Vec<crate::MirStmt>)>, HirError> {
    let Some(plan) = plan(ctx, statement)? else {
        return Ok(None);
    };
    let mut output = Vec::new();
    let mut captured = false;
    for event in plan.events {
        if let FixedTargetEvent::Receiver {
            expression,
            destination,
        } = event
        {
            let value = super::expr::lower_expr_with_replacements(
                ctx,
                expression,
                &mut output,
                replacements,
            )?;
            output.push(crate::MirStmt {
                kind: crate::MirStmtKind::Assign {
                    place: MirPlace::Local(destination),
                    value,
                },
                span: expression.span,
            });
            continue;
        }
        if !captured {
            push_phase(
                &mut output,
                &plan.owner,
                crate::MirFixedPreparationPhase::CaptureReceiver,
                statement.span,
            );
            captured = true;
        }
        let phase = match event {
            FixedTargetEvent::BeginIndex { path_step } => {
                crate::MirFixedPreparationPhase::BeginIndex { path_step }
            }
            FixedTargetEvent::FinishStep { path_step } => {
                crate::MirFixedPreparationPhase::FinishStep { path_step }
            }
            FixedTargetEvent::Selector {
                path_step,
                component,
                source,
                destination,
            } => {
                let region = match source {
                    SelectorSource::Expression(expression) => {
                        expression_region(ctx, expression, replacements, destination)?
                    }
                    SelectorSource::End { offset, span } => {
                        super::expr::contextual_end_argument_region(ctx, offset, span)
                    }
                };
                crate::MirFixedPreparationPhase::EvaluateSelector {
                    path_step,
                    component,
                    region: Box::new(region),
                    destination,
                }
            }
            FixedTargetEvent::Receiver { .. } => {
                unreachable!("receiver handled before phase capture")
            }
        };
        push_phase(&mut output, &plan.owner, phase, statement.span);
    }
    Ok(Some((plan.owner, output)))
}

pub(super) fn push_phase(
    output: &mut Vec<crate::MirStmt>,
    assignment: &MirPreparedAssignment,
    phase: crate::MirFixedPreparationPhase,
    span: Span,
) {
    output.push(crate::MirStmt {
        kind: crate::MirStmtKind::PrepareFixedAssignment(crate::MirFixedPreparation {
            assignment: assignment.clone(),
            phase,
        }),
        span,
    });
}
