use super::super::NativeBlock;
use serde::{Deserialize, Serialize};

/// Source-derived upper bounds for one suspended native MIR site. The executor
/// admits the physical continuation backing before activating that site.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeExpressionContinuationGeometry {
    pub region_capacity: u32,
    pub retained_input_capacity: u32,
}

impl NativeExpressionContinuationGeometry {
    pub fn for_blocks(blocks: &[NativeBlock]) -> Option<Self> {
        let mut region_capacity = 0usize;
        let mut retained_input_capacity = 0usize;
        // The one verified fixed destination slot can coexist with any RHS
        // site: original arena generation, bound RHS, and completed root
        // publication. Selector values use the destination's exact Runtime
        // root leases, not additional source-quoted arena references.
        let persistent_fixed_references = if blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .any(|instruction| {
                matches!(
                    &instruction.operation,
                    super::super::NativeOperation::Statement(
                        runmat_mir::MirStmtKind::PrepareFixedAssignment(_)
                    )
                )
            }) {
            3usize
        } else {
            0
        };
        for instruction in blocks.iter().flat_map(|block| &block.instructions) {
            let mut operands = Some(0usize);
            let mut regions = Some(0usize);
            match &instruction.operation {
                super::super::NativeOperation::Rvalue { value, .. } => {
                    value.visit_operands(|_| {
                        operands = operands.and_then(|count| count.checked_add(1))
                    });
                    value.visit_expression_regions(|_| {
                        regions = regions.and_then(|count| count.checked_add(1))
                    });
                }
                super::super::NativeOperation::Statement(statement) => {
                    statement.visit_operands(|_| {
                        operands = operands.and_then(|count| count.checked_add(1))
                    });
                    statement.visit_expression_regions(|_| {
                        regions = regions.and_then(|count| count.checked_add(1))
                    });
                }
            }
            let cursors = operation_cursor_regions(&instruction.operation)?;
            let explicit_regions = regions?;
            if explicit_regions == 0 && cursors == 0 {
                continue;
            }
            // A participating site retains no more inputs than its typed MIR
            // operand occurrences, prepared results and root/RHS. Constants
            // are counted by MIR traversal even when SSA has no input slot.
            let inputs = 2usize
                .checked_add(operands?)?
                .checked_add(explicit_regions)?
                .checked_add(cursors)?
                .checked_add(instruction.inputs.len())?;
            retained_input_capacity = retained_input_capacity.max(inputs);
            // Count the actual typed expression regions and cursor-entry
            // roles, including roles nested inside another region's steps.
            // An embedded construct is not necessarily a cursor (and one
            // construct may contain several dotted call-argument cursors).
            let region_slots = explicit_regions.checked_add(cursors)?;
            region_capacity = region_capacity.max(region_slots);
        }
        Some(Self {
            region_capacity: u32::try_from(region_capacity).ok()?,
            retained_input_capacity: u32::try_from(
                retained_input_capacity.checked_add(persistent_fixed_references)?,
            )
            .ok()?,
        })
    }
}

fn operation_cursor_regions(operation: &super::super::NativeOperation) -> Option<usize> {
    match operation {
        super::super::NativeOperation::Rvalue { value, result } => {
            let value_regions = rvalue_cursor_regions(value)?;
            let target_regions = match result {
                super::super::NativeRvalueResult::MultiAssignment(targets) => {
                    target_list_cursor_regions(targets)?
                }
                super::super::NativeRvalueResult::SequenceAssignment(target) => {
                    sequence_cursor_regions(target)?
                }
                super::super::NativeRvalueResult::Assignment
                | super::super::NativeRvalueResult::Expression
                | super::super::NativeRvalueResult::Discard
                | super::super::NativeRvalueResult::Terminator => 0,
            };
            value_regions.checked_add(target_regions)
        }
        super::super::NativeOperation::Statement(statement) => statement_cursor_regions(statement),
    }
}

fn statement_cursor_regions(statement: &runmat_mir::MirStmtKind) -> Option<usize> {
    match statement {
        runmat_mir::MirStmtKind::PrepareFixedAssignment(phase) => match phase.selector_region() {
            Some(region) => 1usize
                .checked_add(expression_steps_cursor_regions(&region.steps)?)?
                .checked_add(call_argument_cursor_regions(&region.argument)?),
            None => Some(0),
        },
        runmat_mir::MirStmtKind::PreparedAssign { .. } => Some(1),
        runmat_mir::MirStmtKind::Assign { place, value } => {
            place_cursor_regions(place)?.checked_add(rvalue_cursor_regions(value)?)
        }
        runmat_mir::MirStmtKind::MultiAssign { targets, value } => {
            target_list_cursor_regions(targets)?.checked_add(rvalue_cursor_regions(value)?)
        }
        runmat_mir::MirStmtKind::SequenceAssign { target, value } => {
            sequence_cursor_regions(target)?.checked_add(rvalue_cursor_regions(value)?)
        }
        runmat_mir::MirStmtKind::Expr(value) => rvalue_cursor_regions(value),
        runmat_mir::MirStmtKind::CaptureSequence { source, .. } => expansion_cursor_regions(source),
        runmat_mir::MirStmtKind::PlaceMutation(_)
        | runmat_mir::MirStmtKind::StatementOutput(_)
        | runmat_mir::MirStmtKind::WorkspaceEffect { .. }
        | runmat_mir::MirStmtKind::EnvironmentEffect(_) => Some(0),
    }
}

fn rvalue_cursor_regions(value: &runmat_mir::MirRvalue) -> Option<usize> {
    match value {
        runmat_mir::MirRvalue::ShortCircuit { right_temps, .. } => {
            right_temps.iter().try_fold(1usize, |count, step| {
                count.checked_add(statement_cursor_regions(&step.kind)?)
            })
        }
        runmat_mir::MirRvalue::Index { indexing, .. } => indexing_cursor_regions(indexing),
        runmat_mir::MirRvalue::SubscriptChain(chain) => chain_cursor_regions(chain),
        runmat_mir::MirRvalue::Call(call) => call_arguments_cursor_regions(&call.args),
        runmat_mir::MirRvalue::Future { args, .. }
        | runmat_mir::MirRvalue::ExternalFuture { args, .. } => call_arguments_cursor_regions(args),
        runmat_mir::MirRvalue::Use(_)
        | runmat_mir::MirRvalue::Unary(_, _)
        | runmat_mir::MirRvalue::Binary(_, _, _)
        | runmat_mir::MirRvalue::Range { .. }
        | runmat_mir::MirRvalue::Aggregate { .. }
        | runmat_mir::MirRvalue::StructLiteral { .. }
        | runmat_mir::MirRvalue::ObjectLiteral { .. }
        | runmat_mir::MirRvalue::Member { .. }
        | runmat_mir::MirRvalue::DynamicMember { .. }
        | runmat_mir::MirRvalue::WorkspaceFirstStaticProperty { .. }
        | runmat_mir::MirRvalue::MetaClass(_)
        | runmat_mir::MirRvalue::Colon
        | runmat_mir::MirRvalue::End
        | runmat_mir::MirRvalue::Spawn(_)
        | runmat_mir::MirRvalue::Distributed(_)
        | runmat_mir::MirRvalue::Collective(_) => Some(0),
    }
}

fn call_arguments_cursor_regions(args: &[runmat_mir::MirCallArg]) -> Option<usize> {
    args.iter().try_fold(0usize, |count, argument| {
        count.checked_add(call_argument_cursor_regions(argument)?)
    })
}

fn call_argument_cursor_regions(argument: &runmat_mir::MirCallArg) -> Option<usize> {
    match argument {
        runmat_mir::MirCallArg::Expansion(source) => expansion_cursor_regions(source),
        runmat_mir::MirCallArg::Single(_) | runmat_mir::MirCallArg::CapturedSequence(_) => Some(0),
    }
}

fn expression_steps_cursor_regions(steps: &[runmat_mir::MirExpressionStep]) -> Option<usize> {
    steps.iter().try_fold(0usize, |count, step| {
        let nested = match step {
            runmat_mir::MirExpressionStep::Let { value, .. } => rvalue_cursor_regions(value)?,
            runmat_mir::MirExpressionStep::CaptureSequence { source, .. } => {
                expansion_cursor_regions(source)?
            }
        };
        count.checked_add(nested)
    })
}

fn indexing_cursor_regions(indexing: &runmat_mir::MirIndexing) -> Option<usize> {
    indexing
        .components
        .iter()
        .try_fold(0usize, |count, component| {
            let nested = match component {
                runmat_mir::MirIndexComponent::ContextualExpr(region) => {
                    expression_steps_cursor_regions(region.steps())?
                }
                runmat_mir::MirIndexComponent::PreparedExpr(region) => {
                    expression_steps_cursor_regions(region.steps())?
                }
                runmat_mir::MirIndexComponent::Expansion(region) => 1usize
                    .checked_add(expression_steps_cursor_regions(&region.steps)?)?
                    .checked_add(call_argument_cursor_regions(&region.argument)?)?,
                runmat_mir::MirIndexComponent::Colon | runmat_mir::MirIndexComponent::Expr(_) => 0,
            };
            count.checked_add(nested)
        })
}

fn expansion_cursor_regions(source: &runmat_mir::MirExpansionSource) -> Option<usize> {
    match source {
        runmat_mir::MirExpansionSource::SubscriptChain(chain) => chain_cursor_regions(chain),
        runmat_mir::MirExpansionSource::CellContents { indexing, .. } => {
            indexing_cursor_regions(indexing)
        }
        runmat_mir::MirExpansionSource::ReturnedOutputs(_)
        | runmat_mir::MirExpansionSource::Member { .. }
        | runmat_mir::MirExpansionSource::DynamicMember { .. } => Some(0),
    }
}

fn chain_cursor_regions(chain: &runmat_mir::MirSubscriptChain) -> Option<usize> {
    chain.steps.iter().try_fold(0usize, |count, step| {
        let nested = match step {
            runmat_mir::MirSubscriptStep::Index(indexing) => indexing_cursor_regions(indexing)?,
            runmat_mir::MirSubscriptStep::PreparedDynamicMember(region) => {
                expression_steps_cursor_regions(region.steps())?
            }
            runmat_mir::MirSubscriptStep::DottedInvoke { arguments, .. } => {
                arguments.iter().try_fold(0usize, |total, argument| {
                    let next = match argument {
                        runmat_mir::MirDottedInvokeArgument::Colon => 0,
                        runmat_mir::MirDottedInvokeArgument::Contextual(region) => {
                            expression_steps_cursor_regions(region.steps())?
                        }
                        runmat_mir::MirDottedInvokeArgument::Call(region) => 1usize
                            .checked_add(expression_steps_cursor_regions(&region.steps)?)?
                            .checked_add(call_argument_cursor_regions(&region.argument)?)?,
                    };
                    total.checked_add(next)
                })?
            }
            runmat_mir::MirSubscriptStep::Member(_)
            | runmat_mir::MirSubscriptStep::DynamicMember(_) => 0,
        };
        count.checked_add(nested)
    })
}

fn target_list_cursor_regions(targets: &runmat_mir::MirOutputTargetList) -> Option<usize> {
    targets.targets.iter().try_fold(0usize, |count, target| {
        let next = match target {
            runmat_mir::MirOutputTarget::Place(place) => place_cursor_regions(place)?,
            runmat_mir::MirOutputTarget::Sequence(target) => sequence_cursor_regions(target)?,
            runmat_mir::MirOutputTarget::Discard => 0,
        };
        count.checked_add(next)
    })
}

fn sequence_cursor_regions(target: &runmat_mir::MirSequenceTarget) -> Option<usize> {
    // Sequence preparation and replay each own an expression-progress cursor.
    2usize.checked_add(place_cursor_regions(target.base())?)
}

fn place_cursor_regions(place: &runmat_mir::MirPlace) -> Option<usize> {
    match place {
        runmat_mir::MirPlace::Local(_) | runmat_mir::MirPlace::Binding(_) => Some(0),
        runmat_mir::MirPlace::Member(base, _) | runmat_mir::MirPlace::DynamicMember(base, _) => {
            // A nonlocal mutation has preparation, completed-prefix skip and
            // drive cursors. Count all three roles at each nested place step.
            3usize.checked_add(place_cursor_regions(base)?)
        }
        runmat_mir::MirPlace::Index(base, indexing) => 3usize
            .checked_add(place_cursor_regions(base)?)?
            .checked_add(indexing_cursor_regions(indexing)?),
    }
}
