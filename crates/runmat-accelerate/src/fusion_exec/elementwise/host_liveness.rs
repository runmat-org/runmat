//! Exact whole-tensor liveness for selected VM host numerical regions.
//!
//! This plan never grants execution or allocation authority. It authenticates
//! when complete real tensor owners can move back into a vacant physical slot
//! after their final VM operand read. Named calls enter this plan only through
//! the already-admitted exact-real output specification; catalog names do not
//! grant recycling authority.

use crate::fusion::{FusionGroupPlan, FusionOp};
use crate::graph::{PrimitiveOp, ValueId};
use runmat_runtime::context::{
    VmFusionHostInstruction, VmFusionHostOperation, VmFusionHostRealSlotSpec,
};
use runmat_types::{NumericClass, OperatorKind};
use runmat_value::NumericDType;

#[path = "host_liveness/aliases.rs"]
mod aliases;
use aliases::Aliases;
#[path = "host_liveness/quote.rs"]
mod quote;

#[cfg(test)]
#[path = "host_liveness/tests.rs"]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct HostRealSlot {
    pub dtype: NumericDType,
    pub shape: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HostOperandRetirement {
    pub operand_position: usize,
    pub slot: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct HostOperationLiveness {
    pub operation: VmFusionHostOperation,
    pub instruction: VmFusionHostInstruction,
    pub output_slot: Option<usize>,
    pub retirements: [Option<HostOperandRetirement>; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct HostLivenessPlan {
    pub slots: Vec<HostRealSlot>,
    pub operations: Vec<HostOperationLiveness>,
}

/// Checked upper bound for the planner vectors that coexist with the selected
/// semantic program. The retained plan owns its slot/operation vectors and
/// slot shapes through output installation; the alias tables coexist while
/// that plan is constructed, and the borrowed slot-spec projection coexists
/// with it during Runtime carrier preparation.
pub(super) fn preparation_bytes(
    value_ids_upper: usize,
    operations: usize,
    max_rank: usize,
) -> Option<u64> {
    quote::preparation_bytes(value_ids_upper, operations, max_rank)
}

impl HostLivenessPlan {
    pub(super) fn slot_specs(&self) -> Vec<VmFusionHostRealSlotSpec<'_>> {
        let mut specs = Vec::with_capacity(self.slots.len());
        specs.extend(self.slots.iter().map(|slot| VmFusionHostRealSlotSpec {
            dtype: slot.dtype,
            shape: &slot.shape,
        }));
        specs
    }
}

/// Plan only operations whose moved operands have a joined VM retirement
/// consumer. Named eligibility comes from the already-admitted output owner;
/// returning `None` keeps the legacy exact output plan.
pub(super) fn plan(
    source: &FusionGroupPlan,
    program: &crate::fusion::semantics::ResolvedProgram,
    canonical: &[VmFusionHostOperation],
    named_instruction: impl Fn(
        usize,
        runmat_builtins::BuiltinBindingIdentity,
        usize,
    ) -> Option<VmFusionHostInstruction>,
) -> Option<HostLivenessPlan> {
    plan_with(
        source,
        canonical,
        |id| slot_for(program, id),
        named_instruction,
    )
}

fn plan_with(
    source: &FusionGroupPlan,
    canonical: &[VmFusionHostOperation],
    mut slot_for: impl FnMut(ValueId) -> Option<HostRealSlot>,
    named_instruction: impl Fn(
        usize,
        runmat_builtins::BuiltinBindingIdentity,
        usize,
    ) -> Option<VmFusionHostInstruction>,
) -> Option<HostLivenessPlan> {
    if canonical.len() != source.operations.len() || source.operations.is_empty() {
        return None;
    }

    let id_occurrences = source
        .inputs
        .len()
        .checked_add(usize::from(source.output.is_some()))?
        .checked_add(source.materialized_stores.len())?
        .checked_add(
            source
                .operations
                .iter()
                .try_fold(0usize, |count, operation| {
                    let (inputs, _) = parts(operation)?;
                    count.checked_add(inputs.len().checked_add(1)?)
                })?,
        )?;
    let mut ids = Vec::with_capacity(id_occurrences);
    ids.extend_from_slice(&source.inputs);
    if let Some(output) = source.output {
        ids.push(output);
    }
    ids.extend(
        source
            .materialized_stores
            .iter()
            .map(|store| store.value_id),
    );
    for operation in &source.operations {
        let (inputs, output) = parts(operation)?;
        ids.extend_from_slice(inputs);
        ids.push(output);
    }
    ids.sort_unstable();
    ids.dedup();
    let mut aliases = Aliases::new(ids);

    for (index, (operation, identity)) in source.operations.iter().zip(canonical).enumerate() {
        let (inputs, output) = parts(operation)?;
        match (operation, identity) {
            (
                FusionOp::Primitive {
                    op: PrimitiveOp::UPlus,
                    ..
                },
                VmFusionHostOperation::Intrinsic(OperatorKind::UnaryPlus),
            ) if inputs.len() == 1 => aliases.union(inputs[0], output)?,
            (
                FusionOp::Primitive {
                    op: PrimitiveOp::Neg,
                    ..
                },
                VmFusionHostOperation::Intrinsic(OperatorKind::UnaryMinus),
            ) if inputs.len() == 1 => {}
            (
                FusionOp::Primitive {
                    op: PrimitiveOp::Mul,
                    ..
                },
                VmFusionHostOperation::Intrinsic(OperatorKind::MatrixMultiply),
            ) if inputs.len() == 2 => {}
            (FusionOp::Builtin { .. }, VmFusionHostOperation::Builtin(binding))
                if named_instruction(index, *binding, inputs.len()).is_some() => {}
            _ => return None,
        }
    }

    let mut producer = vec![None; aliases.len()];
    for (index, operation) in source.operations.iter().enumerate() {
        let (_, output) = parts(operation)?;
        if matches!(
            operation,
            FusionOp::Primitive {
                op: PrimitiveOp::UPlus,
                ..
            }
        ) {
            continue;
        }
        let root = aliases.root_index(output)?;
        if producer[root].replace(index).is_some() {
            return None;
        }
    }

    let mut pinned = vec![false; aliases.len()];
    for root in 0..aliases.len() {
        pinned[root] = producer[root].is_none();
    }
    for id in source.inputs.iter().copied().chain(source.output).chain(
        source
            .materialized_stores
            .iter()
            .map(|store| store.value_id),
    ) {
        pinned[aliases.root_index(id)?] = true;
    }

    let mut remaining = vec![0usize; aliases.len()];
    for operation in &source.operations {
        let (inputs, _) = parts(operation)?;
        for &input in inputs {
            let root = aliases.root_index(input)?;
            remaining[root] = remaining[root].checked_add(1)?;
        }
    }

    let mut occupied = vec![None; aliases.len()];
    let mut slots: Vec<HostRealSlot> = Vec::with_capacity(source.operations.len());
    let mut vacant: Vec<bool> = Vec::with_capacity(source.operations.len());
    let mut operations = Vec::with_capacity(source.operations.len());

    for (index, operation) in source.operations.iter().enumerate() {
        let (inputs, output) = parts(operation)?;
        for &input in inputs {
            let root = aliases.root_index(input)?;
            if producer[root].is_some_and(|producer| producer >= index) {
                return None;
            }
        }

        let output_slot = if matches!(
            operation,
            FusionOp::Primitive {
                op: PrimitiveOp::UPlus,
                ..
            }
        ) {
            None
        } else {
            let key = slot_for(output)?;
            let slot = slots
                .iter()
                .zip(&vacant)
                .position(|(candidate, vacant)| *vacant && *candidate == key)
                .unwrap_or_else(|| {
                    slots.push(key);
                    vacant.push(false);
                    slots.len() - 1
                });
            vacant[slot] = false;
            let root = aliases.root_index(output)?;
            if occupied[root].replace(slot).is_some() {
                return None;
            }
            Some(slot)
        };

        let mut retirements = [None, None];
        for (position, &input) in inputs.iter().enumerate() {
            let root = aliases.root_index(input)?;
            remaining[root] = remaining[root].checked_sub(1)?;
            if remaining[root] == 0 && !pinned[root] {
                if let Some(slot) = occupied[root].take() {
                    // Preserve every occurrence in this operation. The
                    // Runtime batch authenticates that repeated operands
                    // really share one backing before dropping aliases and
                    // checking uniqueness on the survivor.
                    for (candidate, candidate_id) in inputs[..=position].iter().enumerate() {
                        if aliases.root_index(*candidate_id)? == root {
                            retirements[candidate] = Some(HostOperandRetirement {
                                operand_position: candidate,
                                slot,
                            });
                        }
                    }
                    vacant[slot] = true;
                }
            }
        }
        // Unary plus publishes the same Tensor owner it consumed. If that
        // alias dies at this instruction there is no distinct moved operand
        // to return: the result still owns it until the VM subsequently pops
        // or publishes it. Defer that cohort to the later generic VM value
        // retirement hook instead of claiming an impossible slot return.
        if matches!(
            operation,
            FusionOp::Primitive {
                op: PrimitiveOp::UPlus,
                ..
            }
        ) && retirements.iter().any(Option::is_some)
        {
            return None;
        }
        let operation = canonical[index];
        let instruction = match operation {
            VmFusionHostOperation::Intrinsic(operator) => {
                VmFusionHostInstruction::Intrinsic(operator)
            }
            VmFusionHostOperation::Builtin(binding) => {
                named_instruction(index, binding, inputs.len())?
            }
        };
        operations.push(HostOperationLiveness {
            operation,
            instruction,
            output_slot,
            retirements,
        });
    }

    if remaining.iter().any(|remaining| *remaining != 0) {
        return None;
    }
    Some(HostLivenessPlan { slots, operations })
}

fn parts(operation: &FusionOp) -> Option<(&[ValueId], ValueId)> {
    match operation {
        FusionOp::Primitive { inputs, output, .. } | FusionOp::Builtin { inputs, output, .. } => {
            Some((inputs, (*output)?))
        }
    }
}

fn slot_for(
    program: &crate::fusion::semantics::ResolvedProgram,
    id: ValueId,
) -> Option<HostRealSlot> {
    let dtype = match program.floating_class(id).ok()? {
        NumericClass::Single => NumericDType::F32,
        NumericClass::Double => NumericDType::F64,
        _ => return None,
    };
    let known = program.fact(id).ok()?.shape.known_dims()?;
    let mut shape = Vec::with_capacity(known.len());
    for extent in known {
        shape.push(extent?);
    }
    Some(HostRealSlot { dtype, shape })
}
