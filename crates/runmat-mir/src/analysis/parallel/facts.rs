use runmat_types::{
    DynamicReason, ParallelTransferability, ProgramFunctionId, ProgramPointId, RegionValueId,
    ValueFact,
};

use crate::{BasicBlockId, MirOperand, MirRvalue};

use super::super::AnalysisStore;

pub(super) fn value_fact(
    store: &AnalysisStore,
    function: ProgramFunctionId,
    block: BasicBlockId,
    position: usize,
    local: crate::MirLocalId,
) -> ValueFact {
    let point = u32::try_from(block.0)
        .ok()
        .zip(u32::try_from(position).ok())
        .map(|(block, position)| ProgramPointId {
            function,
            block,
            position,
        });
    point
        .and_then(|point| store.facts_at(point))
        .and_then(|facts| {
            facts.local(RegionValueId {
                function,
                local: u32::try_from(local.0).ok()?,
            })
        })
        .and_then(|local| local.fact.clone())
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
}

pub(super) fn rvalue_fact(
    store: &AnalysisStore,
    function: ProgramFunctionId,
    block: BasicBlockId,
    position: usize,
    value: &MirRvalue,
) -> ValueFact {
    match value {
        MirRvalue::Use(MirOperand::Local(local)) => {
            value_fact(store, function, block, position, *local)
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

pub(super) fn operand_fact(
    store: &AnalysisStore,
    function: ProgramFunctionId,
    block: BasicBlockId,
    position: usize,
    value: &MirOperand,
) -> ValueFact {
    match value {
        MirOperand::Local(local) => value_fact(store, function, block, position, *local),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

pub(super) fn transferability(fact: &ValueFact) -> ParallelTransferability {
    ParallelTransferability::for_fact(fact)
}
