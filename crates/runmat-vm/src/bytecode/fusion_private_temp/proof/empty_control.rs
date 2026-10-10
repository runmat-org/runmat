//! Exact sealed empty-tensor reduction control proof. This is distinct from
//! a shader producer edge: the selected [] value is consumed only as a
//! nondata terminal reduction control.

use super::{
    selected_statement, terminal_reduction_effects_are_compatible, unique_statement_signal,
    BridgeProof,
};
use crate::bytecode::{FusionCandidateGroup, FusionInstructionKind, Instr};
use runmat_mir::analysis::FusionStageFactStatus;

pub(super) fn candidate_empty_control(
    proof: &BridgeProof<'_>,
    candidate: &FusionCandidateGroup,
    pair: &crate::bytecode::fusion_stage::CompiledPrivateTemporarySlot,
) -> Option<super::super::FusionPrivateEmptyControl> {
    let (store_pc, load_pc) = proof.accesses.unique_store_load(pair.slot)?;
    let producer_pc = store_pc.checked_sub(1)?;
    let instructions = &proof.bytecode.instructions;
    let candidate_block = u32::try_from(candidate.block.0).ok()?;
    let (definition, use_site) = proof.statements.unique_pair(proof.projection, pair.local)?;
    let [defined] = definition.contract.defs.as_slice() else {
        return None;
    };
    let [Instr::CreateMatrix(0, 0), Instr::StoreVar(slot)] =
        instructions.get(definition.pc.start..definition.pc.end)?
    else {
        return None;
    };
    let (consumer_pc, consumer_kind) = unique_statement_signal(instructions, use_site)?;
    if *slot != pair.slot
        || producer_pc != definition.pc.start
        || load_pc <= store_pc
        || consumer_pc <= load_pc
        || consumer_kind != FusionInstructionKind::Reduction
        || usize::try_from(use_site.contract.point.position)
            .ok()
            .and_then(|position| position.checked_add(1))
            != Some(candidate.stmt_end)
        || !selected_statement(candidate, candidate_block, definition, store_pc)
        || !selected_statement(candidate, candidate_block, use_site, load_pc)
        || definition.contract.status != FusionStageFactStatus::Inferred
        || !definition.contract.uses.is_empty()
        || !definition.contract.effects.0.is_empty()
        || !terminal_reduction_effects_are_compatible(use_site)
        || !definition.contract.live_after.contains(defined)
        || !use_site.contract.uses.contains(defined)
        || use_site.contract.live_after.contains(defined)
        || proof.bytecode.catch_scopes[producer_pc..=consumer_pc]
            .iter()
            .any(|scope| scope != &proof.bytecode.catch_scopes[producer_pc])
        || (producer_pc + 1..=consumer_pc).any(|pc| proof.control_targets.contains(&pc))
        || !proof.bytecode.instr_spans[producer_pc..=consumer_pc]
            .iter()
            .all(|span| {
                candidate.source_span.start <= span.start && span.end <= candidate.source_span.end
            })
        || !proof
            .statements
            .overlapping(proof.projection, producer_pc, consumer_pc)
            .all(|statement| {
                selected_statement(candidate, candidate_block, statement, statement.pc.start)
                    && (statement.pc.start == use_site.pc.start
                        || statement.contract.effects.0.is_empty())
            })
    {
        return None;
    }
    Some(super::super::FusionPrivateEmptyControl {
        slot: pair.slot,
        store_pc,
        producer_pc,
        consumer_pc,
    })
}
