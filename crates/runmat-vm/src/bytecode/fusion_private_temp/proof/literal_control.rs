//! A sealed compiler-private single use of an immutable numeric control.
//! The graph checks the literal bytes and exact consumer separately. This
//! source certificate must precede image recognition, so it cannot depend on
//! a fusion candidate that recognition has not produced yet.

use super::{
    elementwise_statement_effects_are_compatible, terminal_reduction_effects_are_compatible,
    BridgeProof,
};
use runmat_mir::analysis::FusionStageFactStatus;

pub(super) fn sealed_literal_control(
    proof: &BridgeProof<'_>,
    pair: &crate::bytecode::fusion_stage::CompiledPrivateTemporarySlot,
) -> Option<super::super::FusionPrivateLiteralControl> {
    let (store_pc, load_pc) = proof.accesses.unique_store_load(pair.slot)?;
    let producer_pc = store_pc.checked_sub(1)?;
    let (definition, use_site) = proof.statements.unique_pair(proof.projection, pair.local)?;
    let consumer_pc = use_site.pc.end.checked_sub(1)?;
    if load_pc <= store_pc
        || producer_pc < definition.pc.start
        || store_pc >= definition.pc.end
        || load_pc < use_site.pc.start
        || load_pc >= use_site.pc.end
        || consumer_pc <= load_pc
        || definition.contract.point.function != proof.projection.owner
        || use_site.contract.point.function != proof.projection.owner
        || definition.contract.point.block != use_site.contract.point.block
        || definition.contract.status != FusionStageFactStatus::Inferred
        || !matches!(
            use_site.contract.status,
            FusionStageFactStatus::Inferred | FusionStageFactStatus::UnknownInputs
        )
        || !elementwise_statement_effects_are_compatible(&proof.bytecode.instructions, definition)
        || !terminal_reduction_effects_are_compatible(use_site)
        || !definition
            .contract
            .live_after
            .iter()
            .any(|value| value.local == pair.local)
        || !use_site
            .contract
            .uses
            .iter()
            .any(|value| value.local == pair.local)
        || use_site
            .contract
            .live_after
            .iter()
            .any(|value| value.local == pair.local)
        || proof.bytecode.catch_scopes[producer_pc..=consumer_pc]
            .iter()
            .any(|scope| scope != &proof.bytecode.catch_scopes[producer_pc])
        || (producer_pc + 1..=consumer_pc).any(|pc| proof.control_targets.contains(&pc))
        || !proof
            .statements
            .overlapping(proof.projection, producer_pc, consumer_pc)
            .all(|statement| {
                statement.contract.point.function == proof.projection.owner
                    && statement.contract.point.block == definition.contract.point.block
                    && (statement.pc.start == use_site.pc.start
                        || elementwise_statement_effects_are_compatible(
                            &proof.bytecode.instructions,
                            statement,
                        ))
            })
    {
        return None;
    }
    Some(super::super::FusionPrivateLiteralControl {
        slot: pair.slot,
        store_pc,
        load_pc,
        producer_pc,
        consumer_pc,
        projected_value: None,
    })
}
