//! Exact compiler-selected private temporary use proof for one bytecode stream.

use std::collections::{BTreeMap, HashSet};

use crate::bytecode::instr::VmSlotAccess;
use crate::bytecode::{
    Bytecode, CompiledFusionStageStatement, FusionCandidateGroup, FusionInstructionKind,
    FusionStageProjection, FusionStageSelectionOrigin, Instr,
};
use runmat_mir::analysis::FusionStageFactStatus;
use runmat_types::EffectKind;

use super::FusionPrivateTempBridge;
use crate::bytecode::fusion_effects::elementwise_statement_effects_are_compatible;

#[path = "proof/empty_control.rs"]
mod empty_control;
#[path = "proof/literal_control.rs"]
mod literal_control;

struct SlotAccessIndex(BTreeMap<usize, Vec<(usize, VmSlotAccess)>>);

struct StatementLocalIndex {
    definitions: BTreeMap<u32, Vec<usize>>,
    uses: BTreeMap<u32, Vec<usize>>,
    ordered_ranges: Vec<(usize, usize, usize)>,
    prefix_max_end: Vec<usize>,
}

impl StatementLocalIndex {
    fn from_projection(projection: &FusionStageProjection) -> Self {
        let mut definitions: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        let mut uses: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        let mut ordered_ranges = Vec::with_capacity(projection.statements.len());
        for (index, statement) in projection.statements.iter().enumerate() {
            ordered_ranges.push((statement.pc.start, statement.pc.end, index));
            for value in &statement.contract.defs {
                let entries = definitions.entry(value.local).or_default();
                if entries.last() != Some(&index) {
                    entries.push(index);
                }
            }
            for value in &statement.contract.uses {
                let entries = uses.entry(value.local).or_default();
                if entries.last() != Some(&index) {
                    entries.push(index);
                }
            }
        }
        ordered_ranges.sort_unstable();
        let mut max_end = 0;
        let prefix_max_end = ordered_ranges
            .iter()
            .map(|(_, end, _)| {
                max_end = max_end.max(*end);
                max_end
            })
            .collect();
        Self {
            definitions,
            uses,
            ordered_ranges,
            prefix_max_end,
        }
    }

    fn unique_pair<'a>(
        &self,
        projection: &'a FusionStageProjection,
        local: u32,
    ) -> Option<(
        &'a CompiledFusionStageStatement,
        &'a CompiledFusionStageStatement,
    )> {
        let [definition] = self.definitions.get(&local)?.as_slice() else {
            return None;
        };
        let [use_site] = self.uses.get(&local)?.as_slice() else {
            return None;
        };
        Some((
            projection.statements.get(*definition)?,
            projection.statements.get(*use_site)?,
        ))
    }

    fn overlapping<'a>(
        &'a self,
        projection: &'a FusionStageProjection,
        start: usize,
        end: usize,
    ) -> impl Iterator<Item = &'a CompiledFusionStageStatement> + 'a {
        let upper = self
            .ordered_ranges
            .partition_point(|(from, _, _)| *from <= end);
        let lower = self.prefix_max_end[..upper].partition_point(|to| *to <= start);
        self.ordered_ranges[lower..upper]
            .iter()
            .filter(move |(_, to, _)| *to > start)
            .filter_map(move |(_, _, index)| projection.statements.get(*index))
    }
}

impl SlotAccessIndex {
    fn from_instructions(instructions: &[Instr], projection: &FusionStageProjection) -> Self {
        let private_slots = projection
            .private_temporary_slots
            .iter()
            .map(|pair| pair.slot)
            .collect::<HashSet<_>>();
        let mut by_slot: BTreeMap<usize, Vec<(usize, VmSlotAccess)>> = BTreeMap::new();
        for (pc, instruction) in instructions.iter().enumerate() {
            instruction.visit_vm_slot_accesses(|slot, access| {
                if private_slots.contains(&slot) {
                    by_slot.entry(slot).or_default().push((pc, access));
                }
            });
        }
        Self(by_slot)
    }

    fn unique_store_load(&self, slot: usize) -> Option<(usize, usize)> {
        let [(store, VmSlotAccess::StoreVar), (load, VmSlotAccess::LoadVar)] =
            self.0.get(&slot)?.as_slice()
        else {
            return None;
        };
        Some((*store, *load))
    }
}

struct BridgeProof<'a> {
    bytecode: &'a Bytecode,
    projection: &'a FusionStageProjection,
    accesses: SlotAccessIndex,
    statements: StatementLocalIndex,
    control_targets: HashSet<usize>,
}

pub(super) fn bridges(bytecode: &Bytecode) -> Vec<FusionPrivateTempBridge> {
    let Some(proof) = validated_proof(bytecode) else {
        return Vec::new();
    };
    let mut result = Vec::new();
    for candidate in &bytecode.fusion_metadata.mir_fusion_candidate_groups {
        for pair in &proof.projection.private_temporary_slots {
            if private_pair_owned(&proof, candidate, pair) {
                if let Some(bridge) = proof.candidate_pair(candidate, pair) {
                    result.push(bridge);
                }
            }
        }
    }
    result.sort_by_key(|bridge| bridge.store_pc);
    result.dedup();
    result
}

pub(super) fn empty_controls(bytecode: &Bytecode) -> Vec<super::FusionPrivateEmptyControl> {
    let Some(proof) = validated_proof(bytecode) else {
        return Vec::new();
    };
    let mut result = Vec::new();
    for candidate in &bytecode.fusion_metadata.mir_fusion_candidate_groups {
        for pair in &proof.projection.private_temporary_slots {
            if private_pair_owned(&proof, candidate, pair) {
                if let Some(control) =
                    empty_control::candidate_empty_control(&proof, candidate, pair)
                {
                    result.push(control);
                }
            }
        }
    }
    result.sort_by_key(|control| control.store_pc);
    result.dedup();
    result
}

pub(super) fn literal_controls(bytecode: &Bytecode) -> Vec<super::FusionPrivateLiteralControl> {
    let Some(proof) = validated_proof(bytecode) else {
        return Vec::new();
    };
    let mut result = Vec::new();
    for pair in &proof.projection.private_temporary_slots {
        if private_pair_owned_by_layout(&proof, None, pair) {
            if let Some(control) = literal_control::sealed_literal_control(&proof, pair) {
                result.push(control);
            }
        }
    }
    result.sort_by_key(|control| control.store_pc);
    result.dedup();
    result
}

fn validated_proof(bytecode: &Bytecode) -> Option<BridgeProof<'_>> {
    if bytecode.validate_fusion_stage_projection().is_err()
        || bytecode.catch_scopes.len() != bytecode.instructions.len()
        || bytecode.instr_spans.len() != bytecode.instructions.len()
    {
        return None;
    }
    let projection = bytecode.fusion_metadata.stage_projection.as_ref()?;
    projection.selection_origin?;
    Some(BridgeProof {
        bytecode,
        projection,
        accesses: SlotAccessIndex::from_instructions(&bytecode.instructions, projection),
        statements: StatementLocalIndex::from_projection(projection),
        control_targets: super::super::multi_store::interior_control_targets(
            &bytecode.instructions,
        ),
    })
}

fn private_pair_owned(
    proof: &BridgeProof<'_>,
    candidate: &FusionCandidateGroup,
    pair: &crate::bytecode::fusion_stage::CompiledPrivateTemporarySlot,
) -> bool {
    private_pair_owned_by_layout(proof, Some(candidate.function), pair)
}

fn private_pair_owned_by_layout(
    proof: &BridgeProof<'_>,
    function_id: Option<runmat_hir::FunctionId>,
    pair: &crate::bytecode::fusion_stage::CompiledPrivateTemporarySlot,
) -> bool {
    let bytecode = proof.bytecode;
    let projection = proof.projection;
    match projection.selection_origin {
        Some(FusionStageSelectionOrigin::EntrypointStream) => {
            bytecode.layout.as_ref().is_some_and(|layout| {
                layout
                    .functions
                    .values()
                    .filter(|function| {
                        function_id.is_none_or(|expected| function.function == expected)
                            && function.resume_points == bytecode.resume_points
                            && function
                                .resume_points
                                .keys()
                                .all(|point| point.function == projection.owner)
                            && function
                                .mir_local_slots
                                .get(&runmat_mir::MirLocalId(pair.local as usize))
                                == Some(&crate::layout::VmSlotId(pair.slot))
                            && !function
                                .binding_slots
                                .values()
                                .any(|bound| bound.0 == pair.slot)
                    })
                    .count()
                    == 1
            })
        }
        Some(FusionStageSelectionOrigin::FunctionBody) => bytecode
            .function_private_layout
            .as_ref()
            .is_some_and(|layout| {
                bytecode.active_function.is_some_and(|active| {
                    function_id.is_none_or(|expected| active == expected)
                        && layout.matches_candidate(
                            active,
                            &bytecode.resume_points,
                            projection.portable_origin,
                        )
                        && layout.owns_private_pair(*pair)
                })
            }),
        None => false,
    }
}

impl BridgeProof<'_> {
    fn candidate_pair(
        &self,
        candidate: &FusionCandidateGroup,
        pair: &crate::bytecode::fusion_stage::CompiledPrivateTemporarySlot,
    ) -> Option<FusionPrivateTempBridge> {
        let (store_pc, load_pc) = self.accesses.unique_store_load(pair.slot)?;
        let producer_pc = store_pc.checked_sub(1)?;
        let instructions = &self.bytecode.instructions;
        if load_pc <= store_pc || load_pc >= instructions.len() {
            return None;
        }
        let candidate_block = u32::try_from(candidate.block.0).ok()?;
        let (definition, use_site) = self.statements.unique_pair(self.projection, pair.local)?;
        let (unique_producer_pc, producer_kind) =
            unique_statement_signal(instructions, definition)?;
        let (consumer_pc, consumer_kind) = unique_statement_signal(instructions, use_site)?;
        if unique_producer_pc != producer_pc
            || !private_elementwise_signal(instructions, producer_pc, producer_kind)
            || consumer_pc <= load_pc
            || !(consumer_kind == FusionInstructionKind::Reduction
                || private_elementwise_signal(instructions, consumer_pc, consumer_kind))
            || self.bytecode.catch_scopes[producer_pc..=consumer_pc]
                .iter()
                .any(|scope| scope != &self.bytecode.catch_scopes[producer_pc])
            || (producer_pc + 1..=consumer_pc).any(|pc| self.control_targets.contains(&pc))
            || !self.bytecode.instr_spans[producer_pc..=consumer_pc]
                .iter()
                .all(|span| {
                    candidate.source_span.start <= span.start
                        && span.end <= candidate.source_span.end
                })
        {
            return None;
        }
        let terminal_reduction = consumer_kind == FusionInstructionKind::Reduction
            && usize::try_from(use_site.contract.point.position)
                .ok()
                .and_then(|position| position.checked_add(1))
                == Some(candidate.stmt_end);
        if !selected_statement(candidate, candidate_block, definition, store_pc)
            || !selected_statement(candidate, candidate_block, use_site, load_pc)
            || !elementwise_statement_effects_are_compatible(instructions, definition)
            || !(if terminal_reduction {
                terminal_reduction_effects_are_compatible(use_site)
            } else {
                private_elementwise_signal(instructions, consumer_pc, consumer_kind)
                    && elementwise_statement_effects_are_compatible(instructions, use_site)
            })
            || !self
                .statements
                .overlapping(self.projection, producer_pc, consumer_pc)
                .all(|statement| {
                    selected_statement(candidate, candidate_block, statement, statement.pc.start)
                        && (statement.pc.start == use_site.pc.start && terminal_reduction
                            || elementwise_statement_effects_are_compatible(
                                instructions,
                                statement,
                            ))
                })
            || !definition
                .contract
                .live_after
                .iter()
                .any(|value| value.local == pair.local)
            || use_site
                .contract
                .live_after
                .iter()
                .any(|value| value.local == pair.local)
        {
            return None;
        }
        Some(FusionPrivateTempBridge {
            slot: pair.slot,
            store_pc,
            load_pc,
            producer_pc,
            consumer_pc,
        })
    }
}

// `Mul` is structurally Matmul until the authoritative graph proves a scalar
// operand. This only permits an edge proposal; closure never selects a
// Matmul group, and graph binding projection still checks the exact edge.
fn private_elementwise_signal(
    instructions: &[Instr],
    pc: usize,
    kind: FusionInstructionKind,
) -> bool {
    kind == FusionInstructionKind::Elementwise
        || (kind == FusionInstructionKind::Matmul
            && matches!(instructions.get(pc), Some(Instr::Mul)))
}

fn unique_statement_signal(
    instructions: &[Instr],
    statement: &CompiledFusionStageStatement,
) -> Option<(usize, FusionInstructionKind)> {
    let mut signals = instructions
        .get(statement.pc.start..statement.pc.end)?
        .iter()
        .enumerate()
        .filter_map(|(offset, instruction)| {
            super::super::compile::instr_fusion_signal_kind(instruction)
                .map(|kind| (statement.pc.start + offset, kind))
        });
    let signal = signals.next()?;
    signals.next().is_none().then_some(signal)
}

fn selected_statement(
    candidate: &FusionCandidateGroup,
    candidate_block: u32,
    statement: &CompiledFusionStageStatement,
    pc: usize,
) -> bool {
    statement.contract.point.block == candidate_block
        && usize::try_from(statement.contract.point.position)
            .is_ok_and(|position| candidate.stmt_start <= position && position < candidate.stmt_end)
        && statement.pc.start <= pc
        && pc < statement.pc.end
        && matches!(
            statement.contract.status,
            FusionStageFactStatus::Inferred | FusionStageFactStatus::UnknownInputs
        )
}

/// The private producer is pure. The terminal, separately authenticated
/// reduction call may throw or suspend through the ordinary VM boundary; no
/// workspace/environment mutation or unknown effect is omitted by fusion.
fn terminal_reduction_effects_are_compatible(statement: &CompiledFusionStageStatement) -> bool {
    statement
        .contract
        .effects
        .0
        .iter()
        .all(|effect| matches!(effect, EffectKind::MayThrow | EffectKind::MaySuspend))
}
