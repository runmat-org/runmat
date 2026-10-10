//! Image-only statement/store reconciliation from the sealed selected stream.
//! Generic consecutive-signal MIR hints do not span nested mean/sigma source
//! statements, so they cannot be promoted into fabricated image candidates.

use std::ops::Range;

use runmat_accelerate::fusion::{
    image_reduction::publication_follows_stage, FusionGroup, FusionPattern,
};
use runmat_accelerate::graph::{VarBinding, VarKind};
use runmat_mir::analysis::FusionStageFactStatus;
use runmat_types::{EffectKind, RegionValueId};

use super::super::{
    interior_control_targets, is_structural_region_instruction, ProspectiveStore,
    SelectedMultiStoreGraph,
};
use super::{ImageLiveOut, ImageLiveOutSource, ImageStoreOrdinal};
use crate::bytecode::{Bytecode, CompiledFusionStageStatement, FusionStageProjection, Instr};
use crate::layout::VmSlotId;

#[derive(Clone)]
pub(super) struct SelectedRegionStore<'a> {
    pub store: ProspectiveStore,
    pub stage: &'a CompiledFusionStageStatement,
    pub slot: VmSlotId,
    pub defs: Vec<RegionValueId>,
    pub named: bool,
}

#[derive(Clone)]
pub(super) struct ImageStatementRegion<'a> {
    pub pcs: Range<usize>,
    pub stores: Vec<SelectedRegionStore<'a>>,
    pub live_outs: Vec<ImageLiveOut>,
}

pub(super) fn selected_image_region<'a>(
    bytecode: &Bytecode,
    projection: &'a FusionStageProjection,
    selected: &SelectedMultiStoreGraph<'_>,
    group: &FusionGroup,
) -> Option<ImageStatementRegion<'a>> {
    let function = selected.selected_function()?;
    let layout = bytecode.layout.as_ref()?.functions.get(&function)?;
    if layout.function != function || layout.resume_points != bytecode.resume_points {
        return None;
    }
    // The final arithmetic node can precede its MIR `Use`/named-store
    // statement. Extend only through exact source publications in the same
    // bytecode graph, not through a guessed number of later statements.
    let FusionPattern::ImageNormalize(pattern) = group.pattern.as_ref()? else {
        return None;
    };
    let producers = pattern.reduction_routes.as_ref()?.stage_producers()?;
    let first = projection
        .statements
        .iter()
        .position(|stage| stage.pc.start <= group.span.start && group.span.start < stage.pc.end)?;
    let source_start = projection.statements[first].pc.start;
    let mut sources = vec![producers.mu, producers.sigma];
    sources.extend(producers.pre_clamp_output);
    sources.extend(producers.clamp_output);
    sources.push(producers.output);
    let mut next_store_pc = group.span.start;
    let mut last_store_pc = None;
    for source in sources {
        let store = selected.stores.iter().find(|store| {
            store.pc >= next_store_pc
                && layout
                    .binding_slots
                    .values()
                    .any(|slot| slot.0 == store.binding.index)
                && projection.statements.iter().any(|stage| {
                    stage.pc.start <= store.pc
                        && store.pc < stage.pc.end
                        && publication_follows_stage(
                            &selected.graph,
                            source,
                            store.value_id,
                            source_start..stage.pc.end,
                        )
                })
        });
        let store = store?;
        last_store_pc = Some(store.pc);
        next_store_pc = store.pc.checked_add(1)?;
    }
    let last_pc = group.span.end.max(last_store_pc?);
    let last = projection
        .statements
        .iter()
        .position(|stage| stage.pc.start <= last_pc && last_pc < stage.pc.end)?;
    let stages = projection.statements.get(first..=last)?;
    let pcs = stages.first()?.pc.start..stages.last()?.pc.end;
    if pcs.start >= pcs.end
        || group.span.start < pcs.start
        || group.span.end >= pcs.end
        || interior_control_targets(&bytecode.instructions)
            .iter()
            .any(|target| pcs.start < *target && *target < pcs.end)
    {
        return None;
    }
    let owner = stages[0].contract.point.function;
    let block = stages[0].contract.point.block;
    let scope = *bytecode.catch_scopes.get(pcs.start)?;
    if scope.is_some() {
        return None;
    }
    let mut next_pc = pcs.start;
    for (offset, stage) in stages.iter().enumerate() {
        if stage.pc.start != next_pc
            || stage.pc.statement != stage.contract.point
            || stage.contract.point.function != owner
            || stage.contract.point.block != block
            || stage.contract.point.position
                != stages[0]
                    .contract
                    .point
                    .position
                    .checked_add(u32::try_from(offset).ok()?)?
            // UnknownInputs means this sealed statement has complete PC,
            // defs, liveness, and known effects, but its numeric inputs must
            // be resolved at invocation time. This remains a non-executable
            // publication proposal; no arithmetic proof is minted here.
            || !matches!(
                stage.contract.status,
                FusionStageFactStatus::Inferred | FusionStageFactStatus::UnknownInputs
            )
            || stage
                .contract
                .uses
                .iter()
                .chain(&stage.contract.defs)
                .chain(&stage.contract.live_after)
                .any(|value| {
                    !layout
                        .mir_local_slots
                        .contains_key(&runmat_mir::MirLocalId(value.local as usize))
                })
            || !stage
                .contract
                .effects
                .0
                .iter()
                .all(|effect| matches!(effect, EffectKind::MayThrow | EffectKind::MaySuspend))
        {
            return None;
        }
        next_pc = stage.pc.end;
    }
    if next_pc != pcs.end {
        return None;
    }
    let mut depth = 0usize;
    for pc in pcs.clone() {
        let instruction = bytecode.instructions.get(pc)?;
        if bytecode.catch_scopes.get(pc) != Some(&scope)
            || !is_structural_region_instruction(instruction)
        {
            return None;
        }
        let effect = instruction.stack_effect()?;
        let next_depth = depth
            .checked_sub(effect.pops)
            .and_then(|depth| depth.checked_add(effect.pushes));
        depth = next_depth?;
    }
    if depth != 0 {
        return None;
    }
    let visible_slots: std::collections::HashSet<_> =
        layout.binding_slots.values().map(|slot| slot.0).collect();
    let mut transcript = selected
        .stores
        .iter()
        .filter(|store| pcs.start <= store.pc && store.pc < pcs.end);
    let mut stores = Vec::new();
    for pc in pcs.clone() {
        let binding = match bytecode.instructions[pc] {
            Instr::StoreVar(index) => Some(VarBinding {
                kind: VarKind::Global,
                index,
            }),
            Instr::StoreLocal(index) => Some(VarBinding {
                kind: VarKind::Local,
                index,
            }),
            _ => None,
        };
        let Some(binding) = binding else { continue };
        let store = transcript.next()?;
        if store.pc != pc
            || store.binding != binding
            || selected.graph.value(store.value_id).is_none()
        {
            return None;
        }
        let named = visible_slots.contains(&binding.index);
        // Both StoreLocal and StoreVar can target a compiler MIR temporary.
        // Only the selected layout's exact MIR def-to-slot mapping below can
        // classify an unbound slot as internal; opcode spelling is not owner
        // authority. A source-named workspace slot must never be demoted to
        // an intermediate by a missing layout binding.
        if !named && bytecode.var_names.contains_key(&binding.index) {
            return None;
        }
        let mut containing = stages
            .iter()
            .filter(|stage| stage.pc.start <= pc && pc < stage.pc.end);
        let stage = containing.next()?;
        let defs = stage
            .contract
            .defs
            .iter()
            .copied()
            .filter(|defined| {
                layout
                    .mir_local_slots
                    .get(&runmat_mir::MirLocalId(defined.local as usize))
                    .is_some_and(|slot| slot.0 == binding.index)
            })
            .collect::<Vec<_>>();
        if containing.next().is_some() || defs.is_empty() {
            return None;
        }
        let slot = VmSlotId(binding.index);
        stores.push(SelectedRegionStore {
            store: ProspectiveStore {
                pc,
                value_id: store.value_id,
                binding,
            },
            stage,
            slot,
            defs,
            named,
        });
    }
    if transcript.next().is_some() || stores.iter().filter(|store| store.named).count() < 3 {
        return None;
    }
    let mut live_outs = Vec::new();
    for value in &stages.last()?.contract.live_after {
        let slot = *layout
            .mir_local_slots
            .get(&runmat_mir::MirLocalId(value.local as usize))?;
        let last_store = stores.iter().rposition(|store| store.slot == slot);
        let defined_inside = stages
            .iter()
            .any(|stage| stage.contract.defs.contains(value));
        let source = match last_store {
            Some(index) if stores[index].defs.contains(value) => {
                ImageLiveOutSource::LastStore(ImageStoreOrdinal(index))
            }
            Some(_) => return None,
            None if !defined_inside => ImageLiveOutSource::PreserveEntry,
            None => return None,
        };
        live_outs.push(ImageLiveOut {
            value: *value,
            slot,
            source,
        });
    }
    Some(ImageStatementRegion {
        pcs,
        stores,
        live_outs,
    })
}
