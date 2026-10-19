//! Selected MIR statement facts joined to the existing structural store bridge.
//! This proposal still cannot authorize provider execution or publication.

use runmat_accelerate::graph::VarKind;
use runmat_mir::analysis::FusionStageFactStatus;
use runmat_types::{EffectKind, ProgramPointId};

use super::ProspectiveMultiStoreRegion;
use crate::bytecode::{Bytecode, CompiledFusionStageStatement, FusionStageProjection};

pub struct ProspectiveSemanticMultiStoreRegion<'a> {
    pub structural: ProspectiveMultiStoreRegion,
    pub stages: Vec<&'a CompiledFusionStageStatement>,
}

impl Bytecode {
    pub fn prospective_semantic_multi_store_regions(
        &self,
    ) -> Vec<ProspectiveSemanticMultiStoreRegion<'_>> {
        let Some(projection) = self.fusion_metadata.stage_projection.as_ref() else {
            return Vec::new();
        };
        if self.validate_fusion_stage_projection().is_err() {
            return Vec::new();
        }
        self.prospective_multi_store_regions()
            .into_iter()
            .filter_map(|structural| semantic_region(self, projection, structural))
            .collect()
    }
}

pub(super) fn semantic_region<'a>(
    bytecode: &Bytecode,
    projection: &'a FusionStageProjection,
    structural: ProspectiveMultiStoreRegion,
) -> Option<ProspectiveSemanticMultiStoreRegion<'a>> {
    let mut candidates = bytecode
        .fusion_metadata
        .mir_fusion_candidate_groups
        .iter()
        .filter(|candidate| candidate.id == structural.mir_candidate_id);
    let candidate = candidates.next()?;
    if candidates.next().is_some()
        || u32::try_from(candidate.stmt_start).ok()? != structural.first_statement.position
        || u32::try_from(candidate.stmt_end.checked_sub(1)?).ok()?
            != structural.last_statement.position
    {
        return None;
    }
    let layout = bytecode
        .layout
        .as_ref()?
        .functions
        .get(&candidate.function)?;
    if layout.function != candidate.function || layout.resume_points != bytecode.resume_points {
        return None;
    }
    let selected_local = |local: u32| {
        let local = runmat_mir::MirLocalId(usize::try_from(local).ok()?);
        layout.mir_local_slots.get(&local).copied()
    };
    let mut stages = Vec::with_capacity(candidate.stmt_end - candidate.stmt_start);
    let mut next_pc = structural.pcs.start;
    for position in candidate.stmt_start..candidate.stmt_end {
        let point = ProgramPointId {
            function: structural.first_statement.function,
            block: structural.first_statement.block,
            position: u32::try_from(position).ok()?,
        };
        let stage = projection.statement_at(point)?;
        if stage.pc.start != next_pc
            || stage.contract.status != FusionStageFactStatus::Inferred
            || stage
                .contract
                .uses
                .iter()
                .chain(&stage.contract.defs)
                .chain(&stage.contract.live_after)
                .any(|value| selected_local(value.local).is_none())
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
        stages.push(stage);
    }
    if next_pc != structural.pcs.end
        || structural.stores.iter().any(|store| {
            let owner_slot = match store.binding.kind {
                VarKind::Global => layout
                    .binding_slots
                    .values()
                    .any(|slot| slot.0 == store.binding.index),
                VarKind::Local => layout
                    .mir_local_slots
                    .values()
                    .any(|slot| slot.0 == store.binding.index),
            };
            if !owner_slot {
                return true;
            }
            let mut containing = stages
                .iter()
                .filter(|stage| stage.pc.start <= store.pc && store.pc < stage.pc.end);
            let Some(stage) = containing.next() else {
                return true;
            };
            containing.next().is_some()
                || !stage.contract.defs.iter().any(|defined| {
                    selected_local(defined.local).is_some_and(|slot| slot.0 == store.binding.index)
                })
        })
    {
        return None;
    }
    Some(ProspectiveSemanticMultiStoreRegion { structural, stages })
}

#[cfg(test)]
#[path = "semantic/tests.rs"]
mod tests;
