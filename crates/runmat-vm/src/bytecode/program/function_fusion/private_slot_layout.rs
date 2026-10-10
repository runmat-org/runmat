//! Compiler-owned function frame slice for a private MIR temporary bridge.
//! Ordered records keep portable definition identity independent of map order.

use std::collections::BTreeMap;

use runmat_types::{ProgramFunctionId, ProgramPointId};
use serde::{Deserialize, Serialize};

use crate::bytecode::{fusion_stage::CompiledPrivateTemporarySlot, FunctionBytecode};
use crate::layout::VmFunctionLayout;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FunctionPrivateSlotLayout {
    function: runmat_hir::FunctionId,
    portable_origin: ProgramFunctionId,
    local_count: usize,
    local_slots: Vec<CompiledPrivateTemporarySlot>,
    bound_slots: Vec<usize>,
    resume_points: Vec<(ProgramPointId, usize)>,
}

impl FunctionPrivateSlotLayout {
    pub(crate) fn from_compiler(
        layout: &VmFunctionLayout,
        resume_points: &BTreeMap<ProgramPointId, usize>,
        portable_origin: ProgramFunctionId,
    ) -> Option<Self> {
        let mut local_slots = layout
            .mir_local_slots
            .iter()
            .map(|(local, slot)| {
                Some(CompiledPrivateTemporarySlot {
                    local: u32::try_from(local.0).ok()?,
                    slot: slot.0,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        local_slots.sort_unstable_by_key(|pair| pair.local);
        let mut bound_slots = layout
            .binding_slots
            .values()
            .map(|slot| slot.0)
            .collect::<Vec<_>>();
        bound_slots.sort_unstable();
        bound_slots.dedup();
        Some(Self {
            function: layout.function,
            portable_origin,
            local_count: layout.local_count,
            local_slots,
            bound_slots,
            resume_points: resume_points
                .iter()
                .map(|(point, pc)| (*point, *pc))
                .collect(),
        })
    }

    pub(crate) fn rebind(&mut self, function: runmat_hir::FunctionId, owner: ProgramFunctionId) {
        self.function = function;
        for (point, _) in &mut self.resume_points {
            point.function = owner;
        }
    }

    pub(crate) fn matches_candidate(
        &self,
        function: runmat_hir::FunctionId,
        resume_points: &BTreeMap<ProgramPointId, usize>,
        portable_origin: ProgramFunctionId,
    ) -> bool {
        self.function == function
            && self.portable_origin == portable_origin
            && self
                .resume_points
                .iter()
                .copied()
                .eq(resume_points.iter().map(|(point, pc)| (*point, *pc)))
    }

    pub(crate) fn owns_private_pair(&self, pair: CompiledPrivateTemporarySlot) -> bool {
        self.local_slots
            .binary_search_by_key(&pair.local, |row| row.local)
            .is_ok_and(|index| self.local_slots[index] == pair)
            && self.bound_slots.binary_search(&pair.slot).is_err()
    }

    pub(super) fn valid_for(&self, body: &FunctionBytecode) -> bool {
        let Some(projection) = body.fusion_stage_projection.as_ref() else {
            return false;
        };
        projection.selection_origin
            == Some(crate::bytecode::FusionStageSelectionOrigin::FunctionBody)
            && self.matches_candidate(
                body.function,
                &body.resume_points,
                projection.portable_origin,
            )
            // The compiler may allocate additional scratch slots after layout.
            && self.local_count <= body.var_count
            && self.local_slots.len() == projection.mir_local_count
            && self
                .local_slots
                .windows(2)
                .all(|pair| pair[0].local < pair[1].local)
            && self
                .local_slots
                .iter()
                .all(|pair| pair.slot < self.local_count)
            && self.bound_slots.windows(2).all(|pair| pair[0] < pair[1])
            && self.bound_slots.iter().all(|slot| *slot < self.local_count)
            // Every compiler-visible binding, including non-ABI locals, is
            // named in the FunctionBytecode projection. No omitted binding
            // may be relabeled as a private MIR temporary.
            && self.bound_slots.len() == body.var_names.len()
            && self
                .bound_slots
                .iter()
                .all(|slot| body.var_names.contains_key(slot))
            && projection.private_temporary_slots.iter().all(|pair| {
                self.owns_private_pair(*pair) && !body.var_names.contains_key(&pair.slot)
            })
            && body
                .input_slots
                .iter()
                .chain(body.output_slots.iter())
                .chain(body.capture_slots.iter())
                .chain(body.varargin_slot.iter())
                .chain(body.varargout_slot.iter())
                .chain(body.implicit_nargin_slot.iter())
                .chain(body.implicit_nargout_slot.iter())
                .all(|slot| self.bound_slots.binary_search(slot).is_ok())
    }
}

#[cfg(test)]
#[path = "private_slot_layout/tests.rs"]
mod tests;
