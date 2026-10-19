//! Function-owned fusion optimization hints. Admission binds them to the
//! selected sealed statement-PC projection and exact executable stream once
//! per immutable registry snapshot; serialized counts alone authorize nothing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use runmat_mir::analysis::FusionStageFactStatus;
use runmat_types::{EffectKind, ProgramPointId, RegionValueId};
use serde::{Deserialize, Serialize};

use super::{FunctionBytecode, FunctionRegistry, FusionCandidateGroup, FusionInstructionWindow};
use crate::bytecode::{
    compile::{
        derive_semantic_fusion_instruction_windows, fusion_windows::FusionWindowGroup,
        instr_fusion_signal_kind,
    },
    FusionInstructionKind, Instr,
};

/// Only a catalog-declared pure, uniquely bound, arity-compatible reduction
/// may carry the terminal call's ordinary throw/suspend effects. The generic
/// fusion signal fallback for old tags is not an effect authority.
fn canonical_terminal_reduction(instruction: &Instr) -> bool {
    let Instr::CallBuiltinMulti(name, arguments, outputs) = instruction else {
        return false;
    };
    *outputs == 1
        && runmat_builtins::builtin_catalog_entry_by_name(name).is_some_and(|entry| {
            crate::accel::fusion_declaration::canonical_kind(entry, Some(*arguments))
                == Some(FusionInstructionKind::Reduction)
        })
}

#[path = "function_fusion/private_slot_layout.rs"]
mod private_slot_layout;
pub(crate) use private_slot_layout::FunctionPrivateSlotLayout;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionFusionScaffold {
    pub signal_count: usize,
    pub candidates: Vec<FusionCandidateGroup>,
    pub windows: Vec<FusionInstructionWindow>,
    /// Absent in old artifacts: function-local private-store coalescing is
    /// unavailable, while the established function fusion hint still works.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) private_layout: Option<FunctionPrivateSlotLayout>,
}

impl FunctionFusionScaffold {
    /// Structural check against the actual instructions, not a MIR proof.
    pub(crate) fn valid_for(
        &self,
        function: runmat_hir::FunctionId,
        instructions: &[Instr],
        instr_spans: &[runmat_hir::Span],
    ) -> bool {
        let Some(candidate_signals) = self.candidates.iter().try_fold(0usize, |sum, candidate| {
            sum.checked_add(candidate.signal_count)
        }) else {
            return false;
        };
        self.signal_count >= candidate_signals
            && !self.candidates.is_empty()
            && !self.windows.is_empty()
            && instr_spans.len() == instructions.len()
            && self.candidates.iter().enumerate().all(|(id, candidate)| {
                candidate.id == id
                    && candidate.function == function
                    && candidate.signal_count >= 2
                    && candidate.stmt_start < candidate.stmt_end
            })
            && self.windows.iter().all(|window| {
                window.span.start <= window.span.end && window.span.end < instructions.len()
            })
            && self
                .windows
                .windows(2)
                .all(|pair| pair[0].span.end < pair[1].span.start)
    }

    /// MIR source spans are a useful initial hint, but only the selected
    /// statement projection can identify which emitted signal PCs belong to
    /// a function candidate after constructor/default lowering.
    pub(crate) fn source_windows_match(
        &self,
        instructions: &[Instr],
        instr_spans: &[runmat_hir::Span],
    ) -> bool {
        let mut derived =
            derive_semantic_fusion_instruction_windows(instructions, instr_spans, &self.candidates);
        derived.sort_by_key(|window| (window.span.start, window.span.end));
        derived == self.windows
    }
}

impl FunctionBytecode {
    pub(crate) fn validate_fusion_scaffold(&self) -> Result<(), String> {
        let Some(scaffold) = &self.fusion_scaffold else {
            return Ok(());
        };
        self.validate_fusion_stage_projection()?;
        let index = FunctionFusionCertificateIndex::new(self)?;
        if !scaffold.valid_for(self.function, &self.instructions, &self.instr_spans) {
            return Err("function fusion hint disagrees with executable windows".into());
        }
        if scaffold
            .private_layout
            .as_ref()
            .is_some_and(|layout| !layout.valid_for(self))
        {
            return Err("function private layout disagrees with selected body".into());
        }
        if scaffold.windows
            != selected_function_windows(
                self,
                &index,
                &scaffold.candidates,
                scaffold.private_layout.as_ref(),
            )?
        {
            return Err("function fusion hint disagrees with selected signal statements".into());
        }
        Ok(())
    }
}

/// Rebuild the exact function-local windows from certified statement PCs.
/// Source-span containment is not a PC authority: synthesized constructor
/// defaults can share spans with other emitted instructions.
pub(crate) fn selected_function_windows(
    body: &FunctionBytecode,
    index: &FunctionFusionCertificateIndex<'_>,
    candidates: &[FusionCandidateGroup],
    layout: Option<&FunctionPrivateSlotLayout>,
) -> Result<Vec<FusionInstructionWindow>, String> {
    let mut windows = Vec::new();
    let mut used = HashSet::new();
    for candidate in candidates {
        let mut signals = index.certify_candidate(candidate, layout)?;
        signals.sort_unstable();
        let mut group = FusionWindowGroup::default();
        for pc in signals {
            if !used.insert(pc) {
                return Err("function fusion hint reuses a selected signal".into());
            }
            let kind = body
                .instructions
                .get(pc)
                .and_then(instr_fusion_signal_kind)
                .ok_or_else(|| "function fusion hint names a non-signal instruction".to_string())?;
            group.push(pc, kind, &mut windows);
        }
        group.finish(&mut windows);
    }
    windows.sort_by_key(|window| (window.span.start, window.span.end));
    if windows
        .windows(2)
        .any(|pair| pair[0].span.end >= pair[1].span.start)
    {
        return Err("function fusion hint has overlapping selected windows".into());
    }
    Ok(windows)
}

/// Reusable selected-statement authority for both compiler slicing and strict
/// artifact admission. The projection and use index are built once per body.
pub(crate) struct FunctionFusionCertificateIndex<'a> {
    body: &'a FunctionBytecode,
    projection: &'a crate::bytecode::FusionStageProjection,
    rows: HashMap<ProgramPointId, usize>,
    signal_pcs: Vec<Vec<usize>>,
    use_counts: HashMap<RegionValueId, usize>,
    unique_users: HashMap<RegionValueId, ProgramPointId>,
    private_temporary_pairs: HashSet<(u32, usize)>,
}

impl<'a> FunctionFusionCertificateIndex<'a> {
    pub(crate) fn owner(&self) -> runmat_types::ProgramFunctionId {
        self.projection.owner
    }

    pub(crate) fn new(body: &'a FunctionBytecode) -> Result<Self, String> {
        let projection = body.fusion_stage_projection.as_ref().ok_or_else(|| {
            "function fusion hint has no selected statement authority".to_string()
        })?;
        let mut rows = HashMap::with_capacity(projection.statements.len());
        let mut signal_pcs = Vec::with_capacity(projection.statements.len());
        let mut use_counts = HashMap::new();
        let mut unique_users = HashMap::new();
        let private_temporary_pairs = projection
            .private_temporary_slots
            .iter()
            .map(|pair| (pair.local, pair.slot))
            .collect();
        for (index, statement) in projection.statements.iter().enumerate() {
            rows.insert(statement.contract.point, index);
            signal_pcs.push(
                (statement.pc.start..statement.pc.end)
                    .filter(|pc| {
                        body.instructions.get(*pc).is_some_and(|instruction| {
                            instr_fusion_signal_kind(instruction).is_some()
                        })
                    })
                    .collect(),
            );
            for used in &statement.contract.uses {
                *use_counts.entry(*used).or_insert(0) += 1;
                unique_users.insert(*used, statement.contract.point);
            }
        }
        Ok(Self {
            body,
            projection,
            rows,
            signal_pcs,
            use_counts,
            unique_users,
            private_temporary_pairs,
        })
    }

    pub(crate) fn statement_at(
        &self,
        point: ProgramPointId,
    ) -> Option<&'a crate::bytecode::CompiledFusionStageStatement> {
        self.rows
            .get(&point)
            .map(|index| &self.projection.statements[*index])
    }

    pub(crate) fn signal_pcs(&self, point: ProgramPointId) -> Option<&[usize]> {
        self.rows
            .get(&point)
            .map(|index| self.signal_pcs[*index].as_slice())
    }

    pub(crate) fn unique_user(&self, definition: RegionValueId) -> Option<ProgramPointId> {
        (self.use_counts.get(&definition) == Some(&1))
            .then(|| self.unique_users.get(&definition).copied())
            .flatten()
    }

    pub(crate) fn certified_empty_control(
        &self,
        statement: &crate::bytecode::CompiledFusionStageStatement,
        terminal: &crate::bytecode::CompiledFusionStageStatement,
        layout: Option<&FunctionPrivateSlotLayout>,
    ) -> bool {
        let Some(layout) = layout else {
            return false;
        };
        selected_empty_tensor_control(self, layout, statement, terminal)
    }

    pub(crate) fn certify_candidate(
        &self,
        candidate: &FusionCandidateGroup,
        layout: Option<&FunctionPrivateSlotLayout>,
    ) -> Result<Vec<usize>, String> {
        let block = u32::try_from(candidate.block.0)
            .map_err(|_| "function fusion hint block exceeds statement identity".to_string())?;
        let point_at = |position: usize| -> Result<ProgramPointId, String> {
            Ok(ProgramPointId {
                function: self.projection.owner,
                block,
                position: u32::try_from(position).map_err(|_| {
                    "function fusion hint position exceeds statement identity".to_string()
                })?,
            })
        };
        let terminal = self
            .statement_at(point_at(candidate.stmt_end.checked_sub(1).ok_or_else(
                || "function fusion hint has no selected terminal".to_string(),
            )?)?)
            .ok_or_else(|| "function fusion hint has no selected terminal".to_string())?;
        let mut authorized = Vec::new();
        for position in candidate.stmt_start..candidate.stmt_end {
            let point = point_at(position)?;
            let statement = self.statement_at(point).ok_or_else(|| {
                "function fusion hint names a missing selected statement".to_string()
            })?;
            if !matches!(
                statement.contract.status,
                FusionStageFactStatus::Inferred | FusionStageFactStatus::UnknownInputs
            ) {
                return Err(
                    "function fusion hint crosses an effectful or uncertain statement".into(),
                );
            }
            let signal_pcs = self.signal_pcs(point).ok_or_else(|| {
                "function fusion hint names a missing selected statement".to_string()
            })?;
            if signal_pcs.is_empty() {
                if !self.certified_empty_control(statement, terminal, layout) {
                    return Err("function fusion hint has an uncertified control statement".into());
                }
                continue;
            }
            if signal_pcs.len() != 1 {
                return Err("function fusion hint has multiple signals in one statement".into());
            }
            if !crate::bytecode::fusion_effects::elementwise_statement_effects_are_compatible(
                &self.body.instructions,
                statement,
            ) && (layout.is_none()
                || position.checked_add(1) != Some(candidate.stmt_end)
                || !self
                    .body
                    .instructions
                    .get(signal_pcs[0])
                    .is_some_and(canonical_terminal_reduction)
                || !statement
                    .contract
                    .effects
                    .0
                    .iter()
                    .all(|effect| matches!(effect, EffectKind::MayThrow | EffectKind::MaySuspend)))
            {
                return Err("function fusion hint has nonterminal or impure effects".into());
            }
            authorized.extend_from_slice(signal_pcs);
        }
        if authorized.len() != candidate.signal_count {
            return Err("function fusion hint signal count disagrees with selected body".into());
        }
        Ok(authorized)
    }
}

fn selected_empty_tensor_control(
    index: &FunctionFusionCertificateIndex<'_>,
    layout: &FunctionPrivateSlotLayout,
    statement: &crate::bytecode::CompiledFusionStageStatement,
    terminal: &crate::bytecode::CompiledFusionStageStatement,
) -> bool {
    let [definition] = statement.contract.defs.as_slice() else {
        return false;
    };
    let Some([Instr::CreateMatrix(0, 0), Instr::StoreVar(slot)]) = index
        .body
        .instructions
        .get(statement.pc.start..statement.pc.end)
    else {
        return false;
    };
    let pair = crate::bytecode::fusion_stage::CompiledPrivateTemporarySlot {
        local: definition.local,
        slot: *slot,
    };
    statement.contract.status == FusionStageFactStatus::Inferred
        && statement.contract.uses.is_empty()
        && statement.contract.effects.0.is_empty()
        && statement.contract.live_after.contains(definition)
        && definition.function == index.projection.owner
        && layout.owns_private_pair(pair)
        && index
            .private_temporary_pairs
            .contains(&(pair.local, pair.slot))
        && terminal.contract.uses.contains(definition)
        && !terminal.contract.live_after.contains(definition)
        && index.use_counts.get(definition) == Some(&1)
        && {
            index
                .signal_pcs(terminal.contract.point)
                .is_some_and(|signals| {
                    signals.len() == 1
                        && index
                            .body
                            .instructions
                            .get(signals[0])
                            .is_some_and(canonical_terminal_reduction)
                })
        }
}

impl FunctionRegistry {
    /// The pointer key is an exact body inside this immutable Arc snapshot,
    /// including retained leased bodies. A cloned or modified external body
    /// cannot reuse another definition's admission.
    pub(crate) fn admitted_fusion_scaffold(
        &self,
        body: &FunctionBytecode,
    ) -> Option<&Arc<FunctionFusionScaffold>> {
        self.data
            .admitted_fusion_scaffolds
            .get_or_init(|| {
                let mut admitted = HashMap::new();
                for body in self
                    .functions
                    .values()
                    .chain(self.retired.values().flatten())
                {
                    if body.validate_fusion_scaffold().is_ok() {
                        if let Some(scaffold) = &body.fusion_scaffold {
                            admitted.insert(body as *const _ as usize, Arc::new(scaffold.clone()));
                        }
                    }
                }
                admitted
            })
            .get(&(body as *const _ as usize))
    }

    pub(crate) fn validate_fusion_scaffolds_for_artifact(&self) -> Result<(), String> {
        let mut admitted = HashMap::new();
        for body in self.functions.values() {
            body.validate_fusion_scaffold()?;
            if let Some(scaffold) = &body.fusion_scaffold {
                admitted.insert(body as *const _ as usize, Arc::new(scaffold.clone()));
            }
        }
        // Retired definitions are not serialized into ProgramV2, so their
        // invalid hints cannot reject that artifact. They remain executable
        // through exact local leases in this same snapshot, however; seeding
        // the shared cache from current bodies alone would suppress a valid
        // retained hint permanently after encoding.
        for body in self.retired.values().flatten() {
            if body.validate_fusion_scaffold().is_ok() {
                if let Some(scaffold) = &body.fusion_scaffold {
                    admitted.insert(body as *const _ as usize, Arc::new(scaffold.clone()));
                }
            }
        }
        let _ = self.data.admitted_fusion_scaffolds.set(admitted);
        Ok(())
    }
}

#[cfg(test)]
#[path = "function_fusion/terminal_effect_tests.rs"]
mod terminal_effect_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admitted_hint_is_shared_per_snapshot_and_cow_mutation_invalidates_it() {
        let ast =
            runmat_parser::parse("function y=f()\npersistent p\nt=p.*2;\np=p+t+1;\ny=p;\nend\n")
                .unwrap();
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
        let registry = FunctionRegistry::new(
            crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir).unwrap(),
        );
        let function = hir.assembly.functions[0].id;
        let body = registry.get(function).unwrap();
        assert!(registry.data.admitted_fusion_scaffolds.get().is_none());
        let first = registry
            .admitted_fusion_scaffold(body)
            .expect("admitted hint");
        let second = registry.admitted_fusion_scaffold(body).unwrap();
        assert!(Arc::ptr_eq(first, second));

        let mut changed = registry.clone();
        changed
            .functions
            .get_mut(&function)
            .unwrap()
            .fusion_scaffold = None;
        assert!(changed.data.admitted_fusion_scaffolds.get().is_none());
        assert!(changed
            .admitted_fusion_scaffold(changed.get(function).unwrap())
            .is_none());
        assert!(registry.admitted_fusion_scaffold(body).is_some());
    }

    #[test]
    fn program_encode_does_not_hide_retired_owned_hint_in_shared_snapshot() {
        let ast =
            runmat_parser::parse("function y=f()\npersistent p\nt=p.*2;\np=p+t+1;\ny=p;\nend\n")
                .unwrap();
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
        let mut registry = FunctionRegistry::new(
            crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir).unwrap(),
        );
        let function = hir.assembly.functions[0].id;
        let owners = registry
            .definition_dependency_owners(function)
            .expect("selected definition graph");
        let issuer = Arc::new(());
        let authority = runmat_value::BoundFunctionAuthority::local(
            Arc::downgrade(&issuer),
            runmat_value::FunctionDefinitionOwner::issue(&owners[0].1, owners[0].0 .0),
            owners[1..]
                .iter()
                .map(|(id, owner)| runmat_value::FunctionDefinitionOwner::issue(owner, id.0))
                .collect(),
            runmat_value::LocalExport::Ephemeral,
        );
        let mut replacement = registry.get(function).unwrap().clone();
        replacement.definition_owner = Arc::new(runmat_value::FunctionDefinitionOwner::default());
        replacement.fusion_scaffold = None;
        registry
            .remove(function)
            .expect("retire old owned function");
        registry.insert_replacing_name(replacement);
        assert!(registry
            .retired
            .get(&function)
            .is_some_and(|bodies| !bodies.is_empty()));

        crate::bytecode::encode_interpreter_program_v2(&registry)
            .expect("portable current registry encode seeds admission");
        let owned = registry
            .owned_view(function, &authority)
            .expect("exact old definition remains selected");
        let body = owned
            .get(function)
            .expect("retired body behind bound lease");
        assert!(body.fusion_scaffold.is_some());
        assert!(owned.admitted_fusion_scaffold(body).is_some());
        let layout = crate::layout::derive_layout(&hir.assembly, &mir).unwrap();
        let executable = crate::Bytecode::for_function(body, owned.clone(), layout);
        assert!(!executable.runtime_fusion_groups().is_empty());
    }
}
