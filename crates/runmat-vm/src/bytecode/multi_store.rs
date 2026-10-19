//! Structural MIR-to-bytecode candidates for ordered multi-store fusion.
//!
//! This projection is deliberately not an executable fusion plan. Provider
//! output completeness, external uses, error ordering, and publication costs
//! must be proved before a candidate can replace any bytecode instructions.

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

use runmat_accelerate::graph::{AccelGraph, ValueId, VarBinding, VarKind};
use runmat_types::{ProgramFunctionId, ProgramPointId};

use crate::accel::graph::{
    build_accel_graph_with_stores, build_accel_graph_with_transcript, AccelStore,
};
use crate::bytecode::{
    Bytecode, FusionCandidateGroup, FusionStageSelectionOrigin, Instr, StatementPcRange,
};

mod semantic;
pub use semantic::ProspectiveSemanticMultiStoreRegion;

mod image_normalize;
pub(crate) use image_normalize::{
    CapturedImageInputs, SelectedImageRegion, SelectedImageRegionSet,
};
pub use image_normalize::{
    ImageCheckpoint, ImageLiveOut, ImageLiveOutSource, ImageNormalizePublicationJoin,
    ImageRegionStore, ImageStoreDestination, ImageStoreOrdinal,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProspectiveStore {
    pub pc: usize,
    pub value_id: ValueId,
    pub binding: VarBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProspectiveMultiStoreRegion {
    pub mir_candidate_id: usize,
    pub first_statement: ProgramPointId,
    pub last_statement: ProgramPointId,
    pub pcs: Range<usize>,
    pub stores: Vec<ProspectiveStore>,
}

impl Bytecode {
    /// Reconcile the selected MIR candidate and the same graph walk that owns
    /// bytecode value IDs. No result is executable without later effect,
    /// liveness, output, placement, and failure-order admission.
    pub fn prospective_multi_store_regions(&self) -> Vec<ProspectiveMultiStoreRegion> {
        SelectedMultiStoreGraph::new(self)
            .map(|selected| selected.structural_regions())
            .unwrap_or_default()
    }
}

/// The sole image-proposal graph/store transcript owner. Its private
/// constructor derives both from the exact selected bytecode; a caller cannot
/// substitute an unrelated graph with coincident local ValueIds.
pub(super) struct SelectedMultiStoreGraph<'a> {
    bytecode: &'a Bytecode,
    pub(super) graph: AccelGraph,
    stores: Vec<AccelStore>,
}

impl<'a> SelectedMultiStoreGraph<'a> {
    pub(super) fn new(bytecode: &'a Bytecode) -> Option<Self> {
        if bytecode
            .fusion_metadata
            .mir_fusion_candidate_groups
            .is_empty()
        {
            return None;
        }
        let (graph, stores) =
            build_accel_graph_with_stores(&bytecode.instructions, &bytecode.var_types);
        Some(Self {
            bytecode,
            graph,
            stores,
        })
    }

    /// Image source reconciliation is owned by the sealed statement
    /// projection, not by the narrower consecutive-signal MIR hint. Nested
    /// calls and materializations can separate its source-visible stores.
    pub(super) fn for_image(bytecode: &'a Bytecode) -> Option<Self> {
        bytecode.fusion_metadata.stage_projection.as_ref()?;
        let (mut graph, stores, loads) =
            build_accel_graph_with_transcript(&bytecode.instructions, &bytecode.var_types);
        let mut literals = crate::bytecode::fusion_private_temp::literal_controls(bytecode);
        let groups = graph
            .detect_fusion_groups()
            .into_iter()
            .filter(|group| group.kind == runmat_accelerate::FusionKind::ImageNormalize)
            .collect::<Vec<_>>();
        if groups.is_empty()
            || !crate::bytecode::fusion_private_temp::project_private_bindings(
                &mut graph,
                &groups,
                &crate::bytecode::fusion_private_temp::bridges(bytecode),
                &crate::bytecode::fusion_private_temp::empty_controls(bytecode),
                &mut literals,
                &stores,
                &loads,
            )
        {
            return None;
        }
        Some(Self {
            bytecode,
            graph,
            stores,
        })
    }

    pub(super) fn selected_function(&self) -> Option<runmat_hir::FunctionId> {
        let bytecode = self.bytecode;
        let layout = bytecode.layout.as_ref()?;
        let selection_origin = bytecode
            .fusion_metadata
            .stage_projection
            .as_ref()?
            .selection_origin?;
        if let Some(active) = bytecode.active_function {
            let function = layout.functions.get(&active)?;
            if function.function != active || function.resume_points != bytecode.resume_points {
                return None;
            }
            match selection_origin {
                FusionStageSelectionOrigin::FunctionBody => {
                    // Function-only views must retain the exact selected immutable
                    // body. Matching function IDs or coincident statement-PC maps
                    // do not transfer its portable MIR projection authority.
                    let body = bytecode.function_registry.get(active)?;
                    let selected_stage = body.fusion_stage_projection.as_ref()?;
                    let view_stage = bytecode.fusion_metadata.stage_projection.as_ref()?;
                    let selected_layout = view_stage.selected_function_layout.as_ref()?;
                    if body.validate_fusion_stage_projection().is_err()
                        || bytecode.validate_fusion_stage_projection().is_err()
                        || selected_stage.portable_origin != body.portable_function
                        || selected_stage.as_ref() != view_stage.as_ref()
                        || selected_layout != function
                        || body.resume_points != bytecode.resume_points
                        || body.source_id != bytecode.source_id
                        || body.var_count != bytecode.var_count
                    {
                        return None;
                    }
                }
                FusionStageSelectionOrigin::EntrypointStream => {
                    if bytecode.function_registry.get(active).is_some()
                        || !layout
                            .entrypoints
                            .values()
                            .any(|entry| entry.target == active)
                    {
                        // The compiler omits a selected entrypoint body from
                        // the registry; this route cannot borrow a function.
                        return None;
                    }
                }
            }
            Some(active)
        } else {
            match selection_origin {
                FusionStageSelectionOrigin::EntrypointStream => {}
                FusionStageSelectionOrigin::FunctionBody => return None,
            }
            let mut selected = layout.entrypoints.values().filter_map(|entrypoint| {
                let function = layout.functions.get(&entrypoint.target)?;
                (function.function == entrypoint.target
                    && function.resume_points == bytecode.resume_points
                    && bytecode.function_registry.get(entrypoint.target).is_none())
                .then_some(entrypoint.target)
            });
            let function = selected.next()?;
            selected
                .all(|candidate| candidate == function)
                .then_some(function)
        }
    }

    pub(super) fn structural_regions(&self) -> Vec<ProspectiveMultiStoreRegion> {
        let bytecode = self.bytecode;
        let Ok(ranges) = bytecode.statement_pc_ranges() else {
            return Vec::new();
        };
        if bytecode.catch_scopes.len() != bytecode.instructions.len() {
            return Vec::new();
        }
        let Some(program_owner) = ranges.first().map(|range| range.statement.function) else {
            return Vec::new();
        };
        let Some(selected_function) = self.selected_function() else {
            return Vec::new();
        };
        let selected_functions = HashSet::from([selected_function]);
        let by_statement: BTreeMap<_, _> = ranges
            .iter()
            .map(|range| (range.statement, range))
            .collect();
        let interior_targets = interior_control_targets(&bytecode.instructions);
        let context = ReconciliationContext {
            bytecode,
            ranges: &by_statement,
            stores: &self.stores,
            graph: &self.graph,
            interior_targets: &interior_targets,
            program_owner,
            selected_functions: &selected_functions,
        };
        bytecode
            .fusion_metadata
            .mir_fusion_candidate_groups
            .iter()
            .filter_map(|candidate| reconcile_candidate(candidate, &context))
            .collect()
    }
}

struct ReconciliationContext<'a> {
    bytecode: &'a Bytecode,
    ranges: &'a BTreeMap<ProgramPointId, &'a StatementPcRange>,
    stores: &'a [AccelStore],
    graph: &'a runmat_accelerate::graph::AccelGraph,
    interior_targets: &'a HashSet<usize>,
    program_owner: ProgramFunctionId,
    selected_functions: &'a HashSet<runmat_hir::FunctionId>,
}

fn reconcile_candidate(
    candidate: &FusionCandidateGroup,
    context: &ReconciliationContext<'_>,
) -> Option<ProspectiveMultiStoreRegion> {
    // The selected layout is the serialized authority connecting this MIR
    // candidate's FunctionId to the program-point owner's separate identity
    // domain. Never compare the two identifiers by numeric coincidence.
    if !context.selected_functions.contains(&candidate.function)
        || context
            .bytecode
            .active_function
            .is_some_and(|active| active != candidate.function)
        || candidate.stmt_end <= candidate.stmt_start
    {
        return None;
    }
    let function = context.program_owner;
    let block = u32::try_from(candidate.block.0).ok()?;
    let first_position = u32::try_from(candidate.stmt_start).ok()?;
    let last_position = u32::try_from(candidate.stmt_end.checked_sub(1)?).ok()?;
    let first_statement = ProgramPointId {
        function,
        block,
        position: first_position,
    };
    let last_statement = ProgramPointId {
        function,
        block,
        position: last_position,
    };
    let first = *context.ranges.get(&first_statement)?;
    let last = *context.ranges.get(&last_statement)?;
    if first.start >= last.end {
        return None;
    }
    let mut previous_end = first.start;
    for position in candidate.stmt_start..candidate.stmt_end {
        let point = ProgramPointId {
            function,
            block,
            position: u32::try_from(position).ok()?,
        };
        let range = *context.ranges.get(&point)?;
        if range.start != previous_end {
            return None;
        }
        previous_end = range.end;
    }
    if previous_end != last.end
        || context
            .interior_targets
            .iter()
            .any(|target| first.start < *target && *target < last.end)
    {
        return None;
    }
    let scope = context.bytecode.catch_scopes[first.start];
    let mut depth = 0usize;
    for pc in first.start..last.end {
        let instruction = &context.bytecode.instructions[pc];
        if context.bytecode.catch_scopes[pc] != scope
            || !is_structural_region_instruction(instruction)
        {
            return None;
        }
        let effect = instruction.stack_effect()?;
        depth = depth.checked_sub(effect.pops)?.checked_add(effect.pushes)?;
    }
    if depth != 0 {
        return None;
    }

    let mut ordered_stores = Vec::new();
    let mut transcript = context
        .stores
        .iter()
        .filter(|store| first.start <= store.pc && store.pc < last.end);
    for pc in first.start..last.end {
        let expected = match context.bytecode.instructions[pc] {
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
        let Some(binding) = expected else { continue };
        let store = transcript.next()?;
        if store.pc != pc
            || store.binding != binding
            || context.graph.value(store.value_id).is_none()
        {
            return None;
        }
        ordered_stores.push(ProspectiveStore {
            pc,
            value_id: store.value_id,
            binding,
        });
    }
    if ordered_stores.len() < 2 || transcript.next().is_some() {
        return None;
    }
    Some(ProspectiveMultiStoreRegion {
        mir_candidate_id: candidate.id,
        first_statement,
        last_statement,
        pcs: first.start..last.end,
        stores: ordered_stores,
    })
}

pub(super) fn interior_control_targets(instructions: &[Instr]) -> HashSet<usize> {
    instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instr::Jump(target)
            | Instr::JumpIfFalse(target)
            | Instr::AndAnd(target)
            | Instr::OrOr(target) => Some(*target),
            Instr::StochasticEvolution { fallback_pc, .. } => Some(*fallback_pc),
            Instr::EnterTry { catch_pc, .. } => Some(*catch_pc),
            _ => None,
        })
        .collect()
}

pub(super) fn is_structural_region_instruction(instruction: &Instr) -> bool {
    matches!(
        instruction,
        Instr::LoadConst(_)
            | Instr::LoadInt(_)
            | Instr::LoadComplex(_, _)
            | Instr::LoadBool(_)
            | Instr::LoadString(_)
            | Instr::LoadCharRow(_)
            | Instr::LoadVar(_)
            | Instr::LoadLocal(_)
            | Instr::StoreVar(_)
            | Instr::StoreLocal(_)
            | Instr::Add
            | Instr::Sub
            | Instr::Mul
            | Instr::RightDiv
            | Instr::LeftDiv
            | Instr::Pow
            | Instr::Neg
            | Instr::UPlus
            | Instr::ElemMul
            | Instr::ElemDiv
            | Instr::ElemPow
            | Instr::ElemLeftDiv
            // The current graph builder records one ValueId per builtin call.
            // A different bytecode output arity would corrupt store identity.
            | Instr::CallBuiltinMulti(_, _, 1)
            | Instr::CreateMatrix(_, _)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_char_row_control_has_one_structural_stack_push() {
        let control = Instr::LoadCharRow("native".into());
        assert!(is_structural_region_instruction(&control));
        let effect = control.stack_effect().expect("literal stack effect");
        assert_eq!((effect.pops, effect.pushes), (0, 1));
    }

    fn compile_source(source: &str) -> Bytecode {
        let ast = runmat_parser::parse(source).expect("parse source");
        let hir =
            runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
        crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
            .expect("compile bytecode")
    }

    fn named_stores(bytecode: &Bytecode, region: &ProspectiveMultiStoreRegion) -> Vec<String> {
        region
            .stores
            .iter()
            .filter_map(|store| bytecode.var_names.get(&store.binding.index).cloned())
            .collect()
    }

    #[test]
    fn compiled_generic_run_reconciles_exact_ordered_stores() {
        let bytecode = compile_source("a=1; mu=a+1; sigma=mu.*2; out=sigma-3;");
        let regions = bytecode.prospective_multi_store_regions();
        let region = regions
            .iter()
            .find(|region| {
                named_stores(&bytecode, region)
                    .iter()
                    .map(String::as_str)
                    .eq(["mu", "sigma", "out"])
            })
            .expect("same-block MIR run and all three exact bytecode stores");
        assert!(region.stores.windows(2).all(|pair| pair[0].pc < pair[1].pc));
        assert_eq!(
            region.first_statement.function,
            region.last_statement.function
        );
        assert_eq!(region.first_statement.block, region.last_statement.block);
        assert!(region.pcs.start <= region.stores[0].pc);
        assert!(region.stores.last().unwrap().pc < region.pcs.end);
    }

    #[test]
    fn branch_and_catch_do_not_merge_distinct_mir_runs() {
        let bytecode = compile_source(
            "a=1; if a; mu=a+1; else; mu=a-1; end; \
             try; sigma=mu.*2; catch e; sigma=0; end; out=sigma-3;",
        );
        assert!(bytecode.prospective_multi_store_regions().is_empty());
    }

    #[test]
    fn damaged_statement_or_catch_authority_rejects_before_publication() {
        let mut bytecode = compile_source("a=1; mu=a+1; sigma=mu.*2; out=sigma-3;");
        let region = bytecode
            .prospective_multi_store_regions()
            .into_iter()
            .next()
            .expect("valid compiled candidate");
        bytecode.catch_scopes[region.stores[1].pc] = Some(runmat_mir::CatchScopeId(7));
        assert!(bytecode.prospective_multi_store_regions().is_empty());

        bytecode.catch_scopes[region.stores[1].pc] = None;
        bytecode.resume_points.remove(&region.last_statement);
        assert!(bytecode.prospective_multi_store_regions().is_empty());
    }

    #[test]
    fn script_artifact_preserves_candidate_and_selected_layout_authority() {
        let bytecode = compile_source("a=1; mu=a+1; sigma=mu.*2; out=sigma-3;");
        let expected = bytecode.prospective_multi_store_regions();
        assert!(!expected.is_empty());
        let artifact = crate::bytecode::encode_interpreter_script_v2(&bytecode).unwrap();
        let mut restored = crate::bytecode::decode_interpreter_script_v2(&artifact).unwrap();
        assert!(
            restored.active_function.is_none(),
            "session identity is not serialized"
        );
        assert_eq!(restored.prospective_multi_store_regions(), expected);

        let layout = restored.layout.as_mut().expect("selected compiler layout");
        let entrypoint = layout.entrypoints.values_mut().next().expect("entrypoint");
        let original_target = entrypoint.target;
        entrypoint.target = runmat_hir::FunctionId(usize::MAX);
        assert!(restored.prospective_multi_store_regions().is_empty());

        let layout = restored.layout.as_mut().unwrap();
        layout.entrypoints.values_mut().next().unwrap().target = original_target;
        layout
            .functions
            .get_mut(&original_target)
            .unwrap()
            .resume_points
            .clear();
        assert!(restored.prospective_multi_store_regions().is_empty());

        let mut mixed = bytecode;
        let (mut point, pc) = mixed.resume_points.pop_last().unwrap();
        point.function = ProgramFunctionId(point.function.0 + 1);
        mixed.resume_points.insert(point, pc);
        assert!(mixed.prospective_multi_store_regions().is_empty());
    }

    #[test]
    fn graph_output_arity_must_match_bytecode_before_store_ids_are_used() {
        assert!(is_structural_region_instruction(&Instr::CallBuiltinMulti(
            "single".into(),
            1,
            1,
        )));
        for outputs in [0, 2] {
            assert!(!is_structural_region_instruction(&Instr::CallBuiltinMulti(
                "single".into(),
                1,
                outputs,
            )));
        }
    }
}
