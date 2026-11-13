use super::*;

fn compiled(source: &str) -> Bytecode {
    let ast = runmat_parser::parse(source).expect("parse source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile selected source")
}

fn compiled_with_status(
    source: &str,
    status: FusionStageFactStatus,
    effect: Option<EffectKind>,
) -> Bytecode {
    let ast = runmat_parser::parse(source).expect("parse source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    let mut analysis = runmat_mir::analysis::analyze_assembly(&mir);
    for contract in &mut analysis.fusion_stage_contracts {
        contract.status = status;
        if let Some(effect) = effect {
            contract.effects.0.insert(effect);
        }
    }
    crate::bytecode::compile_with_analysis(
        &hir.assembly,
        &mir,
        hir.assembly.entrypoints[0].id,
        &analysis,
    )
    .expect("compile selected negative analysis")
}

#[test]
fn selected_facts_join_compiled_store_pcs_without_authorizing_execution() {
    let bytecode = compiled("a=1; mu=a+1; sigma=mu.*2; out=sigma-3;");
    let proposals = bytecode.prospective_semantic_multi_store_regions();
    let proposal = proposals
        .iter()
        .find(|proposal| {
            proposal
                .structural
                .stores
                .iter()
                .filter_map(|store| bytecode.var_names.get(&store.binding.index))
                .map(String::as_str)
                .eq(["mu", "sigma", "out"])
        })
        .expect("selected MIR facts and exact ordered stores");
    assert_eq!(proposal.stages.len(), 3);
    assert!(proposal.stages.iter().all(|stage| {
        stage.contract.point == stage.pc.statement
            && stage.contract.status == FusionStageFactStatus::Inferred
    }));
    let artifact = crate::bytecode::encode_interpreter_script_v2(&bytecode).unwrap();
    let restored = crate::bytecode::decode_interpreter_script_v2(&artifact).unwrap();
    assert_eq!(
        restored.prospective_semantic_multi_store_regions().len(),
        proposals.len()
    );
}

#[test]
fn uncertain_and_effectful_selected_facts_never_qualify_stores() {
    let source = "a=1; mu=a+1; sigma=mu.*2; out=sigma-3;";
    for status in [
        FusionStageFactStatus::UnknownInputs,
        FusionStageFactStatus::UnknownEffects,
        FusionStageFactStatus::WidenedFacts,
        FusionStageFactStatus::Nonconverged,
    ] {
        let bytecode = compiled_with_status(source, status, None);
        assert!(bytecode
            .prospective_semantic_multi_store_regions()
            .is_empty());
    }
    let bytecode = compiled_with_status(
        source,
        FusionStageFactStatus::Inferred,
        Some(EffectKind::WorkspaceWrite),
    );
    assert!(bytecode
        .prospective_semantic_multi_store_regions()
        .is_empty());
    for effect in [EffectKind::MayThrow, EffectKind::MaySuspend] {
        let bytecode = compiled_with_status(source, FusionStageFactStatus::Inferred, Some(effect));
        assert!(
            !bytecode
                .prospective_semantic_multi_store_regions()
                .is_empty(),
            "{effect:?} is not itself an observable side effect"
        );
    }
}

#[test]
fn known_but_swapped_store_slots_do_not_match_their_statement_defs() {
    let bytecode = compiled("a=1; mu=a+1; sigma=mu.*2; out=sigma-3;");
    let projection = bytecode.fusion_metadata.stage_projection.as_ref().unwrap();
    let structural = bytecode
        .prospective_semantic_multi_store_regions()
        .into_iter()
        .find(|region| region.structural.stores.len() >= 2)
        .expect("ordered selected stores")
        .structural;
    assert_ne!(
        structural.stores[0].binding.index,
        structural.stores[1].binding.index
    );
    assert!(semantic_region(&bytecode, projection, structural.clone()).is_some());

    let mut swapped = structural;
    let first = swapped.stores[0].binding.index;
    swapped.stores[0].binding.index = swapped.stores[1].binding.index;
    swapped.stores[1].binding.index = first;
    assert!(semantic_region(&bytecode, projection, swapped).is_none());
}
