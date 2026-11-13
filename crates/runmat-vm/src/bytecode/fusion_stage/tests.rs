use super::*;

fn compile_source(source: &str) -> super::super::Bytecode {
    let ast = runmat_parser::parse(source).expect("parse source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    super::super::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile selected analysis")
}

#[test]
fn compiled_script_retains_exact_selected_statement_projection() {
    let bytecode = compile_source("a=1; mu=a+1; sigma=mu.*2; out=sigma-3;");
    let projection = bytecode
        .fusion_metadata
        .stage_projection
        .as_ref()
        .expect("selected MIR statement facts");
    assert_eq!(
        projection.selection_origin,
        Some(FusionStageSelectionOrigin::EntrypointStream)
    );
    assert_eq!(
        projection.statements.len(),
        bytecode.statement_pc_ranges().unwrap().len()
    );
    assert!(bytecode.validate_fusion_stage_projection().is_ok());
    assert!(projection
        .statements
        .iter()
        .all(|statement| statement.contract.point == statement.pc.statement));

    let artifact = super::super::encode_interpreter_script_v2(&bytecode).unwrap();
    let restored = super::super::decode_interpreter_script_v2(&artifact).unwrap();
    assert!(restored.validate_fusion_stage_projection().is_ok());
    assert_eq!(
        restored.fusion_metadata.stage_projection,
        bytecode.fusion_metadata.stage_projection
    );
}

#[test]
fn legacy_missing_selection_origin_retains_old_seal_without_image_authority() {
    let mut bytecode = compile_source(
        "imgs=ones(2,4,5,'single'); mu=single(mean(imgs,[2 3],'native')); sigma=single(sqrt(mean((imgs-mu).^2,[2 3],'native')+single(1e-6))); out=single(((imgs-mu)./sigma).^single(1.8));",
    );
    assert!(!bytecode
        .prospective_image_normalize_publications()
        .is_empty());
    std::sync::Arc::make_mut(bytecode.fusion_metadata.stage_projection.as_mut().unwrap())
        .selection_origin = None;
    bytecode.reseal_fusion_stage_after_owned_remap();
    assert!(bytecode.validate_fusion_stage_projection().is_ok());
    let projection = bytecode.fusion_metadata.stage_projection.as_ref().unwrap();
    let wire = serde_json::to_value(projection).unwrap();
    assert!(wire.get("selection_origin").is_none());
    let restored: FusionStageProjection = serde_json::from_value(wire).unwrap();
    assert_eq!(restored.selection_origin, None);
    assert!(bytecode
        .prospective_image_normalize_publications()
        .is_empty());
}

#[test]
fn malformed_owner_local_pc_and_stream_are_not_fusion_stage_evidence() {
    let original = compile_source("a=1; mu=a+1; sigma=mu.*2; out=sigma-3;");
    let mut damaged = original.clone();
    std::sync::Arc::make_mut(damaged.fusion_metadata.stage_projection.as_mut().unwrap())
        .statements[0]
        .pc
        .end += 1;
    assert!(damaged.validate_fusion_stage_projection().is_err());

    let mut damaged = original.clone();
    std::sync::Arc::make_mut(damaged.fusion_metadata.stage_projection.as_mut().unwrap()).statements
        [0]
    .contract
    .uses
    .push(RegionValueId {
        function: ProgramFunctionId(u32::MAX),
        local: 0,
    });
    assert!(damaged.validate_fusion_stage_projection().is_err());

    let mut damaged = original.clone();
    damaged.instructions.push(Instr::Return);
    damaged.catch_scopes.push(None);
    assert!(damaged.validate_fusion_stage_projection().is_err());
}

#[test]
fn possibly_unassigned_branch_input_remains_unknown_in_compiled_function() {
    let bytecode = compile_source(
        "y=f(1); function y=f(flag); if flag; a=1; end; mu=a+1; sigma=mu.*2; y=sigma-3; end",
    );
    let function = bytecode
        .bound_functions
        .values()
        .find(|function| function.fusion_stage_projection.is_some())
        .expect("selected function MIR facts");
    let projection = function.fusion_stage_projection.as_ref().unwrap();
    assert_eq!(
        projection.selection_origin,
        Some(FusionStageSelectionOrigin::FunctionBody)
    );
    assert!(function.validate_fusion_stage_projection().is_ok());
    assert!(projection.statements.iter().any(|statement| {
        statement.contract.status == runmat_mir::analysis::FusionStageFactStatus::UnknownInputs
    }));
}

#[test]
fn selected_function_stage_owner_tracks_session_and_closed_unit_rebinding() {
    let bytecode =
        compile_source("a=helper(1); function y=helper(x); mu=x+1; sigma=mu.*2; y=sigma-3; end");
    let original = bytecode
        .bound_functions
        .values()
        .find(|function| function.fusion_stage_projection.is_some())
        .expect("selected helper body")
        .clone();
    assert!(original.validate_fusion_stage_projection().is_ok());
    let execution = original.execution_bytecode(&super::super::FunctionRegistry::default());
    assert!(std::sync::Arc::ptr_eq(
        execution.fusion_metadata.stage_projection.as_ref().unwrap(),
        original.fusion_stage_projection.as_ref().unwrap()
    ));

    let mut session = original.clone();
    session.rebind_identity(runmat_hir::FunctionId(701));
    assert!(session.validate_fusion_stage_projection().is_ok());
    assert!(session
        .fusion_stage_projection
        .as_ref()
        .unwrap()
        .statements
        .iter()
        .all(|row| row.contract.point.function == original.portable_function));
    assert_eq!(session.portable_function, original.portable_function);
    assert_eq!(
        session
            .fusion_stage_projection
            .as_ref()
            .unwrap()
            .selection_origin,
        Some(FusionStageSelectionOrigin::FunctionBody)
    );
    assert_eq!(
        session
            .fusion_stage_projection
            .as_ref()
            .unwrap()
            .portable_origin,
        original.portable_function
    );
    assert_eq!(
        session
            .fusion_stage_projection
            .as_ref()
            .unwrap()
            .portable_definition_identity(),
        original
            .fusion_stage_projection
            .as_ref()
            .unwrap()
            .portable_definition_identity()
    );

    let mut closed = original.clone();
    crate::bytecode::program::function_remap::remap_closed_function(
        &mut closed,
        &std::collections::HashMap::from([(original.function, runmat_hir::FunctionId(702))]),
    )
    .expect("closed projection");
    assert_eq!(closed.function, runmat_hir::FunctionId(702));
    assert_eq!(closed.portable_function, original.portable_function);
    assert!(closed.validate_fusion_stage_projection().is_ok());
    assert_eq!(
        closed.fusion_stage_projection.as_ref().unwrap().owner,
        original.portable_function
    );

    let mut eval = original.clone();
    crate::bytecode::program::function_remap::remap_eval_local_function(
        &mut eval,
        &std::collections::HashMap::from([(original.function, runmat_hir::FunctionId(703))]),
    )
    .expect("eval-local projection");
    assert_eq!(eval.function, runmat_hir::FunctionId(703));
    assert!(eval.validate_fusion_stage_projection().is_ok());
    assert_eq!(
        eval.fusion_stage_projection
            .as_ref()
            .unwrap()
            .portable_definition_identity(),
        original
            .fusion_stage_projection
            .as_ref()
            .unwrap()
            .portable_definition_identity()
    );
}

#[test]
fn session_remap_seals_local_helper_callback_and_source_only_after_owned_transform() {
    let mut bytecode = compile_source(
        "y=outer(1); function y=outer(x); f=@inner; q=f(x); mu=q+1; sigma=mu.*2; y=sigma-3; end; function z=inner(x); z=x+4; end",
    );
    let registry = bytecode.function_registry();
    let outer_id = registry.resolve_name("outer").expect("outer function");
    let inner_id = registry.resolve_name("inner").expect("local helper");
    let remap = std::collections::HashMap::from([
        (outer_id, runmat_hir::FunctionId(701)),
        (inner_id, runmat_hir::FunctionId(702)),
    ]);
    let mut outer = registry.get(outer_id).expect("outer body").clone();
    assert!(outer.fusion_stage_projection.is_some());
    assert!(outer.validate_fusion_stage_projection().is_ok());
    let portable_origin = outer.portable_function;
    let source = outer.source_id.or(bytecode.source_id);
    crate::bytecode::program::remap_session_function(&mut outer, remap[&outer_id], source, &remap)
        .expect("exact session remap");
    assert_eq!(outer.function, runmat_hir::FunctionId(701));
    assert_eq!(outer.portable_function, portable_origin);
    assert!(outer.validate_fusion_stage_projection().is_ok());
    assert!(outer.instructions.iter().any(|instruction| {
        matches!(
            instruction,
            Instr::CreateBoundFunctionHandle(runmat_hir::FunctionId(702), ..)
        )
    }));

    let prior_fingerprint = outer
        .fusion_stage_projection
        .as_ref()
        .unwrap()
        .source_fingerprint
        .clone();
    crate::bytecode::program::rebind_session_source(
        &mut outer,
        Some(crate::bytecode::BytecodeSourceRef::Program(
            runmat_types::ProgramSourceId(42),
        )),
    )
    .expect("owned source rebind");
    assert!(outer.validate_fusion_stage_projection().is_ok());
    assert_ne!(
        outer
            .fusion_stage_projection
            .as_ref()
            .unwrap()
            .source_fingerprint,
        prior_fingerprint
    );

    crate::bytecode::program::remap_session_entrypoint(&mut bytecode, &remap, &[])
        .expect("entrypoint call remap");
    assert!(bytecode.validate_fusion_stage_projection().is_ok());
    assert!(bytecode.instructions.iter().any(|instruction| {
        matches!(
            instruction,
            Instr::CallSemanticFunctionMulti(runmat_hir::FunctionId(701), ..)
        )
    }));
}

#[test]
fn malformed_selected_projection_cannot_be_resealed_by_session_remap() {
    let bytecode =
        compile_source("y=helper(1); function y=helper(x); mu=x+1; sigma=mu.*2; y=sigma-3; end");
    let mut function = bytecode
        .bound_functions
        .values()
        .find(|function| function.fusion_stage_projection.is_some())
        .expect("selected helper")
        .clone();
    let original_id = function.function;
    std::sync::Arc::make_mut(function.fusion_stage_projection.as_mut().unwrap())
        .source_fingerprint = "forged".into();
    let source = function.source_id;
    assert!(crate::bytecode::program::remap_session_function(
        &mut function,
        runmat_hir::FunctionId(701),
        source,
        &std::collections::HashMap::from([(original_id, runmat_hir::FunctionId(701))]),
    )
    .is_err());
    assert_eq!(function.function, original_id);
    assert_eq!(
        function
            .fusion_stage_projection
            .as_ref()
            .unwrap()
            .source_fingerprint,
        "forged"
    );
}
