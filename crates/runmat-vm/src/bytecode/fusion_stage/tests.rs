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
    session.rebind_identity(runmat_hir::FunctionId(701), ProgramFunctionId(901));
    assert!(session.validate_fusion_stage_projection().is_ok());
    assert!(session
        .fusion_stage_projection
        .as_ref()
        .unwrap()
        .statements
        .iter()
        .all(|row| row.contract.point.function == ProgramFunctionId(901)));
    assert_eq!(session.portable_function, original.portable_function);
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
    );
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
    );
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
