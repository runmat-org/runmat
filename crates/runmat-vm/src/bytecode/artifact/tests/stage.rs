use super::*;

#[cfg(feature = "native-accel")]
#[test]
fn pre_stage_field_canonical_script_and_registry_remain_decodable() {
    let bytecode = Bytecode::with_instructions(vec![Instr::Return], 0);
    let script_value = serde_json::to_value(&bytecode).unwrap();
    assert!(script_value["fusion_metadata"]
        .as_object()
        .unwrap()
        .get("stage_projection")
        .is_none());
    let prefield_script = super::super::envelope::encode_payload(
        InterpreterPayloadForm::ScriptV2,
        SCRIPT_V2_PREAMBLE,
        &[("bytecode", BYTECODE_SCHEMA_VERSION)],
        &script_value,
    )
    .unwrap();
    assert!(decode_interpreter_script_v2(&prefield_script).is_ok());

    let registry = FunctionRegistry::new(HashMap::from([(
        FunctionId(1),
        FunctionBytecode {
            function: FunctionId(1),
            instructions: vec![Instr::Return],
            catch_scopes: vec![None],
            ..Default::default()
        },
    )]));
    let registry_value = serde_json::to_value(&registry).unwrap();
    assert!(registry_value["functions"]
        .as_object()
        .unwrap()
        .values()
        .all(|function| function
            .as_object()
            .unwrap()
            .get("fusion_stage_projection")
            .is_none()));
    let prefield_registry = super::super::envelope::encode_payload(
        InterpreterPayloadForm::ProgramV2,
        b"runmat-interpreter-program-v2\n",
        &[
            ("bytecode", BYTECODE_SCHEMA_VERSION),
            ("function_registry", FUNCTION_REGISTRY_SCHEMA_VERSION),
        ],
        &registry_value,
    )
    .unwrap();
    assert!(decode_interpreter_program_v2(&prefield_registry).is_ok());
}

#[cfg(feature = "native-accel")]
#[test]
fn staged_statement_artifacts_reject_forged_pc_owner_and_duplicates() {
    let ast = runmat_parser::parse("a=1; mu=a+1; sigma=mu.*2; out=sigma-3;").unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    let bytecode =
        crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap();
    assert!(bytecode.fusion_metadata.stage_projection.is_some());
    let encoded = encode_interpreter_script_v2(&bytecode).unwrap();
    assert!(decode_interpreter_script_v2(&encoded).is_ok());

    let canonical = serde_json::to_value(&bytecode).unwrap();
    for mutation in 0..3 {
        let mut forged = canonical.clone();
        let projection = &mut forged["fusion_metadata"]["stage_projection"];
        match mutation {
            0 => projection["statements"][0]["pc"]["end"] = serde_json::json!(usize::MAX),
            1 => {
                projection["statements"][0]["contract"]["point"]["function"] =
                    serde_json::json!(u32::MAX)
            }
            _ => {
                let duplicate = projection["statements"][0].clone();
                projection["statements"]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
        }
        let artifact = super::super::envelope::encode_payload(
            InterpreterPayloadForm::ScriptV2,
            SCRIPT_V2_PREAMBLE,
            &[("bytecode", BYTECODE_SCHEMA_VERSION)],
            &forged,
        )
        .unwrap();
        assert!(decode_interpreter_script_v2(&artifact).is_err());
    }
}

#[cfg(feature = "native-accel")]
#[test]
fn function_artifact_rejects_forged_selected_stage_source() {
    let ast = runmat_parser::parse(
        "a=helper(1); function y=helper(x); mu=x+1; sigma=mu.*2; y=sigma-3; end",
    )
    .unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    let bytecode =
        crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap();
    let registry = FunctionRegistry::new(bytecode.bound_functions);
    assert!(registry
        .functions
        .values()
        .any(|function| function.fusion_stage_projection.is_some()));
    let encoded = encode_interpreter_program_v2(&registry).unwrap();
    assert!(decode_interpreter_program_v2(&encoded).is_ok());

    let mut forged = serde_json::to_value(&registry).unwrap();
    let function = forged["functions"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .find(|function| !function["fusion_stage_projection"].is_null())
        .unwrap();
    function["fusion_stage_projection"]["source_fingerprint"] =
        serde_json::json!("stale-selected-stream");
    let artifact = super::super::envelope::encode_payload(
        InterpreterPayloadForm::ProgramV2,
        b"runmat-interpreter-program-v2\n",
        &[
            ("bytecode", BYTECODE_SCHEMA_VERSION),
            ("function_registry", FUNCTION_REGISTRY_SCHEMA_VERSION),
        ],
        &forged,
    )
    .unwrap();
    assert!(decode_interpreter_program_v2(&artifact).is_err());
}
