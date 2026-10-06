use super::*;
use crate::{Bytecode, FunctionBytecode, FunctionRegistry, Instr};
use runmat_hir::FunctionId;
use runmat_value::Value;

#[test]
fn scripts_preserve_empty_outputs_and_never_fabricate_ans() {
    let empty = || Bytecode::with_instructions(vec![Instr::Return], 0);
    let response = futures::executor::block_on(execute_script_bytecode(empty(), 0));
    assert_eq!(
        response,
        ProgramExecutionResponse::from_success_outputs(vec![], vec![])
    );
    for requested in [1, 2] {
        assert!(matches!(
            futures::executor::block_on(execute_script_bytecode(empty(), requested)),
            ProgramExecutionResponse::Failure { .. }
        ));
    }
    let mut unassigned = Bytecode::with_instructions(vec![Instr::Return], 1);
    unassigned.var_names.insert(0, "ans".into());
    assert!(matches!(
        futures::executor::block_on(execute_script_bytecode(unassigned, 1)),
        ProgramExecutionResponse::Failure { .. }
    ));
    let mut bytecode = Bytecode::with_instructions(
        vec![Instr::LoadConst(42.0), Instr::StoreVar(0), Instr::Return],
        1,
    );
    bytecode.var_names.insert(0, "ans".into());
    let ProgramExecutionResponse::Success { value } =
        futures::executor::block_on(execute_script_bytecode(bytecode, 1))
    else {
        panic!("actual ans must survive");
    };
    assert_eq!(
        runmat_runtime::execution::value_codec::decode_inline_value(&value).unwrap(),
        Value::Num(42.0)
    );
}

#[test]
fn original_function_outputs_preserve_zero_one_and_multiple_ordinals() {
    let function = FunctionBytecode {
        function: FunctionId(7),
        display_name: "original_outputs".into(),
        instructions: vec![
            Instr::LoadConst(13.0),
            Instr::StoreVar(0),
            Instr::LoadConst(29.0),
            Instr::StoreVar(1),
            Instr::Return,
        ],
        var_count: 2,
        output_slots: vec![0, 1],
        ..Default::default()
    };
    let registry = FunctionRegistry::new(std::collections::HashMap::from([(
        function.function,
        function,
    )]));
    for requested_outputs in [0, 1, 2] {
        let revision = ProgramRevision::new(
            Digest::sha256(b"outputs-graph"),
            Digest::sha256(b"outputs-source"),
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "7".into(),
            outputs: OutputContract { requested_outputs },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("portable-output-test"),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: vec![],
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV1,
            serde_json::to_vec(&registry).unwrap(),
        )
        .unwrap();
        let request = ProgramExecutionRequest {
            schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe,
            artifact,
            function: 7,
            arguments: vec![],
            requested_outputs,
        };
        let response = futures::executor::block_on(execute_program_request(request.clone()));
        response.validate_against(&request).unwrap();
        let outputs = match response {
            ProgramExecutionResponse::Success { value } => vec![value],
            ProgramExecutionResponse::ExternalizedSuccess {
                outputs,
                result_objects,
            } => {
                assert!(result_objects.is_empty());
                outputs
            }
            other => panic!("original function failed: {other:?}"),
        };
        let value = runmat_runtime::execution::value_codec::decode_inline_outputs(
            &outputs,
            usize::from(requested_outputs),
        )
        .unwrap();
        assert_eq!(
            value,
            match requested_outputs {
                0 => Value::OutputList(vec![]),
                1 => Value::Num(13.0),
                _ => Value::OutputList(vec![Value::Num(13.0), Value::Num(29.0)]),
            }
        );
    }
}
