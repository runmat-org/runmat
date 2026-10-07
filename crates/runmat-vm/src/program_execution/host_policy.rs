//! Exercise policy through the original legacy worker entry, including child isolation.
use super::*;
use crate::{Bytecode, Instr};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::ProgramBuildRecipe;

fn request(instructions: Vec<Instr>) -> ProgramExecutionRequest {
    let revision = ProgramRevision::new(
        Digest::sha256(b"legacy-host-graph"),
        Digest::sha256(b"legacy-host-source"),
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
        entrypoint: "script".into(),
        outputs: OutputContract {
            requested_outputs: 0,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("legacy-host-policy-test"),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: vec![],
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterScriptV1,
        serde_json::to_vec(&Bytecode::with_instructions(instructions, 0)).unwrap(),
    )
    .unwrap();
    ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_V3,
        recipe,
        artifact,
        function: 0,
        arguments: vec![],
        requested_outputs: 0,
    }
}

#[test]
fn legacy_worker_cannot_consume_interactive_parent_input() {
    let parent = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let _scope = parent.enter();
    runmat_runtime::interaction::push_queued_response(Ok(
        runmat_runtime::interaction::InteractionResponse::Line("original parent input".into()),
    ));
    let response = futures::executor::block_on(execute_program_request(request(vec![
        Instr::LoadString("prompt".into()),
        Instr::LoadString("s".into()),
        Instr::CallBuiltinMulti("input".into(), 2, 1),
        Instr::Pop,
        Instr::Return,
    ])));
    let ProgramExecutionResponse::Failure { message } = response else {
        panic!("a portable worker must reject interactive input");
    };
    assert!(
        message.contains("portable execution does not permit interactive input"),
        "{message}"
    );
    assert_eq!(
        futures::executor::block_on(runmat_runtime::interaction::request_line_async("", false))
            .unwrap(),
        "original parent input",
    );
}
