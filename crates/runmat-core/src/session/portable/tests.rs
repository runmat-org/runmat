use super::*;
use crate::{CompatMode, ExecutableSource};
use futures::executor::block_on;
use runmat_execution::OutputContract;
use runmat_execution_artifact::*;
use runmat_value::Value;

mod console;

fn request(text: &str, selected: Option<&str>, outputs: u16) -> ProgramExecutionRequest {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = ExecutableSource::frozen(
        "portable-session",
        "outer.m",
        directory.path().join("outer.m").to_string_lossy(),
        text,
    );
    let unit = block_on(session.compile_executable_unit(source, None)).unwrap();
    let envelope = unit.portable_envelope_for(selected).unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: envelope.manifest.identity.entrypoint.clone(),
        outputs: OutputContract {
            requested_outputs: outputs,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("session-portable-test"),
        interop: envelope.manifest.interop.clone(),
        accelerators: Vec::new(),
        foreign_policy: None,
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        callable: runmat_execution::ProgramCallable::semantic(
            envelope.manifest.identity.entrypoint_function.unwrap(),
            None,
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        artifact: ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::ExecutableUnitV4,
            envelope.canonical_bytes().unwrap(),
        )
        .unwrap(),
        recipe,
        arguments: Vec::new(),
        requested_outputs: outputs,
    }
}

fn invoke_session(
    session: &mut RunMatSession,
    request: &ProgramExecutionRequest,
    control: &InvocationControl,
) -> ProgramExecutionResponse {
    let result = (|| {
        request
            .validate_for_portable_host()
            .map_err(|error| error.to_string())?;
        let envelope = request
            .artifact
            .executable_unit()
            .map_err(|error| error.to_string())?
            .unwrap();
        let arguments = request
            .arguments
            .iter()
            .map(runmat_runtime::execution::value_codec::decode_inline_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        let value = block_on(session.invoke_portable_unit(
            &envelope,
            arguments,
            usize::from(request.requested_outputs),
            control,
        ))?;
        runmat_runtime::execution::value_codec::encode_inline_sequence(
            value.value(),
            usize::from(request.requested_outputs),
        )
        .map_err(|error| error.to_string())
    })();
    match result {
        Ok(outputs) => ProgramExecutionResponse::from_success_outputs(outputs, vec![]),
        Err(message) => ProgramExecutionResponse::Failure { message },
    }
}

fn invoke(request: &ProgramExecutionRequest) -> ProgramExecutionResponse {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    invoke_session(&mut session, request, &InvocationControl::default())
}

fn values(response: ProgramExecutionResponse) -> Vec<Value> {
    let outputs = match response {
        ProgramExecutionResponse::Success { outputs } => outputs,
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => {
            assert!(result_objects.is_empty());
            outputs
        }
        other => panic!("portable execution failed: {other:?}"),
    };
    outputs
        .iter()
        .map(|value| runmat_runtime::execution::value_codec::decode_inline_value(value).unwrap())
        .collect()
}

#[test]
fn exact_nonprimary_procedure_preserves_captures_and_output_ordinals() {
    let mut request = request(
        concat!(
            "function y = outer(); y = 999; end\n",
            "function [y,z] = selected(x)\ng = @(t) helper(t) + x;\n",
            "y = g(3); z = helper(x); end\n",
            "function y = helper(x); y = 2*x; end\n",
        ),
        Some("selected"),
        2,
    );
    assert_ne!(request.callable.semantic_function().unwrap().0, 0);
    request.arguments =
        vec![
            runmat_runtime::execution::value_codec::encode_inline_value(&Value::Num(10.0)).unwrap(),
        ];
    let cold = serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
    let response = invoke(&cold);
    response.validate_against(&request).unwrap();
    assert_eq!(values(response), vec![Value::Num(16.0), Value::Num(20.0)]);
}

#[test]
fn script_outputs_require_actual_ans_assignment_not_last_storage_slot() {
    let assigned = request("ans = 42; unrelated = 999;", None, 1);
    assert_eq!(values(invoke(&assigned)), vec![Value::Num(42.0)]);
    for text in [
        "unrelated = 999;",
        "if false; ans = 42; end; unrelated = 999;",
    ] {
        let absent = request(text, None, 1);
        assert!(matches!(
            invoke(&absent),
            ProgramExecutionResponse::Failure { .. }
        ));
        let no_outputs = request(text, None, 0);
        assert!(values(invoke(&no_outputs)).is_empty());
    }
    assert!(matches!(
        invoke(&request("ans = 42;", None, 2)),
        ProgramExecutionResponse::Failure { .. }
    ));
}

#[test]
fn original_environment_and_cancellation_are_enforced_before_execution() {
    let request = request("ans = 42;", None, 1);
    let mut session = RunMatSession::with_options(false, false).unwrap();
    session.set_compat_mode(CompatMode::Strict);
    assert!(matches!(
        invoke_session(&mut session, &request, &InvocationControl::default()),
        ProgramExecutionResponse::Failure { .. }
    ));
    session.set_compat_mode(CompatMode::Matlab);
    let cancellation = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    assert!(matches!(
        invoke_session(
            &mut session,
            &request,
            &InvocationControl::default().with_cancellation(
                runmat_execution::CancellationToken::from_shared(cancellation)
            )
        ),
        ProgramExecutionResponse::Failure { .. }
    ));
}

#[test]
fn complete_units_execute_through_owned_vm_and_session_hosts() {
    let original = request("ans = 42;", None, 1);
    // The VM entry owns workspace, lifecycle and local-task supervision; it
    // is not the former bare-context partial embedding.
    let executed = block_on(runmat_vm::execute_program_request(original.clone()));
    executed.validate_against(&original).unwrap();
    assert_eq!(values(executed), vec![Value::Num(42.0)]);
    assert_eq!(values(invoke(&original)), vec![Value::Num(42.0)]);
}

mod environment;
