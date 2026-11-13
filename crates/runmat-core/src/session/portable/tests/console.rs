use super::*;
use runmat_execution::console::{ConsoleCompletion, ConsoleLimits, ConsoleStream};
use runmat_runtime::context::PortableConsoleHost;
use std::rc::Rc;

fn control(native: bool) -> InvocationControl {
    #[cfg(not(target_arch = "wasm32"))]
    if native {
        return InvocationControl::default().force_generic_native();
    }
    let _ = native;
    InvocationControl::default()
}

fn session(native: bool, bytes: u64) -> (RunMatSession, Rc<PortableConsoleHost>) {
    let mut session = RunMatSession::with_options(native, false).unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    if native {
        session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
            generic_hot_threshold: 1,
            specialized_hot_threshold: 3,
            deterministic: true,
            ..Default::default()
        });
    }
    let host = Rc::new(
        PortableConsoleHost::new(ConsoleLimits {
            max_records: 20,
            max_text_bytes: bytes,
        })
        .unwrap(),
    );
    session.install_host_service(host.clone());
    (session, host)
}

#[test]
fn original_portable_console_retains_nested_writes_and_original_output_ordinals() {
    let request = request(
        "function [a,b] = outer(); fprintf(1,'first'); inner(); clc; fprintf(1,'last'); a=3; b=7; end\nfunction inner(); fprintf(2,'middle'); end",
        None, 2);
    for native in [false, true] {
        let (mut session, host) = session(native, 100);
        let response = invoke_session(&mut session, &request, &control(native));
        response.validate_against(&request).unwrap();
        assert_eq!(values(response), vec![Value::Num(3.0), Value::Num(7.0)]);
        #[cfg(not(target_arch = "wasm32"))]
        if native {
            assert!(
                session.generic_native_cache_counts().0 > 0,
                "native path must actually compile"
            );
        }
        let transcript = host.finish(true).unwrap();
        transcript.validate().unwrap();
        assert_eq!(transcript.completion(), ConsoleCompletion::Complete);
        assert_eq!(
            transcript
                .records()
                .iter()
                .map(|r| (r.stream, r.text.as_ref()))
                .collect::<Vec<_>>(),
            vec![
                (ConsoleStream::Stdout, "first"),
                (ConsoleStream::Stderr, "middle"),
                (ConsoleStream::ClearScreen, ""),
                (ConsoleStream::Stdout, "last")
            ]
        );
        assert!(block_on(
            session
                .runtime_context()
                .scope(async { runmat_runtime::console::take_thread_buffer() })
        )
        .is_empty());
    }
}

#[test]
fn caught_capture_overflow_cannot_return_portable_success() {
    let request = request(
        "function y = outer(); try; fprintf(1,'too much'); catch; end; y=7; end",
        None,
        1,
    );
    for native in [false, true] {
        let (mut session, host) = session(native, 2);
        let response = invoke_session(&mut session, &request, &control(native));
        assert!(
            matches!(response, ProgramExecutionResponse::Failure { message } if message.contains("console UTF-8 bytes"))
        );
        assert!(matches!(
            host.finish(true).unwrap().completion(),
            ConsoleCompletion::Rejected { .. }
        ));
    }
}

#[test]
fn portable_input_fails_explicitly_after_preserving_the_emitted_prefix() {
    let request = request(
        "function y = outer(); fprintf(1,'before input'); y=input('prompt','s'); end",
        None,
        1,
    );
    let (mut session, host) = session(false, 100);
    let response = invoke_session(&mut session, &request, &InvocationControl::default());
    assert!(
        matches!(response, ProgramExecutionResponse::Failure { message } if message.contains("portable execution does not permit interactive input"))
    );
    let transcript = host.finish(false).unwrap();
    assert_eq!(transcript.completion(), ConsoleCompletion::Interrupted);
    assert_eq!(transcript.records()[0].text.as_ref(), "before input");
}

#[test]
fn original_capture_controls_bind_request_identity_and_cold_parts() {
    let mut original = request("function y=outer(); y=7; end", None, 1);
    let disabled = original.canonical_digest().unwrap();
    original.console = Some(ConsoleLimits {
        max_records: 20,
        max_text_bytes: 100,
    });
    original.validate_for_portable_host().unwrap();
    let digest = original.canonical_digest().unwrap();
    assert_ne!(disabled, digest);
    assert_eq!(digest, original.as_borrowed().canonical_digest().unwrap());
    assert_eq!(
        serde_json::to_vec(&original).unwrap(),
        serde_json::to_vec(&original.as_borrowed()).unwrap()
    );
    let cold: ProgramExecutionRequest =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    assert_eq!(cold.canonical_digest().unwrap(), digest);
    let descriptor = runmat_execution_artifact::ProgramExecutionDescriptor {
        schema_version: original.schema_version,
        recipe: original.recipe.clone(),
        artifact: original.artifact.clone(),
        callable: original.callable.clone(),
        requested_outputs: original.requested_outputs,
        environment: original.environment.clone(),
        console: original.console,
    };
    let rebuilt = ProgramExecutionRequest::from_parts(
        descriptor,
        runmat_execution_artifact::ProgramExecutionInputs {
            schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            context: original.context.clone(),
            objects: Vec::new(),
            arguments: original.arguments.clone(),
        },
    )
    .unwrap();
    assert_eq!(rebuilt.canonical_digest().unwrap(), digest);
    original.console.as_mut().unwrap().max_text_bytes += 1;
    assert_ne!(original.canonical_digest().unwrap(), digest);
}

#[test]
fn ordinary_executable_invocation_rejects_sealed_host_without_running_source() {
    let original = request("function y=outer(); y=7; end", None, 1);
    let envelope = original.artifact.executable_unit().unwrap().unwrap();
    let unit = crate::ExecutableUnit::from_portable_envelope(&envelope).unwrap();
    for native in [false, true] {
        let (mut session, host) = session(native, 100);
        host.finish(true).unwrap();
        let result = block_on(session.invoke_executable(
            &unit,
            crate::ProcedureInvocation {
                target: crate::ProcedureTarget::SemanticFunction(
                    envelope.manifest.identity.entrypoint_function.unwrap(),
                ),
                arguments: crate::ProcedureArguments::Source(Vec::new()),
                requested_outputs: 1,
            },
            &control(native),
        ));
        assert!(result.is_err());
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(session.generic_native_cache_counts().0, 0);
    }
}
