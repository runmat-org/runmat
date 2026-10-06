use super::{execution_diagnostic, standalone_runtime};
use runmat_execution::{Digest, LanguageCompatibilityMode, ProgramEnvironment, ProgramRevision};
use runmat_runtime::context::RuntimeLanguageMode;

fn revision(mode: LanguageCompatibilityMode) -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"aot-policy-graph"),
        Digest::sha256(b"aot-policy-source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"aot-policy-runtime"),
            Digest::sha256(b"aot-policy-catalog"),
            mode,
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn standalone_policy_comes_from_program_revision() {
    for (mode, expected, extensions) in [
        (
            LanguageCompatibilityMode::RunMat,
            RuntimeLanguageMode::RunMat,
            true,
        ),
        (
            LanguageCompatibilityMode::Matlab,
            RuntimeLanguageMode::Matlab,
            false,
        ),
        (
            LanguageCompatibilityMode::Strict,
            RuntimeLanguageMode::Strict,
            false,
        ),
    ] {
        // A conflicting ambient policy must not change the compiled program.
        let _ambient = runmat_runtime::compatibility::push_runmat_extensions_enabled(!extensions);
        let program = revision(mode);
        let runtime = standalone_runtime(&program);
        assert!(runtime.service_ports().placement().is_some());
        assert!(runtime.service_ports().acceleration().is_some());
        assert_eq!(runtime.program_revision(), Some(&program));
        assert_eq!(runtime.language_mode(), expected);
        assert_eq!(runtime.runmat_extensions_enabled(), extensions);
        {
            let _scope = runtime.enter();
            assert_eq!(
                runmat_runtime::compatibility::runmat_extensions_enabled(),
                extensions
            );
            assert_eq!(
                runmat_runtime::compatibility::ensure_sparse_integer_extension_enabled(
                    "aot-policy-test"
                )
                .is_ok(),
                extensions
            );
        }
        assert_eq!(
            runmat_runtime::compatibility::runmat_extensions_enabled(),
            !extensions
        );
    }
}

#[test]
fn standalone_program_contexts_keep_independent_policies() {
    let runmat = standalone_runtime(&revision(LanguageCompatibilityMode::RunMat));
    let matlab = standalone_runtime(&revision(LanguageCompatibilityMode::Matlab));
    let _outer = runmat.enter();
    assert!(runmat_runtime::compatibility::runmat_extensions_enabled());
    {
        let _inner = matlab.enter();
        assert!(!runmat_runtime::compatibility::runmat_extensions_enabled());
        assert_eq!(matlab.language_mode(), RuntimeLanguageMode::Matlab);
    }
    assert!(runmat_runtime::compatibility::runmat_extensions_enabled());
    assert_eq!(runmat.language_mode(), RuntimeLanguageMode::RunMat);
    // Program invocation clones the context and changes service ports; neither
    // operation may replace the compiled language policy.
    let callback = runmat
        .clone()
        .with_service_ports(runmat.service_ports().clone().without_workspace());
    assert_eq!(callback.language_mode(), RuntimeLanguageMode::RunMat);
    assert!(callback.runmat_extensions_enabled());
}

#[test]
fn standalone_execution_diagnostic_keeps_original_identifier_and_message() {
    let original = runmat_runtime::build_runtime_error("aot source failure")
        .with_identifier("RunMat:test:AotSource")
        .build();
    assert_eq!(
        execution_diagnostic(
            original,
            &standalone_runtime(&revision(LanguageCompatibilityMode::Matlab)),
        ),
        "standalone native execution failed: RunMat:test:AotSource: aot source failure"
    );
}

#[test]
fn standalone_diagnostic_renders_bound_file_frame_span_and_original_cause() {
    use std::error::Error as _;

    let text = "first line\nerror('source failure')";
    let map = runmat_execution::ExecutableSourceMapV2::new(vec![
        runmat_execution::ExecutableSourceRecordV2 {
            source_id: runmat_types::ProgramSourceId(0),
            owner_identity: "aot-fixture".into(),
            relative_path: "src/main.m".into(),
            display_name: "src/main.m".into(),
            text: text.into(),
            content_revision: Digest::sha256(text.as_bytes()),
        },
    ])
    .unwrap();
    let runtime = runmat_runtime::source_context::bind_portable_source_map_v2(
        standalone_runtime(&revision(LanguageCompatibilityMode::Matlab)),
        &map,
    )
    .unwrap();
    let issued = runtime
        .program_source_id(runmat_types::ProgramSourceId(0))
        .unwrap();
    let original = runmat_runtime::build_runtime_error("source failure")
        .with_identifier("RunMat:test:AotSource")
        .with_span((11usize, 5usize).into())
        .with_call_frames(vec![runmat_runtime::CallFrame {
            function: "main".into(),
            source_id: Some(issued.0),
            span: Some((11, 5)),
        }])
        .with_source(std::io::Error::other("typed source cause"))
        .build();
    assert!(original.source().unwrap().is::<std::io::Error>());
    let rendered = execution_diagnostic(original, &runtime);
    assert!(rendered.contains("RunMat:test:AotSource: source failure"));
    assert!(rendered.contains("[source span 11..16]"));
    assert!(rendered.contains("main @ src/main.m:2:1"));
    assert!(rendered.contains("caused by: typed source cause"));
}
