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

fn ordinary_runtime(mode: LanguageCompatibilityMode) -> runmat_runtime::context::RuntimeContext {
    standalone_runtime(&revision(mode), None, None)
        .expect("ordinary standalone runtime should construct without selected inputs")
}

#[test]
fn standalone_fea_admission_is_lazy_and_keeps_each_originating_host() {
    use runmat_runtime::analysis::fea_run_admission::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Host(AtomicUsize);
    impl FeaRunAdmission for Host {
        fn admit(&self, _: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(FeaRunDenied::LicenseInvalid)
        }
    }
    let first = Arc::new(Host(AtomicUsize::new(0)));
    let second = Arc::new(Host(AtomicUsize::new(0)));
    let make = |host: Arc<dyn FeaRunAdmission>| {
        FeaRunAdmissionContext::new(host)
            .run_sync(|| ordinary_runtime(LanguageCompatibilityMode::RunMat))
    };
    let runtime_first = make(first.clone());
    let runtime_second = make(second.clone());
    assert_eq!(first.0.load(Ordering::SeqCst), 0);
    assert_eq!(second.0.load(Ordering::SeqCst), 0);
    for (runtime, expected) in [
        (&runtime_first, first.clone()),
        (&runtime_second, second.clone()),
    ] {
        let _runtime = runtime.enter();
        let actual = capture_fea_run_admission().unwrap();
        let expected: Arc<dyn FeaRunAdmission> = expected;
        assert!(Arc::ptr_eq(&actual, &expected));
    }
    let _runtime = runtime_first.enter();
    let request = FeaRunRequest::local(
        FeaRunKind::Solve,
        1000,
        Digest::sha256(b"compiled-fea-root"),
    )
    .unwrap();
    assert!(matches!(
        FeaRunScope::admit(request),
        Err(FeaRunDenied::LicenseInvalid)
    ));
    assert_eq!(first.0.load(Ordering::SeqCst), 1);
    assert_eq!(second.0.load(Ordering::SeqCst), 0);
    // Explicit invocation hosts still override the stored fallback port.
    let second_erased: Arc<dyn FeaRunAdmission> = second.clone();
    FeaRunAdmissionContext::new(second_erased.clone()).run_sync(|| {
        assert!(Arc::ptr_eq(
            &capture_fea_run_admission().unwrap(),
            &second_erased
        ));
    });
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
        let runtime = standalone_runtime(&program, None, None)
            .expect("ordinary standalone runtime should construct without selected inputs");
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
    let runmat = ordinary_runtime(LanguageCompatibilityMode::RunMat);
    let matlab = ordinary_runtime(LanguageCompatibilityMode::Matlab);
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
            &ordinary_runtime(LanguageCompatibilityMode::Matlab),
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
        ordinary_runtime(LanguageCompatibilityMode::Matlab),
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
            source_owner: None,
            diagnostic_source: None,
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
