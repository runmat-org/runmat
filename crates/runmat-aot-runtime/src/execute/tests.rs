use super::standalone_runtime;
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
