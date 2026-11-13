use super::*;
use runmat_runtime::context::IsolatedMemoryPreferenceService;
use std::fs;

struct RestoreCwd(std::path::PathBuf);
impl Drop for RestoreCwd {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.0).expect("restore working directory");
    }
}

#[test]
fn failing_cleanup_still_runs_real_mex_at_exit_before_host_closes() {
    let _cwd_lock = runmat_filesystem::provider_override_lock();
    let directory = tempfile::tempdir().expect("temporary MEX lifecycle fixture");
    let _cwd = RestoreCwd(std::env::current_dir().expect("read working directory"));
    // Ordinary source admits MEX bytes from its current search-path choice;
    // installing metadata alone is not a selected-unit artifact receipt.
    std::env::set_current_dir(directory.path()).expect("select fixture directory");
    let source = directory.path().join("close_probe.c");
    let marker = directory.path().join("at_exit.marker");
    let marker_literal = marker
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    fs::write(
        &source,
        format!(
            r#"
#include "mex.h"
#include <stdio.h>
static void probe_at_exit(void) {{
    FILE *file = fopen("{marker_literal}", "w");
    if (file != NULL) {{ fputs("called", file); fclose(file); }}
}}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {{
    (void)nrhs; (void)prhs;
    mexAtExit(probe_at_exit);
    if (nlhs != 0) plhs[0] = mxCreateDoubleScalar(1.0);
}}
"#
        ),
    )
    .expect("write C MEX fixture");
    let artifact = runmat_mex::MexBuild::new(&source, directory.path())
        .compile()
        .expect("compile C MEX fixture");
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session
        .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
        .expect("isolated settings");
    session
        .install_mex_artifact(&artifact.module)
        .expect("install exact MEX fixture");
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<cleanup-before-mex-shutdown>".into(),
            text: "close_probe(); cleanupObj = onCleanup(@() error('Cleanup:Probe','expected cleanup failure'));".into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    runtime.block_on(async {
        let outcome = session
            .execute_request(request)
            .await
            .result
            .expect("fixture invocation and cleanup registration");
        assert!(
            !outcome.has_errors(),
            "fixture invocation must reach successful cleanup registration: {:?}",
            outcome.diagnostics
        );
        assert!(
            outcome
                .workspace_delta
                .upserts
                .iter()
                .any(|upsert| upsert.name.0 == "cleanupObj"),
            "fixture invocation must publish the cleanup owner"
        );
        drop(outcome);
        let cleanup = match session.close_gracefully_retirement().await {
            SessionRetirement::Retired(Err(error)) => error,
            SessionRetirement::Retired(Ok(())) => {
                panic!("onCleanup callback must fail after host retirement")
            }
            SessionRetirement::Retained(error) => {
                panic!("MEX host unexpectedly retained after cleanup callback: {error}")
            }
        };
        assert!(
            cleanup.to_string().contains("expected cleanup failure"),
            "cleanup diagnostic remains primary: {cleanup}"
        );
    });
    assert_eq!(
        fs::read_to_string(marker).expect("mexAtExit wrote marker before callback host closed"),
        "called"
    );
}

#[test]
fn cleanup_released_by_cpp_engine_unload_reports_its_original_error() {
    let _cwd_lock = runmat_filesystem::provider_override_lock();
    let directory = tempfile::tempdir().expect("temporary retained-engine lifecycle fixture");
    let _cwd = RestoreCwd(std::env::current_dir().expect("read working directory"));
    std::env::set_current_dir(directory.path()).expect("select fixture directory");
    let source = directory.path().join("cpp_probe.cpp");
    let marker = directory.path().join("engine_unload.marker");
    let marker_literal = marker
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    fs::write(
        &source,
        format!(
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <cstdio>
class MexFunction : public matlab::mex::Function {{
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {{
        (void)outputs; (void)inputs;
        // Function caches this explicit engine-context owner until unload.
        (void)getEngine();
    }}
    ~MexFunction() override {{
        FILE *file = fopen("{marker_literal}", "a");
        if (file != NULL) {{ fputs("unloaded\n", file); fclose(file); }}
    }}
}};
"#
        ),
    )
    .expect("write retained-engine C++ MEX fixture");
    let artifact = runmat_mex::MexBuild::new(&source, directory.path())
        .compile()
        .expect("compile retained-engine C++ MEX fixture");
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session
        .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
        .expect("isolated settings");
    session
        .install_mex_artifact(&artifact.module)
        .expect("install exact retained-engine MEX fixture");
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<cleanup-after-engine-unload>".into(),
            // The cached engine legitimately retains this invocation's
            // cleanup owner, so the initial workspace drain cannot run it.
            text: "cleanupObj = onCleanup(@() error('Cleanup:RetainedEngine','expected retained engine cleanup failure')); cpp_probe();".into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    runtime.block_on(async {
        let outcome = session
            .execute_request(request)
            .await
            .result
            .expect("cleanup registration and retained-engine invocation");
        assert!(
            !outcome.has_errors(),
            "fixture must register cleanup and acquire its engine context: {:?}",
            outcome.diagnostics
        );
        assert!(
            outcome
                .workspace_delta
                .upserts
                .iter()
                .any(|upsert| upsert.name.0 == "cleanupObj"),
            "fixture must publish the cleanup owner"
        );
        drop(outcome);
        assert!(
            !marker.exists(),
            "engine owner must remain live until close"
        );
        let cleanup = match session.close_gracefully_retirement().await {
            SessionRetirement::Retired(Err(error)) => error,
            SessionRetirement::Retired(Ok(())) => {
                panic!("cleanup exposed by engine unload must report its failure")
            }
            SessionRetirement::Retained(error) => {
                panic!("retained-engine MEX host unexpectedly remained live: {error}")
            }
        };
        assert_eq!(cleanup.identifier(), Some("Cleanup:RetainedEngine"));
        assert!(
            cleanup
                .to_string()
                .contains("expected retained engine cleanup failure"),
            "original cleanup diagnostic must survive foreign-host retirement: {cleanup}"
        );
    });
    let expected: &[u8] = if cfg!(windows) {
        b"unloaded\r\n"
    } else {
        b"unloaded\n"
    };
    assert_eq!(
        fs::read(marker).expect("retained engine gateway unloaded"),
        expected,
        "gateway destruction must happen exactly once"
    );
}

#[test]
fn cleanup_republished_into_base_during_close_reports_its_original_error() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session
        .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
        .expect("isolated settings");
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<cleanup-repopulates-base-during-close>".into(),
            // The first callback publishes a new canonical base owner. It
            // must be retired while base authority and cleanup host are live,
            // not dropped for the first time by final workspace closure.
            text: r#"
cleanupObj = onCleanup(@publish_cleanup_during_close);
function publish_cleanup_during_close()
    assignin('base','repopulatedCleanup',onCleanup(@() error('Cleanup:Repopulated','expected republished cleanup failure')));
    error('Cleanup:First','expected first cleanup failure');
end
"#.into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    runtime.block_on(async {
        let outcome = session
            .execute_request(request)
            .await
            .result
            .expect("register workspace-publishing cleanup callback");
        assert!(
            !outcome.has_errors(),
            "fixture must register its first cleanup callback: {:?}",
            outcome.diagnostics
        );
        assert!(
            outcome
                .workspace_delta
                .upserts
                .iter()
                .any(|upsert| upsert.name.0 == "cleanupObj"),
            "fixture must publish the first cleanup owner"
        );
        drop(outcome);
        let cleanup = match session.close_gracefully_retirement().await {
            SessionRetirement::Retired(Err(error)) => error,
            SessionRetirement::Retired(Ok(())) => {
                panic!("cleanup republished during close must not be abandoned")
            }
            SessionRetirement::Retained(error) => {
                panic!("pure cleanup fixture unexpectedly retained a foreign host: {error}")
            }
        };
        assert_eq!(cleanup.identifier(), Some("Cleanup:First"));
        assert!(
            cleanup
                .to_string()
                .contains("expected first cleanup failure"),
            "first callback diagnostic must remain primary: {cleanup}"
        );
        let mut source = std::error::Error::source(&cleanup);
        let mut republished_failure = false;
        while let Some(error) = source {
            republished_failure |= error
                .to_string()
                .contains("expected republished cleanup failure");
            source = error.source();
        }
        assert!(
            republished_failure,
            "republished cleanup must execute and retain its failure under the original error"
        );
    });
}

#[test]
fn compiled_mex_runtests_child_preserves_logical_callback_ancestry() {
    let _cwd_lock = runmat_filesystem::provider_override_lock();
    let directory = tempfile::tempdir().expect("temporary MEX ancestry fixture");
    let _cwd = RestoreCwd(std::env::current_dir().expect("read working directory"));
    std::env::set_current_dir(directory.path()).expect("select fixture directory");
    let entered = directory.path().join("inner-entered.marker");
    let effect = directory.path().join("callback-effect.marker");
    let rejected = directory.path().join("depth-rejected.marker");
    let escaped_c_path = |path: &std::path::Path| {
        path.to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    };
    let escaped_matlab_path = effect.to_string_lossy().replace('\'', "''");

    let outer_source = directory.path().join("outer_ancestry.c");
    fs::write(
        &outer_source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    mxArray *test = mxCreateString("testNested.m");
    int status = mexCallMATLAB(0, NULL, 1, &test, "runtests");
    mxDestroyArray(test);
    if (nlhs != 0) plhs[0] = mxCreateDoubleScalar(status == 0 ? 1.0 : 0.0);
}
"#,
    )
    .expect("outer C MEX source");
    let inner_source = directory.path().join("inner_ancestry.c");
    fs::write(
        &inner_source,
        format!(
            r#"
#include "mex.h"
#include <stdio.h>
#include <string.h>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {{
    (void)nrhs; (void)prhs;
    FILE *entered = fopen("{}", "a");
    if (entered != NULL) {{ fputs("1", entered); fclose(entered); }}
    mxArray *effect = NULL;
    mxArray *error = mexCallMATLABWithTrap(1, &effect, 0, NULL, "effect_probe");
    int succeeded = error == NULL;
    if (error != NULL) {{
        mxArray *field = mxGetField(error, 0, "identifier");
        char *identifier = field != NULL ? mxArrayToString(field) : NULL;
        int expected = identifier != NULL && strcmp(identifier, "RunMat:MEX:CallbackDepth") == 0;
        if (identifier != NULL) mxFree(identifier);
        mxDestroyArray(error);
        if (!expected) mexErrMsgTxt("unexpected inner callback failure");
        FILE *rejected = fopen("{}", "a");
        if (rejected != NULL) {{ fputs("1", rejected); fclose(rejected); }}
    }}
    if (nlhs != 0) plhs[0] = mxCreateDoubleScalar(succeeded ? 1.0 : 0.0);
    if (effect != NULL) mxDestroyArray(effect);
}}
"#,
            escaped_c_path(&entered),
            escaped_c_path(&rejected)
        ),
    )
    .expect("inner C MEX source");
    fs::write(
        directory.path().join("testNested.m"),
        "assert(inner_ancestry() == 1);",
    )
    .expect("child test source");
    fs::write(
        directory.path().join("effect_probe.m"),
        format!(
            "function y = effect_probe()\nfid = fopen('{escaped_matlab_path}', 'a');\nassert(fid > 0);\nfprintf(fid, '1');\nfclose(fid);\ny = 1;\nend\n"
        ),
    )
    .expect("callback effect source");

    let outer = runmat_mex::MexBuild::new(&outer_source, directory.path())
        .compile()
        .expect("compile outer C MEX");
    let _inner = runmat_mex::MexBuild::new(&inner_source, directory.path())
        .compile()
        .expect("compile inner C MEX");
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("host reactor");

    for parent_depth in [62, 63] {
        let mut session = RunMatSession::with_options(false, false).expect("parent session");
        session
            .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
            .expect("install parent user-path settings");
        assert!(session
            .runtime_context
            .clone()
            .with_test_mex_callback_ancestry_depth(65)
            .is_none());
        session.runtime_context = session
            .runtime_context
            .clone()
            .with_test_mex_callback_ancestry_depth(parent_depth)
            .expect("checked fixture depth");
        session
            .install_mex_artifact(&outer.module)
            .expect("install outer MEX artifact");
        let request = crate::abi::ExecutionRequest::for_source(
            crate::abi::SourceInput::Text {
                name: "<compiled-mex-ancestry>".into(),
                text: "outer_status = outer_ancestry();".into(),
            },
            session.compat_mode(),
            crate::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        let result = {
            let _entered = host.enter();
            host.block_on(session.execute_request(request)).result
        };
        if parent_depth == 62 {
            result.expect("parent62 → outer63 → inner64 callback succeeds");
        }
        assert_eq!(
            fs::read_to_string(&entered).expect("child MEX entered"),
            if parent_depth == 62 { "1" } else { "11" },
            "the testing child must execute the compiled inner MEX in both cases"
        );
        assert_eq!(
            fs::read_to_string(&effect).expect("first admitted callback wrote marker"),
            "1",
            "a 65th callback must be rejected before executing effect_probe"
        );
        if parent_depth == 63 {
            assert_eq!(
                fs::read_to_string(&rejected).expect("exact depth rejection marker"),
                "1",
                "the inner callback must report RunMat:MEX:CallbackDepth"
            );
        } else {
            assert!(
                !rejected.exists(),
                "the admitted callback must not report an error"
            );
        }
        let _entered = host.enter();
        host.block_on(session.close_gracefully())
            .expect("fixture session retires its compiled MEX host");
    }
}
