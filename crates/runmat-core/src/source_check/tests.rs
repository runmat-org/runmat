use super::*;
use runmat_filesystem::{FsProvider, MemoryFsProvider};
use std::sync::Arc;

#[test]
fn ordinary_check_matrix_uses_project_policy_unreached_roots_and_typed_function_selection() {
    let _serial = runmat_filesystem::provider_override_lock();
    let fs = Arc::new(MemoryFsProvider::with_current_dir("/project"));
    fs.write_project_path("/project/runmat.toml", b"[package]\nname=\"check-fixture\"\n[sources]\nroots=[\"src\"]\n[runtime.language]\ncompat=\"matlab\"\n").unwrap();
    fs.write_project_path("/project/src/main.m", b"answer=helper(41);\n")
        .unwrap();
    fs.write_project_path(
        "/project/src/helper.m",
        b"function y=helper(x)\ny=x+1;\nend\n",
    )
    .unwrap();
    fs.write_project_path(
        "/project/src/unused.m",
        b"% unused source must still be checked\nx = ;\n",
    )
    .unwrap();
    fs.write_project_path(
        "/project/src/app/server.m",
        b"function y=main(x)\ny=x+1;\nend\n",
    )
    .unwrap();
    fs.write_project_path(
        "/project/src/private/secret.m",
        b"function y=secret(x)\ny=x+1;\nend\n",
    )
    .unwrap();
    let provider: Arc<dyn FsProvider> = fs;
    let _provider = runmat_filesystem::replace_provider(provider.clone());
    for (target, kind) in [
        (
            serde_json::json!({"kind":"script","path":"/project/src/main.m"}),
            "script",
        ),
        (
            serde_json::json!({"kind":"function","path":"/project/src/helper.m","function":"helper"}),
            "function",
        ),
        (
            serde_json::json!({"kind":"project","path":"/project/runmat.toml"}),
            "project",
        ),
        (
            serde_json::json!({"kind":"function","path":"/project/src/app/server.m","function":"main"}),
            "function",
        ),
        (
            serde_json::json!({"kind":"function","path":"/project/src/private/secret.m","function":"secret"}),
            "function",
        ),
    ] {
        let request: SourceCheckRequest =
            serde_json::from_value(serde_json::json!({"target":target,
            "userSettingsJson":"{\"runtime\":{\"language\":{\"compat\":\"strict\"}}}"}))
            .unwrap();
        let report = futures::executor::block_on(check_source(request)).unwrap();
        assert_eq!(report.document_kind, kind);
        assert_eq!(report.language_compatibility, LanguageCompatMode::Matlab);
        assert!(report.project_revision.is_some());
        if kind == "project" {
            assert_eq!(report.documents.len(), 5);
            assert_eq!(report.outcome, CheckOutcome::Failed);
            let unused = report
                .documents
                .iter()
                .find(|d| d.path.ends_with("unused.m"))
                .unwrap();
            let syntax = unused
                .diagnostics
                .iter()
                .find(|d| d.code == "RunMat:ParseError")
                .unwrap();
            assert!(matches!(
                syntax.stage,
                runmat_static_analysis::project::check_report::CheckStage::Syntax
            ));
            assert_eq!(
                syntax.location.path.as_deref(),
                Some(std::path::Path::new("/project/src/unused.m"))
            );
        } else {
            assert_eq!(report.documents.len(), 1);
            assert!(!report.documents[0]
                .diagnostics
                .iter()
                .any(|d| d.code == "RunMat:Check:FunctionNotFound"));
        }
        assert!(Arc::ptr_eq(
            &runmat_filesystem::current_provider(),
            &provider
        ));
    }
    let selected: SourceCheckRequest = serde_json::from_value(serde_json::json!({
        "target":{"kind":"function","path":"/project/src/helper.m","function":"not_helper"}
    }))
    .unwrap();
    let missing = futures::executor::block_on(check_source(selected)).unwrap();
    assert!(missing.documents[0]
        .diagnostics
        .iter()
        .any(|d| d.code == "RunMat:Check:FunctionNotFound"));
    let companion: SourceCheckRequest = serde_json::from_value(serde_json::json!({
        "target":{"kind":"function","path":"/project/src/main.m","function":"helper"}
    }))
    .unwrap();
    let wrong_primary = futures::executor::block_on(check_source(companion)).unwrap();
    assert!(wrong_primary.documents[0]
        .diagnostics
        .iter()
        .any(|d| d.code == "RunMat:Check:FunctionNotFound"));
    let malformed: SourceCheckRequest = serde_json::from_value(serde_json::json!({
        "target":{"kind":"function","path":"/project/src/unused.m","function":"missing"}
    }))
    .unwrap();
    let syntax = futures::executor::block_on(check_source(malformed)).unwrap();
    assert!(syntax.documents[0]
        .diagnostics
        .iter()
        .any(|d| d.code == "RunMat:ParseError"));
    assert!(!syntax.documents[0]
        .diagnostics
        .iter()
        .any(|d| d.code == "RunMat:Check:FunctionNotFound"));
    // A native embedding resolves its actual host layers at this same captured
    // boundary. It cannot accidentally fall back to browser defaults or select
    // a second project from the process CWD. Exercise the existing canonical
    // explicit-file layer above the captured project's MATLAB policy.
    let host_settings = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    std::fs::write(
        host_settings.path(),
        "[runtime.language]\ncompat='strict'\n",
    )
    .unwrap();
    let native: SourceCheckRequest = serde_json::from_value(serde_json::json!({
        "target":{"kind":"script","path":"/project/src/main.m"}
    }))
    .unwrap();
    let native_report =
        futures::executor::block_on(check_source_with_runtime_settings(native, |settings| {
            use runmat_config::runtime::{ConfigLoadOptions, ConfigLoader, UserSettingsSource};
            let project = settings
                .project
                .expect("Core must retain the captured project");
            assert!(project.source.contains("check-fixture"));
            ConfigLoader::load_with_options(&ConfigLoadOptions {
                user: UserSettingsSource::Disabled,
                project_document: Some((project.source, project.format)),
                explicit_file: Some(host_settings.path().into()),
                policy: settings.policy,
                environment: false,
                ..Default::default()
            })
            .map_err(|error| error.to_string())
        }))
        .unwrap();
    assert_eq!(
        native_report.language_compatibility,
        LanguageCompatMode::Strict
    );
    assert_eq!(native_report.project_revision, syntax.project_revision);
    // The selected version does not exist: an unavailable immutable input
    // must reject instead of silently falling back to the live workspace bytes.
    let pinned: SourceCheckRequest = serde_json::from_value(serde_json::json!({
        "target":{"kind":"script","path":"/project/src/main.m"},
        "inputManifest":{"snapshotId":"missing-version","entries":[{"path":"/project/src/main.m","content":{"kind":"file","versionId":"absent","size":19}}]}
    })).unwrap();
    let resolved = std::cell::Cell::new(false);
    assert!(
        futures::executor::block_on(check_source_with_runtime_settings(pinned, |settings| {
            resolved.set(true);
            settings.resolve().map_err(|error| error.to_string())
        }))
        .is_err()
    );
    assert!(
        !resolved.get(),
        "unavailable captured inputs precede host resolution"
    );
    assert!(Arc::ptr_eq(
        &runmat_filesystem::current_provider(),
        &provider
    ));
}
