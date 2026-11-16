use super::*;
use runmat_filesystem::{FsProvider, MemoryFsProvider, ProviderGuard};
use std::sync::Arc;
use wasm_bindgen_test::wasm_bindgen_test;

struct Fixture {
    files: Arc<MemoryFsProvider>,
    _restore: ProviderGuard,
}

impl Fixture {
    fn new() -> Self {
        DOCS.with(|documents| documents.borrow_mut().docs.clear());
        let files = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
        let restore = runmat_filesystem::replace_provider(files.clone());
        workspace_changed();
        Self {
            files,
            _restore: restore,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        DOCS.with(|documents| documents.borrow_mut().docs.clear());
        workspace_changed();
    }
}

fn has_unresolved(uri: &str) -> bool {
    let diagnostics: Vec<Diagnostic> =
        serde_wasm_bindgen::from_value(diagnostics(uri.to_owned()).unwrap()).unwrap();
    diagnostics.iter().any(|diagnostic| {
        matches!(&diagnostic.code,
        Some(lsp_types::NumberOrString::String(code)) if code == "RM-RES0001")
    })
}

async fn helper_navigation_uses_editor_sources(manifest: bool) {
    let fixture = Fixture::new();
    let (main_path, helper_path) = if manifest {
        fixture
            .files
            .write_project_path(
                "/workspace/runmat.toml",
                b"[package]\nname = 'lsp-fixture'\n[sources]\nroots = ['src', 'lib']\n",
            )
            .unwrap();
        ("/workspace/src/main.m", "/workspace/lib/helper.m")
    } else {
        ("/workspace/main.m", "/workspace/helper.m")
    };
    let source = "value = helper(1);";
    fixture
        .files
        .write_project_path(main_path, source.as_bytes())
        .unwrap();
    fixture
        .files
        .write_project_path(
            helper_path,
            b"function out = helper(value)\nout = value;\nend\n",
        )
        .unwrap();
    let uri = format!("file://{main_path}");
    let helper_uri = format!("file://{helper_path}");
    update_document_lexical(
        helper_uri.clone(),
        1,
        "% unsaved editor text\nfunction out = helper(value)\nout = value;\nend\n".into(),
    )
    .unwrap();
    open_document(uri.clone(), source.into()).await;
    assert!(!has_unresolved(&uri));
    let position =
        crate::core::position::offset_to_position(source, source.find("helper").unwrap() + 1);
    let locations: Vec<lsp_types::Location> = serde_wasm_bindgen::from_value(
        definition(uri.clone(), position.line, position.character)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        locations
            .iter()
            .any(|location| location.uri.as_str() == helper_uri && location.range.start.line == 1),
        "definition must use the unsaved companion: {locations:?}"
    );
    let references: Vec<lsp_types::Location> = serde_wasm_bindgen::from_value(
        references(uri.clone(), position.line, position.character)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(references
        .iter()
        .any(|location| location.uri.as_str() == uri));

    if manifest {
        let handoff = project_handoff(main_path.into()).await.unwrap();
        let expected = install_project_handoff(handoff).unwrap();
        analyze_document(uri.clone(), 1).await.unwrap();
        let revision: runmat_package::ProjectRevision =
            serde_wasm_bindgen::from_value(expected).unwrap();
        DOCS.with(|documents| {
            assert_eq!(
                documents.borrow().docs[&uri]
                    .analysis
                    .as_ref()
                    .unwrap()
                    .project_revision
                    .as_ref(),
                Some(&revision)
            )
        });
    }

    close_document(helper_uri);
    fixture
        .files
        .remove_file(Path::new(helper_path))
        .await
        .unwrap();
    workspace_changed();
    analyze_document(uri.clone(), 1).await.unwrap();
    assert!(
        has_unresolved(&uri),
        "deleted companion must leave a resolution diagnostic"
    );
    fixture
        .files
        .write_project_path(
            helper_path,
            b"function out = helper(value)\nout = value;\nend\n",
        )
        .unwrap();
    workspace_changed();
    analyze_document(uri.clone(), 1).await.unwrap();
    assert!(
        !has_unresolved(&uri),
        "added companion must be rediscovered"
    );
    let renamed = format!("{helper_path}.renamed");
    fixture
        .files
        .rename(Path::new(helper_path), Path::new(&renamed))
        .await
        .unwrap();
    workspace_changed();
    analyze_document(uri.clone(), 1).await.unwrap();
    assert!(
        has_unresolved(&uri),
        "renamed companion must leave the source catalog"
    );
}

#[wasm_bindgen_test]
async fn loose_siblings_support_navigation_editor_overlays_and_file_changes() {
    helper_navigation_uses_editor_sources(false).await;
}

#[wasm_bindgen_test]
async fn manifest_source_roots_share_diagnostics_navigation_and_frozen_identity() {
    helper_navigation_uses_editor_sources(true).await;
}

#[wasm_bindgen_test]
async fn workspace_changes_discard_pending_source_results() {
    let _fixture = Fixture::new();
    update_document_lexical(
        "file:///workspace/unsaved%20file.m".into(),
        1,
        "value = 1;".into(),
    )
    .unwrap();
    let captured = SourceWorkspace::capture().unwrap();
    let (resume, changed) = futures::channel::oneshot::channel();
    wasm_bindgen_futures::spawn_local(async move {
        assert!(
            runmat_filesystem::read_to_string_async("/workspace/unsaved file.m")
                .await
                .is_err(),
            "the pending request must not expose editor text to another task"
        );
        update_document_lexical(
            "file:///workspace/unsaved%20file.m".into(),
            2,
            "value = 2;".into(),
        )
        .unwrap();
        resume.send(()).unwrap();
    });
    let result = captured
        .run(async {
            changed.await.unwrap();
            let source = runmat_filesystem::read_to_string_async("/workspace/unsaved file.m")
                .await
                .unwrap();
            assert_eq!(source, "value = 1;");
            source
        })
        .await
        .unwrap();
    assert!(result.is_none());
    assert_eq!(
        source_name_from_uri("file:///workspace/./unsaved%20file.m").as_deref(),
        Some("/workspace/unsaved file.m")
    );
}
