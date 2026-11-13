use super::*;
use crate::{BuiltinDocumentExpectation, BuiltinDocumentSelection};

fn errors_for(request: BuiltinDocumentExpectation) -> Vec<BuiltinCatalogValidationError> {
    let workspace = DesktopWorkspacePaths {
        files: BTreeSet::from(["editor.m"]),
        directories: BTreeSet::from([""]),
    };
    let mut errors = Vec::new();
    validate_interaction(
        "opentoline",
        &BuiltinDesktopHostInteraction::Document(request),
        &workspace,
        &mut errors,
    );
    errors
}

#[test]
fn document_paths_are_confined_and_materialized() {
    assert!(errors_for(BuiltinDocumentExpectation::OpenFile {
        relative_path: "editor.m"
    })
    .is_empty());
    for relative_path in ["../editor.m", "/editor.m", "missing.m"] {
        assert!(!errors_for(BuiltinDocumentExpectation::OpenFile { relative_path }).is_empty());
    }
    assert!(errors_for(BuiltinDocumentExpectation::OpenVariable { name: "results" }).is_empty());
    for name in ["", "a.b", "../x", "a\0b"] {
        assert!(!errors_for(BuiltinDocumentExpectation::OpenVariable { name }).is_empty());
    }
}

#[test]
fn navigation_serializes_portable_typed_position() {
    let request = BuiltinDocumentExpectation::Navigate {
        relative_path: "editor.m",
        line: std::num::NonZeroUsize::new(12).unwrap(),
        column: std::num::NonZeroUsize::new(4).unwrap(),
        selection: BuiltinDocumentSelection::Select,
    };
    assert!(errors_for(request).is_empty());
    let value = serde_json::to_value(BuiltinDesktopHostInteraction::Document(request)).unwrap();
    assert_eq!(
        value,
        serde_json::json!({"Document": {"Navigate": {
            "relative_path": "editor.m", "line": 12, "column": 4, "selection": "Select"
        }}})
    );
}
