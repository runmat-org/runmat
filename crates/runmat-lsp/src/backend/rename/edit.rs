use std::collections::HashMap;
use std::path::Path;

use tower_lsp::lsp_types::{
    DocumentChangeOperation, DocumentChanges, Location, OneOf,
    OptionalVersionedTextDocumentIdentifier, RenameFile, ResourceOp, TextDocumentEdit, TextEdit,
    Url, WorkspaceEdit,
};

use crate::core::fea::FeaFunctionImportPath;
use runmat_lexer::{fea::FeaKeyword, Token};

use super::super::RunMatLanguageServer;

pub(super) async fn workspace_edit(
    server: &RunMatLanguageServer,
    locations: Vec<Location>,
    new_name: &str,
    import_paths: Vec<FeaFunctionImportPath>,
    file_rename: Option<(Url, Url)>,
) -> WorkspaceEdit {
    let mut edits = location_edits(locations, new_name);
    if file_rename.is_some() {
        for import in import_paths {
            edits
                .entry(import.location.uri)
                .or_default()
                .push(TextEdit {
                    range: import.location.range,
                    new_text: quoted_renamed_import(&import.source, new_name),
                });
        }
    }
    for edits in edits.values_mut() {
        edits.sort_by_key(|edit| (edit.range.start.line, edit.range.start.character));
        edits.dedup_by(|left, right| left.range == right.range);
    }
    let Some((old_uri, new_uri)) = file_rename else {
        return WorkspaceEdit {
            changes: Some(edits),
            document_changes: None,
            change_annotations: None,
        };
    };
    let versions = server
        .state
        .read()
        .await
        .documents
        .iter()
        .map(|(uri, document)| (uri.clone(), document.version))
        .collect::<HashMap<_, _>>();
    let mut operations = edits
        .into_iter()
        .map(|(uri, edits)| {
            DocumentChangeOperation::Edit(TextDocumentEdit {
                text_document: OptionalVersionedTextDocumentIdentifier {
                    version: versions.get(&uri).copied().flatten(),
                    uri,
                },
                edits: edits.into_iter().map(OneOf::Left).collect(),
            })
        })
        .collect::<Vec<_>>();
    operations.sort_by_key(|operation| match operation {
        DocumentChangeOperation::Edit(edit) => edit.text_document.uri.as_str().to_string(),
        DocumentChangeOperation::Op(_) => String::new(),
    });
    operations.push(DocumentChangeOperation::Op(ResourceOp::Rename(
        RenameFile {
            old_uri,
            new_uri,
            options: None,
            annotation_id: None,
        },
    )));
    WorkspaceEdit {
        changes: None,
        document_changes: Some(DocumentChanges::Operations(operations)),
        change_annotations: None,
    }
}

pub(super) fn location_edits(
    locations: Vec<Location>,
    new_name: &str,
) -> HashMap<Url, Vec<TextEdit>> {
    let mut edits = HashMap::<Url, Vec<TextEdit>>::new();
    for location in locations {
        edits.entry(location.uri).or_default().push(TextEdit {
            range: location.range,
            new_text: new_name.to_string(),
        });
    }
    edits
}

pub(super) fn file_rename(uri: &Url, old_name: &str, new_name: &str) -> Option<(Url, Url)> {
    let mut path = uri.to_file_path().ok()?;
    if path.file_stem()?.to_str()? != old_name {
        return None;
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("m");
    path.set_file_name(format!("{new_name}.{extension}"));
    Some((uri.clone(), Url::from_file_path(path).ok()?))
}

fn quoted_renamed_import(source: &str, new_name: &str) -> String {
    let mut path = Path::new(source).to_path_buf();
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("m");
    path.set_file_name(format!("{new_name}.{extension}"));
    serde_json::to_string(&path.to_string_lossy().replace('\\', "/"))
        .expect("import paths serialize as JSON strings")
}

pub(super) fn valid_fea_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    (first == '_' || first.is_alphabetic())
        && characters.all(|character| character == '_' || character.is_alphanumeric())
        && FeaKeyword::from_source(value).is_none()
}

pub(super) fn valid_runmat_identifier(value: &str) -> bool {
    matches!(runmat_lexer::tokenize(value).as_slice(), [Token::Ident])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_reject_keywords_and_non_identifiers() {
        assert!(valid_fea_identifier("impact_response"));
        assert!(valid_fea_identifier("Δload"));
        assert!(!valid_fea_identifier(""));
        assert!(!valid_fea_identifier("2load"));
        assert!(!valid_fea_identifier("impact response"));
        assert!(!valid_fea_identifier("study"));
        assert!(valid_runmat_identifier("impact_response"));
        assert!(!valid_runmat_identifier("Δload"));
        assert!(!valid_runmat_identifier("function"));
    }

    #[test]
    fn file_and_import_renames_preserve_location_and_extension() {
        let project = tempfile::tempdir().unwrap();
        let directory = project.path().join("project/signals");
        let old_uri = Url::from_file_path(directory.join("impactPulse.m")).unwrap();
        let (_, new_uri) = file_rename(&old_uri, "impactPulse", "impactResponse").unwrap();
        assert_eq!(
            new_uri,
            Url::from_file_path(directory.join("impactResponse.m")).unwrap()
        );
        assert_eq!(
            quoted_renamed_import("../signals/impactPulse.m", "impactResponse"),
            "\"../signals/impactResponse.m\""
        );
        assert!(file_rename(&old_uri, "anotherFunction", "renamed").is_none());
    }
}
