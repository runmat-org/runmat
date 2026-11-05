use tower_lsp::jsonrpc::{Error as RpcError, Result as RpcResult};
use tower_lsp::lsp_types::{
    PrepareRenameResponse, RenameParams, TextDocumentPositionParams, Url, WorkspaceEdit,
};

use crate::core::analysis::references_locations_at;
use crate::core::fea::FeaRenameTarget;

use super::{rename_subject_at, rename_subject_has_collision, RunMatLanguageServer};

mod edit;
mod runmat;

use edit::{
    file_rename, location_edits, valid_fea_identifier, valid_runmat_identifier, workspace_edit,
};
use runmat::{dedupe_locations, function_definitions, function_locations};

enum RenameTarget {
    Fea(FeaRenameTarget),
    RunMat {
        name: String,
        is_local: bool,
        parameter_owner: Option<String>,
        range: tower_lsp::lsp_types::Range,
    },
}

pub(super) async fn prepare(
    server: &RunMatLanguageServer,
    params: TextDocumentPositionParams,
) -> RpcResult<Option<PrepareRenameResponse>> {
    if let Some(design) = server.design_of(&params.text_document.uri).await {
        return Ok(design
            .prepare_rename(&params.position)
            .map(
                |(range, placeholder)| PrepareRenameResponse::RangeWithPlaceholder {
                    range,
                    placeholder,
                },
            ));
    }
    let Some(target) = target_at(server, &params).await else {
        return Ok(None);
    };
    let (range, placeholder) = match target {
        RenameTarget::Fea(target) => (target.range, target.key.name),
        RenameTarget::RunMat { name, range, .. } => (range, name),
    };
    Ok(Some(PrepareRenameResponse::RangeWithPlaceholder {
        range,
        placeholder,
    }))
}

pub(super) async fn rename(
    server: &RunMatLanguageServer,
    params: RenameParams,
) -> RpcResult<Option<WorkspaceEdit>> {
    let request = params.text_document_position;
    if let Some(design) = server.design_of(&request.text_document.uri).await {
        return design
            .rename(&request.position, &params.new_name)
            .map_err(RpcError::invalid_params);
    }
    let Some(target) = target_at(server, &request).await else {
        return Ok(None);
    };
    match target {
        RenameTarget::Fea(target) => {
            let valid = if target.key.kind == runmat_analysis_language::FeaSymbolKind::Function {
                valid_runmat_identifier(&params.new_name) && valid_fea_identifier(&params.new_name)
            } else {
                valid_fea_identifier(&params.new_name)
            };
            if !valid {
                return Ok(None);
            }
            rename_fea(server, target, &params.new_name).await
        }
        RenameTarget::RunMat {
            name,
            is_local,
            parameter_owner,
            ..
        } => {
            if !valid_runmat_identifier(&params.new_name) {
                return Ok(None);
            }
            rename_runmat(
                server,
                &request.text_document.uri,
                &request.position,
                &name,
                is_local,
                parameter_owner.as_deref(),
                &params.new_name,
            )
            .await
        }
    }
}

async fn target_at(
    server: &RunMatLanguageServer,
    params: &TextDocumentPositionParams,
) -> Option<RenameTarget> {
    let state = server.state.read().await;
    let document = state.documents.get(&params.text_document.uri)?;
    if let Some(analysis) = &document.fea_analysis {
        return analysis
            .rename_target(&document.text, &params.position)
            .map(RenameTarget::Fea);
    }
    let analysis = document.analysis.as_ref()?;
    let subject = rename_subject_at(&document.text, analysis, &params.position)?;
    Some(RenameTarget::RunMat {
        name: subject.name,
        is_local: subject.is_local,
        parameter_owner: subject.parameter_owner,
        range: subject.range,
    })
}

async fn rename_fea(
    server: &RunMatLanguageServer,
    target: FeaRenameTarget,
    new_name: &str,
) -> RpcResult<Option<WorkspaceEdit>> {
    let analyses = super::fea::workspace_analyses(server)
        .await
        .map_err(RpcError::invalid_params)?;
    if analyses
        .iter()
        .any(|analysis| analysis.rename_has_collision(&target, new_name))
    {
        return Ok(None);
    }
    let mut locations = analyses
        .iter()
        .flat_map(|analysis| analysis.rename_locations(&target))
        .collect::<Vec<_>>();
    let import_paths = analyses
        .iter()
        .flat_map(|analysis| analysis.function_import_paths(&target))
        .collect::<Vec<_>>();
    if !import_paths.is_empty() && !valid_fea_identifier(new_name) {
        return Ok(None);
    }
    if target.key.kind == runmat_analysis_language::FeaSymbolKind::Function {
        let definitions = function_definitions(server, &target.key.name).await;
        if definitions.len() != 1 || definitions[0].uri != target.definition_uri {
            return Ok(None);
        }
        locations
            .extend(function_locations(server, &target.definition_uri, &target.key.name).await);
    }
    dedupe_locations(&mut locations);
    if locations.is_empty() {
        return Ok(None);
    }
    let source_file_rename = (target.key.kind == runmat_analysis_language::FeaSymbolKind::Function)
        .then(|| file_rename(&target.definition_uri, &target.key.name, new_name))
        .flatten();
    Ok(Some(
        workspace_edit(
            server,
            locations,
            new_name,
            import_paths,
            source_file_rename,
        )
        .await,
    ))
}

async fn rename_runmat(
    server: &RunMatLanguageServer,
    uri: &Url,
    position: &tower_lsp::lsp_types::Position,
    old_name: &str,
    is_local: bool,
    parameter_owner: Option<&str>,
    new_name: &str,
) -> RpcResult<Option<WorkspaceEdit>> {
    if is_local {
        let mut locations = {
            let state = server.state.read().await;
            let Some(document) = state.documents.get(uri) else {
                return Ok(None);
            };
            let Some(analysis) = document.analysis.as_ref() else {
                return Ok(None);
            };
            if rename_subject_has_collision(&document.text, analysis, position, new_name) {
                return Ok(None);
            }
            references_locations_at(&document.text, analysis, position, uri)
        };
        if let Some(function_name) = parameter_owner {
            let analyses = super::fea::workspace_analyses(server)
                .await
                .map_err(RpcError::invalid_params)?;
            if old_name != new_name
                && analyses.iter().any(|analysis| {
                    !analysis
                        .function_parameter_locations(uri, function_name, new_name)
                        .is_empty()
                })
            {
                return Ok(None);
            }
            let parameter_locations = analyses
                .iter()
                .flat_map(|analysis| {
                    analysis.function_parameter_locations(uri, function_name, old_name)
                })
                .collect::<Vec<_>>();
            if !parameter_locations.is_empty() && !valid_fea_identifier(new_name) {
                return Ok(None);
            }
            locations.extend(parameter_locations);
            dedupe_locations(&mut locations);
        }
        return Ok((!locations.is_empty()).then(|| WorkspaceEdit {
            changes: Some(location_edits(locations, new_name)),
            document_changes: None,
            change_annotations: None,
        }));
    }

    let definitions = function_definitions(server, old_name).await;
    let [definition] = definitions.as_slice() else {
        return Ok(None);
    };
    let definition_uri = definition.uri.clone();
    let new_name_definitions = function_definitions(server, new_name).await;
    if new_name != old_name && !new_name_definitions.is_empty() {
        return Ok(None);
    }
    let mut locations = function_locations(server, &definition_uri, old_name).await;
    let analyses = super::fea::workspace_analyses(server)
        .await
        .map_err(RpcError::invalid_params)?;
    let target = FeaRenameTarget {
        definition_uri: definition_uri.clone(),
        key: runmat_analysis_language::FeaSymbolKey {
            kind: runmat_analysis_language::FeaSymbolKind::Function,
            name: old_name.to_string(),
        },
        range: tower_lsp::lsp_types::Range::default(),
    };
    if analyses
        .iter()
        .any(|analysis| analysis.rename_has_collision(&target, new_name))
    {
        return Ok(None);
    }
    locations.extend(
        analyses
            .iter()
            .flat_map(|analysis| analysis.rename_locations(&target)),
    );
    let import_paths = analyses
        .iter()
        .flat_map(|analysis| analysis.function_import_paths(&target))
        .collect::<Vec<_>>();
    if !import_paths.is_empty() && !valid_fea_identifier(new_name) {
        return Ok(None);
    }
    dedupe_locations(&mut locations);
    if locations.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        workspace_edit(
            server,
            locations,
            new_name,
            import_paths,
            file_rename(&definition_uri, old_name, new_name),
        )
        .await,
    ))
}
