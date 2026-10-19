use std::collections::HashSet;

use tower_lsp::lsp_types::{Location, Url};

use crate::core::analysis::{
    function_definitions_in_document, function_rename_references_in_document,
};

use super::super::RunMatLanguageServer;

pub(super) async fn function_definitions(
    server: &RunMatLanguageServer,
    symbol: &str,
) -> Vec<Location> {
    let cache = server.ensure_project_cache(None).await;
    let state = server.state.read().await;
    let mut locations = state
        .documents
        .iter()
        .filter_map(|(uri, document)| Some((uri, &document.text, document.analysis.as_ref()?)))
        .flat_map(|(uri, text, analysis)| {
            function_definitions_in_document(text, analysis, symbol)
                .into_iter()
                .map(|range| Location {
                    uri: uri.clone(),
                    range,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if let Some(cache) = cache {
        locations.extend(cache.files.values().flat_map(|document| {
            function_definitions_in_document(&document.text, &document.analysis, symbol)
                .into_iter()
                .map(|range| Location {
                    uri: document.uri.clone(),
                    range,
                })
                .collect::<Vec<_>>()
        }));
    }
    dedupe_locations(&mut locations);
    locations
}

pub(super) async fn function_locations(
    server: &RunMatLanguageServer,
    definition_uri: &Url,
    symbol: &str,
) -> Vec<Location> {
    let cache = server.ensure_project_cache(Some(definition_uri)).await;
    let state = server.state.read().await;
    let mut locations = Vec::new();
    for (uri, document) in &state.documents {
        let Some(analysis) = &document.analysis else {
            continue;
        };
        append_function_locations(&mut locations, uri, &document.text, analysis, symbol);
    }
    if let Some(cache) = cache {
        for document in cache.files.values() {
            append_function_locations(
                &mut locations,
                &document.uri,
                &document.text,
                &document.analysis,
                symbol,
            );
        }
    }
    dedupe_locations(&mut locations);
    locations
}

fn append_function_locations(
    locations: &mut Vec<Location>,
    uri: &Url,
    text: &str,
    analysis: &super::super::DocumentAnalysis,
    symbol: &str,
) {
    locations.extend(
        function_definitions_in_document(text, analysis, symbol)
            .into_iter()
            .chain(function_rename_references_in_document(
                text, analysis, symbol,
            ))
            .map(|range| Location {
                uri: uri.clone(),
                range,
            }),
    );
}

pub(super) fn dedupe_locations(locations: &mut Vec<Location>) {
    let mut seen = HashSet::new();
    locations.retain(|location| {
        seen.insert((
            location.uri.clone(),
            location.range.start.line,
            location.range.start.character,
            location.range.end.line,
            location.range.end.character,
        ))
    });
}
