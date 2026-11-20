//! Catalog lists for analysis authoring: the workflow steps, procedure ids, output operation
//! ids and language reference paths.
use runmat_analysis_core::AnalysisProcedure;
use runmat_analysis_language::output_operation_extension;

pub fn analysis_authoring_workflow() -> Vec<String> {
    vec![
        "edit <study>.fea with workspace file tools".to_string(),
        "runmat check <study>.fea [--study <id> | --sweep <id>]".to_string(),
        "runmat plan <study>.fea [--study <id> | --sweep <id>] --json".to_string(),
        "runmat run <study>.fea [--study <id> | --sweep <id>] [--cluster <profile>]".to_string(),
    ]
}

pub fn analysis_procedure_ids() -> Vec<String> {
    AnalysisProcedure::ALL
        .into_iter()
        .map(|procedure| {
            serde_json::to_value(procedure)
                .expect("analysis procedure must serialize")
                .as_str()
                .expect("analysis procedure must serialize as a string")
                .to_string()
        })
        .collect()
}

pub fn analysis_output_operation_ids() -> Vec<String> {
    output_operation_extension()
        .operations
        .into_iter()
        .map(|operation| operation.identity.to_string())
        .collect()
}

pub fn analysis_language_reference_paths() -> Vec<String> {
    runmat_analysis_language::reference::FEA_LANGUAGE_REFERENCE_IDS
        .iter()
        .map(|id| (*id).to_owned())
        .collect()
}
