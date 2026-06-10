use anyhow::{Context, Result};
use log::info;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_runtime::analysis::{
    analysis_run_study_op, analysis_run_study_sweep_op,
    load_analysis_study_document_from_path_async, AnalysisStudyDocument,
};
use runmat_runtime::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use serde::Serialize;
use std::path::PathBuf;

use crate::cli::Cli;
use crate::commands::accel::dump_provider_telemetry_if_requested;
use crate::AlreadyReportedCliError;

pub async fn execute_study_path(
    path: PathBuf,
    _cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    info!("Executing analysis study: {path:?}");
    let document = load_analysis_study_document_from_path_async(&path)
        .await
        .map_err(|err| {
            anyhow::anyhow!(
                "Failed to load analysis study file {}: {err}",
                path.display()
            )
        })?;
    let context = OperationContext::new(None, None);

    match document {
        AnalysisStudyDocument::Study(spec) => {
            let envelope = analysis_run_study_op(&spec, context).map_err(report_operation_error)?;
            print_envelope(&envelope, config.runtime.verbose)?;
        }
        AnalysisStudyDocument::Sweep(spec) => {
            let envelope =
                analysis_run_study_sweep_op(&spec, context).map_err(report_operation_error)?;
            print_envelope(&envelope, config.runtime.verbose)?;
        }
    }

    dump_provider_telemetry_if_requested();
    Ok(())
}

fn print_envelope<T: Serialize>(
    envelope: &OperationEnvelope<T>,
    include_metadata: bool,
) -> Result<()> {
    if include_metadata {
        println!(
            "{}",
            serde_json::to_string_pretty(envelope).context("Failed to serialize study result")?
        );
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&envelope.data)
                .context("Failed to serialize study result")?
        );
    }
    Ok(())
}

fn report_operation_error(error: OperationErrorEnvelope) -> anyhow::Error {
    match serde_json::to_string_pretty(&error) {
        Ok(payload) => eprintln!("{payload}"),
        Err(_) => eprintln!("{}: {}", error.error_code, error.message.replace('\n', " ")),
    }
    AlreadyReportedCliError.into()
}
