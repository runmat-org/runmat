//! One result projection for native and browser document execution.
use super::*;

pub fn analysis_document_study_result(
    path: String,
    run: AnalysisStudyRunData,
    progress_events: Vec<FeaProgressEvent>,
) -> Result<AnalysisDocumentRunResult, String> {
    let step_results = run
        .steps
        .iter()
        .map(|step| {
            analysis_results_by_run_id_op(
                &step.run_id,
                AnalysisResultsQuery::metadata_only(),
                crate::operations::OperationContext::new(None, None),
            )
            .map(|result| (step, result.data))
            .map_err(|error| error.message)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let field_bindings = project_study_field_bindings(&step_results);
    let (results, result_summary) = project_study_result_views(&step_results)?;
    Ok(AnalysisDocumentRunResult {
        assessments: run.assessments.clone(),
        case_assessments: None,
        path,
        document_kind: AnalysisDocumentKind::Study,
        run: serde_json::to_value(run).map_err(|e| e.to_string())?,
        results,
        field_bindings,
        result_summary,
        figure_handles: Vec::new(),
        artifact_manifest: None,
        diagnostics: Vec::new(),
        progress_events,
    })
}
#[doc(hidden)]
pub fn project_study_field_bindings(
    step_results: &[(&AnalysisStudyStepRunData, AnalysisResultsData)],
) -> Vec<AnalysisFieldBinding> {
    step_results
        .iter()
        .flat_map(|(step, results)| {
            results
                .field_descriptors
                .iter()
                .map(move |descriptor| AnalysisFieldBinding {
                    step_id: step.procedure_step.step_id.clone(),
                    run_id: step.run_id.clone(),
                    descriptor: descriptor.clone(),
                })
        })
        .collect()
}
#[doc(hidden)]
pub fn project_study_result_views(
    step_results: &[(&AnalysisStudyStepRunData, AnalysisResultsData)],
) -> Result<(Option<serde_json::Value>, Option<serde_json::Value>), String> {
    match step_results {
        [] => Ok((None, None)),
        [(_, results)] => Ok((
            Some(serde_json::to_value(results).map_err(|e| e.to_string())?),
            Some(serde_json::to_value(&results.summary).map_err(|e| e.to_string())?),
        )),
        steps => Ok((
            Some(
                serde_json::json!({"steps":steps.iter().map(|(step,results)|serde_json::json!({"stepId":step.procedure_step.step_id,"runId":step.run_id,"results":results})).collect::<Vec<_>>()}),
            ),
            Some(
                serde_json::json!({"steps":steps.iter().map(|(step,results)|serde_json::json!({"stepId":step.procedure_step.step_id,"runId":step.run_id,"summary":results.summary})).collect::<Vec<_>>()}),
            ),
        )),
    }
}
