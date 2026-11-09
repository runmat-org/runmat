//! Metadata queries over every structural frequency publication sample.
use super::*;

pub(super) fn results_op(
    step: &AnalysisStudyStepRunData,
    query: AnalysisResultsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsData>, OperationErrorEnvelope> {
    let AnalysisStudyStepNumericalResultData::StructuralFrequency(result) = &step.numerical_result
    else {
        return Err(operation_error(
            ANALYSIS_RESULTS_OPERATION,
            ANALYSIS_RESULTS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS.ARTIFACT_INVALID",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "structural frequency store returned another numerical result kind",
            BTreeMap::from([("run_id".to_string(), step.run_id.clone())]),
        ));
    };
    let publication = &result.publication;
    let mut descriptors = publication
        .samples()
        .iter()
        .flat_map(|sample| sample.fields.fields.iter().map(frequency_field_descriptor))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            operation_error(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESULTS.ARTIFACT_INVALID",
                    error_type: OperationErrorType::Contract,
                    retryable: false,
                    severity: OperationErrorSeverity::Fatal,
                },
                format!("structural frequency field catalog is invalid: {error}"),
                BTreeMap::from([("run_id".to_string(), step.run_id.clone())]),
            )
        })?;
    if !query.include_fields.is_empty() {
        for requested in &query.include_fields {
            if !descriptors.iter().any(|descriptor| {
                crate::analysis::field_selection::field_id_matches(&descriptor.field_id, requested)
            }) {
                return Err(operation_error(
                    ANALYSIS_RESULTS_OPERATION,
                    ANALYSIS_RESULTS_OP_VERSION,
                    &context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RESULTS.FIELD_NOT_FOUND",
                        error_type: OperationErrorType::Input,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    format!("requested FEA field '{requested}' was not produced by run"),
                    BTreeMap::from([("requested_field".to_string(), requested.clone())]),
                ));
            }
        }
        descriptors.retain(|descriptor| {
            query.include_fields.iter().any(|requested| {
                crate::analysis::field_selection::field_id_matches(&descriptor.field_id, requested)
            })
        });
    }
    let frequencies = publication
        .result()
        .samples()
        .iter()
        .map(|sample| sample.plan_entry().frequency_hz())
        .collect::<Vec<_>>();
    let receipt = publication.receipt();
    let truncation = receipt.modal_truncation_balance();
    let summary = AnalysisResultsSummary {
        field_count: descriptors.len(),
        total_elements: descriptors
            .iter()
            .map(|descriptor| descriptor.element_count)
            .sum(),
        frequency_sample_count: frequencies.len(),
        available_frequencies_hz: frequencies.clone(),
        max_frequency_solver_residual: receipt
            .samples()
            .iter()
            .map(|sample| sample.solver_relative_residual())
            .reduce(f64::max),
        max_frequency_balance_error: truncation
            .is_none()
            .then(|| {
                receipt
                    .samples()
                    .iter()
                    .map(|sample| sample.physical_relative_balance_error())
                    .reduce(f64::max)
            })
            .flatten(),
        modal_truncation_balance_error: truncation
            .map(|measure| measure.maximum_relative_balance_error),
        modal_truncation_balance_frequency_hz: truncation
            .and_then(|measure| frequencies.get(measure.sample_index).copied()),
        ..AnalysisResultsSummary::default()
    };
    Ok(OperationEnvelope::new(
        ANALYSIS_RESULTS_OPERATION,
        ANALYSIS_RESULTS_OP_VERSION,
        &context,
        AnalysisResultsData {
            identity: step.identity.clone(),
            field_descriptors: descriptors,
            fields: Vec::new(),
            modal_results: None,
            buckling_results: None,
            thermal_results: None,
            transient_results: None,
            nonlinear_results: None,
            structural_frequency_results: Some(result.as_ref().clone()),
            acoustic_frequency_results: None,
            diagnostics: query.include_diagnostics.then(Vec::new),
            run_status: step.run_status,
            numerically_acceptable: step.numerically_acceptable,
            quality_reasons: step.quality_reasons.clone(),
            provenance: step.provenance.clone(),
            summary,
        },
    ))
}
