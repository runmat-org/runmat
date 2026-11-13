//! Metadata queries over every authenticated acoustic frequency sample.
use super::*;

pub(super) fn results_op(
    step: &AnalysisStudyStepRunData,
    query: AnalysisResultsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsData>, OperationErrorEnvelope> {
    let AnalysisStudyStepNumericalResultData::AcousticFrequency(result) = &step.numerical_result
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
            "acoustic frequency store returned another numerical result kind",
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
                format!("acoustic frequency field catalog is invalid: {error}"),
                BTreeMap::from([("run_id".to_string(), step.run_id.clone())]),
            )
        })?;
    // Authored outputs evaluated at each frequency are small inline results, one per sample.
    // They are returned inline in `fields`; `field_descriptors` lists only the recovered field
    // artifacts the field-page reader serves, as for structural frequency runs.
    let observation_fields = result
        .entries
        .iter()
        .flat_map(|entry| entry.observations.iter())
        .map(|observation| observation.field.clone())
        .collect::<Vec<_>>();
    if !query.include_fields.is_empty() {
        for requested in &query.include_fields {
            let matches = |field_id: &str| {
                crate::analysis::field_selection::field_id_matches(field_id, requested)
            };
            if !descriptors
                .iter()
                .any(|descriptor| matches(&descriptor.field_id))
                && !observation_fields
                    .iter()
                    .any(|field| matches(&field.field_id))
            {
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
    let frequencies = result
        .entries
        .iter()
        .map(|entry| entry.entry.frequency_hz())
        .collect::<Vec<_>>();
    let summary = AnalysisResultsSummary {
        field_count: descriptors.len(),
        total_elements: descriptors
            .iter()
            .map(|descriptor| descriptor.element_count)
            .sum(),
        frequency_sample_count: frequencies.len(),
        available_frequencies_hz: frequencies,
        max_frequency_solver_residual: result
            .entries
            .iter()
            .map(|entry| entry.result.relative_residual)
            .reduce(f64::max),
        max_frequency_balance_error: result
            .entries
            .iter()
            .map(|entry| entry.result.power_relative_error)
            .reduce(f64::max),
        ..AnalysisResultsSummary::default()
    };
    Ok(OperationEnvelope::new(
        ANALYSIS_RESULTS_OPERATION,
        ANALYSIS_RESULTS_OP_VERSION,
        &context,
        AnalysisResultsData {
            identity: step.identity.clone(),
            field_descriptors: descriptors,
            fields: observation_fields
                .into_iter()
                .filter(|field| {
                    query.include_fields.is_empty()
                        || query.include_fields.iter().any(|requested| {
                            crate::analysis::field_selection::field_id_matches(
                                &field.field_id,
                                requested,
                            )
                        })
                })
                .collect(),
            modal_results: None,
            buckling_results: None,
            thermal_results: None,
            transient_results: None,
            nonlinear_results: None,
            structural_frequency_results: None,
            acoustic_frequency_results: Some(result.as_ref().clone()),
            diagnostics: query.include_diagnostics.then(Vec::new),
            run_status: step.run_status,
            numerically_acceptable: step.numerically_acceptable,
            quality_reasons: step.quality_reasons.clone(),
            provenance: step.provenance.clone(),
            summary,
        },
    ))
}
