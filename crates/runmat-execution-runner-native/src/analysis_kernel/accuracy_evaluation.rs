use super::*;

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::AccuracyEvaluationRequest,
    )?;
    let numerical_receipt =
        unique_input(&invocation, AnalysisNumericalObjectKind::NumericalReceipt)?;
    let observations =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::ObservationCatalog)?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::AccuracyReport] {
        return Err(AnalysisBridgeError::Invalid(
            "accuracy report output differs from the workload contract".into(),
        ));
    }
    let request_manifest_identity = request.manifest.logical_identity;
    let receipt_manifest_identity = numerical_receipt.manifest.logical_identity;
    let admitted_plan = runmat_analysis_execution::decode_admitted_resource_plan(
        resource_plan_input,
        invocation.output_policy.inventory_limits(),
    )?;
    runmat_analysis_execution::require_accuracy_source_decode_budget(
        invocation.inputs,
        &admitted_plan,
        invocation.output_policy.inventory_limits(),
    )?;
    let request = AccuracyEvaluationRequest::from_canonical_bytes(&request.payload())?;
    if request.identity().digest() != request_manifest_identity
        || request.producer != invocation.host.workload.step
        || request.source_observations.is_some() != observations.is_some()
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy inputs differ from the admitted evaluation request",
        ));
    }
    let observations = if let Some(observations) = observations {
        let manifest_identity = observations.manifest.logical_identity;
        let observations =
            runmat_analysis_fea::results::observation::ObservationPublicationCatalog::from_canonical_bytes(&observations.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        if observations.identity().digest() != manifest_identity
            || Some(observations.identity()) != request.source_observations
        {
            return Err(AnalysisBridgeError::Identity(
                "accuracy observation artifact differs from its admitted source",
            ));
        }
        Some(observations)
    } else {
        None
    };
    let numerical_receipt =
        runmat_analysis_fea::execution::AccuracySourceReceipt::from_canonical_bytes(
            request.source_kind,
            &numerical_receipt.payload(),
        )?;
    if numerical_receipt.identity().digest() != receipt_manifest_identity
        || numerical_receipt.source_result() != request.source_result
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy numerical receipt differs from its admitted source result",
        ));
    }
    if resource_plan_input.manifest.logical_identity != admitted_plan.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "accuracy resource identity differs from admitted inputs",
        ));
    }
    runmat_analysis_execution::validate_admitted_accuracy_source(
        &request,
        &numerical_receipt,
        observations.as_ref(),
        invocation.inputs,
        &admitted_plan,
    )?;
    let evidence = numerical_receipt.evidence(&request)?;
    let artifact =
        runmat_analysis_fea::results::accuracy::evaluate_accuracy_artifact(&request, evidence)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::AccuracyReport,
        artifact.identity().digest(),
        artifact
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )])
}
