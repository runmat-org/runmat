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
    let source_context: Vec<_> = invocation
        .inputs
        .iter()
        .filter(|input| {
            // Phase-owned inputs are authenticated above. Pass every remaining
            // object to the shared source validator, which owns each physics
            // family's exact closed inventory; do not duplicate its solver list.
            !matches!(
                input.manifest.kind,
                AnalysisNumericalObjectKind::AccuracyEvaluationRequest
                    | AnalysisNumericalObjectKind::NumericalReceipt
                    | AnalysisNumericalObjectKind::ObservationCatalog
                    | AnalysisNumericalObjectKind::ResourcePlan
            )
        })
        .collect();
    if invocation.inputs.len() != 3 + usize::from(observations.is_some()) + source_context.len()
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::AccuracyReport]
    {
        return Err(AnalysisBridgeError::Invalid(
            "accuracy evaluation inputs or report output differ from the workload contract".into(),
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
        observations.validate_source_result(
            request.producer,
            request.source_kind,
            request.source_result,
        )?;
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
    let expected_plan = runmat_analysis_execution::build_prepared_accuracy_resource_plan(
        &request,
        &numerical_receipt,
        invocation.inputs,
    )?;
    if resource_plan_input.manifest.logical_identity != admitted_plan.identity.digest()
        || admitted_plan != expected_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy resource plan differs from admitted inputs",
        ));
    }
    // Authenticate the exact admitted resource plan before reconstructing source topology
    // and constraints. The authored source checks remain mandatory after admission.
    runmat_analysis_execution::validate_accuracy_source_context(
        &request,
        &numerical_receipt,
        observations.as_ref(),
        &source_context,
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
