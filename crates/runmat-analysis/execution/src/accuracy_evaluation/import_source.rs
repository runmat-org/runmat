//! Cold accuracy claims are rederived from the admitted original sources for every family.
use super::*;
use runmat_analysis_fea::planning::resource::FeaResourceDemand;

fn unique(
    prepared: &PreparedAccuracy,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
    let mut matching = prepared.inputs.iter().filter(|i| i.manifest.kind == kind);
    match (matching.next(), matching.next()) {
        (Some(input), None) => Ok(input),
        _ => Err(AnalysisBridgeError::Identity(
            "accuracy requires a unique original phase input",
        )),
    }
}

pub(super) fn validate_output_bounds(
    prepared: &PreparedAccuracy,
    imported: &PreparedAnalysisObjects,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    crate::transport::validate_prepared_analysis_inputs(&prepared.host, &prepared.inputs)?;
    crate::require_accuracy_source_decode_budget(
        &prepared.inputs,
        &prepared.resource_plan,
        limits,
    )?;
    let FeaResourceDemand::AccuracyEvaluation(demand) = &prepared.resource_plan.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "accuracy output resource family",
        ));
    };
    if imported.manifest.encoded_length > demand.output.encoded_payload_bytes_upper_bound.0 {
        return Err(AnalysisBridgeError::Identity(
            "accuracy output exceeds admitted wire bound",
        ));
    }
    let payload = imported.payload();
    let resident =
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&payload)
            .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    if resident > demand.output.construction_resident_bytes_admission_budget.0 {
        return Err(AnalysisBridgeError::Identity(
            "accuracy output exceeds admitted decode bound",
        ));
    }
    Ok(())
}

pub(super) fn validate_original_evidence(
    prepared: &PreparedAccuracy,
    artifact: &AccuracyArtifact,
) -> AnalysisBridgeResult<()> {
    let request_objects = unique(
        prepared,
        AnalysisNumericalObjectKind::AccuracyEvaluationRequest,
    )?;
    let request = AccuracyEvaluationRequest::from_canonical_bytes(&request_objects.payload())?;
    if request != prepared.request
        || request.identity().digest() != request_objects.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy original request differs",
        ));
    }
    unique(prepared, AnalysisNumericalObjectKind::ResourcePlan)?;
    let receipt_objects = unique(prepared, AnalysisNumericalObjectKind::NumericalReceipt)?;
    let receipt = AccuracySourceReceipt::from_canonical_bytes(
        request.source_kind,
        &receipt_objects.payload(),
    )?;
    if receipt.identity().digest() != receipt_objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "accuracy original receipt logical identity",
        ));
    }
    let mut catalogs = prepared
        .inputs
        .iter()
        .filter(|i| i.manifest.kind == AnalysisNumericalObjectKind::ObservationCatalog);
    let catalog = match (catalogs.next(), catalogs.next()) {
        (None, None) => None,
        (Some(objects), None) => {
            let catalog = match &prepared.source_observations {
                // The catalog prepared this input in process: same identity, no decode.
                Some(catalog)
                    if catalog.identity().digest() == objects.manifest.logical_identity =>
                {
                    catalog.clone()
                }
                _ => ObservationPublicationCatalog::from_canonical_bytes(&objects.payload())
                    .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?,
            };
            if catalog.identity().digest() != objects.manifest.logical_identity {
                return Err(AnalysisBridgeError::Identity(
                    "accuracy original catalog logical identity",
                ));
            }
            Some(catalog)
        }
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "accuracy requires a unique original catalog",
            ))
        }
    };
    crate::validate_admitted_accuracy_source(
        &request,
        &receipt,
        catalog.as_ref(),
        &prepared.inputs,
        &prepared.resource_plan,
    )?;
    if artifact.evidence() != receipt.evidence(&request)?.as_slice() {
        return Err(AnalysisBridgeError::Identity(
            "accuracy differs from original source-derived evidence",
        ));
    }
    Ok(())
}
