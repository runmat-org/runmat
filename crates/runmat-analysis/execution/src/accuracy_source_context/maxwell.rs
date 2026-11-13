//! Original complex Maxwell authority, not a regenerated solve or spatial certificate.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::{
    execution::{
        AccuracyEvaluationRequest, AccuracySourceKind, AnalysisNumericalObjectKind as Kind,
        MaxwellHarmonicSolverResult, MaxwellHarmonicWorkload, MaxwellNumericalReceiptArtifact,
        ObservationHistorySourceContext, SequenceIdentity, StepIdentity,
    },
    results::observation::{ObservationFieldCatalogSource, ObservationPublicationCatalog},
};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[cfg(test)]
mod tests;

pub(super) fn validate(
    request: &AccuracyEvaluationRequest,
    receipt: &MaxwellNumericalReceiptArtifact,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[&PreparedAnalysisObjects],
) -> AnalysisBridgeResult<()> {
    request.validate()?;
    receipt.validate()?;
    if inputs.len() != 3
        || request.source_kind != AccuracySourceKind::MaxwellHarmonic
        || request.source_result != receipt.source_result()
        || request.source_observations != observations.map(|catalog| catalog.identity())
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell accuracy requires exact receipt and workload/mesh/retained-result closure",
        ));
    }
    let unique = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matching = inputs
            .iter()
            .copied()
            .filter(|input| input.manifest.kind == kind);
        match (matching.next(), matching.next()) {
            (Some(input), None) => Ok(input),
            _ => Err(AnalysisBridgeError::Identity(
                "Maxwell accuracy requires unique original source inputs",
            )),
        }
    };
    let workload_objects = unique(Kind::MaxwellHarmonicWorkload)?;
    let mesh_objects = unique(Kind::Mesh)?;
    let result_objects = unique(Kind::SolverResult)?;
    for objects in [workload_objects, mesh_objects, result_objects] {
        crate::maxwell_source_admission::authenticate_maxwell_source_objects(
            objects,
            objects.manifest.kind,
        )?;
    }
    // Authenticate immutable envelopes before materializing any numerical source domain.
    let workload_bytes = workload_objects.payload();
    let mesh_bytes = mesh_objects.payload();
    let workload = MaxwellHarmonicWorkload::from_canonical_bytes(&workload_bytes)?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_bytes)
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    mesh.validate_canonical()
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let bytes = result_objects.payload();
    let result = MaxwellHarmonicSolverResult::from_canonical_bytes_for(
        &bytes,
        result_objects.manifest.payload_digest,
        &workload,
        &mesh,
    )?;
    if workload.identity()?.digest() != workload_objects.manifest.logical_identity
        || Digest::from_bytes(*mesh.canonical_digest.bytes())
            != mesh_objects.manifest.logical_identity
        || result.identity()?.digest() != result_objects.manifest.logical_identity
        || receipt.source_workload() != workload.identity()?
        || receipt.source_result() != result.identity()?
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell accuracy canonical source identities differ",
        ));
    }
    receipt.validate_against(&workload, &result, &mesh)?;
    let step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or(AnalysisBridgeError::Identity(
            "Maxwell accuracy authored producing step absent",
        ))?;
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
        step,
    )?;
    if producer != request.producer
        || request.plan != step.accuracy
        || observations.is_some() == step.observations.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell accuracy differs from original authored requirements",
        ));
    }
    if let Some(catalog) = observations {
        catalog.validate_source_result(
            request.producer,
            request.source_kind,
            request.source_result,
        )?;
        catalog.validate_for(
            step,
            ObservationHistorySourceContext::MaxwellHarmonic {
                workload: &workload,
                mesh: &mesh,
            },
        )?;
        let ObservationFieldCatalogSource::SampledRecovery {
            recovery_request,
            sampling,
            ..
        } = catalog.source().source()
        else {
            return Err(AnalysisBridgeError::Identity(
                "Maxwell observations require exact original complex sampled recovery",
            ));
        };
        if recovery_request.as_ref() != &result.native_field_request_from_result()?
            || sampling.as_ref() != result.sampling_topology()
        {
            return Err(AnalysisBridgeError::Identity(
                "Maxwell observation catalog differs from retained native complex result",
            ));
        }
    }
    Ok(())
}
