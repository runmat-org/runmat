//! Exact original finite-volume authority; never substitutes nodal fields or solves.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::{
    execution::{
        AccuracyEvaluationRequest, AnalysisNumericalObjectKind as Kind,
        FluidNumericalReceiptArtifact, FluidSteadySolverResult, FluidSteadyWorkload,
        ObservationHistorySourceContext,
    },
    results::observation::{ObservationFieldCatalogSource, ObservationPublicationCatalog},
};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
fn unique<'a>(
    inputs: &[&'a PreparedAnalysisObjects],
    kind: Kind,
) -> AnalysisBridgeResult<&'a PreparedAnalysisObjects> {
    let mut matching = inputs.iter().copied().filter(|i| i.manifest.kind == kind);
    match (matching.next(), matching.next()) {
        (Some(i), None) => Ok(i),
        _ => Err(invalid("fluid accuracy exact source inventory")),
    }
}
fn original_source(
    request: &AccuracyEvaluationRequest,
    inputs: &[&PreparedAnalysisObjects],
) -> AnalysisBridgeResult<(FluidSteadyWorkload, SolverMeshArtifact)> {
    if inputs.len() != 3 {
        return Err(invalid(
            "fluid accuracy requires original workload/mesh/result",
        ));
    }
    for i in inputs {
        crate::accuracy_resource::authenticate_accuracy_objects(i)?;
    }
    let w = unique(inputs, Kind::FluidSteadyWorkload)?;
    let m = unique(inputs, Kind::Mesh)?;
    unique(inputs, Kind::SolverResult)?;
    let facts = crate::object::prepared_input_resource_facts_from_refs(inputs.iter().copied())?;
    if facts.decoded_resident_bytes_admission_budget.0
        > request.plan.resources.resident_byte_budget()
    {
        return Err(invalid(
            "fluid accuracy source decode exceeds authored ceiling",
        ));
    }
    let workload = FluidSteadyWorkload::from_canonical_bytes(&w.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&m.payload()).map_err(invalid)?;
    let [step] = workload.model.sequence.steps.as_slice() else {
        return Err(invalid("fluid accuracy source step"));
    };
    if workload.identity()?.digest() != w.manifest.logical_identity
        || Digest::from_bytes(mesh.canonical_digest.0) != m.manifest.logical_identity
        || workload.producer_identity()? != request.producer
        || request.plan != step.accuracy
    {
        return Err(invalid(
            "fluid accuracy differs from original authored source",
        ));
    }
    Ok((workload, mesh))
}

pub(crate) fn resource_workspace(
    request: &AccuracyEvaluationRequest,
    inputs: &[&PreparedAnalysisObjects],
    maximum: u64,
) -> AnalysisBridgeResult<u64> {
    let (w, m) = original_source(request, inputs)?;
    Ok(runmat_analysis_fea::planning::resource::fluid_accuracy_source_workspace(&w, &m, maximum)?)
}

pub(super) fn validate(
    request: &AccuracyEvaluationRequest,
    receipt: &FluidNumericalReceiptArtifact,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[&PreparedAnalysisObjects],
) -> AnalysisBridgeResult<()> {
    let (w, m) = original_source(request, inputs)?;
    let original = unique(inputs, Kind::SolverResult)?;
    let payload = original.payload();
    let result = FluidSteadySolverResult::decode_on_source(
        &payload,
        original.manifest.payload_digest,
        &w,
        &m,
    )?;
    if result.result().identity()?.digest() != original.manifest.logical_identity {
        return Err(invalid("fluid accuracy retained result identity"));
    }
    receipt.validate_against_on(&result, &m)?;
    let step = &w.model.sequence.steps[0];
    if observations.is_some() == step.observations.is_empty() {
        return Err(invalid("fluid accuracy original observations presence"));
    }
    if let Some(catalog) = observations {
        catalog.validate_source_result(
            request.producer,
            request.source_kind,
            request.source_result,
        )?;
        catalog.validate_for(
            step,
            ObservationHistorySourceContext::VolumeMesh { mesh: &m },
        )?;
        let ObservationFieldCatalogSource::VolumeMeshRecovery {
            recovery_request,
            manifest,
            ..
        } = catalog.source().source()
        else {
            return Err(invalid(
                "fluid accuracy requires the original native volume-mesh catalog",
            ));
        };
        let original_request = result.native_field_request()?;
        if recovery_request.as_ref() != &original_request {
            return Err(invalid(
                "fluid accuracy catalog differs from original native source",
            ));
        }
        let original_fields = result.recover_native_fields(&original_request)?;
        if manifest.as_ref() != &original_fields.manifest().map_err(invalid)? {
            return Err(invalid(
                "fluid accuracy catalog field storage differs from original U/p/q",
            ));
        }
    }
    Ok(())
}
