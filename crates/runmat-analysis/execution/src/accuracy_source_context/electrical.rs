//! Exact electrical numerical intent AND retained result bind accuracy evidence.
//! Reconstructing this authority performs no electrical assembly or numerical solve.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::{
    execution::{
        AccuracyEvaluationRequest, AnalysisNumericalObjectKind as Kind,
        ElectricalNumericalReceiptArtifact, ElectricalStaticSolverResult, ElectricalStaticWorkload,
        ObservationHistorySourceContext, SequenceIdentity, StepIdentity,
    },
    results::observation::{ObservationFieldCatalogSource, ObservationPublicationCatalog},
};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub(super) fn validate(
    request: &AccuracyEvaluationRequest,
    receipt: &ElectricalNumericalReceiptArtifact,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[&PreparedAnalysisObjects],
) -> AnalysisBridgeResult<()> {
    if inputs.len() != 3 {
        return Err(AnalysisBridgeError::Identity(
            "electrical accuracy requires exact workload, mesh and retained result",
        ));
    }
    let unique = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = inputs
            .iter()
            .copied()
            .filter(|input| input.manifest.kind == kind);
        match (matches.next(), matches.next()) {
            (Some(input), None) => Ok(input),
            _ => Err(AnalysisBridgeError::Identity(
                "electrical accuracy requires unique source inputs",
            )),
        }
    };
    let workload_objects = unique(Kind::ElectricalStaticWorkload)?;
    let mesh_objects = unique(Kind::Mesh)?;
    let result_objects = unique(Kind::SolverResult)?;
    let workload = ElectricalStaticWorkload::from_canonical_bytes(&workload_objects.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let result = ElectricalStaticSolverResult::from_canonical_bytes(&result_objects.payload())?;
    if workload.identity()?.digest() != workload_objects.manifest.logical_identity
        || Digest::from_bytes(*mesh.canonical_digest.bytes())
            != mesh_objects.manifest.logical_identity
        || result.source_result_identity()?.digest() != result_objects.manifest.logical_identity
        || receipt.source_workload() != workload.identity()?
        || receipt.source_result() != result.source_result_identity()?
    {
        return Err(AnalysisBridgeError::Identity(
            "electrical accuracy source canonical identities differ",
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
            "electrical accuracy source step is absent",
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
            "electrical accuracy differs from authored requirements",
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
            ObservationHistorySourceContext::VolumeMesh { mesh: &mesh },
        )?;
        let ObservationFieldCatalogSource::VolumeMeshRecovery {
            recovery_request, ..
        } = catalog.source().source()
        else {
            return Err(AnalysisBridgeError::Identity(
                "electrical accuracy observations require the original volume-mesh recovery",
            ));
        };
        if recovery_request.as_ref() != &result.native_field_request(&workload, &mesh)? {
            return Err(AnalysisBridgeError::Identity(
                "electrical accuracy observations differ from original native result",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
