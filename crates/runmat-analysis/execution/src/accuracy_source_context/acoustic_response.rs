//! Exact acoustic response numerical intent AND retained result bind accuracy evidence.
//! Reconstructing this authority performs no acoustic response assembly or numerical solve.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::{
    execution::{
        AccuracyEvaluationRequest, AnalysisNumericalObjectKind as Kind,
        AcousticResponseNumericalReceiptArtifact, AcousticResponseSolverResult, AcousticResponseWorkload,
        SequenceIdentity, StepIdentity,
    },
    results::observation::ObservationPublicationCatalog,
};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub(super) fn validate(
    request: &AccuracyEvaluationRequest,
    receipt: &AcousticResponseNumericalReceiptArtifact,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[&PreparedAnalysisObjects],
) -> AnalysisBridgeResult<()> {
    if observations.is_some() || inputs.len() != 3 {
        return Err(AnalysisBridgeError::Identity(
            "acoustic response accuracy requires exact workload, mesh and retained result, without unsupported observations",
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
                "acoustic response accuracy requires unique source inputs",
            )),
        }
    };
    let workload_objects = unique(Kind::AcousticResponseWorkload)?;
    let mesh_objects = unique(Kind::Mesh)?;
    let result_objects = unique(Kind::SolverResult)?;
    let workload = AcousticResponseWorkload::from_canonical_bytes(&workload_objects.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let result = AcousticResponseSolverResult::from_canonical_bytes(&result_objects.payload())?;
    if workload.identity()?.digest() != workload_objects.manifest.logical_identity
        || Digest::from_bytes(*mesh.canonical_digest.bytes())
            != mesh_objects.manifest.logical_identity
        || result.source_result_identity()?.digest() != result_objects.manifest.logical_identity
        || receipt.source_workload() != workload.identity()?
        || receipt.source_result() != result.source_result_identity()?
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic response accuracy source canonical identities differ",
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
            "acoustic response accuracy source step is absent",
        ))?;
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
        step,
    )?;
    if producer != request.producer
        || request.plan != step.accuracy
        || request.formulation_requirement != workload.formulation_requirement
        || !step.observations.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic response accuracy differs from authored requirements",
        ));
    }
    Ok(())
}

