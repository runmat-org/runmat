//! Reprojection shared by admission and workers. Context is bound to the producing receipt,
//! independently of the observation catalog's own declarations and hashes.
mod acoustic;
mod acoustic_response;
mod electrical;
pub(crate) mod fluid;
mod magnetic;
mod maxwell;
mod nonlinear;
mod structural;
use runmat_analysis_fea::{
    execution::{
        AccuracyEvaluationRequest, AccuracySourceReceipt, AnalysisNumericalObjectKind as Kind,
        ObservationHistorySourceContext, SequenceIdentity, StepIdentity, ThermalWorkload,
    },
    results::observation::ObservationPublicationCatalog,
};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
pub(crate) use structural::prepare_contract as prepare_accuracy_source_contract;

use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};

pub fn validate_accuracy_source_context(
    request: &AccuracyEvaluationRequest,
    receipt: &AccuracySourceReceipt,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[&PreparedAnalysisObjects],
) -> AnalysisBridgeResult<()> {
    request.validate()?;
    receipt.validate()?;
    if request.source_kind != receipt.kind()
        || request.source_result != receipt.source_result()
        || request.source_observations != observations.map(|catalog| catalog.identity())
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy source context differs from request",
        ));
    }
    if let AccuracySourceReceipt::FluidSteady(receipt) = receipt {
        return fluid::validate(request, receipt, observations, inputs);
    }
    if let AccuracySourceReceipt::NonlinearSolid(receipt) = receipt {
        return nonlinear::validate(request, receipt, observations, inputs);
    }
    if let AccuracySourceReceipt::AcousticHarmonic(receipt) = receipt {
        return acoustic::validate(request, receipt, observations, inputs);
    }
    if let AccuracySourceReceipt::AcousticResponse(receipt) = receipt {
        return acoustic_response::validate(request, receipt, observations, inputs);
    }
    if let AccuracySourceReceipt::ElectricalStatic(receipt) = receipt {
        return electrical::validate(request, receipt, observations, inputs);
    }
    if let AccuracySourceReceipt::Magnetostatic(receipt) = receipt {
        return magnetic::validate(request, receipt, observations, inputs);
    }
    if let AccuracySourceReceipt::MaxwellHarmonic(receipt) = receipt {
        return maxwell::validate(request, receipt, observations, inputs);
    }
    let AccuracySourceReceipt::Thermal(receipt) = receipt else {
        return structural::validate(request, receipt, observations, inputs);
    };
    let unique = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matching = inputs.iter().filter(|input| input.manifest.kind == kind);
        match (matching.next(), matching.next()) {
            (Some(input), None) => Ok(*input),
            _ => Err(AnalysisBridgeError::Identity(
                "thermal accuracy requires unique workload and mesh inputs",
            )),
        }
    };
    if inputs.len() != 2 {
        return Err(AnalysisBridgeError::Identity(
            "thermal accuracy requires original workload and mesh",
        ));
    }
    let workload_objects = unique(Kind::ThermalWorkload)?;
    let mesh_objects = unique(Kind::Mesh)?;
    let workload = ThermalWorkload::from_canonical_bytes(&workload_objects.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let workload_identity = workload.identity()?;
    let mesh_identity = Digest::from_bytes(*mesh.canonical_digest.bytes());
    if workload_identity.digest() != workload_objects.manifest.logical_identity
        || workload_identity != receipt.source_workload
        || mesh_identity != mesh_objects.manifest.logical_identity
        || mesh_identity != receipt.receipt().topology_digest
        || receipt.receipt().producer != request.producer
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal accuracy context differs from numerical receipt",
        ));
    }
    // Validation of the workload guarantees a unique selected step.
    let step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or(AnalysisBridgeError::Identity(
            "thermal accuracy source step is absent",
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
            "thermal accuracy differs from authored requirements",
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
            ObservationHistorySourceContext::Thermal {
                workload: &workload,
                mesh: &mesh,
            },
        )?;
    }
    Ok(())
}
