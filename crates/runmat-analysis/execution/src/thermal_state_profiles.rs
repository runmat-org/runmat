//! Source-bound thermal publication profiles derived from metadata, never field arrays.
use crate::{
    decode_prepared_thermal_manifest, AnalysisBridgeError, AnalysisBridgeResult,
    PreparedAnalysisObjects,
};
use runmat_analysis_fea::execution::{
    SemanticStatePublicationProfileSource, SequenceIdentity, StatePublicationProfileEntry,
    StatePublicationProfiles, StepIdentity, ThermalWorkload,
};

/// Project published-state wire shapes from the prepared result's metadata partition only.
pub fn thermal_state_publication_profiles(
    workload: &ThermalWorkload,
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StatePublicationProfiles> {
    workload.validate()?;
    let manifest = decode_prepared_thermal_manifest(objects)?;
    let (producer, _) = selected_step(workload)?;
    if manifest.source_workload != workload.identity()?
        || manifest.numerical_intent != workload.numerical_intent_identity()?
        || manifest.producer != producer
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal publication profile source differs from workload or producer",
        ));
    }
    let entries = workload
        .state_publication_requests(manifest.topology_digest, manifest.identity())?
        .into_iter()
        .map(|request| {
            let profile = request
                .encoding_profile(SemanticStatePublicationProfileSource::Thermal(&manifest))?;
            Ok(StatePublicationProfileEntry::new(request, profile)?)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    Ok(StatePublicationProfiles::new(
        producer,
        manifest.identity(),
        manifest.topology_digest,
        objects.manifest.payload_digest,
        entries,
    )?)
}

fn selected_step(
    workload: &ThermalWorkload,
) -> AnalysisBridgeResult<(StepIdentity, &runmat_analysis_core::ProcedureStep)> {
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence)?;
    let step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or_else(|| AnalysisBridgeError::Invalid("thermal step absent".into()))?;
    Ok((StepIdentity::from_solver_contract(sequence, step)?, step))
}
