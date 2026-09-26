//! Authored publication inventory shared by producer admission and result publication.
use super::{
    AnalysisExecutionResult, SemanticStatePublicationRequest, SemanticStatePublicationSource,
    SequenceIdentity, SolverResultIdentity, StatePublicationProfiles, StepIdentity,
};
use runmat_analysis_core::{
    AnalysisModel, ProcedureStep, StateCompleteness, StateKind, StatePersistence,
};
use runmat_canonical_codec::SerializedCanonicalProfile;
use runmat_execution::Digest;

pub fn structural_state_publication_requests(
    model: &AnalysisModel,
    step: &ProcedureStep,
    sources: &[SemanticStatePublicationSource],
    topology_digest: Digest,
    source_result: SolverResultIdentity,
) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
    if !model.sequence.steps.iter().any(|authored| authored == step) {
        return Err(super::AnalysisExecutionError::Invalid(
            "state profile producer step differs from authored sequence".into(),
        ));
    }
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&model.sequence)?,
        step,
    )?;
    step.state_outputs
        .iter()
        .filter(|output| {
            output.completeness == StateCompleteness::Complete
                && output.persistence != StatePersistence::CheckpointOnly
        })
        .filter_map(|output| {
            let source = match output.kind {
                StateKind::StructuralPrimary => {
                    SemanticStatePublicationSource::StructuralPrimaryFields
                }
                StateKind::GeometricStiffness => {
                    SemanticStatePublicationSource::StructuralLinearization
                }
                StateKind::ModalBasis => SemanticStatePublicationSource::ModalBasis,
                StateKind::StructuralResponseEnvelope => {
                    SemanticStatePublicationSource::StructuralResponseEnvelope
                }
                StateKind::StructuralRandomStatistics => {
                    SemanticStatePublicationSource::StructuralRandomStatistics
                }
                _ => return None,
            };
            sources.contains(&source).then(|| {
                SemanticStatePublicationRequest::new(
                    source,
                    producer,
                    output.clone(),
                    topology_digest,
                    source_result,
                )
            })
        })
        .collect()
}

/// Only fixed-width identities are placeholders; the authored declaration inventory is exact.
pub fn structural_state_publication_profiles_encoding_profile(
    model: &AnalysisModel,
    step: &ProcedureStep,
    sources: &[SemanticStatePublicationSource],
    topology_digest: Digest,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&model.sequence)?,
        step,
    )?;
    StatePublicationProfiles::encoding_profile_for_requests(&structural_state_publication_requests(
        model,
        step,
        sources,
        topology_digest,
        SolverResultIdentity::from_step_result(producer, Digest::sha256([]))?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_core::StateId;

    #[test]
    fn authored_inventory_selects_all_structural_sources_and_filters_nonpersistent_states() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let mut step = ProcedureStep::structural_static("profile-inventory");
        let template = step.state_outputs[0].clone();
        let pairs = [
            (
                StateKind::StructuralPrimary,
                SemanticStatePublicationSource::StructuralPrimaryFields,
            ),
            (
                StateKind::GeometricStiffness,
                SemanticStatePublicationSource::StructuralLinearization,
            ),
            (
                StateKind::ModalBasis,
                SemanticStatePublicationSource::ModalBasis,
            ),
            (
                StateKind::StructuralResponseEnvelope,
                SemanticStatePublicationSource::StructuralResponseEnvelope,
            ),
            (
                StateKind::StructuralRandomStatistics,
                SemanticStatePublicationSource::StructuralRandomStatistics,
            ),
        ];
        step.state_outputs = pairs
            .iter()
            .enumerate()
            .map(|(index, (kind, _))| {
                let mut output = template.clone();
                output.kind = *kind;
                output.state_id = StateId(format!("state-{index}"));
                output
            })
            .collect();
        let mut checkpoint = template.clone();
        checkpoint.persistence = StatePersistence::CheckpointOnly;
        checkpoint.state_id = StateId("checkpoint".into());
        step.state_outputs.push(checkpoint);
        model.sequence.steps = vec![step.clone()];
        let topology = Digest::sha256(b"topology");
        let producer = StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&model.sequence).unwrap(),
            &step,
        )
        .unwrap();
        let result =
            SolverResultIdentity::from_step_result(producer, Digest::sha256(b"result")).unwrap();
        for (kind, source) in pairs {
            let requests =
                structural_state_publication_requests(&model, &step, &[source], topology, result)
                    .unwrap();
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].source, source);
            assert_eq!(requests[0].declaration.kind, kind);
            assert_eq!(requests[0].producer, producer);
            let sizing = structural_state_publication_profiles_encoding_profile(
                &model,
                &step,
                &[source],
                topology,
            )
            .unwrap();
            assert_eq!(
                sizing,
                StatePublicationProfiles::encoding_profile_for_requests(&requests).unwrap()
            );
        }
        let mut altered = step.clone();
        altered.state_outputs[0].state_id = StateId("substituted".into());
        assert!(
            structural_state_publication_requests(&model, &altered, &[], topology, result).is_err()
        );
    }
}
