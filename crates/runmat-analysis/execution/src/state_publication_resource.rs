//! Shared preparation/worker admission for physical-state publication.
use crate::{
    prepared_input_resource_facts_without_resource_plan, projected_output_set_resource_facts,
    AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, SemanticStatePublicationRequest, StatePublicationProfiles,
    },
    planning::resource::{
        build_state_publication_resource_plan, FeaResourcePlan, PreparedPhaseResourceFacts,
    },
};
use runmat_execution::Digest;

/// Inspects canonical metadata and shared input storage profiles, not decoded numerical fields.
/// The worker additionally reprojects the selected state from its decoded numerical source
/// before constructing that state; a catalog hash alone does not certify its size.
pub fn build_prepared_state_publication_resource_plan(
    request: &SemanticStatePublicationRequest,
    catalog: &StatePublicationProfiles,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let numerical: Vec<_> = inputs
        .iter()
        .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .collect();
    if numerical.len() != 3
        || inputs
            .iter()
            .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .count()
            > 1
    {
        return Err(AnalysisBridgeError::Invalid(
            "state publication requires exactly request, source result and profile catalog".into(),
        ));
    }
    let unique = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = numerical
            .iter()
            .copied()
            .filter(|input| input.manifest.kind == kind);
        let value = matches.next().ok_or_else(|| {
            AnalysisBridgeError::Invalid(format!("state publication is missing {kind:?}"))
        })?;
        if matches.next().is_some() {
            return Err(AnalysisBridgeError::Invalid(format!(
                "state publication repeats {kind:?}"
            )));
        }
        Ok(value)
    };
    let source = unique(AnalysisNumericalObjectKind::SolverResult)?;
    let profiles = unique(AnalysisNumericalObjectKind::StatePublicationProfiles)?;
    let request_objects = unique(AnalysisNumericalObjectKind::StatePublicationRequest)?;
    if source.manifest.logical_identity != request.source_result.digest()
        || profiles.manifest.logical_identity != catalog.identity()
        || request_objects.manifest.logical_identity != Digest::sha256(request.canonical_bytes()?)
    {
        return Err(AnalysisBridgeError::Identity(
            "state publication resource inputs differ from their canonical identities",
        ));
    }
    catalog.validate_source(
        request.producer,
        request.source_result,
        request.topology_digest,
        source.manifest.payload_digest,
    )?;
    let profile = catalog.entry(request)?.profile();
    Ok(build_state_publication_resource_plan(
        request,
        catalog,
        source.manifest.payload_digest,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output:
                runmat_analysis_fea::planning::resource::profiled_canonical_output_resource_facts(
                    profile,
                )?,
            output_set: projected_output_set_resource_facts(),
        },
    )?)
}
