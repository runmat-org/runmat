//! One admission plan for receipt publication; source plans live in the usage catalog.
use crate::{
    prepared_input_resource_facts_without_resource_plan, projected_output_resource_facts,
    projected_output_set_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    PreparedAnalysisObjects,
};
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind, ResultPublicationRequest},
    planning::resource::{
        build_result_publication_resource_plan, FeaResourcePlan, PreparedPhaseResourceFacts,
        ResourceUsageEvidence,
    },
    results::{accuracy::AccuracyArtifact, publication::published_result_encoding_profile},
};

pub fn build_prepared_result_publication_resource_plan(
    request: &ResultPublicationRequest,
    accuracy: &AccuracyArtifact,
    usage: &ResourceUsageEvidence,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let numerical: Vec<_> = inputs
        .iter()
        .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .collect();
    if numerical.len() != 3 + usize::from(request.source_observations.is_some())
        || inputs.len() - numerical.len() > 1
    {
        return Err(AnalysisBridgeError::Invalid(
            "publication requires request, accuracy, usage, optional observations and at most one admission plan".into(),
        ));
    }
    let unique = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = numerical
            .iter()
            .copied()
            .filter(|input| input.manifest.kind == kind);
        let value = matches.next().ok_or_else(|| {
            AnalysisBridgeError::Invalid(format!("publication is missing {kind:?}"))
        })?;
        if matches.next().is_some() {
            return Err(AnalysisBridgeError::Invalid(format!(
                "publication repeats {kind:?}"
            )));
        }
        Ok(value)
    };
    if unique(AnalysisNumericalObjectKind::PublicationRequest)?
        .manifest
        .logical_identity
        != request.identity().digest()
        || unique(AnalysisNumericalObjectKind::AccuracyReport)?
            .manifest
            .logical_identity
            != accuracy.identity().digest()
        || unique(AnalysisNumericalObjectKind::ResourceUsage)?
            .manifest
            .logical_identity
            != usage.identity().digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "publication resource inputs differ from canonical metadata",
        ));
    }
    if let Some(observations) = request.source_observations {
        if unique(AnalysisNumericalObjectKind::ObservationCatalog)?
            .manifest
            .logical_identity
            != observations.digest()
        {
            return Err(AnalysisBridgeError::Identity(
                "publication observation input differs from request",
            ));
        }
    }
    // Reconstruct from the source catalog at both worker admission and import. A receipt's
    // hashes alone cannot prove that an otherwise-valid usage graph retains required tasks.
    let observations = request.source_observations.map(|_| {
        runmat_analysis_fea::results::observation::ObservationPublicationCatalog::from_canonical_bytes(
            &unique(AnalysisNumericalObjectKind::ObservationCatalog)?.payload(),
        ).map_err(AnalysisBridgeError::from)
    }).transpose()?;
    if ResultPublicationRequest::new(accuracy, observations.as_ref(), usage)? != *request {
        return Err(AnalysisBridgeError::Identity(
            "publication request differs from source-bound task accounting",
        ));
    }
    let profile = published_result_encoding_profile(request, accuracy, usage)?;
    Ok(build_result_publication_resource_plan(
        request,
        accuracy,
        usage,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_output_resource_facts(profile.encoded_bytes_upper_bound)?,
            output_set: projected_output_set_resource_facts(),
        },
    )?)
}
