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
    results::{
        accuracy::AccuracyArtifact, observation::ObservationPublicationCatalog,
        publication::published_result_encoding_profile,
    },
};
use std::borrow::Cow;

pub fn build_prepared_result_publication_resource_plan(
    request: &ResultPublicationRequest,
    accuracy: &AccuracyArtifact,
    usage: &ResourceUsageEvidence,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<FeaResourcePlan> {
    build_plan(request, accuracy, usage, inputs, None)
}

/// As [`build_prepared_result_publication_resource_plan`], with the typed observation catalog
/// whose canonical bytes the caller just prepared as the observation input: no decode.
pub(crate) fn build_result_publication_resource_plan_on(
    request: &ResultPublicationRequest,
    accuracy: &AccuracyArtifact,
    usage: &ResourceUsageEvidence,
    inputs: &[PreparedAnalysisObjects],
    observations: Option<&ObservationPublicationCatalog>,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    build_plan(
        request,
        accuracy,
        usage,
        inputs,
        observations.map(Cow::Borrowed),
    )
}

fn build_plan(
    request: &ResultPublicationRequest,
    accuracy: &AccuracyArtifact,
    usage: &ResourceUsageEvidence,
    inputs: &[PreparedAnalysisObjects],
    typed_observations: Option<Cow<'_, ObservationPublicationCatalog>>,
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
    // A typed catalog stands for its input only when its identity is the input's (checked
    // against the request above); otherwise the input bytes are decoded.
    let observations = match (request.source_observations, typed_observations) {
        (None, _) => None,
        (Some(identity), Some(catalog)) if catalog.identity() == identity => Some(catalog),
        (Some(_), _) => Some(Cow::Owned(
            ObservationPublicationCatalog::from_canonical_bytes(
                &unique(AnalysisNumericalObjectKind::ObservationCatalog)?.payload(),
            )?,
        )),
    };
    let profile = published_result_encoding_profile(request, accuracy, usage)?;
    Ok(build_result_publication_resource_plan(
        request,
        accuracy,
        usage,
        observations.as_deref(),
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_output_resource_facts(profile.encoded_bytes_upper_bound)?,
            output_set: projected_output_set_resource_facts(),
        },
    )?)
}
