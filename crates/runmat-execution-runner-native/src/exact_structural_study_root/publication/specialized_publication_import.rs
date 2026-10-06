use runmat_analysis_execution::{
    validate_retained_result_publication_resource_usage, ImportedPublishedResult,
    PreparedAnalysisObjects, PreparedResourceUsage,
};
use runmat_analysis_fea::planning::resource::{
    validate_resource_usage_evidence, ResourceUsageEvidence,
};
use runmat_analysis_fea::results::accuracy::AccuracyArtifact;
use runmat_analysis_fea::results::publication::PublishedAnalysisResult;
use runmat_execution::Digest;

use super::protocol;
use crate::NativeExecutionResult;

pub(super) fn import_specialized_publication(
    publication_baseline_usage_objects: &PreparedAnalysisObjects,
    resource_usage_objects: &PreparedAnalysisObjects,
    publication_objects: &PreparedAnalysisObjects,
    workload_digest: Digest,
    result_digest: Digest,
    accuracy: &AccuracyArtifact,
) -> NativeExecutionResult<(
    PreparedResourceUsage,
    PreparedResourceUsage,
    ImportedPublishedResult,
)> {
    let publication_baseline_usage = import_resource_usage(
        publication_baseline_usage_objects,
        workload_digest,
        result_digest,
        "publication baseline",
    )?;
    let resource_usage = import_resource_usage(
        resource_usage_objects,
        workload_digest,
        result_digest,
        "terminal",
    )?;

    let publication = PublishedAnalysisResult::from_canonical_bytes(&publication_objects.payload())
        .map_err(protocol)?;
    publication
        .validate_sources(accuracy, &publication_baseline_usage.evidence)
        .map_err(protocol)?;
    if publication.identity().digest() != publication_objects.manifest.logical_identity
        || publication.producer != accuracy.producer
        || publication.source_result != accuracy.source_result
        || publication.source_observations != accuracy.source_observations
        || publication.source_accuracy != accuracy.identity()
        || publication.source_resource_plan != publication_baseline_usage.evidence.primary_plan
        || publication.source_resource_usage != publication_baseline_usage.evidence.identity()
    {
        return Err(protocol(
            "specialized published result differs from its exact governed sources",
        ));
    }
    validate_retained_result_publication_resource_usage(
        &publication,
        &publication_baseline_usage.evidence,
        &resource_usage.evidence,
    )
    .map_err(protocol)?;

    Ok((
        publication_baseline_usage,
        resource_usage,
        ImportedPublishedResult {
            result: publication,
            objects: publication_objects.clone(),
        },
    ))
}

fn import_resource_usage(
    objects: &PreparedAnalysisObjects,
    workload_digest: Digest,
    result_digest: Digest,
    role: &str,
) -> NativeExecutionResult<PreparedResourceUsage> {
    let evidence =
        ResourceUsageEvidence::from_canonical_bytes(&objects.payload()).map_err(protocol)?;
    let primary_plan = evidence
        .plans
        .iter()
        .find(|plan| plan.identity == evidence.primary_plan)
        .ok_or_else(|| protocol(format!("{role} resource evidence omits its primary plan")))?;
    validate_resource_usage_evidence(primary_plan, &evidence).map_err(protocol)?;
    if evidence.identity().digest() != objects.manifest.logical_identity
        || evidence.workload_digest != workload_digest
        || evidence.result_digest != result_digest
        || !evidence.overruns.is_empty()
    {
        return Err(protocol(format!(
            "specialized {role} resource evidence differs from its exact workload or result"
        )));
    }
    Ok(PreparedResourceUsage {
        evidence,
        objects: objects.clone(),
    })
}
