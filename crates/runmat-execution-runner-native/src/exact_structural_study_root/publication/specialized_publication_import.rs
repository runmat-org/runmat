use runmat_analysis_execution::{
    ImportedPublishedResult, PreparedAnalysisObjects, PreparedStructuralResourceUsage,
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
    resource_objects: &PreparedAnalysisObjects,
    publication_objects: &PreparedAnalysisObjects,
    workload_digest: Digest,
    result_digest: Digest,
    accuracy: &AccuracyArtifact,
) -> NativeExecutionResult<(PreparedStructuralResourceUsage, ImportedPublishedResult)> {
    let evidence = ResourceUsageEvidence::from_canonical_bytes(&resource_objects.payload())
        .map_err(protocol)?;
    let primary_plan = evidence
        .plans
        .iter()
        .find(|plan| plan.identity == evidence.primary_plan)
        .ok_or_else(|| protocol("resource evidence omits its primary plan"))?;
    validate_resource_usage_evidence(primary_plan, &evidence).map_err(protocol)?;
    if evidence.identity().digest() != resource_objects.manifest.logical_identity
        || evidence.workload_digest != workload_digest
        || evidence.result_digest != result_digest
        || !evidence.overruns.is_empty()
    {
        return Err(protocol(
            "specialized resource evidence differs from its exact workload or result",
        ));
    }

    let publication = PublishedAnalysisResult::from_canonical_bytes(&publication_objects.payload())
        .map_err(protocol)?;
    publication
        .validate_sources(accuracy, primary_plan, &evidence)
        .map_err(protocol)?;
    if publication.identity().digest() != publication_objects.manifest.logical_identity
        || publication.producer != accuracy.producer
        || publication.source_result != accuracy.source_result
        || publication.source_observations != accuracy.source_observations
        || publication.source_accuracy != accuracy.identity()
        || publication.source_resource_plan != evidence.primary_plan
        || publication.source_resource_usage != evidence.identity()
    {
        return Err(protocol(
            "specialized published result differs from its exact governed sources",
        ));
    }

    Ok((
        PreparedStructuralResourceUsage {
            evidence,
            objects: resource_objects.clone(),
        },
        ImportedPublishedResult {
            result: publication,
            objects: publication_objects.clone(),
        },
    ))
}
