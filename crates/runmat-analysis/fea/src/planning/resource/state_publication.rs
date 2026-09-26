//! Publication admission from a producer-bound canonical profile, without constructing a state.
use super::*;
use crate::execution::{SemanticStatePublicationRequest, StatePublicationProfiles};
use runmat_execution::Digest;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod workspace_tests;

/// `facts.inputs` must describe the request, source result and profile catalog. The transport
/// boundary rederives those facts from the exact immutable inputs on both driver and worker;
/// the resource-plan input itself is reserved by the shared estimator. A catalog hash is not
/// proof of a truthful profile: the worker must also reproject its source before publication.
pub fn build_state_publication_resource_plan(
    request: &SemanticStatePublicationRequest,
    catalog: &StatePublicationProfiles,
    source_payload_digest: Digest,
    facts: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let invalid = |error: &dyn std::fmt::Display| ResourcePlanningError::Invalid(error.to_string());
    catalog
        .validate_source(
            request.producer,
            request.source_result,
            request.topology_digest,
            source_payload_digest,
        )
        .map_err(|error| invalid(&error))?;
    let entry = catalog.entry(request).map_err(|error| invalid(&error))?;
    let request_bytes = request.canonical_bytes().map_err(|error| invalid(&error))?;
    let request_identity = Digest::sha256(&request_bytes);
    let metadata_bytes = request_bytes
        .len()
        .checked_add(
            catalog
                .canonical_bytes()
                .map_err(|error| invalid(&error))?
                .len(),
        )
        .ok_or(ResourcePlanningError::Overflow(
            "publication metadata bytes",
        ))?;
    if facts.inputs.encoded_payload_bytes.0 <= metadata_bytes as u64 {
        return Err(ResourcePlanningError::Invalid(
            "publication input facts omit source result or metadata".into(),
        ));
    }
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::SemanticStatePublication(request_identity),
        demand: FeaResourceDemand::SemanticStatePublication(
            SemanticStatePublicationResourceDemand {
                request: request_identity,
                source: request.source,
                source_result: request.source_result,
                source_payload_digest,
                topology_digest: request.topology_digest,
                profile_catalog: catalog.identity(),
                state_encoded_bytes_upper_bound: ByteCount(entry.encoded_bytes_upper_bound),
                state_materialization_resident_bytes_estimate: ByteCount(
                    entry.materialization_resident_bytes_estimate,
                ),
                source_validation_workspace_bytes: facts
                    .inputs
                    .decoded_resident_bytes_admission_budget,
                inputs: facts.inputs,
                output: facts.output,
            },
        ),
        output_set: Some(facts.output_set),
        worker: super::post_process::post_process_worker(),
        accuracy_ceiling: super::post_process::post_process_ceiling(),
    })
}
