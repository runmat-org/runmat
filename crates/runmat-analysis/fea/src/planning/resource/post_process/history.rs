//! Checkpoint transition admission, including retained execution provenance.
use super::*;
use crate::results::observation::OBSERVATION_LIMITS;
use runmat_canonical_codec::{profile_serialized, SerializedCanonicalProfile};

pub fn build_observation_history_checkpoint_resource_plan(
    request: &ObservationHistoryCheckpointRequest,
    previous: Option<&ObservationHistoryCheckpoint>,
    sample: &ObservationResult,
    inputs: PreparedInputResourceFacts,
    output: PreparedOutputResourceFacts,
    output_set: PreparedOutputSetResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    request
        .validate_advance_inputs(previous, sample)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let observation = request
        .plan
        .observations
        .iter()
        .find(|observation| observation.spec.observation_id == request.observation_id)
        .ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "history resource request names no resolved observation".into(),
            )
        })?;
    let previous_retained = previous.map_or(0, |checkpoint| checkpoint.samples().len());
    let retained_sample_count = previous_retained
        .checked_add(1)
        .map(|count| count.min(observation.sample_bounds.maximum_retained_samples))
        .ok_or(ResourcePlanningError::Overflow(
            "retained history sample count",
        ))?;
    let output_payload_upper_bound =
        observation_history_checkpoint_payload_bytes_upper_bound(request, previous, sample)?;
    validate_projected_output(&output, output_payload_upper_bound, "observation history")?;
    let workspace = checked_add(
        byte_len(
            &sample
                .canonical_bytes()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        )?,
        previous.map_or(Ok(0), observation_history_checkpoint_payload_bytes)?,
        "history transition workspace",
    )?;
    let evidence = execution_evidence_profile(previous, sample)?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralObservationHistoryCheckpoint(request.identity),
        demand: FeaResourceDemand::StructuralObservationHistoryCheckpoint(
            ObservationHistoryCheckpointResourceDemand {
                topology_digest: sample.topology_digest,
                predecessor_present: previous.is_some(),
                predecessor_sample_count: Count(
                    previous.map_or(0, |value| value.manifest().observed_sample_count),
                ),
                retained_sample_count: Count(count(
                    retained_sample_count,
                    "retained history samples",
                )?),
                inputs,
                transition_workspace_bytes_upper_bound: ByteCount(workspace),
                execution_evidence_encoded_bytes_upper_bound: ByteCount(
                    evidence.encoded_bytes_upper_bound,
                ),
                execution_evidence_workspace_bytes_upper_bound: ByteCount(
                    evidence.materialization_resident_bytes_estimate,
                ),
                output,
            },
        ),
        output_set: Some(output_set),
        worker: post_process_worker(),
        accuracy_ceiling: post_process_ceiling(),
    })
}

pub fn observation_history_checkpoint_payload_bytes_upper_bound(
    request: &ObservationHistoryCheckpointRequest,
    previous: Option<&ObservationHistoryCheckpoint>,
    sample: &ObservationResult,
) -> Result<u64, ResourcePlanningError> {
    let observation = request
        .plan
        .observations
        .iter()
        .find(|observation| observation.spec.observation_id == request.observation_id)
        .ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "history resource request names no resolved observation".into(),
            )
        })?;
    let retained_sample_count = previous
        .map_or(0, |checkpoint| checkpoint.samples().len())
        .checked_add(1)
        .map(|count| count.min(observation.sample_bounds.maximum_retained_samples))
        .ok_or(ResourcePlanningError::Overflow(
            "retained history sample count",
        ))?;
    // The next checkpoint keeps at most the previous checkpoint's samples and manifest plus the
    // new sample, with one record envelope per retained sample and one for the manifest.
    let copied = checked_add(
        previous.map_or(Ok(0), observation_history_checkpoint_payload_bytes)?,
        byte_len(
            &sample
                .canonical_bytes()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        )?,
        "history checkpoint copied bytes",
    )?;
    let base = copied_output_payload_bytes_upper_bound(
        copied,
        checked_add(
            count(retained_sample_count, "retained history samples")?,
            1,
            "history output records",
        )?,
        0,
    )?;
    checked_add(
        base,
        execution_evidence_profile(previous, sample)?.encoded_bytes_upper_bound,
        "history execution evidence output",
    )
}

pub(in crate::planning::resource) fn observation_history_checkpoint_payload_bytes(
    checkpoint: &ObservationHistoryCheckpoint,
) -> Result<u64, ResourcePlanningError> {
    let manifest = checkpoint
        .manifest()
        .canonical_bytes()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    checkpoint
        .samples()
        .iter()
        .try_fold(byte_len(&manifest)?, |total, sample| {
            checked_add(
                total,
                byte_len(
                    &sample
                        .canonical_bytes()
                        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
                )?,
                "history checkpoint payload bytes",
            )
        })
}

// Manifest evidence is an additional copy of each retained result's variable-length execution
// record. Count it explicitly rather than spending a fixed per-sample overhead on arbitrarily
// many partition IDs/provider records. Including the sample about to be evicted is conservative.
// The borrowed profiler neither evaluates fields nor constructs canonical buffers/value trees.
fn execution_evidence_profile(
    previous: Option<&ObservationHistoryCheckpoint>,
    sample: &ObservationResult,
) -> Result<SerializedCanonicalProfile, ResourcePlanningError> {
    let mut evidence = previous
        .into_iter()
        .flat_map(|checkpoint| checkpoint.manifest().partition_evidence.iter())
        .chain(std::iter::once(&sample.partition_evidence));
    evidence.try_fold(
        SerializedCanonicalProfile {
            encoded_bytes_upper_bound: 0,
            materialization_resident_bytes_estimate: 0,
        },
        |total, item| {
            // Match the observation contract's bounds. Profiling is a conservative admission
            // estimate, not an allocator measurement or a replacement for domain validation.
            let profile = profile_serialized(item, OBSERVATION_LIMITS)
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            Ok(SerializedCanonicalProfile {
                encoded_bytes_upper_bound: checked_add(
                    total.encoded_bytes_upper_bound,
                    profile.encoded_bytes_upper_bound,
                    "history execution evidence encoding",
                )?,
                materialization_resident_bytes_estimate: checked_add(
                    total.materialization_resident_bytes_estimate,
                    profile.materialization_resident_bytes_estimate,
                    "history execution evidence materialization",
                )?,
            })
        },
    )
}
