//! Original stage provenance is serialized from borrowed fields, never a cloned
//! observation. The legacy clone remains only as an independent test oracle.
use super::*;
#[derive(Serialize)]
struct OriginalObservation<'a> {
    stage: MeshingStageKind,
    partition: &'a MeshingPartitionDescriptor,
    stage_result_digest: StableDigest,
    entity_counts: &'a BTreeMap<String, u64>,
    invariants: &'a [InvariantEvidence],
    achieved_error_distributions: &'a BTreeMap<String, ErrorDistribution>,
    completed_work: u64,
    estimated_work: u64,
    peak_memory_bytes: u64,
    peak_scratch_bytes: u64,
    search_work: u64,
    maximum_recursion_depth: u32,
    iterations: u64,
    elapsed_time_ms: u64,
    cancellation_checkpoints: u64,
}
impl<'a> From<&'a MeshingStageEvidence> for OriginalObservation<'a> {
    fn from(value: &'a MeshingStageEvidence) -> Self {
        Self {
            stage: value.stage,
            partition: &value.partition,
            stage_result_digest: value.stage_result_digest,
            entity_counts: &value.entity_counts,
            invariants: &value.invariants,
            achieved_error_distributions: &value.achieved_error_distributions,
            completed_work: value.completed_work,
            estimated_work: value.estimated_work,
            peak_memory_bytes: value.peak_memory_bytes,
            peak_scratch_bytes: value.peak_scratch_bytes,
            search_work: value.search_work,
            maximum_recursion_depth: value.maximum_recursion_depth,
            iterations: value.iterations,
            elapsed_time_ms: value.elapsed_time_ms,
            cancellation_checkpoints: value.cancellation_checkpoints,
        }
    }
}
pub(super) fn checkpoint(control: &dyn CanonicalCodecControl) -> Result<(), MeshingContractError> {
    if control.is_cancelled() {
        return Err(MeshingContractError::invalid(
            "meshing evidence validation",
            "operation interrupted",
        ));
    }
    Ok(())
}
pub(super) fn original_digest(
    value: &MeshingStageEvidence,
    control: &dyn CanonicalCodecControl,
) -> Result<StableDigest, MeshingContractError> {
    super::super::codec::digest_borrowed_contract(
        MeshingStageEvidence::DOMAIN,
        &OriginalObservation::from(value),
        MeshingStageEvidence::LIMITS,
        control,
    )
}
pub(super) fn validation_workspace(
    value: &MeshingEvidence,
    control: &dyn CanonicalCodecControl,
) -> Result<u64, MeshingContractError> {
    value
        .stages
        .iter()
        .filter(|stage| stage.reused_from.is_some())
        .try_fold(0, |maximum, stage| {
            checkpoint(control)?;
            let profile = runmat_canonical_codec::profile_canonical_encoding_with_control(
                &OriginalObservation::from(stage),
                MeshingStageEvidence::LIMITS.into(),
                control,
            )
            .map_err(|error| MeshingContractError::invalid(error.field, error.reason))?;
            Ok(maximum.max(profile.resident_workspace_bytes_estimate))
        })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_original_observation_matches_all_owned_canonical_fields() {
        let artifact = super::super::super::artifact_tests::artifact();
        let evidence = super::super::super::artifact_tests::evidence(&artifact);
        let control = super::super::super::MeshingCanonicalCodecControl::new(
            &super::super::super::NeverCancelled,
        );
        for stage in evidence.stages {
            let expected = stage.canonical_digest().unwrap();
            let mut reused = stage;
            reused.reused_from = Some(ReusedMeshingStageObservation {
                original_observation_digest: expected,
                origin: ReusedMeshingStageOrigin::ImplementationCache {
                    implementation_digest: StableDigest::from_bytes([9; 32]),
                },
            });
            assert_eq!(original_digest(&reused, &control).unwrap(), expected);
            reused.validate().unwrap();
            reused.completed_work = reused.completed_work.saturating_sub(1);
            // Field equivalence is asserted independently of whether this change
            // remains valid as a successful observation.
            let mut owned = reused.clone();
            owned.reused_from = None;
            let expected = super::super::super::codec::digest_borrowed_contract(
                MeshingStageEvidence::DOMAIN,
                &owned,
                MeshingStageEvidence::LIMITS,
                &control,
            )
            .unwrap();
            assert_eq!(original_digest(&reused, &control).unwrap(), expected);
        }
    }

    #[test]
    fn reused_stage_timings_may_exceed_the_restoring_run_wall_time() {
        let artifact = super::super::super::artifact_tests::artifact();
        let mut evidence = super::super::super::artifact_tests::evidence(&artifact);
        evidence.stages[0].elapsed_time_ms = 1_000;
        evidence.resources.wall_time_ms = 10;
        let fresh = evidence.validate(&artifact).unwrap_err();
        assert!(fresh
            .to_string()
            .contains("stage resource measurements exceed the aggregate resource evidence"));
        let original = evidence.stages[0].canonical_digest().unwrap();
        evidence.stages[0].reused_from = Some(ReusedMeshingStageObservation {
            original_observation_digest: original,
            origin: ReusedMeshingStageOrigin::ImplementationCache {
                implementation_digest: StableDigest::from_bytes([9; 32]),
            },
        });
        evidence.validate(&artifact).unwrap();
        evidence.stages[0].reused_from.as_mut().unwrap().origin =
            ReusedMeshingStageOrigin::AcceptedInvocation {
                invocation_identity_digest: StableDigest::from_bytes([10; 32]),
            };
        evidence.validate(&artifact).unwrap();
        evidence.stages[0].reused_from.as_mut().unwrap().origin =
            ReusedMeshingStageOrigin::AcceptedInvocation {
                invocation_identity_digest: StableDigest::from_bytes([0; 32]),
            };
        assert!(evidence.validate(&artifact).is_err());
    }
}
