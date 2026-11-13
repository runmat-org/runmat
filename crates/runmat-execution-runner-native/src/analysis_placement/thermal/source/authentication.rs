//! Compare the original independently replayed completion without projecting a second DTO.
//! Metadata is borrowed; one original field partition is decoded at a time.
use super::super::super::source_failure::{artifact, bridge};
use super::*;
use runmat_execution_artifact::ArtifactError;
use runmat_runtime::analysis::storage::AnalysisPublicationError;

impl<S: CacheImport> RetainedThermalSource<S> {
    /// Preserve typed provider failures until the public string-only object boundary.
    pub(super) fn read_catalog_object(&self, reference: &ValueRef) -> ArtifactResult<Vec<u8>> {
        self.completed
            .result
            .host
            .artifact_access
            .validate_reference(
                reference,
                runmat_execution::value::ValueRefKind::ResultObject,
                &reference.media_type,
                &reference.value_schema,
            )
            .map_err(|error| ArtifactError::Identity(error.to_string()))?;
        if !self.catalog.objects.contains(reference) {
            return Err(ArtifactError::Identity(
                "thermal source reference differs from original catalog".into(),
            ));
        }
        let bytes = CacheImport::read_verified_bounded(
            self,
            reference.logical_digest,
            reference.encoded_length,
        )?
        .ok_or_else(|| ArtifactError::Identity("thermal source omitted a catalog object".into()))?;
        if Digest::sha256(&bytes) != reference.logical_digest
            || bytes.len() as u64 != reference.encoded_length
        {
            return Err(ArtifactError::Identity(
                "thermal source object digest or length differs".into(),
            ));
        }
        Ok(bytes)
    }

    /// Independently replay original phases and retain typed admission failures.
    pub(super) fn authenticate_original(
        &self,
        candidate: &ThermalStudyCompletion,
    ) -> Result<(), AnalysisPublicationError> {
        let replay = import_thermal_study_completion(
            self,
            self.completed.preparation(),
            self.completed.result_closures().to_vec(),
        )
        .map_err(bridge)?;
        if replay != self.completed {
            return Err("thermal completion differs from authenticated replay".into());
        }
        validate_completion(&replay, candidate)?;
        for reference in &candidate.catalog.objects {
            self.read_catalog_object(reference).map_err(artifact)?;
        }
        Ok(())
    }
}

pub(super) fn validate_completion(
    replay: &CompletedThermalStudy,
    candidate: &ThermalStudyCompletion,
) -> Result<(), AnalysisPublicationError> {
    if candidate.host_workload_digest != replay.result.host.workload.identity.digest()
        || candidate.compiled != replay.compiled
        || candidate.workload != replay.workload
        || candidate.manifest != replay.result.manifest
        || candidate.numerical_receipt != replay.result.numerical_receipt
        || candidate.accuracy != replay.accuracy.artifact
        || candidate.baseline_usage != replay.baseline_usage.evidence
        || candidate.terminal_usage != replay.terminal_usage.evidence
        || candidate.publication != replay.publication.result
        || candidate.frozen_project_digest != replay.frozen_project_digest()
        || candidate.states.len() != replay.states.len()
        || candidate
            .states
            .iter()
            .zip(&replay.states)
            .any(|(actual, original)| actual != &original.state)
    {
        return Err("runtime thermal completion differs from authenticated replay".into());
    }
    let root = prepare_thermal_study_root(replay).map_err(bridge)?;
    if candidate.root != root.root
        || candidate.catalog != root.catalog
        || candidate.observations != observations::publication(replay, &root.catalog)?
    {
        return Err("runtime thermal catalog differs from authenticated replay".into());
    }
    // Root bytes are not needed during field comparison. Preserve only its authenticated
    // metadata; there is no complete copied field inventory alongside the original DTO.
    drop(root);
    let identities = replay
        .result
        .manifest
        .fields
        .iter()
        .flatten()
        .flat_map(|field| field.partitions.iter().map(|p| p.partition.digest()))
        .collect::<std::collections::BTreeSet<_>>();
    if candidate.field_partitions.len() != identities.len() {
        return Err("runtime thermal partition inventory differs from authenticated replay".into());
    }
    // The public projection has a canonical identity order. Matching that exact order
    // also rejects duplicates, omissions, foreign entries and reordering before decoding.
    for (actual, identity) in candidate.field_partitions.iter().zip(identities) {
        if actual.identity.digest() != identity {
            return Err("runtime thermal partition order differs from authenticated replay".into());
        }
        let bytes = replay
            .result
            .objects
            .partition_payload(identity)
            .map_err(bridge)?;
        let original = RecoveredFieldPartition::from_canonical_bytes(&bytes)
            .map_err(|error| error.to_string())?;
        if actual != &original {
            return Err("runtime thermal field values differ from authenticated replay".into());
        }
    }
    Ok(())
}
