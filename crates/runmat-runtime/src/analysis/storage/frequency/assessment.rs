//! Original scalar partition/history extraction with bounded shared materialization.
use super::*;
use crate::analysis::storage::{AssessmentRunEvidence, AssessmentStoreError};
mod resources;
#[cfg(test)]
mod tests;

/// Read only scalar observation partitions. Histories remain with the observation owner;
/// a catalog sample is never reinterpreted as a temporal or frequency series reduction.
pub(in crate::analysis::storage) fn assessment_evidence(
    artifact: &PersistedStructuralFrequencyArtifact,
    remaining_bytes: u64,
    maximum_resident_bytes: u64,
    read: impl FnMut(Digest, u64) -> Result<Option<Vec<u8>>, AssessmentStoreError>,
) -> Result<AssessmentRunEvidence, AssessmentStoreError> {
    use runmat_analysis_core::ObservationValueShape;
    use runmat_analysis_fea::results::observation::ObservationResult;
    use AssessmentStoreError as Error;
    validate_artifact(artifact, &artifact.step.run_id).map_err(Error::Invalid)?;
    let AnalysisStudyStepNumericalResultData::StructuralFrequency(result) =
        &artifact.step.numerical_result
    else {
        return Err(Error::Invalid(
            "frequency evidence has another result kind".into(),
        ));
    };
    let mut reader = resources::Reader::new(
        &artifact.object_catalog,
        remaining_bytes,
        maximum_resident_bytes,
        read,
    );
    let mut scalars = Vec::new();
    for (sample, roots) in result
        .publication
        .samples()
        .iter()
        .zip(artifact.artifact_roots.samples())
    {
        let (Some(observations), Some(root)) = (&sample.observations, roots.observations()) else {
            continue;
        };
        let mut entries = observations
            .results
            .iter()
            .filter(|entry| {
                entry.value_shape == ObservationValueShape::Scalar
                    && !result
                        .publication
                        .histories()
                        .iter()
                        .any(|history| history.observation_id == entry.observation_id)
            })
            .peekable();
        if entries.peek().is_none() {
            continue;
        }
        let retained = reader.resident(&(&artifact.step.identity, &scalars))?;
        let manifest = reader.manifest(root.reference().logical_digest, retained)?;
        let manifest_resident = reader.resident(&manifest)?;
        if manifest.kind != root.expected_kind()
            || manifest.logical_identity != root.expected_logical_identity()
            || manifest.payload_digest != root.expected_payload_digest()
        {
            return Err(Error::Invalid(
                "observation root differs from its semantic role".into(),
            ));
        }
        for entry in entries {
            let partition = manifest
                .partitions
                .iter()
                .find(|partition| partition.logical_identity == entry.result.digest())
                .ok_or_else(|| Error::Invalid("scalar observation partition is missing".into()))?;
            // Check the declared allocation before constructing a payload buffer.
            let first = partition.first_chunk_ordinal as usize;
            let end = first
                .checked_add(partition.chunk_count as usize)
                .ok_or_else(|| Error::Invalid("observation chunk interval overflows".into()))?;
            let chunks = manifest
                .chunks
                .get(first..end)
                .ok_or_else(|| Error::Invalid("observation chunk interval is invalid".into()))?;
            let retained = resources::add(
                reader.resident(&(&artifact.step.identity, &scalars))?,
                manifest_resident,
            )?;
            let payload = reader.payload(
                chunks,
                partition.encoded_length,
                partition.payload_digest,
                retained,
            )?;
            reader.admit_decode(&payload, retained)?;
            let observation = ObservationResult::from_canonical_bytes(&payload)
                .map_err(|error| Error::Invalid(error.to_string()))?;
            entry
                .validate_result(&observation)
                .map_err(|error| Error::Invalid(error.to_string()))?;
            // Forecast the attributed scalar copy beside the original decoded result/cache.
            reader.admit(resources::add(
                retained,
                reader.resident(&(&observation, &observation))?,
            )?)?;
            scalars.extend(
                runmat_analysis_fea::results::assessment::scalar_observation_evidence(&observation),
            );
        }
    }
    for (history, roots) in result
        .publication
        .histories()
        .iter()
        .zip(artifact.artifact_roots.histories())
    {
        let (Some(publication), Some(roots)) = (&history.finalization, roots.finalization()) else {
            continue;
        };
        let (Some(series), Some(root)) = (&publication.series, &roots.series) else {
            continue;
        };
        let retained = reader.resident(&(&artifact.step.identity, &scalars))?;
        let manifest = reader.manifest(root.reference().logical_digest, retained)?;
        let manifest_resident = reader.resident(&manifest)?;
        if manifest.kind != root.expected_kind()
            || manifest.logical_identity != root.expected_logical_identity()
            || manifest.payload_digest != root.expected_payload_digest()
        {
            return Err(Error::Invalid(
                "series root differs from its semantic role".into(),
            ));
        }
        manifest
            .whole_payload_partition()
            .map_err(|error| Error::Invalid(error.to_string()))?;
        let retained = resources::add(retained, manifest_resident)?;
        let payload = reader.payload(
            &manifest.chunks,
            manifest.encoded_length,
            manifest.payload_digest,
            retained,
        )?;
        reader.admit_decode(&payload, retained)?;
        let reduction =
            runmat_analysis_fea::results::observation::SeriesReductionResult::from_canonical_bytes(
                &payload,
            )
            .map_err(|error| Error::Invalid(error.to_string()))?;
        series
            .validate_published_result(&reduction, &publication.history, &publication.request)
            .map_err(|error| Error::Invalid(error.to_string()))?;
        reader.admit(resources::add(
            retained,
            reader.resident(&(&reduction, &reduction))?,
        )?)?;
        scalars
            .extend(runmat_analysis_fea::results::assessment::scalar_series_evidence(&reduction));
    }
    reader.admit(reader.resident(&(&artifact.step.identity, &scalars))?)?;
    Ok(crate::analysis::storage::AssessmentRunEvidence {
        identity: artifact.step.identity.clone(),
        numerically_acceptable: artifact.step.numerically_acceptable,
        observation_scalars: scalars,
    })
}
