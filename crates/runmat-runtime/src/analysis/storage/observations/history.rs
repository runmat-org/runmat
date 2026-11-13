use super::*;
use runmat_analysis_core::PartialAccessPolicy;
use runmat_analysis_fea::results::observation::{
    ObservationHistory, ObservationHistoryManifest, ObservationPublicationEntry,
};

pub(super) fn load(
    attachment: &ObservationAttachment,
    observation_id: &ObservationId,
    coordinate: RecoveredFieldSample,
    mut read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<ObservationHistoryLookup, String> {
    let Some(history) =
        attachment
            .publication
            .catalog
            .entries()
            .iter()
            .find_map(|entry| match entry {
                ObservationPublicationEntry::History(history)
                    if &history.history.observation_id == observation_id =>
                {
                    Some(&history.history)
                }
                _ => None,
            })
    else {
        return Ok(ObservationHistoryLookup::ObservationNotFound);
    };
    let Some(entry) = history
        .samples
        .iter()
        .find(|entry| entry.coordinate == coordinate)
    else {
        return Ok(ObservationHistoryLookup::SampleNotFound);
    };
    if history
        .samples
        .iter()
        .filter(|entry| entry.coordinate == coordinate)
        .count()
        != 1
    {
        return Err("history coordinate identifies multiple event occurrences".into());
    }
    let root = &attachment
        .publication
        .histories
        .iter()
        .find(|root| root.history == history.identity)
        .ok_or("history has no retained semantic root")?
        .object;
    let mut verified = |digest| {
        let reference = attachment
            .objects
            .objects
            .iter()
            .find(|reference| reference.logical_digest == digest)
            .ok_or("history read names an undeclared object")?;
        let bytes = read(digest)?.ok_or("history object is missing")?;
        if Digest::sha256(&bytes) != digest || bytes.len() as u64 != reference.encoded_length {
            return Err("history object content differs from catalog".into());
        }
        Ok(Some(bytes))
    };
    let bytes = verified(root.reference.logical_digest)?.ok_or("history root missing")?;
    let manifest =
        AnalysisObjectManifest::from_canonical_bytes(&bytes).map_err(|error| error.to_string())?;
    if manifest.kind != AnalysisNumericalObjectKind::History
        || manifest.logical_identity != history.identity.digest()
        || manifest.payload_digest != root.payload_digest
    {
        return Err("history object root differs from semantic role".into());
    }
    let sample = match history.partial_access {
        PartialAccessPolicy::ManifestAndChunks => {
            let semantic = super::super::object_payload::read_payload_partition(
                &manifest,
                history.identity.digest(),
                &mut verified,
            )?;
            let semantic = ObservationHistoryManifest::from_canonical_bytes(&semantic)
                .map_err(|error| error.to_string())?;
            if &semantic != history {
                return Err("history manifest differs from published catalog".into());
            }
            let expected: Vec<_> = std::iter::once(history.identity.digest())
                .chain(history.samples.iter().map(|entry| entry.sample.digest()))
                .collect();
            if !manifest
                .partitions
                .iter()
                .map(|partition| partition.logical_identity)
                .eq(expected)
            {
                return Err("history partition inventory differs from semantic manifest".into());
            }
            let payload = super::super::object_payload::read_payload_partition(
                &manifest,
                entry.sample.digest(),
                &mut verified,
            )?;
            ObservationHistorySample::from_canonical_bytes(&payload)
                .map_err(|error| error.to_string())?
        }
        PartialAccessPolicy::WholeArtifact => {
            let mut payload = Vec::new();
            for chunk in &manifest.chunks {
                let bytes = verified(chunk.digest)?.ok_or("whole history chunk missing")?;
                super::super::validate_chunk(chunk, &bytes)?;
                payload.extend_from_slice(&bytes);
            }
            if payload.len() as u64 != manifest.encoded_length
                || Digest::sha256(&payload) != manifest.payload_digest
            {
                return Err("whole history payload differs from manifest".into());
            }
            let whole = ObservationHistory::from_canonical_bytes(&payload)
                .map_err(|error| error.to_string())?;
            if whole.manifest() != history {
                return Err("whole history differs from published catalog".into());
            }
            whole
                .samples()
                .iter()
                .find(|sample| sample.identity == entry.sample)
                .ok_or("whole history sample missing")?
                .clone()
        }
    };
    if sample.identity != entry.sample
        || sample.coordinate != coordinate
        || sample.result.logical_identity != entry.result
        || sample.event_attribution != entry.event_attribution
    {
        return Err("history sample differs from retained semantic entry".into());
    }
    Ok(ObservationHistoryLookup::Sample(Box::new(sample)))
}
