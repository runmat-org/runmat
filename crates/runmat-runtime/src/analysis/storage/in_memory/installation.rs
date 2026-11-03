//! Standalone installation shares complete inventory retention and reader lock order.
use super::*;

fn created_at(record: &PersistedAnalysisArtifact) -> &str {
    match record {
        PersistedAnalysisArtifact::Standard(r) => &r.created_at,
        PersistedAnalysisArtifact::StructuralFrequency(r) => &r.created_at,
        PersistedAnalysisArtifact::AcousticFrequency(r) => &r.created_at,
    }
}
fn same_content(left: &PersistedAnalysisArtifact, right: &PersistedAnalysisArtifact) -> bool {
    match (left, right) {
        (PersistedAnalysisArtifact::Standard(a), PersistedAnalysisArtifact::Standard(b)) => {
            same_persisted_content(a, b)
        }
        (
            PersistedAnalysisArtifact::StructuralFrequency(a),
            PersistedAnalysisArtifact::StructuralFrequency(b),
        ) => frequency::same_content(a, b),
        (
            PersistedAnalysisArtifact::AcousticFrequency(a),
            PersistedAnalysisArtifact::AcousticFrequency(b),
        ) => frequency_sample::same_content(a, b),
        _ => false,
    }
}

impl InMemoryAnalysisArtifactStore {
    pub(super) fn install_standalone(
        &self,
        id: &str,
        record: PersistedAnalysisArtifact,
        objects: Vec<(Digest, Vec<u8>)>,
    ) -> Result<String, String> {
        validate_run_id_token(id)?;
        let mut observations = self
            .observations
            .write()
            .map_err(|_| "observation lock poisoned")?;
        let mut runs = self.runs.write().map_err(|_| "run lock poisoned")?;
        let mut chunks = self.chunks.write().map_err(|_| "chunk lock poisoned")?;
        let mut publications = self
            .publications
            .write()
            .map_err(|_| "publication lock poisoned")?;
        let mut owners = self
            .publication_runs
            .write()
            .map_err(|_| "publication ownership lock poisoned")?;
        if owners.contains_key(id) {
            return Err("run identity is reserved by another publication".into());
        }
        if let Some(existing) = runs.get(id) {
            match existing {
                PersistedAnalysisArtifact::Standard(r) => validate_persisted_run(r, id)?,
                PersistedAnalysisArtifact::StructuralFrequency(r) => {
                    frequency::validate_artifact(r, id)?
                }
                PersistedAnalysisArtifact::AcousticFrequency(r) => {
                    frequency_sample::validate_artifact(r, id)?
                }
            }
            if !same_content(existing, &record) {
                return Err("analysis run identity already contains different content".into());
            }
            return Ok(created_at(existing).to_owned());
        }
        for (digest, bytes) in &objects {
            if Digest::sha256(bytes) != *digest
                || chunks
                    .get(digest)
                    .is_some_and(|old| old.as_ref() != bytes.as_slice())
            {
                return Err("analysis object digest collision".into());
            }
        }
        let mut incoming = HashMap::from([(id.to_owned(), record)]);
        let evictions = if self.retain_standalone {
            let descriptors = publications
                .iter()
                .map(|(root, bytes)| {
                    let (descriptor, _): (
                        publication::PublicationDescriptor,
                        serde::de::IgnoredAny,
                    ) = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
                    Ok((*root, descriptor))
                })
                .collect::<Result<HashMap<_, _>, String>>()?;
            publication::retention::plan_inventory(
                &publication::retention::inventory_facts(&incoming),
                None,
                &publication::retention::inventory_facts(&runs),
                &owners,
                &descriptors,
            )?
        } else {
            Vec::new()
        };
        let mut reachable =
            publication::reclamation::memory_reachable(&runs, &observations, &evictions)?;
        let streamed = self
            .streamed
            .read()
            .map_err(|_| "streamed chunk lock poisoned")?;
        if reachable
            .iter()
            .any(|digest| !chunks.contains_key(digest) && !streamed.contains(digest))
        {
            return Err("retained publication references absent objects".into());
        }
        reachable.extend(objects.iter().map(|(digest, _)| *digest));
        let record = incoming
            .remove(id)
            .ok_or("original incoming record absent")?;
        let timestamp = created_at(&record).to_owned();
        crate::analysis::fea_run_admission::FeaRunScope::check_publication_checked()
            .map_err(|denied| denied.to_string())?;
        if crate::interrupt::is_cancelled() {
            return Err("analysis publication was cancelled".into());
        }
        // All validation precedes the first reader-visible mutation.
        for eviction in evictions {
            for id in eviction.runs {
                runs.remove(&id);
                owners.remove(&id);
                observations.remove(&id);
            }
            if let Some(root) = eviction.publication {
                publications.remove(&root);
            }
        }
        chunks.retain(|digest, _| reachable.contains(digest));
        runs.insert(id.to_owned(), record);
        chunks.extend(
            objects
                .into_iter()
                .map(|(digest, bytes)| (digest, Arc::from(bytes))),
        );
        Ok(timestamp)
    }
}
