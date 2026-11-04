use super::*;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};

struct Values(usize);
impl Serialize for Values {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(std::iter::repeat_n(f64::MAX, self.0))
    }
}
pub(in crate::analysis::storage) fn workspace(
    field: &PersistedFieldArtifact,
    maximum: u64,
) -> Result<u64, AnalysisPublicationError> {
    let values = checked_element_count(&field.shape)?;
    let profile = profile_serialized(
        &Values(values),
        CanonicalLimits::new(
            usize::try_from(maximum).map_err(|_| "field read ceiling exceeds address space")?,
            100_000_000,
            65_536,
            64,
        ),
    )?;
    // Original byte payload, decoded values and exact content/manifest construction coexist.
    profile
        .materialization_resident_bytes_estimate
        .checked_mul(4)
        .filter(|n| *n <= maximum)
        .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
            message: "original field construction exceeds its remaining read ceiling".into(),
        })
}
fn reconstruct(
    run_id: &str,
    catalog: &RecoveredFieldArtifactManifest,
    entry: &RecoveredFieldArtifactEntry,
    field: &PersistedFieldArtifact,
    mut read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<InitialFieldBinding, AnalysisPublicationError> {
    let payload = super::super::field_reader::read_field_payload(field, &mut read)?;
    let values = decode_field_values(&payload)?;
    let recovered = RecoveredField::new(
        entry.definition.clone(),
        entry.sample,
        AnalysisField::host_f64(
            &field.field_id,
            // The staged field keeps the published state (ordinal) its producer stored.
            entry
                .definition
                .layout()
                .with_ordinal(field.template.layout.ordinal),
            field.shape.clone(),
            values,
        ),
    )
    .map_err(|e| e.to_string())?;
    if recovered.identity() != entry.field {
        return Err("original field differs from authenticated source catalog".into());
    }
    let artifact = RecoveredFieldArtifact::new(
        catalog.recovery_request,
        catalog.source,
        catalog.topology_digest,
        catalog.entity_counts,
        vec![recovered],
    )
    .map_err(|e| e.to_string())?;
    let reference = runmat_analysis_core::InitialFieldArtifactRef {
        artifact_id: run_id.into(),
        content_digest: artifact.identity().digest().to_string(),
        topology_digest: artifact.topology_digest.to_string(),
        field_id: field.field_id.clone(),
    };
    InitialFieldBinding::new(reference, artifact).map_err(|e| e.to_string().into())
}

impl InMemoryAnalysisArtifactStore {
    pub(in crate::analysis::storage) fn export_original_field(
        &self,
        run_id: &str,
        field_id: &str,
        maximum: u64,
    ) -> Result<Option<InitialFieldBinding>, AnalysisPublicationError> {
        validate_run_id_token(run_id)?;
        validate_field_id(field_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "field source store poisoned")?;
        let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get(run_id) else {
            return Ok(None);
        };
        let metadata = publication::resident(run.as_ref(), maximum)?;
        let remaining = maximum.checked_sub(metadata).ok_or_else(|| {
            AnalysisPublicationError::ResourceLimit {
                message: "field source metadata exceeds read ceiling".into(),
            }
        })?;
        validate_persisted_run(run, run_id)?;
        let Some(catalog::OriginalFieldSource {
            catalog,
            entry,
            field,
        }) = catalog::select(run, field_id)?
        else {
            return Ok(None);
        };
        workspace(field, remaining)?;
        let chunks = self
            .chunks
            .read()
            .map_err(|_| "field source chunks poisoned")?;
        reconstruct(run_id, catalog, entry, field, |digest| {
            let Some(bytes) = chunks.get(&digest) else {
                return Ok(None);
            };
            if bytes.len() as u64 > remaining {
                return Err("original field chunk exceeds read ceiling".into());
            }
            Ok(Some(bytes.to_vec()))
        })
        .map(Some)
    }
}
impl access::FilesystemAccess<'_> {
    pub(in crate::analysis::storage) fn export_original_field(
        &self,
        run_id: &str,
        field_id: &str,
        maximum: u64,
    ) -> Result<Option<InitialFieldBinding>, AnalysisPublicationError> {
        validate_field_id(field_id)?;
        let Some((path, expected, remaining)) = self.assessment_record_path(run_id, maximum)?
        else {
            return Ok(None);
        };
        let Some(bytes) = self.read_bounded_artifact(&path, remaining)? else {
            return Ok(None);
        };
        let metadata = self.admit_json_metadata(&bytes)?;
        let _retained = self.reserve_resident(metadata)?;
        let record = decode_persisted_artifact(&bytes, run_id)?;
        if let Some(expected) = expected {
            if publication::record_digest(&record)? != expected {
                return Err("field source differs from committed publication".into());
            }
        }
        let PersistedAnalysisArtifact::Standard(run) = record else {
            return Ok(None);
        };
        let Some(catalog::OriginalFieldSource {
            catalog,
            entry,
            field,
        }) = catalog::select(&run, field_id)?
        else {
            return Ok(None);
        };
        let _construction = self.reserve_resident(workspace(field, self.remaining_bytes())?)?;
        reconstruct(run_id, catalog, entry, field, |digest| {
            self.read_chunk(digest).map_err(|e| e.to_string())
        })
        .map(Some)
    }
}
