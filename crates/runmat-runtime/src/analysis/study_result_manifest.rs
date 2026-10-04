//! Physics-neutral whole-study evidence envelope shared by typed publication adapters.
use super::AnalysisStudyRunData;
use runmat_analysis_fea::study::CompiledAnalysisStudy;
use runmat_execution::Digest;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact, StableDigest};

/// Persist a complete authenticated inventory through the shared immutable artifact owner.
/// Family adapters authenticate their semantic completion before calling this transport helper.
pub(super) fn persist_catalog_objects(
    source: &impl runmat_execution::value::PublishedObjectSource,
    catalog: &runmat_execution::value::PublishedObjectCatalog,
    definition: Digest,
    object_stage_prefix: &str,
    staging: &super::storage::AnalysisPublicationBuilder,
) -> Result<Vec<String>, super::storage::AnalysisPublicationError> {
    catalog.validate().map_err(|e| e.to_string())?;
    staging.require_source_objects(&catalog.objects)?;
    catalog
        .objects
        .iter()
        .map(|reference| {
            let bytes = source.read_verified(reference)?;
            if Digest::sha256(&bytes) != reference.logical_digest
                || bytes.len() as u64 != reference.encoded_length
            {
                return Err("study catalog object digest or length mismatch".into());
            }
            staging.persist_evidence(
                &definition.to_string(),
                &format!(
                    "{object_stage_prefix}-{}",
                    reference
                        .logical_digest
                        .to_string()
                        .trim_start_matches("sha256:")
                ),
                serde_json::json!({"reference": reference, "bytes": bytes}),
            )
        })
        .collect()
}

/// The generic axes retain each physics owner's actual workload and numerical evidence schema.
/// Field names and ordering preserve the existing structural typed-result JSON contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StudyResultManifest<Workloads, NumericalEvidence> {
    schema_version: u16,
    study_definition_digest: Digest,
    compiled: CompiledAnalysisStudy,
    workload_digest: Digest,
    workloads: Workloads,
    solver_mesh_digest: StableDigest,
    meshing_evidence: MeshingEvidence,
    numerical_evidence: NumericalEvidence,
    public_run: AnalysisStudyRunData,
}

pub(super) fn load_for_reassessment(
    definition: Digest,
    path: &std::path::Path,
    maximum_bytes: u64,
) -> Result<
    (CompiledAnalysisStudy, AnalysisStudyRunData),
    super::structural_study::StudyEvidenceReadError,
> {
    let store = super::storage::capture_artifact_store()?;
    let payload = store
        .load_publication_manifest_path(path, maximum_bytes)
        .map_err(|e| match e {
            super::storage::AssessmentStoreError::ResourceLimit => {
                super::structural_study::StudyEvidenceReadError::ResourceLimit
            }
            super::storage::AssessmentStoreError::Invalid(message) => {
                super::structural_study::StudyEvidenceReadError::Invalid(message)
            }
        })?;
    // Opaque family-owned numerical values are not duplicated into an assessment input. The
    // evidence envelope authenticates them; scalar lookup uses the individual run's store.
    let manifest: StudyResultManifest<serde::de::IgnoredAny, serde::de::IgnoredAny> =
        serde_json::from_value(payload).map_err(|error| error.to_string())?;
    if manifest.schema_version != 1
        || manifest.study_definition_digest != definition
        || manifest
            .compiled
            .definition_digest()
            .map_err(|error| error.to_string())?
            != definition
        || manifest.public_run.study_fingerprint != definition.to_string()
        || manifest.public_run.study_id != manifest.compiled.study_id
        || manifest.public_run.model_id != manifest.compiled.model.model_id.0
        || manifest.solver_mesh_digest != manifest.compiled.solver_mesh_digest
    {
        return Err("saved study publication differs from its compiled revision".into());
    }
    Ok((manifest.compiled, manifest.public_run))
}

impl<Workloads, NumericalEvidence> StudyResultManifest<Workloads, NumericalEvidence> {
    pub(super) fn compiled(&self) -> &CompiledAnalysisStudy {
        &self.compiled
    }
    pub(super) fn set_public_run(&mut self, public_run: AnalysisStudyRunData) {
        self.public_run = public_run;
    }
    pub(super) fn new(
        compiled: &CompiledAnalysisStudy,
        workload_digest: Digest,
        workloads: &Workloads,
        mesh: &SolverMeshArtifact,
        meshing_evidence: &MeshingEvidence,
        numerical_evidence: NumericalEvidence,
        publication: &super::structural_study::publication::DeferredStudyRun,
    ) -> Result<Self, super::storage::AnalysisPublicationError>
    where
        Workloads: serde::Serialize + Clone,
        NumericalEvidence: serde::Serialize,
    {
        let public_run = &publication.published.data;
        publication.staging.admit_manifest_copy(&(
            compiled,
            workloads,
            meshing_evidence,
            &numerical_evidence,
            public_run,
        ))?;
        let definition = compiled
            .definition_digest()
            .map_err(|error| error.to_string())?;
        if public_run.study_fingerprint != definition.to_string() {
            return Err("study result manifest differs from published definition identity".into());
        }
        Ok(Self {
            schema_version: 1,
            study_definition_digest: definition,
            compiled: compiled.clone(),
            workload_digest,
            workloads: workloads.clone(),
            solver_mesh_digest: mesh.canonical_digest,
            meshing_evidence: meshing_evidence.clone(),
            numerical_evidence,
            public_run: public_run.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::value::{
        artifact_object_value_id, PublishedObjectCatalog, PublishedObjectSource, ValueRef,
        ValueRefKind,
    };
    struct SubstitutedSource(std::sync::atomic::AtomicUsize);
    impl PublishedObjectSource for SubstitutedSource {
        fn read_verified(&self, _: &ValueRef) -> Result<Vec<u8>, String> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(b"other".to_vec())
        }
    }
    #[test]
    fn shared_catalog_persistence_rejects_substituted_payload_before_writing() {
        let logical_digest = Digest::sha256(b"original");
        let encryption_context = Digest::sha256(b"authority");
        let reference = ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1,
            id: artifact_object_value_id(logical_digest, encryption_context, "authority"),
            logical_digest,
            encoded_length: 8,
            media_type: "application/vnd.runmat.test".into(),
            value_schema: "runmat.test".into(),
            encryption_context,
            kind: ValueRefKind::ResultObject,
            authorization_scope: "authority".into(),
            resident_fence: None,
        };
        let catalog =
            PublishedObjectCatalog::new(vec![reference.clone()], vec![reference]).unwrap();
        let mut staging = crate::analysis::storage::AnalysisPublicationBuilder::new(
            Digest::sha256(b"test"),
            128 << 20,
        )
        .unwrap();
        let source = SubstitutedSource(std::sync::atomic::AtomicUsize::new(0));
        assert!(matches!(
            persist_catalog_objects(
                &source,
                &catalog,
                Digest::sha256(b"study"),
                "test-object",
                &staging
            ),
            Err(crate::analysis::storage::AnalysisPublicationError::ResourceLimit { .. })
        ));
        assert_eq!(source.0.load(std::sync::atomic::Ordering::SeqCst), 0);
        staging.admit_source_objects(&catalog.objects).unwrap();
        assert!(persist_catalog_objects(
            &source,
            &catalog,
            Digest::sha256(b"study"),
            "test-object",
            &staging,
        )
        .unwrap_err()
        .to_string()
        .contains("digest or length mismatch"));
        assert_eq!(source.0.load(std::sync::atomic::Ordering::SeqCst), 1);
        let mut absent = catalog;
        absent.objects.clear();
        assert!(persist_catalog_objects(
            &source,
            &absent,
            Digest::sha256(b"study"),
            "test-object",
            &staging,
        )
        .is_err());
    }
}
