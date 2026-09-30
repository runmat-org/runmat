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
) -> Result<Vec<String>, String> {
    catalog.validate().map_err(|e| e.to_string())?;
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
            super::structural_study::persist_content_addressed_study_evidence(
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

impl<Workloads, NumericalEvidence> StudyResultManifest<Workloads, NumericalEvidence> {
    pub(super) fn new(
        compiled: &CompiledAnalysisStudy,
        workload_digest: Digest,
        workloads: Workloads,
        mesh: &SolverMeshArtifact,
        meshing_evidence: &MeshingEvidence,
        numerical_evidence: NumericalEvidence,
        public_run: AnalysisStudyRunData,
    ) -> Result<Self, String> {
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
            workloads,
            solver_mesh_digest: mesh.canonical_digest,
            meshing_evidence: meshing_evidence.clone(),
            numerical_evidence,
            public_run,
        })
    }
}

impl<Workloads: serde::Serialize, NumericalEvidence: serde::Serialize>
    StudyResultManifest<Workloads, NumericalEvidence>
{
    pub(super) fn persist(&self) -> Result<String, String> {
        let value = serde_json::to_value(self)
            .map_err(|error| format!("failed to encode typed study result manifest: {error}"))?;
        super::structural_study::persist_content_addressed_study_evidence(
            &self.study_definition_digest.to_string(),
            "study-result-manifest",
            value,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::value::{
        artifact_object_value_id, PublishedObjectCatalog, PublishedObjectSource, ValueRef,
        ValueRefKind,
    };
    struct SubstitutedSource;
    impl PublishedObjectSource for SubstitutedSource {
        fn read_verified(&self, _: &ValueRef) -> Result<Vec<u8>, String> {
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
        assert!(persist_catalog_objects(
            &SubstitutedSource,
            &catalog,
            Digest::sha256(b"study"),
            "test-object"
        )
        .unwrap_err()
        .contains("digest or length mismatch"));
        let mut absent = catalog;
        absent.objects.clear();
        assert!(persist_catalog_objects(
            &SubstitutedSource,
            &absent,
            Digest::sha256(b"study"),
            "test-object"
        )
        .is_err());
    }
}
