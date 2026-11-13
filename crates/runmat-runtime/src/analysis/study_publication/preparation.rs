//! Capture host publication authority before original-input authentication/copying.
use crate::analysis::{configuration, storage};
use runmat_execution::Digest;
use std::sync::Arc;

pub(in crate::analysis) struct PublicationPreparation {
    pub(in crate::analysis) staging: storage::AnalysisPublicationBuilder,
    pub(in crate::analysis) store: Arc<dyn storage::AnalysisArtifactStore>,
    pub(in crate::analysis) retained_input: storage::publication::PreparationReservation,
}
impl PublicationPreparation {
    /// Persist mesh objects only after the captured publication ceiling admits
    /// their codec and immutable-collision workspaces.
    pub(in crate::analysis) fn persist_mesh(
        &self,
        mesh: &runmat_meshing_core::SolverMeshArtifact,
        evidence: &runmat_meshing_core::MeshingEvidence,
    ) -> Result<crate::analysis::meshing::ResolvedStudyMesh, storage::AnalysisPublicationError>
    {
        self.staging.admit_mesh_persistence(mesh, evidence)?;
        crate::analysis::meshing::persist_mesh_borrowed(mesh, evidence)
            .map_err(storage::AnalysisPublicationError::from)
    }

    pub(in crate::analysis) fn capture(
        definition: Digest,
        input: &impl serde::Serialize,
    ) -> Result<Self, storage::AnalysisPublicationError> {
        let maximum = configuration::publication_staging_budget()?;
        Self::capture_with_budget(definition, input, maximum)
    }

    /// Keep admission and failure diagnostics under the same captured host ceiling.
    pub(in crate::analysis) fn capture_with_budget(
        definition: Digest,
        input: &impl serde::Serialize,
        maximum: u64,
    ) -> Result<Self, storage::AnalysisPublicationError> {
        let mut staging = storage::AnalysisPublicationBuilder::new(definition, maximum)?;
        let retained_input = staging.admit_preparation(input)?;
        Ok(Self {
            staging,
            store: storage::capture_artifact_store()?,
            retained_input,
        })
    }
    /// Reserve typed completion owners separately from the retained original.
    pub(in crate::analysis) fn capture_with_completion_storage(
        definition: Digest,
        original_bytes: u64,
        copy_bytes: u64,
        maximum: u64,
    ) -> Result<(Self, storage::publication::OwnedCopyReservation), storage::AnalysisPublicationError>
    {
        let mut staging = storage::AnalysisPublicationBuilder::new(definition, maximum)?;
        let (retained_input, copies) =
            staging.admit_completion_storage(original_bytes, copy_bytes)?;
        Ok((
            Self {
                staging,
                store: storage::capture_artifact_store()?,
                retained_input,
            },
            copies,
        ))
    }

    pub(in crate::analysis) fn capture_catalog(
        definition: Digest,
        input: &impl serde::Serialize,
        catalog: &runmat_execution::value::PublishedObjectCatalog,
    ) -> Result<Self, storage::AnalysisPublicationError> {
        let mut prepared = Self::capture(definition, input)?;
        prepared.staging.admit_source_objects(&catalog.objects)?;
        Ok(prepared)
    }

    pub(in crate::analysis) fn admit_structural_sources(
        &mut self,
        result: &crate::analysis::StructuralStudyNumericalResult,
    ) -> Result<(), storage::AnalysisPublicationError> {
        use crate::analysis::StructuralStudyNumericalResult as Result;
        match result {
            Result::ThermalPrestress {
                thermal,
                thermal_source,
                structural,
                ..
            } => {
                thermal_source.admit_publication(thermal, &mut self.staging)?;
                self.admit_structural_sources(structural)
            }
            Result::StructuralFrequency(frequency) => self
                .staging
                .admit_source_objects(&frequency.object_catalog.objects),
            Result::StructuralPrestressedFrequency(result) => self
                .staging
                .admit_source_objects(&result.frequency.object_catalog.objects),
            Result::StructuralModalFrequency(result) => self
                .staging
                .admit_source_objects(&result.frequency.object_catalog.objects),
            Result::LinearSequence(_)
            | Result::StructuralModal(_)
            | Result::StructuralModalTransient(_)
            | Result::StructuralModalResponseSpectrum(_)
            | Result::StructuralRandomResponse(_)
            | Result::StructuralBuckling(_)
            | Result::StructuralPrestressedModal(_)
            | Result::StructuralTransient(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{self, storage::AnalysisArtifactStore};

    #[test]
    fn captured_backend_and_ceiling_survive_ambient_configuration_changes() {
        let _guard = analysis::tests::analysis_test_guard();
        storage::reset_artifact_store_for_tests();
        let original = Arc::new(storage::InMemoryAnalysisArtifactStore::new());
        let replacement = Arc::new(storage::InMemoryAnalysisArtifactStore::new());
        storage::set_artifact_store_for_tests(original.clone());
        let config = configuration::current_fea_runtime_config();
        let run = analysis::analysis_run_linear_static_with_options(
            &analysis::tests::sample_model(),
            runmat_analysis_fea::ComputeBackend::Cpu,
            analysis::AnalysisRunOptions::default(),
            crate::operations::OperationContext::new(None, None),
        )
        .unwrap()
        .data;
        let definition = Digest::sha256(b"captured-preparation");
        let preparation = PublicationPreparation::capture(definition, &run).unwrap();
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
            runmat_meshing_core::ElementOrder::Tet4,
        );
        let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
        let constrained = PublicationPreparation::capture_with_budget(
            definition,
            &(),
            storage::publication::typed_resident(&(), config.publication_staging_bytes).unwrap()
                * 2
                + mesh.canonical_encoding_resident_bytes_estimate().unwrap() / 2,
        )
        .unwrap();
        let isolated = tempfile::tempdir().unwrap();
        let mesh_root = isolated.path().join("artifacts");
        configuration::configure_fea_runtime(configuration::FeaRuntimeConfig {
            artifact_root: Some(mesh_root.clone()),
            ..config.clone()
        })
        .unwrap();
        assert!(matches!(
            constrained.persist_mesh(&mesh, &evidence),
            Err(storage::AnalysisPublicationError::ResourceLimit { .. })
        ));
        assert!(
            !mesh_root.exists(),
            "mesh admission must precede filesystem writes"
        );
        configuration::configure_fea_runtime(config.clone()).unwrap();
        storage::set_artifact_store_for_tests(replacement.clone());
        configuration::configure_fea_runtime(configuration::FeaRuntimeConfig {
            publication_staging_bytes: 1,
            ..config.clone()
        })
        .unwrap();
        let mut child = run;
        child.run_id = "captured_preparation_original".into();
        preparation.staging.stage_run(&child).unwrap();
        let prepared = preparation
            .staging
            .finish(serde_json::json!({
                "study_definition_digest": definition,
                "public_run": {"steps": [{"run_id": child.run_id}]}
            }))
            .unwrap();
        preparation.store.commit_publication(prepared).unwrap();
        assert!(original.load_run(&child.run_id).unwrap().is_some());
        assert!(replacement.load_run(&child.run_id).unwrap().is_none());
        assert!(matches!(
            PublicationPreparation::capture(definition, &child),
            Err(storage::AnalysisPublicationError::ResourceLimit { .. })
        ));
        configuration::configure_fea_runtime(config).unwrap();
        storage::reset_artifact_store_for_tests();
    }
}
