//! Owned imported-source meshing uses the same exact importer and meshing DAG.
use super::{OcctMeshingEvaluatorProvider, SerialGeometryMeshingError};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport},
    object::ObjectInventoryLimits,
    ArtifactAccess, LogicalObject,
};
use runmat_geometry_core::{ExactContactDefinition, PersistentEntityId};
use runmat_geometry_io::{ExactCadImportOptions, GeometryImportContext};
use runmat_meshing_core::{
    CacheAdmissionDecision, MeshingCancellationSignal, MeshingChunkPolicy,
    MeshingExecutionCeilings, MeshingRequestSettings, PlatformBuildIdentity,
};
use runmat_meshing_execution::{
    MeshingKernelDispatcher, MeshingProgressSink, MeshingRunEvidenceContext,
    SerialExactMeshingExecutor,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedMeshContactSelectors {
    pub side_a_face_selectors: Vec<String>,
    pub side_b_face_selectors: Vec<String>,
}

/// Input materialization only. A trusted host separately admits and supervises
/// this exact payload; these fields never authorize execution themselves.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedSourceMeshingRequest {
    pub source_name: String,
    pub source: LogicalObject,
    pub expected_source_digest: Digest,
    pub import_options: ExactCadImportOptions,
    pub settings: MeshingRequestSettings,
    pub contacts: Vec<ImportedMeshContactSelectors>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedSourceMeshingResult {
    pub artifact: runmat_meshing_core::SolverMeshArtifact,
    pub evidence: runmat_meshing_core::MeshingEvidence,
}

#[expect(
    clippy::too_many_arguments,
    reason = "exact meshing receives input, storage and admitted execution controls independently"
)]
pub fn mesh_imported_source_serial<S: CacheImport + CacheExport>(
    request: ImportedSourceMeshingRequest,
    store: &mut S,
    import_context: &GeometryImportContext,
    ceilings: MeshingExecutionCeilings,
    platform: PlatformBuildIdentity,
    inventory: ObjectInventoryLimits,
    cancellation: &dyn MeshingCancellationSignal,
    progress: &mut dyn MeshingProgressSink,
) -> Result<ImportedSourceMeshingResult, SerialGeometryMeshingError> {
    let ImportedSourceMeshingRequest {
        source_name,
        source,
        expected_source_digest,
        import_options,
        settings,
        contacts,
        program_revision,
        artifact_access,
    } = request;
    if source_name.is_empty() || source_name.len() > 4096 || contacts.len() > 1_000_000 {
        return Err("Imported meshing source metadata exceeds its supported bounds.".into());
    }
    source.validate().map_err(|error| error.to_string())?;
    runmat_execution_artifact::object::validate_inventory(std::slice::from_ref(&source), inventory)
        .map_err(|error| error.to_string())?;
    if source.descriptor.digest != expected_source_digest {
        return Err("Geometry source changed after the original asset was loaded.".into());
    }
    settings.validate().map_err(|error| error.to_string())?;
    import_options
        .validate()
        .map_err(|error| error.to_string())?;
    artifact_access
        .validate()
        .map_err(|error| error.to_string())?;
    let format = runmat_geometry_io::detect_geometry_format(&source_name, &source.bytes);
    let imported = runmat_geometry_io::import_exact_cad(
        &source_name,
        &source.bytes,
        format,
        &import_options,
        import_context,
    )
    .map_err(|error| error.to_string())?;
    let mut resolved = Vec::with_capacity(contacts.len());
    for contact in contacts {
        resolved.push(ExactContactDefinition {
            side_a_face_ids: resolve_faces(&imported, &contact.side_a_face_selectors)?,
            side_b_face_ids: resolve_faces(&imported, &contact.side_b_face_selectors)?,
        });
    }
    let imported = imported
        .with_contacts(&resolved)
        .map_err(|error| error.to_string())?;
    let document = imported
        .geometry_document()
        .map_err(|error| error.to_string())?;
    let request = runmat_meshing_core::resolve_meshing_request(document.tolerance, settings)
        .map_err(|error| error.to_string())?;
    let objects = runmat_meshing_execution::prepare_exact_geometry_objects(
        document,
        imported.topology,
        imported.evaluators,
        Some(imported.representation),
        imported.healing_report,
        inventory,
    )
    .map_err(|error| error.to_string())?;
    drop(source);
    for object in &objects.objects {
        store
            .write_verified(object)
            .map_err(|error| error.to_string())?;
    }
    let geometry = runmat_meshing_execution::prepare_exact_geometry_input(
        objects,
        artifact_access.clone(),
        inventory,
    )
    .map_err(|error| error.to_string())?;
    let kernel = MeshingKernelDispatcher::new(OcctMeshingEvaluatorProvider);
    let mut executor = SerialExactMeshingExecutor {
        store,
        kernel: &kernel,
        cancellation,
        progress,
        chunk_policy: MeshingChunkPolicy {
            maximum_chunk_bytes: 8 << 20,
            maximum_records_per_chunk: 65_536,
            maximum_total_encoded_bytes: inventory.max_total_bytes,
        },
        inventory_limits: inventory,
    };
    let completed = runmat_meshing_execution::execute_exact_meshing_dag(
        runmat_meshing_execution::ExactMeshingDagRun {
            geometry: &geometry,
            request,
            ceilings,
            artifact_access,
            capability_cohort: Some(platform.capability_cohort.clone()),
            program_revision,
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
            inventory_limits: inventory,
            evidence: MeshingRunEvidenceContext {
                platform,
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
            named_regions: Default::default(),
        },
        &mut executor,
    )?;
    completed
        .evidence
        .validate(&completed.artifact)
        .map_err(|error| error.to_string())?;
    Ok(ImportedSourceMeshingResult {
        artifact: completed.artifact,
        evidence: completed.evidence,
    })
}

fn resolve_faces(
    imported: &runmat_geometry_io::ImportedExactCad,
    selectors: &[String],
) -> Result<Vec<PersistentEntityId>, SerialGeometryMeshingError> {
    selectors
        .iter()
        .map(|selector| {
            PersistentEntityId::validate_analysis_selection_id(selector)
                .map_err(|error| error.to_string())?;
            let mut matched = imported
                .topology
                .faces
                .iter()
                .filter(|face| face.id.matches_analysis_selection(selector));
            let first = matched
                .next()
                .ok_or_else(|| format!("Contact source face `{selector}` does not exist."))?;
            if matched.next().is_some() {
                return Err(format!(
                    "Contact source face `{selector}` is ambiguous across assembly occurrences."
                )
                .into());
            }
            Ok(first.id.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::ProgramEnvironment;
    use runmat_execution_artifact::{cache::MemoryObjectStore, ObjectNamespace};

    #[test]
    fn changed_source_is_refused_before_import_or_any_meshing_stage() {
        let inventory = ObjectInventoryLimits::default();
        let settings = MeshingRequestSettings::default();
        let ceilings = MeshingExecutionCeilings::for_host(512 << 20, &settings.limits);
        let request = ImportedSourceMeshingRequest {
            source_name: "original.brep".into(),
            source: LogicalObject::new(
                ObjectNamespace::InputValue,
                "input/source",
                "application/octet-stream",
                b"changed-source-not-cad".to_vec(),
            )
            .unwrap(),
            expected_source_digest: Digest::sha256(b"original-source"),
            import_options: Default::default(),
            settings,
            contacts: Vec::new(),
            program_revision: ProgramRevision::new(
                Digest::sha256(b"graph"),
                Digest::sha256(b"source"),
                ProgramEnvironment::new(
                    1,
                    1,
                    Digest::sha256(b"runtime"),
                    Digest::sha256(b"catalog"),
                    "matlab",
                )
                .unwrap(),
            )
            .unwrap(),
            artifact_access: crate::checked_fea_artifact_access("test:original-mesh-root"),
        };
        let mut store = MemoryObjectStore::new(inventory).unwrap();
        let platform = PlatformBuildIdentity {
            capability_cohort: "test".into(),
            target_triple: "test".into(),
            build_digest: runmat_meshing_core::StableDigest::from_bytes([1; 32]),
            exact_kernel_abi: None,
        };
        let result = mesh_imported_source_serial(
            request,
            &mut store,
            &GeometryImportContext::new(),
            ceilings,
            platform,
            inventory,
            &runmat_meshing_core::NeverCancelled,
            &mut runmat_meshing_execution::NoopMeshingProgress,
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("changed after the original asset"));
    }
}
