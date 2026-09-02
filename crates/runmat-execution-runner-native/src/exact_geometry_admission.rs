//! Native exact-CAD admission into a program session's verified object authority.

use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_geometry_core::{ExactContactDefinition, PersistentEntityId};
use runmat_geometry_io::{
    detect_geometry_format, import_exact_cad, ExactCadImportOptions, GeometryFormat,
    GeometryImportContext,
};
use runmat_meshing_execution::{
    prepare_exact_geometry_input, prepare_exact_geometry_objects, MeshingArtifactAccess,
    PreparedExactGeometryInput, PreparedExactGeometryObjects,
};

use crate::NativeProgramSession;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeMeshingContactDefinition {
    pub side_a_face_selectors: Vec<String>,
    pub side_b_face_selectors: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ExactGeometryAdmissionError {
    #[error("exact analysis geometry must be STEP, IGES, or B-rep")]
    UnsupportedFormat,
    #[error("exact geometry import failed: {0}")]
    Import(#[from] runmat_geometry_io::import::GeometryImportError),
    #[error("exact geometry contract failed: {0}")]
    Contract(#[from] runmat_geometry_core::GeometryContractError),
    #[error("exact geometry artifact preparation failed: {0}")]
    Artifact(#[from] runmat_meshing_execution::MeshingExecutionError),
    #[error("exact geometry object persistence failed: {0}")]
    Store(#[from] runmat_execution_artifact::ArtifactError),
}

pub struct PreparedExactGeometryAdmission {
    objects: PreparedExactGeometryObjects,
    source_face_ids: std::collections::BTreeMap<u64, runmat_geometry_core::PersistentEntityId>,
}

impl PreparedExactGeometryAdmission {
    pub fn document(&self) -> &runmat_geometry_core::GeometryDocument {
        &self.objects.document
    }

    pub fn topology(&self) -> &runmat_geometry_core::ExactBRepTopology {
        &self.objects.topology
    }

    pub fn source_face_ids(
        &self,
    ) -> &std::collections::BTreeMap<u64, runmat_geometry_core::PersistentEntityId> {
        &self.source_face_ids
    }

    pub fn healing_report(&self) -> Option<&runmat_geometry_core::GeometryHealingReport> {
        self.objects.healing_report.as_ref()
    }
}

/// Imports and validates exact CAD without selecting an execution session or artifact authority.
pub fn prepare_exact_geometry_admission(
    source_name: &str,
    bytes: &[u8],
    options: &ExactCadImportOptions,
    contact_definitions: &[NativeMeshingContactDefinition],
    import_context: &GeometryImportContext,
    limits: ObjectInventoryLimits,
) -> Result<PreparedExactGeometryAdmission, ExactGeometryAdmissionError> {
    let format = detect_geometry_format(source_name, bytes);
    if !matches!(
        format,
        GeometryFormat::Step | GeometryFormat::Iges | GeometryFormat::Brep
    ) {
        return Err(ExactGeometryAdmissionError::UnsupportedFormat);
    }
    let imported = import_exact_cad(source_name, bytes, format, options, import_context)?;
    let contacts = resolve_contact_definitions(&imported, contact_definitions)?;
    let imported = imported.with_contacts(&contacts)?;
    let document = imported.geometry_document()?;
    let objects = prepare_exact_geometry_objects(
        document,
        imported.topology,
        imported.evaluators,
        Some(imported.representation),
        imported.healing_report,
        limits,
    )?;
    Ok(PreparedExactGeometryAdmission {
        objects,
        source_face_ids: imported.source_face_ids,
    })
}

fn resolve_contact_definitions(
    imported: &runmat_geometry_io::ImportedExactCad,
    definitions: &[NativeMeshingContactDefinition],
) -> Result<Vec<ExactContactDefinition>, ExactGeometryAdmissionError> {
    definitions
        .iter()
        .map(|definition| {
            Ok(ExactContactDefinition {
                side_a_face_ids: resolve_face_selectors(
                    imported,
                    &definition.side_a_face_selectors,
                )?,
                side_b_face_ids: resolve_face_selectors(
                    imported,
                    &definition.side_b_face_selectors,
                )?,
            })
        })
        .collect()
}

fn resolve_face_selectors(
    imported: &runmat_geometry_io::ImportedExactCad,
    selectors: &[String],
) -> Result<Vec<PersistentEntityId>, ExactGeometryAdmissionError> {
    selectors
        .iter()
        .map(|selector| {
            if let Some(ordinal) = selector
                .strip_prefix("face_")
                .and_then(|value| value.parse::<u64>().ok())
            {
                let source_index = ordinal.checked_sub(1).ok_or_else(|| {
                    invalid_contact_selector(selector, "source face ordinal is one-based")
                })?;
                return imported
                    .source_face_ids
                    .get(&source_index)
                    .cloned()
                    .ok_or_else(|| {
                        invalid_contact_selector(selector, "source face ordinal does not exist")
                    });
            }
            let matches = imported
                .topology
                .faces
                .iter()
                .filter(|face| face.id.source_topology_id == *selector)
                .map(|face| face.id.clone())
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [face] => Ok(face.clone()),
                [] => Err(invalid_contact_selector(
                    selector,
                    "persistent source face does not exist",
                )),
                _ => Err(invalid_contact_selector(
                    selector,
                    "persistent source face is ambiguous across assembly occurrences",
                )),
            }
        })
        .collect()
}

fn invalid_contact_selector(selector: &str, reason: &str) -> ExactGeometryAdmissionError {
    ExactGeometryAdmissionError::Contract(runmat_geometry_core::GeometryContractError::invalid(
        "contact face selector",
        format!("{selector}: {reason}"),
    ))
}

/// Binds a prepared exact closure to one session authority and persists it atomically.
pub fn admit_prepared_exact_geometry(
    session: &NativeProgramSession,
    prepared: PreparedExactGeometryAdmission,
    artifact_access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> Result<PreparedExactGeometryInput, ExactGeometryAdmissionError> {
    let mut store = session.object_store();
    for object in &prepared.objects.objects {
        store.write_verified(object)?;
    }
    prepare_exact_geometry_input(prepared.objects, artifact_access, limits)
        .map_err(ExactGeometryAdmissionError::from)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "occt-native")]
    use std::collections::BTreeSet;

    #[cfg(feature = "occt-native")]
    use runmat_execution::resource::Capability;
    #[cfg(feature = "occt-native")]
    use runmat_execution::Digest;
    #[cfg(feature = "occt-native")]
    use runmat_execution_artifact::cache::CacheImport;
    #[cfg(feature = "occt-native")]
    use runmat_meshing_execution::import_exact_geometry_objects;

    use super::*;
    #[cfg(feature = "occt-native")]
    use crate::NativeExecutionConfig;

    #[cfg(feature = "occt-native")]
    const BOX: &[u8] = include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep");
    #[cfg(feature = "occt-native")]
    const INDEPENDENT_CONTACT: &[u8] =
        include_bytes!("../../runmat-geometry/io/tests/fixtures/independent_contact.step");

    #[cfg(feature = "occt-native")]
    fn session() -> (tempfile::TempDir, NativeProgramSession) {
        let directory = tempfile::tempdir().unwrap();
        let session = NativeProgramSession::new(NativeExecutionConfig {
            executable: std::env::current_exe().unwrap(),
            worker_arguments: vec!["--execution-worker".into()],
            max_workers: 1,
            max_message_bytes: 1024,
            max_object_bytes: 512 * 1024 * 1024,
            max_stderr_bytes: 1024,
            store_root: directory.path().to_path_buf(),
            worker_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
        })
        .unwrap();
        (directory, session)
    }

    #[test]
    fn non_cad_input_is_rejected_before_artifact_publication() {
        let result = prepare_exact_geometry_admission(
            "surface.stl",
            b"solid surface\nendsolid surface\n",
            &ExactCadImportOptions::default(),
            &[],
            &GeometryImportContext::new(),
            ObjectInventoryLimits::default(),
        );
        assert!(matches!(
            result,
            Err(ExactGeometryAdmissionError::UnsupportedFormat)
        ));
    }

    #[cfg(feature = "occt-native")]
    #[test]
    fn admitted_cad_is_immediately_reconstructible_from_the_session_store() {
        let (_directory, session) = session();
        let access = MeshingArtifactAccess {
            authorization_scope: "exact-admission-test".into(),
            encryption_context: Digest::sha256(b"exact-admission-test"),
        };
        let limits = ObjectInventoryLimits::default();
        let prepared = prepare_exact_geometry_admission(
            "box.brep",
            BOX,
            &ExactCadImportOptions::default(),
            &[],
            &GeometryImportContext::new(),
            limits,
        )
        .unwrap();
        let admitted = admit_prepared_exact_geometry(&session, prepared, access, limits).unwrap();
        let geometry = admitted.geometry_objects();
        let root = geometry.root_reference();
        let document = geometry.document.clone();
        assert_eq!(
            session
                .object_store()
                .read_verified(root.digest)
                .unwrap()
                .unwrap()
                .len() as u64,
            root.encoded_length
        );
        let reconstructed =
            import_exact_geometry_objects(&session.object_store(), document, root, limits).unwrap();
        assert_eq!(reconstructed, *geometry);
    }

    #[cfg(feature = "occt-native")]
    #[test]
    fn source_face_aliases_author_explicit_contact_before_meshing() {
        let prepared = prepare_exact_geometry_admission(
            "independent_contact.step",
            INDEPENDENT_CONTACT,
            &ExactCadImportOptions::default(),
            &[NativeMeshingContactDefinition {
                side_a_face_selectors: vec!["face_000002".into()],
                side_b_face_selectors: vec!["face_000007".into()],
            }],
            &GeometryImportContext::new(),
            ObjectInventoryLimits::default(),
        )
        .unwrap();

        assert_eq!(prepared.topology().contacts.len(), 1);
        assert_eq!(prepared.topology().contacts[0].side_a_face_ids.len(), 1);
        assert_eq!(prepared.topology().contacts[0].side_b_face_ids.len(), 1);
    }
}
