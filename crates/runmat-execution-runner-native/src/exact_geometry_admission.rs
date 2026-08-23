//! Native exact-CAD admission into a program session's verified object authority.

use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_geometry_io::{
    detect_geometry_format, import_exact_cad, ExactCadImportOptions, GeometryFormat,
    GeometryImportContext,
};
use runmat_meshing_execution::{
    prepare_exact_geometry_input, prepare_exact_geometry_objects, MeshingArtifactAccess,
    PreparedExactGeometryInput,
};

use crate::NativeProgramSession;

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

/// Imports exact CAD, writes its canonical object closure into `session`, and returns the bounded
/// execution input references. No source bytes or physical paths enter the geometry identity.
pub fn admit_exact_geometry(
    session: &NativeProgramSession,
    source_name: &str,
    bytes: &[u8],
    options: &ExactCadImportOptions,
    import_context: &GeometryImportContext,
    artifact_access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> Result<PreparedExactGeometryInput, ExactGeometryAdmissionError> {
    let format = detect_geometry_format(source_name, bytes);
    if !matches!(
        format,
        GeometryFormat::Step | GeometryFormat::Iges | GeometryFormat::Brep
    ) {
        return Err(ExactGeometryAdmissionError::UnsupportedFormat);
    }
    let imported = import_exact_cad(source_name, bytes, format, options, import_context)?;
    let document = imported.geometry_document()?;
    let geometry = prepare_exact_geometry_objects(
        document,
        imported.topology,
        imported.evaluators,
        Some(imported.representation),
        imported.healing_report,
        limits,
    )?;
    let mut store = session.object_store();
    for object in &geometry.objects {
        store.write_verified(object)?;
    }
    prepare_exact_geometry_input(geometry, artifact_access, limits)
        .map_err(ExactGeometryAdmissionError::from)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_execution::resource::Capability;
    use runmat_execution::Digest;
    #[cfg(feature = "occt-native")]
    use runmat_execution_artifact::cache::CacheImport;
    #[cfg(feature = "occt-native")]
    use runmat_meshing_execution::import_exact_geometry_objects;

    use super::*;
    use crate::NativeExecutionConfig;

    #[cfg(feature = "occt-native")]
    const BOX: &[u8] = include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep");

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
        let (_directory, session) = session();
        let result = admit_exact_geometry(
            &session,
            "surface.stl",
            b"solid surface\nendsolid surface\n",
            &ExactCadImportOptions::default(),
            &GeometryImportContext::new(),
            MeshingArtifactAccess {
                authorization_scope: "exact-admission-test".into(),
                encryption_context: Digest::sha256(b"exact-admission-test"),
            },
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
        let admitted = admit_exact_geometry(
            &session,
            "box.brep",
            BOX,
            &ExactCadImportOptions::default(),
            &GeometryImportContext::new(),
            access,
            limits,
        )
        .unwrap();
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
}
