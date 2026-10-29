//! Runtime composition boundary for canonical study meshing.

use std::path::{Path, PathBuf};

use runmat_meshing_core::CanonicalMeshingContract;
use sha2::{Digest as _, Sha256};

use super::{
    atomic_write_bytes, current_fea_runtime_config, default_fea_artifact_root,
    solver_mesh_artifact, AnalysisStudySpec,
};

pub(crate) struct ResolvedStudyMesh {
    pub artifact_path: Option<String>,
    pub evidence_path: Option<String>,
}

pub(super) fn resolve_study_mesh(spec: &AnalysisStudySpec) -> Result<ResolvedStudyMesh, String> {
    if let Some(path) = spec.solver_mesh_artifact_path.as_deref() {
        solver_mesh_artifact::load_solver_mesh_artifact(Path::new(path))
            .map_err(|error| error.to_string())?;
        return Ok(ResolvedStudyMesh {
            artifact_path: Some(path.to_owned()),
            evidence_path: spec.meshing_evidence_artifact_path.clone(),
        });
    }
    let Some(settings) = spec.meshing_settings.clone() else {
        return Ok(ResolvedStudyMesh {
            artifact_path: None,
            evidence_path: None,
        });
    };
    let output = crate::meshing::execute_meshing(&crate::meshing::MeshingExecutionRequest {
        source_path: PathBuf::from(&spec.geometry.source.path),
        source_units: match &spec.geometry.source.unit_evidence {
            runmat_geometry_core::physical::LengthUnitEvidence::ExplicitFallback {
                declaration,
                ..
            } => Some(declaration.unit),
            _ => None,
        },
        expected_source_sha256: crate::meshing::parse_source_sha256(&spec.geometry.source.sha256)?,
        settings,
        contact_definitions: meshing_contact_definitions(spec),
    })
    .map_err(|error| error.to_string())?;
    output
        .evidence
        .validate(&output.artifact)
        .map_err(|error| format!("meshing provider returned invalid evidence: {error}"))?;
    persist_mesh_output(&output)
}

pub(super) fn meshing_contact_definitions(
    spec: &AnalysisStudySpec,
) -> Vec<crate::meshing::MeshingContactDefinition> {
    meshing_contact_definitions_for_model(&spec.model).collect()
}

pub(super) fn meshing_contact_definitions_for_model(
    model: &runmat_analysis_core::AnalysisModel,
) -> impl Iterator<Item = crate::meshing::MeshingContactDefinition> + '_ {
    model
        .interfaces
        .iter()
        .filter(|interface| {
            matches!(
                interface.kind,
                runmat_analysis_core::AnalysisInterfaceKind::Contact(_)
            )
        })
        .map(|interface| crate::meshing::MeshingContactDefinition {
            side_a_face_selectors: vec![interface.primary.region_id.clone()],
            side_b_face_selectors: vec![interface.secondary.region_id.clone()],
        })
}

pub(crate) fn persist_mesh_output(
    output: &crate::meshing::MeshingExecutionOutput,
) -> Result<ResolvedStudyMesh, String> {
    persist_mesh_borrowed(&output.artifact, &output.evidence)
}

/// Persist original admitted inventories without constructing a second mesh owner.
pub(crate) fn persist_mesh_borrowed(
    artifact: &runmat_meshing_core::SolverMeshArtifact,
    evidence: &runmat_meshing_core::MeshingEvidence,
) -> Result<ResolvedStudyMesh, String> {
    let artifact_digest = encode_digest(artifact.canonical_digest.bytes());
    let root = current_fea_runtime_config()
        .artifact_root
        .or_else(|| {
            std::env::var("RUNMAT_FEA_ARTIFACT_ROOT")
                .ok()
                .map(PathBuf::from)
        })
        .unwrap_or_else(default_fea_artifact_root)
        .join("meshes");
    runmat_filesystem::create_dir_all(&root)
        .map_err(|error| format!("failed to create {}: {error}", root.display()))?;
    let artifact_bytes = artifact
        .canonical_encode_with_cancellation(&MeshPersistenceCancellation)
        .map_err(|error| error.to_string())?;
    let evidence_bytes = evidence
        .canonical_encode()
        .map_err(|error| error.to_string())?;
    let evidence_digest = encode_digest(&Sha256::digest(&evidence_bytes).into());
    let artifact_path = root.join(format!("{artifact_digest}.solver-mesh.cbor"));
    let evidence_path = root.join(format!(
        "{artifact_digest}-{evidence_digest}.meshing-evidence.cbor"
    ));
    persist_canonical(&artifact_path, &artifact_bytes)?;
    persist_canonical(&evidence_path, &evidence_bytes)?;
    Ok(ResolvedStudyMesh {
        artifact_path: Some(artifact_path.display().to_string()),
        evidence_path: Some(evidence_path.display().to_string()),
    })
}

struct MeshPersistenceCancellation;
impl runmat_meshing_core::MeshingCancellationSignal for MeshPersistenceCancellation {
    fn is_cancelled(&self) -> bool {
        crate::interrupt::is_cancelled()
    }
}

fn persist_canonical(path: &Path, bytes: &[u8]) -> Result<(), String> {
    // The browser's artifacts belong to the installed provider, not std::fs.
    // Only an authoritative NotFound permits creating a new immutable artifact.
    let exists = match runmat_filesystem::metadata(path) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(format!("failed to inspect {}: {error}", path.display())),
    };
    if exists {
        let mut existing = runmat_filesystem::File::open(path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
        let mut offset = 0;
        let mut buffer = [0_u8; 8192];
        while offset < bytes.len() {
            if crate::interrupt::is_cancelled() {
                return Err("meshing persistence cancelled".into());
            }
            let count = buffer.len().min(bytes.len() - offset);
            let read = std::io::Read::read(&mut existing, &mut buffer[..count])
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            if read == 0 || buffer[..read] != bytes[offset..offset + read] {
                break;
            }
            offset += read;
        }
        if offset == bytes.len()
            && std::io::Read::read(&mut existing, &mut buffer[..1])
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?
                == 0
        {
            return Ok(());
        }
        return Err(format!(
            "canonical meshing output collision at {}",
            path.display()
        ));
    }
    atomic_write_bytes(&path.to_path_buf(), bytes)
}

fn encode_digest(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    encoded
}

#[cfg(test)]
#[path = "meshing/persistence_tests.rs"]
mod persistence_tests;
