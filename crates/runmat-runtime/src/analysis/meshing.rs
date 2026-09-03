//! Runtime composition boundary for canonical study meshing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use runmat_analysis_core::AnalysisModel;
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::CanonicalMeshingContract;
use sha2::{Digest as _, Sha256};

use super::{
    atomic_write_bytes, current_fea_runtime_config, default_fea_artifact_root,
    solver_mesh_artifact, AnalysisStudySpec,
};

pub(crate) struct ResolvedStudyMesh {
    pub artifact_path: Option<String>,
    pub evidence_path: Option<String>,
    pub boundary_region_ids: BTreeMap<String, PersistentEntityId>,
}

pub(super) fn resolve_study_mesh(spec: &AnalysisStudySpec) -> Result<ResolvedStudyMesh, String> {
    if let Some(path) = spec.solver_mesh_artifact_path.as_deref() {
        solver_mesh_artifact::load_solver_mesh_artifact(Path::new(path))
            .map_err(|error| error.to_string())?;
        return Ok(ResolvedStudyMesh {
            artifact_path: Some(path.to_owned()),
            evidence_path: spec.meshing_evidence_artifact_path.clone(),
            boundary_region_ids: BTreeMap::new(),
        });
    }
    let Some(settings) = spec.meshing_settings.clone() else {
        return Ok(ResolvedStudyMesh {
            artifact_path: None,
            evidence_path: None,
            boundary_region_ids: BTreeMap::new(),
        });
    };
    let output = crate::meshing::execute_meshing(&crate::meshing::MeshingExecutionRequest {
        source_path: PathBuf::from(&spec.geometry.source.path),
        source_units: spec.geometry.units,
        expected_source_sha256: crate::meshing::parse_source_sha256(&spec.geometry.source.sha256)?,
        settings,
        contact_definitions: meshing_contact_definitions(spec),
    })?;
    output
        .evidence
        .validate(&output.artifact)
        .map_err(|error| format!("meshing provider returned invalid evidence: {error}"))?;
    persist_mesh_output(&output)
}

fn meshing_contact_definitions(
    spec: &AnalysisStudySpec,
) -> Vec<crate::meshing::MeshingContactDefinition> {
    spec.model
        .iter()
        .flat_map(|model| &model.interfaces)
        .filter(|interface| {
            matches!(
                interface.kind,
                runmat_analysis_core::AnalysisInterfaceKind::Contact(_)
            )
        })
        .map(|interface| crate::meshing::MeshingContactDefinition {
            side_a_face_selectors: vec![interface.primary_region_id.clone()],
            side_b_face_selectors: vec![interface.secondary_region_id.clone()],
        })
        .collect()
}

pub(crate) fn persist_mesh_output(
    output: &crate::meshing::MeshingExecutionOutput,
) -> Result<ResolvedStudyMesh, String> {
    let artifact_digest = encode_digest(output.artifact.canonical_digest.bytes());
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
    let artifact_bytes = output
        .artifact
        .canonical_encode()
        .map_err(|error| error.to_string())?;
    let evidence_bytes = output
        .evidence
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
        boundary_region_ids: output.boundary_region_ids.clone(),
    })
}

pub(crate) fn apply_boundary_region_ids(
    model: &mut AnalysisModel,
    aliases: &BTreeMap<String, PersistentEntityId>,
) {
    for constraint in &mut model.constraints {
        constraint.visit_targets_mut(|target| {
            if let Some(id) = aliases.get(&target.region_id) {
                target.region_id = id.source_topology_id.clone();
            }
        });
    }
    for boundary in &mut model.boundaries {
        if let Some(id) = aliases.get(&boundary.target.region_id) {
            boundary.target.region_id = id.source_topology_id.clone();
        }
    }
    for connector in &mut model.connectors {
        if let Some(id) = aliases.get(&connector.primary.region_id) {
            connector.primary.region_id = id.source_topology_id.clone();
        }
        if let Some(secondary) = &mut connector.secondary {
            if let Some(id) = aliases.get(&secondary.region_id) {
                secondary.region_id = id.source_topology_id.clone();
            }
        }
    }
    for action in &mut model.excitations {
        if let runmat_analysis_core::ExcitationTarget::Region(target) =
            &mut action.excitation_mut().target
        {
            if let Some(id) = aliases.get(&target.region_id) {
                target.region_id = id.source_topology_id.clone();
            }
        }
    }
    for condition in &mut model.initial_conditions {
        if let Some(target) = condition.region_target_mut() {
            if let Some(id) = aliases.get(&target.region_id) {
                target.region_id = id.source_topology_id.clone();
            }
        }
    }
}

fn persist_canonical(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        let existing = runmat_filesystem::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if existing == bytes {
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
