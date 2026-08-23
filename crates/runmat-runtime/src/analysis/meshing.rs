//! Runtime composition boundary for canonical study meshing.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use runmat_analysis_core::AnalysisModel;
use runmat_geometry_core::{PersistentEntityId, UnitSystem};
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingEvidence, MeshingRequestSettings, SolverMeshArtifact,
};
use sha2::{Digest as _, Sha256};

use super::{
    atomic_write_bytes, current_fea_runtime_config, default_fea_artifact_root,
    solver_mesh_artifact, AnalysisStudySpec,
};

#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisMeshingRequest {
    pub source_path: PathBuf,
    pub source_units: UnitSystem,
    pub settings: MeshingRequestSettings,
    pub default_material_id: Option<String>,
    pub region_materials: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisMeshingOutput {
    pub artifact: SolverMeshArtifact,
    pub evidence: MeshingEvidence,
    pub boundary_region_ids: BTreeMap<String, PersistentEntityId>,
}

pub type AnalysisMeshingProvider = Arc<
    dyn Fn(&AnalysisMeshingRequest) -> Result<AnalysisMeshingOutput, String>
        + Send
        + Sync
        + 'static,
>;

fn provider_slot() -> &'static RwLock<Option<AnalysisMeshingProvider>> {
    static PROVIDER: OnceLock<RwLock<Option<AnalysisMeshingProvider>>> = OnceLock::new();
    PROVIDER.get_or_init(|| RwLock::new(None))
}

pub struct AnalysisMeshingProviderGuard {
    previous: Option<AnalysisMeshingProvider>,
}

impl Drop for AnalysisMeshingProviderGuard {
    fn drop(&mut self) {
        if let Ok(mut provider) = provider_slot().write() {
            *provider = self.previous.take();
        }
    }
}

pub fn replace_analysis_meshing_provider(
    provider: Option<AnalysisMeshingProvider>,
) -> Result<AnalysisMeshingProviderGuard, String> {
    let mut slot = provider_slot()
        .write()
        .map_err(|_| "analysis meshing provider lock poisoned".to_string())?;
    let previous = std::mem::replace(&mut *slot, provider);
    Ok(AnalysisMeshingProviderGuard { previous })
}

pub(super) struct ResolvedStudyMesh {
    pub artifact_path: Option<String>,
    pub evidence_path: Option<String>,
    pub boundary_region_ids: BTreeMap<String, PersistentEntityId>,
}

pub(super) fn resolve_study_mesh(
    spec: &AnalysisStudySpec,
    model: &AnalysisModel,
) -> Result<ResolvedStudyMesh, String> {
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
    let provider = provider_slot()
        .read()
        .map_err(|_| "analysis meshing provider lock poisoned".to_string())?
        .clone()
        .ok_or_else(|| {
            "this host cannot execute the requested meshing workload; use a native RunMat host or configure compatible execution capacity".to_string()
        })?;
    let (default_material_id, region_materials) = material_intent(model)?;
    let output = provider(&AnalysisMeshingRequest {
        source_path: PathBuf::from(&spec.geometry.source.path),
        source_units: spec.geometry.units,
        settings,
        default_material_id,
        region_materials,
    })?;
    output
        .evidence
        .validate(&output.artifact)
        .map_err(|error| format!("meshing provider returned invalid evidence: {error}"))?;
    persist_mesh_output(&output)
}

fn material_intent(
    model: &AnalysisModel,
) -> Result<(Option<String>, BTreeMap<String, String>), String> {
    let material_ids = model
        .material_assignments
        .iter()
        .map(|assignment| assignment.assigned_material_id.clone())
        .chain(
            (model.material_assignments.is_empty() && model.materials.len() == 1)
                .then(|| model.materials[0].material_id.clone()),
        )
        .collect::<BTreeSet<_>>();
    if material_ids.is_empty() {
        return Err("solid meshing requires at least one assigned material".to_string());
    }
    if material_ids.len() == 1 {
        return Ok((material_ids.into_iter().next(), BTreeMap::new()));
    }
    let region_materials = model
        .material_assignments
        .iter()
        .map(|assignment| {
            (
                assignment.region_id.clone(),
                assignment.assigned_material_id.clone(),
            )
        })
        .collect();
    Ok((None, region_materials))
}

fn persist_mesh_output(output: &AnalysisMeshingOutput) -> Result<ResolvedStudyMesh, String> {
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

pub(super) fn apply_boundary_region_ids(
    model: &mut AnalysisModel,
    aliases: &BTreeMap<String, PersistentEntityId>,
) {
    for boundary_condition in &mut model.boundary_conditions {
        if let Some(id) = aliases.get(&boundary_condition.region_id) {
            boundary_condition.region_id = id.source_topology_id.clone();
        }
    }
    for load in &mut model.loads {
        if let Some(id) = aliases.get(&load.region_id) {
            load.region_id = id.source_topology_id.clone();
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
