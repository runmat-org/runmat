use serde::Serialize;

use super::contract::AdmissionFailureKind;

#[derive(Debug, Serialize)]
pub struct CorpusReport {
    pub schema_version: u16,
    pub manifest_sha256: String,
    pub fixture_catalog_sha256: String,
    pub selected_tier: String,
    pub fixture_filter: Option<String>,
    pub cases: Vec<CaseReport>,
    pub metamorphic_relations: Vec<MetamorphicReport>,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct CaseReport {
    pub fixture_id: String,
    pub source_sha256: String,
    pub admission_scenarios: Vec<AdmissionReport>,
    pub mesh_scenarios: Vec<MeshReport>,
}

#[derive(Debug, Serialize)]
pub struct AdmissionReport {
    pub id: String,
    pub outcome: String,
    pub solid_regions: Option<usize>,
    pub healing_operations: Option<usize>,
    pub failure: Option<ObservedFailure>,
    pub passed: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MeshReport {
    pub id: String,
    pub outcome: String,
    pub canonical_digest: Option<String>,
    pub node_count: Option<usize>,
    pub element_count: Option<usize>,
    pub boundary_face_count: Option<usize>,
    pub region_count: Option<usize>,
    pub quality: Option<MeshQualityReport>,
    pub failure: Option<ObservedFailure>,
    pub executions: u32,
    pub passed: bool,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MeshQualityReport {
    pub total_volume_m3: f64,
    pub minimum_scaled_jacobian: f64,
    pub maximum_edge_aspect_ratio: f64,
    pub maximum_radius_edge_ratio: f64,
}

#[derive(Debug, Serialize)]
pub struct MetamorphicReport {
    pub id: String,
    pub reference_fixture_id: String,
    pub reference_scenario_id: String,
    pub compared_fixture_ids: Vec<String>,
    pub passed: bool,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservedFailure {
    Admission {
        category: AdmissionFailureKind,
    },
    Accounting {
        cause: runmat_simulation_usage::LedgerError,
    },
    Meshing {
        category: runmat_meshing_core::MeshingFailureCategory,
        stage: runmat_meshing_core::MeshingStageKind,
        operation: runmat_meshing_core::MeshingOperation,
        entity_ids: Vec<runmat_geometry_core::PersistentEntityId>,
        detail: Option<String>,
        remediation: String,
    },
}
