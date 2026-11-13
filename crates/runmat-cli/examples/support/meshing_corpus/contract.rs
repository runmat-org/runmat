use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifest {
    pub schema_version: u16,
    pub fixture_catalog: String,
    pub mesh_profiles: std::collections::BTreeMap<String, MeshRequest>,
    pub cases: Vec<CorpusCase>,
    #[serde(default)]
    pub metamorphic_relations: Vec<MetamorphicRelation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetamorphicRelation {
    pub id: String,
    pub reference: MeshScenarioReference,
    pub comparisons: Vec<MeshScenarioReference>,
    pub require_equal_entity_counts: bool,
    pub volume_relative_tolerance: f64,
    pub quality_relative_tolerance: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshScenarioReference {
    pub fixture_id: String,
    pub scenario_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusCase {
    pub fixture_id: String,
    pub source_units: SourceUnits,
    pub admission_scenarios: Vec<AdmissionScenario>,
    pub mesh_scenarios: Vec<MeshScenario>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceUnits {
    Meter,
    Millimeter,
    Inch,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HealingSettings {
    pub sew: bool,
    pub repair_orientation: bool,
    pub consolidate_duplicates: bool,
    pub repair_tolerance_scale_gaps: bool,
    pub simplify_short_edges_and_sliver_faces: bool,
    pub maximum_displacement_m: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionScenario {
    pub id: String,
    #[serde(default)]
    pub healing: HealingSettings,
    pub expected: AdmissionExpectation,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdmissionExpectation {
    Success {
        solid_regions: usize,
        #[serde(default)]
        healing_operations: Option<usize>,
    },
    Failure {
        kind: AdmissionFailureKind,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionFailureKind {
    UnsupportedFormat,
    Import,
    Contract,
    Artifact,
    Store,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshScenario {
    pub id: String,
    pub tier: ScenarioTier,
    pub admission: String,
    pub profile: String,
    #[serde(default)]
    pub contacts: Vec<ContactDefinition>,
    pub expected: MeshExpectation,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioTier {
    Small,
    Complete,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactDefinition {
    pub side_a_faces: Vec<String>,
    pub side_b_faces: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshRequest {
    pub target_size_m: f64,
    pub maximum_deviation_m: f64,
    pub element_order: ElementOrder,
    pub seed: u64,
    pub maximum_elements: u64,
    pub repeats: u32,
    pub partition_layouts: Vec<PartitionLayout>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementOrder {
    Tet4,
    Tet10,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionLayout {
    pub edges: u32,
    pub faces: u32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeshExpectation {
    Success { regions: usize },
    Failure { failure: FailureExpectation },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FailureExpectation {
    Admission {
        category: AdmissionFailureKind,
    },
    Meshing {
        category: runmat_meshing_core::MeshingFailureCategory,
        stage: runmat_meshing_core::MeshingStageKind,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureCatalog {
    pub schema_version: u16,
    pub corpus_revision: u16,
    pub fixture_root: String,
    pub entries: Vec<FixtureEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureEntry {
    pub id: String,
    pub path: String,
    pub sha256: String,
    pub tier: String,
    pub format: String,
    pub supported_input: bool,
    pub provenance: serde_json::Value,
    pub features: Vec<String>,
    pub expected: serde_json::Value,
    pub test: serde_json::Value,
}
