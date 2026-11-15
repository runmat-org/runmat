//! The persisted boundary a run is drawn from, and the solved bodies a result view names.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRenderTopology {
    pub schema_version: String,
    pub source: AnalysisRenderTopologySource,
    #[serde(default)]
    pub volume_element_count: usize,
    pub meshes: Vec<AnalysisRenderMesh>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisRenderTopologySource {
    SolverPrep,
    AnalysisMesh,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRenderMesh {
    pub mesh_id: String,
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<AnalysisRenderRegion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vertex_volume_node_indices: Vec<Option<usize>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triangle_volume_element_indices: Vec<Option<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisRenderRegion {
    pub region_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triangle_ranges: Vec<AnalysisRenderTriangleRange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisRenderTriangleRange {
    pub start: u32,
    pub count: u32,
}

/// One solved body of a study: a solid region the model assigns a material to, by its region
/// name (the CAD name), and the exact geometry entities its volume elements are meshed from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisStudyRegion {
    pub region_id: String,
    pub entity_ids: Vec<runmat_geometry_core::PersistentEntityId>,
}

impl AnalysisStudyRegion {
    /// The solved bodies of `model`: each solid region it assigns a material to, in assignment
    /// order, once. A model that assigns one material to every solid names no body.
    pub fn of_model(model: &runmat_analysis_core::AnalysisModel) -> Vec<Self> {
        use runmat_analysis_core::{PropertyRegionScope, PropertyRegionTarget};
        let mut regions: Vec<Self> = Vec::new();
        for assignment in &model.material_assignments {
            let (PropertyRegionTarget::Solid(PropertyRegionScope::Region(selection))
            | PropertyRegionTarget::Cell(PropertyRegionScope::Region(selection))) =
                &assignment.target
            else {
                continue;
            };
            if regions
                .iter()
                .any(|region| region.region_id == selection.region_id())
            {
                continue;
            }
            regions.push(Self {
                region_id: selection.region_id().to_string(),
                entity_ids: selection.resolved_entities().to_vec(),
            });
        }
        regions
    }
}
