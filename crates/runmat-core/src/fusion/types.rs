#[derive(Debug, Clone, Default)]
pub struct FusionPlanSnapshot {
    pub nodes: Vec<FusionPlanNode>,
    pub edges: Vec<FusionPlanEdge>,
    pub shaders: Vec<FusionPlanShader>,
    pub decisions: Vec<FusionPlanDecision>,
    pub planner: FusionPlannerMetadata,
}

#[derive(Debug, Clone)]
pub struct FusionPlannerMetadata {
    pub source: String,
    pub mir_local_fact_count: usize,
    pub mir_diagnostic_count: usize,
    pub mir_fusion_signal_count: usize,
}

impl Default for FusionPlannerMetadata {
    fn default() -> Self {
        Self {
            source: "bytecode-accel-graph".to_string(),
            mir_local_fact_count: 0,
            mir_diagnostic_count: 0,
            mir_fusion_signal_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FusionPlanNode {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub shape: Vec<usize>,
    pub residency: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FusionPlanEdge {
    pub from: String,
    pub to: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FusionPlanShader {
    pub name: String,
    pub stage: String,
    pub workgroup_size: Option<[u32; 3]>,
    pub source_hash: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FusionPlanDecision {
    pub node_id: String,
    pub fused: bool,
    pub reason: Option<String>,
    pub thresholds: Option<String>,
}
