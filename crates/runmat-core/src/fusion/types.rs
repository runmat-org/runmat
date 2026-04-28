#[derive(Debug, Clone, Default)]
pub struct FusionPlanSnapshot {
    pub nodes: Vec<FusionPlanNode>,
    pub edges: Vec<FusionPlanEdge>,
    pub shaders: Vec<FusionPlanShader>,
    pub decisions: Vec<FusionPlanDecision>,
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
