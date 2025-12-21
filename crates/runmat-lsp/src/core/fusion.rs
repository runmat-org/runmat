use crate::core::types::{FusionPlanEdgePublic, FusionPlanNodePublic, FusionPlanPublic};
use runmat_core::{FusionPlanEdge, FusionPlanNode, FusionPlanSnapshot};

pub fn fusion_plan_public_from_snapshot(snapshot: FusionPlanSnapshot, notes: Option<String>) -> FusionPlanPublic {
    FusionPlanPublic {
        nodes: snapshot
            .nodes
            .into_iter()
            .map(|n: FusionPlanNode| FusionPlanNodePublic {
                id: n.id,
                kind: n.kind,
                label: n.label,
                residency: n.residency,
                shape: n.shape,
            })
            .collect(),
        edges: snapshot
            .edges
            .into_iter()
            .map(|e: FusionPlanEdge| FusionPlanEdgePublic {
                from: e.from,
                to: e.to,
                label: e.reason,
            })
            .collect(),
        notes,
    }
}

