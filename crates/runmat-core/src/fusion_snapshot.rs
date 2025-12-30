use runmat_accelerate::fusion::FusionGroup;
use runmat_accelerate::graph::{AccelGraph, ShapeInfo};

use crate::{
    FusionPlanDecision, FusionPlanEdge, FusionPlanNode, FusionPlanShader, FusionPlanSnapshot,
};

pub fn build_fusion_snapshot(
    graph: Option<&AccelGraph>,
    groups: &[FusionGroup],
) -> Option<FusionPlanSnapshot> {
    graph?;
    if groups.is_empty() {
        return None;
    }
    let mut nodes = Vec::with_capacity(groups.len());
    let mut edges = Vec::new();
    let mut shaders = Vec::with_capacity(groups.len());
    let mut decisions = Vec::with_capacity(groups.len());

    for (index, group) in groups.iter().enumerate() {
        let node_id = format!("group-{}", group.id);
        nodes.push(FusionPlanNode {
            id: node_id.clone(),
            kind: format!("{:?}", group.kind),
            label: format!(
                "{:?} span=[{}..{}] nodes={}",
                group.kind,
                group.span.start,
                group.span.end,
                group.nodes.len()
            ),
            shape: shape_info(&group.shape),
            residency: Some("gpu".to_string()),
        });
        shaders.push(FusionPlanShader {
            name: node_id.clone(),
            stage: format!("{:?}", group.kind),
            workgroup_size: None,
            source_hash: group.pattern.as_ref().map(|p| format!("{:?}", p)),
        });
        decisions.push(FusionPlanDecision {
            node_id: node_id.clone(),
            fused: true,
            reason: Some(format!(
                "kernel={:?} span=[{}..{}]",
                group.kind, group.span.start, group.span.end
            )),
            thresholds: None,
        });
        if let Some(next) = groups.get(index + 1) {
            edges.push(FusionPlanEdge {
                from: node_id,
                to: format!("group-{}", next.id),
                reason: Some("program-order".to_string()),
            });
        }
    }

    Some(FusionPlanSnapshot {
        nodes,
        edges,
        shaders,
        decisions,
    })
}

fn shape_info(shape: &ShapeInfo) -> Vec<usize> {
    match shape {
        ShapeInfo::Unknown => Vec::new(),
        ShapeInfo::Scalar => vec![1, 1],
        ShapeInfo::Tensor(dims) => dims.iter().map(|d| d.unwrap_or(0)).collect::<Vec<usize>>(),
    }
}
