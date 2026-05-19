use runmat_accelerate::fusion::FusionGroup;
use runmat_accelerate::graph::{AccelGraph, ShapeInfo};

use super::{
    FusionPlanDecision, FusionPlanEdge, FusionPlanNode, FusionPlanShader, FusionPlanSnapshot,
    FusionPlannerMetadata,
};

pub(crate) fn build_fusion_snapshot(
    graph: Option<&AccelGraph>,
    groups: &[FusionGroup],
    semantic_candidate_groups: &[runmat_vm::SemanticFusionCandidateGroup],
    planner: Option<FusionPlannerMetadata>,
) -> Option<FusionPlanSnapshot> {
    let accel_graph_state = if graph.is_some() {
        "present"
    } else {
        "missing"
    };
    let planner = planner.unwrap_or_default();
    if groups.is_empty() {
        if planner.mir_fusion_signal_count == 0 && planner.mir_fusion_candidate_group_count == 0 {
            return None;
        }
        if !semantic_candidate_groups.is_empty() {
            let mut nodes = Vec::with_capacity(semantic_candidate_groups.len());
            let mut edges = Vec::new();
            let mut shaders = Vec::with_capacity(semantic_candidate_groups.len());
            let mut decisions = Vec::with_capacity(semantic_candidate_groups.len());

            for (index, group) in semantic_candidate_groups.iter().enumerate() {
                let node_id = format!("semantic-candidate-{}", group.id);
                nodes.push(FusionPlanNode {
                    id: node_id.clone(),
                    kind: "SemanticCandidate".to_string(),
                    label: format!("semantic-run signals={}", group.signal_count),
                    shape: Vec::new(),
                    residency: None,
                });
                shaders.push(FusionPlanShader {
                    name: node_id.clone(),
                    stage: "semantic-candidate".to_string(),
                    workgroup_size: None,
                    source_hash: None,
                });
                decisions.push(FusionPlanDecision {
                    node_id: node_id.clone(),
                    fused: false,
                    reason: Some(format!(
                        "semantic-candidate signals={} bytecode-groups=0 accel-graph={}",
                        group.signal_count, accel_graph_state
                    )),
                    thresholds: None,
                });
                if let Some(next) = semantic_candidate_groups.get(index + 1) {
                    edges.push(FusionPlanEdge {
                        from: node_id,
                        to: format!("semantic-candidate-{}", next.id),
                        reason: Some("semantic-program-order".to_string()),
                    });
                }
            }

            return Some(FusionPlanSnapshot {
                nodes,
                edges,
                shaders,
                decisions,
                planner,
            });
        }
        return Some(FusionPlanSnapshot {
            nodes: Vec::new(),
            edges: Vec::new(),
            shaders: Vec::new(),
            decisions: vec![FusionPlanDecision {
                node_id: "semantic-candidate-summary".to_string(),
                fused: false,
                reason: Some(format!(
                    "mir-signals={} mir-candidate-groups={} bytecode-groups=0 accel-graph={}",
                    planner.mir_fusion_signal_count,
                    planner.mir_fusion_candidate_group_count,
                    accel_graph_state
                )),
                thresholds: None,
            }],
            planner,
        });
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
        planner,
    })
}

fn shape_info(shape: &ShapeInfo) -> Vec<usize> {
    match shape {
        ShapeInfo::Unknown => Vec::new(),
        ShapeInfo::Scalar => vec![1, 1],
        ShapeInfo::Tensor(dims) => dims.iter().map(|d| d.unwrap_or(0)).collect::<Vec<usize>>(),
    }
}

#[cfg(test)]
mod tests {
    use super::build_fusion_snapshot;
    use crate::fusion::FusionPlannerMetadata;

    #[test]
    fn semantic_candidate_summary_emits_without_accel_graph() {
        let snapshot = build_fusion_snapshot(
            None,
            &[],
            &[],
            Some(FusionPlannerMetadata {
                source: "semantic".to_string(),
                mir_local_fact_count: 0,
                mir_diagnostic_count: 0,
                mir_fusion_signal_count: 2,
                mir_fusion_candidate_group_count: 1,
            }),
        )
        .expect("semantic candidate summary snapshot");

        assert!(snapshot.nodes.is_empty(), "expected no bytecode nodes");
        assert!(
            snapshot
                .decisions
                .iter()
                .any(|decision| decision.node_id == "semantic-candidate-summary"),
            "expected semantic candidate summary decision"
        );
        assert!(
            snapshot.decisions[0]
                .reason
                .as_deref()
                .unwrap_or("")
                .contains("accel-graph=missing"),
            "expected missing accel graph marker in summary reason"
        );
    }

    #[test]
    fn semantic_candidate_groups_emit_nodes_without_bytecode_groups() {
        let snapshot = build_fusion_snapshot(
            None,
            &[],
            &[runmat_vm::SemanticFusionCandidateGroup {
                id: 0,
                signal_count: 3,
            }],
            Some(FusionPlannerMetadata {
                source: "semantic".to_string(),
                mir_local_fact_count: 0,
                mir_diagnostic_count: 0,
                mir_fusion_signal_count: 3,
                mir_fusion_candidate_group_count: 1,
            }),
        )
        .expect("semantic candidate snapshot");

        assert_eq!(
            snapshot.nodes.len(),
            1,
            "expected one semantic candidate node"
        );
        assert_eq!(snapshot.nodes[0].kind, "SemanticCandidate");
        assert!(
            snapshot.decisions[0]
                .reason
                .as_deref()
                .unwrap_or("")
                .contains("semantic-candidate signals=3"),
            "expected semantic candidate signal annotation"
        );
    }
}
