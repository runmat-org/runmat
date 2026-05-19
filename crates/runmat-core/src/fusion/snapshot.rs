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
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            let mut shaders = Vec::new();
            let mut decisions = Vec::new();
            append_semantic_candidate_artifacts(
                semantic_candidate_groups,
                "bytecode-groups=0",
                accel_graph_state,
                &mut nodes,
                &mut edges,
                &mut shaders,
                &mut decisions,
            );

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
    let semantic_gate_open = planner.mir_fusion_candidate_group_count > 0;

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
            fused: semantic_gate_open,
            reason: Some(if semantic_gate_open {
                format!(
                    "kernel={:?} span=[{}..{}]",
                    group.kind, group.span.start, group.span.end
                )
            } else {
                format!(
                    "kernel={:?} span=[{}..{}] semantic-candidate-groups=0 (bytecode compatibility artifact)",
                    group.kind, group.span.start, group.span.end
                )
            }),
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

    append_semantic_candidate_artifacts(
        semantic_candidate_groups,
        &format!("bytecode-groups={}", groups.len()),
        accel_graph_state,
        &mut nodes,
        &mut edges,
        &mut shaders,
        &mut decisions,
    );

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

fn append_semantic_candidate_artifacts(
    semantic_candidate_groups: &[runmat_vm::SemanticFusionCandidateGroup],
    bytecode_group_state: &str,
    accel_graph_state: &str,
    nodes: &mut Vec<FusionPlanNode>,
    edges: &mut Vec<FusionPlanEdge>,
    shaders: &mut Vec<FusionPlanShader>,
    decisions: &mut Vec<FusionPlanDecision>,
) {
    for (index, group) in semantic_candidate_groups.iter().enumerate() {
        let node_id = format!("semantic-candidate-{}", group.id);
        nodes.push(FusionPlanNode {
            id: node_id.clone(),
            kind: "SemanticCandidate".to_string(),
            label: format!(
                "semantic-run f={:?} b={:?} stmts=[{}..{}] signals={}",
                group.function, group.block, group.stmt_start, group.stmt_end, group.signal_count
            ),
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
                "semantic-candidate signals={} {} accel-graph={}",
                group.signal_count, bytecode_group_state, accel_graph_state
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
}

#[cfg(test)]
mod tests {
    use super::build_fusion_snapshot;
    use crate::fusion::FusionPlannerMetadata;
    use runmat_accelerate::fusion::{FusionGroup, FusionKind};
    use runmat_accelerate::graph::{InstrSpan, ShapeInfo};

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
                function: runmat_hir::FunctionId(0),
                block: runmat_mir::BasicBlockId(0),
                stmt_start: 1,
                stmt_end: 4,
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

    #[test]
    fn semantic_candidate_groups_emit_nodes_with_bytecode_groups() {
        let snapshot = build_fusion_snapshot(
            None,
            &[FusionGroup {
                id: 7,
                kind: FusionKind::ElementwiseChain,
                nodes: vec![0, 1],
                shape: ShapeInfo::Scalar,
                span: InstrSpan { start: 3, end: 9 },
                pattern: None,
                stack_layout: None,
            }],
            &[runmat_vm::SemanticFusionCandidateGroup {
                id: 1,
                signal_count: 2,
                function: runmat_hir::FunctionId(1),
                block: runmat_mir::BasicBlockId(2),
                stmt_start: 0,
                stmt_end: 2,
            }],
            Some(FusionPlannerMetadata {
                source: "semantic".to_string(),
                mir_local_fact_count: 0,
                mir_diagnostic_count: 0,
                mir_fusion_signal_count: 2,
                mir_fusion_candidate_group_count: 1,
            }),
        )
        .expect("fusion snapshot");

        assert!(
            snapshot.nodes.iter().any(|node| node.id == "group-7"),
            "expected bytecode fusion node"
        );
        assert!(
            snapshot
                .nodes
                .iter()
                .any(|node| node.id == "semantic-candidate-1"),
            "expected semantic candidate node"
        );
    }

    #[test]
    fn bytecode_groups_without_semantic_candidates_are_marked_non_fused() {
        let snapshot = build_fusion_snapshot(
            None,
            &[FusionGroup {
                id: 3,
                kind: FusionKind::ElementwiseChain,
                nodes: vec![0, 1],
                shape: ShapeInfo::Scalar,
                span: InstrSpan { start: 10, end: 12 },
                pattern: None,
                stack_layout: None,
            }],
            &[],
            Some(FusionPlannerMetadata {
                source: "semantic".to_string(),
                mir_local_fact_count: 0,
                mir_diagnostic_count: 0,
                mir_fusion_signal_count: 2,
                mir_fusion_candidate_group_count: 0,
            }),
        )
        .expect("snapshot");

        let decision = snapshot
            .decisions
            .iter()
            .find(|decision| decision.node_id == "group-3")
            .expect("bytecode group decision");
        assert!(
            !decision.fused,
            "expected bytecode-only group to be marked non-fused when semantic candidates are absent"
        );
        assert!(
            decision
                .reason
                .as_deref()
                .unwrap_or("")
                .contains("semantic-candidate-groups=0"),
            "expected semantic gating reason in decision"
        );
    }
}
