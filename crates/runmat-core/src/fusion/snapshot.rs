use runmat_accelerate::fusion::FusionGroup;
use runmat_accelerate::graph::{AccelGraph, ShapeInfo};
use runmat_builtins::BuiltinSemanticKind;
use runmat_mir::{MirAssembly, MirRvalue, MirStmtKind};

use super::{
    FusionPlanDecision, FusionPlanEdge, FusionPlanNode, FusionPlanShader, FusionPlanSnapshot,
    FusionPlannerMetadata,
};

pub(crate) fn build_fusion_snapshot(
    graph: Option<&AccelGraph>,
    groups: &[FusionGroup],
    planner: Option<FusionPlannerMetadata>,
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
        planner: planner.unwrap_or_default(),
    })
}

fn shape_info(shape: &ShapeInfo) -> Vec<usize> {
    match shape {
        ShapeInfo::Unknown => Vec::new(),
        ShapeInfo::Scalar => vec![1, 1],
        ShapeInfo::Tensor(dims) => dims.iter().map(|d| d.unwrap_or(0)).collect::<Vec<usize>>(),
    }
}

pub(crate) fn semantic_fusion_signal_count(mir: &MirAssembly) -> usize {
    let mut count = 0usize;
    for body in mir.bodies.values() {
        for block in &body.blocks {
            for stmt in &block.statements {
                let value = match &stmt.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => value,
                    MirStmtKind::PlaceMutation(_)
                    | MirStmtKind::WorkspaceEffect { .. }
                    | MirStmtKind::EnvironmentEffect(_) => continue,
                };
                if rvalue_has_fusion_signal(value) {
                    count += 1;
                }
            }
        }
    }
    count
}

pub(crate) fn semantic_fusion_candidate_group_count(mir: &MirAssembly) -> usize {
    let mut groups = 0usize;
    for body in mir.bodies.values() {
        for block in &body.blocks {
            let mut run_len = 0usize;
            for stmt in &block.statements {
                let has_signal = match &stmt.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => rvalue_has_fusion_signal(value),
                    MirStmtKind::PlaceMutation(_)
                    | MirStmtKind::WorkspaceEffect { .. }
                    | MirStmtKind::EnvironmentEffect(_) => false,
                };
                if has_signal {
                    run_len += 1;
                    continue;
                }
                if run_len >= 2 {
                    groups += 1;
                }
                run_len = 0;
            }
            if run_len >= 2 {
                groups += 1;
            }
        }
    }
    groups
}

fn rvalue_has_fusion_signal(value: &MirRvalue) -> bool {
    match value {
        MirRvalue::Unary(_, _) | MirRvalue::Binary(_, _, _) => true,
        MirRvalue::Call(call) => matches!(
            call.semantic_kind,
            BuiltinSemanticKind::Elementwise
                | BuiltinSemanticKind::Reduction
                | BuiltinSemanticKind::LinearAlgebra
                | BuiltinSemanticKind::ShapeTransform(_)
        ),
        MirRvalue::Use(_)
        | MirRvalue::Range { .. }
        | MirRvalue::Aggregate { .. }
        | MirRvalue::Index { .. }
        | MirRvalue::Member { .. }
        | MirRvalue::DynamicMember { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End
        | MirRvalue::Future { .. }
        | MirRvalue::Spawn(_) => false,
    }
}
