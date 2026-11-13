//! Compiler/VM-owned graph witness, before any test publication fault. Raw
//! semantic windows intentionally have no node inventory until this same
//! preparation path reconciles the executable graph.

use runmat_accelerate::graph::{AccelNodeLabel, PrimitiveOp};
use runmat_accelerate::FusionKind;

pub(super) fn assert_compiled_add_min(bytecode: &crate::bytecode::Bytecode) {
    let (graph, groups) = bytecode
        .reconciled_runtime_fusion_graph()
        .expect("compiled source must admit a VM-reconciled fusion plan");
    assert!(
        groups.iter().any(|group| {
            if group.kind != FusionKind::Reduction || group.nodes.len() != 2 {
                return false;
            }
            let Some(nodes) = group
                .nodes
                .iter()
                .map(|id| graph.node(*id))
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            let mut adds = nodes
                .iter()
                .filter(|node| {
                    matches!(&node.label, AccelNodeLabel::Primitive(PrimitiveOp::Add))
                })
                .copied();
            let (Some(add), None) = (adds.next(), adds.next()) else {
                return false;
            };
            let mut reductions = nodes
                .iter()
                .filter(|node| node.is_reduction())
                .copied();
            let (Some(reduction), None) = (reductions.next(), reductions.next()) else {
                return false;
            };
            let inside = |node: &runmat_accelerate::graph::AccelNode| {
                node.span.start >= group.span.start && node.span.end <= group.span.end
            };
            inside(add)
                && inside(reduction)
                && add
                    .outputs
                    .iter()
                    .any(|output| reduction.inputs.contains(output))
        }),
        "compiled source must reconcile one Add feeding its selected reduction; groups={groups:?}; graph={graph:?}; instructions={:?}",
        bytecode.instructions
    );
}
