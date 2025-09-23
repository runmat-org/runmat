use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::graph::{AccelGraph, AccelNode, InstrSpan, NodeId, ShapeInfo, ValueId};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FusionKind {
    ElementwiseChain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionGroup {
    pub id: usize,
    pub kind: FusionKind,
    pub nodes: Vec<NodeId>,
    pub shape: ShapeInfo,
    pub span: InstrSpan,
}

pub fn detect_fusion_groups(graph: &AccelGraph) -> Vec<FusionGroup> {
    if graph.nodes.is_empty() {
        return Vec::new();
    }

    let consumer_map = build_consumer_map(graph);
    let mut assigned: HashSet<NodeId> = HashSet::new();
    let mut groups = Vec::new();
    let mut group_id = 0usize;

    for node in &graph.nodes {
        if !node.is_elementwise() || assigned.contains(&node.id) {
            continue;
        }
        if node.outputs.is_empty() {
            continue;
        }
        let mut current_shape = node_output_shape(graph, node);
        if matches!(current_shape, ShapeInfo::Unknown) {
            continue;
        }
        let mut chain: Vec<NodeId> = Vec::new();
        let mut frontier = node.id;
        let mut local_seen: HashSet<NodeId> = HashSet::new();

        loop {
            if !local_seen.insert(frontier) {
                break;
            }
            chain.push(frontier);
            let next = find_next_elementwise(
                graph,
                frontier,
                &assigned,
                &local_seen,
                &consumer_map,
                &current_shape,
            );
            match next {
                Some((next_id, next_shape)) => {
                    frontier = next_id;
                    current_shape = next_shape;
                }
                None => break,
            }
        }

        if chain.len() > 1 {
            for id in &chain {
                assigned.insert(*id);
            }
            let span = group_span(graph, &chain);
            groups.push(FusionGroup {
                id: group_id,
                kind: FusionKind::ElementwiseChain,
                nodes: chain,
                shape: current_shape.clone(),
                span,
            });
            group_id += 1;
        }
    }

    groups
}

fn build_consumer_map(graph: &AccelGraph) -> HashMap<ValueId, HashSet<NodeId>> {
    let mut map: HashMap<ValueId, HashSet<NodeId>> = HashMap::new();
    for node in &graph.nodes {
        for &input in &node.inputs {
            if let Some(value) = graph.value(input) {
                if matches!(value.origin, crate::graph::ValueOrigin::NodeOutput { .. }) {
                    map.entry(input).or_default().insert(node.id);
                }
            }
        }
    }
    map
}

fn node_output_shape(graph: &AccelGraph, node: &AccelNode) -> ShapeInfo {
    let mut shape = ShapeInfo::Scalar;
    for &output in &node.outputs {
        if let Some(info) = graph.value(output) {
            shape = shape.unify(&info.shape);
        }
    }
    shape
}

fn find_next_elementwise(
    graph: &AccelGraph,
    node_id: NodeId,
    assigned: &HashSet<NodeId>,
    local_seen: &HashSet<NodeId>,
    consumer_map: &HashMap<ValueId, HashSet<NodeId>>,
    current_shape: &ShapeInfo,
) -> Option<(NodeId, ShapeInfo)> {
    let node = graph.node(node_id)?;
    let mut candidate: Option<(NodeId, ShapeInfo)> = None;

    for &output in &node.outputs {
        let consumers = consumer_map.get(&output)?;
        if consumers.len() != 1 {
            return None;
        }
        let next_id = *consumers.iter().next()?;
        if next_id <= node_id || assigned.contains(&next_id) || local_seen.contains(&next_id) {
            return None;
        }
        let next_node = graph.node(next_id)?;
        if !next_node.is_elementwise() {
            return None;
        }
        // Ensure the edge we follow is actually used by next node
        if !next_node.inputs.contains(&output) {
            continue;
        }
        let next_shape = node_output_shape(graph, next_node);
        if matches!(next_shape, ShapeInfo::Unknown) {
            return None;
        }
        let unified = current_shape.unify(&next_shape);
        if matches!(unified, ShapeInfo::Unknown) {
            return None;
        }
        candidate = Some((next_id, unified));
        break;
    }

    candidate
}

fn group_span(graph: &AccelGraph, nodes: &[NodeId]) -> InstrSpan {
    let mut start = usize::MAX;
    let mut end = 0usize;
    for &id in nodes {
        if let Some(node) = graph.node(id) {
            start = start.min(node.span.start);
            end = end.max(node.span.end);
        }
    }
    if start == usize::MAX {
        start = 0;
    }
    InstrSpan { start, end }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan,
        PrimitiveOp, ValueInfo, ValueOrigin, VarKind,
    };
    use runmat_builtins::Type;

    fn simple_elementwise_graph() -> AccelGraph {
        let mut values = Vec::new();
        // Value 0: input tensor
        values.push(ValueInfo {
            id: 0,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
        });
        // Node 0 output value (value id 1)
        values.push(ValueInfo {
            id: 1,
            origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
        });
        // Node 1 output value (value id 2)
        values.push(ValueInfo {
            id: 2,
            origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(4), Some(4)]),
        });

        let node0 = AccelNode {
            id: 0,
            label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
            category: AccelOpCategory::Elementwise,
            inputs: vec![0, 0],
            outputs: vec![1],
            span: InstrSpan { start: 10, end: 10 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        let node1 = AccelNode {
            id: 1,
            label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
            category: AccelOpCategory::Elementwise,
            inputs: vec![1, 0],
            outputs: vec![2],
            span: InstrSpan { start: 11, end: 11 },
            tags: vec![AccelGraphTag::Elementwise],
        };

        AccelGraph {
            nodes: vec![node0, node1],
            values,
        }
    }

    #[test]
    fn detects_chain() {
        let graph = simple_elementwise_graph();
        let groups = detect_fusion_groups(&graph);
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert_eq!(group.nodes, vec![0, 1]);
        assert_eq!(group.kind, FusionKind::ElementwiseChain);
    }
}
