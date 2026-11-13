//! Public-output lifetimes in one validated placement DAG.

use std::collections::BTreeMap;

use runmat_execution::{PlacementPlanRequest, SourceOutputOrdinal};

pub(super) struct RetentionSchedule {
    pub(super) retain_at: Vec<Vec<SourceOutputOrdinal>>,
    pub(super) release_at: Vec<Vec<(usize, SourceOutputOrdinal)>>,
}

impl RetentionSchedule {
    pub(super) fn new(request: &PlacementPlanRequest) -> Self {
        let mut last = BTreeMap::<(usize, SourceOutputOrdinal), usize>::new();
        for edge in &request.graph.edges {
            let output = edge.source_output.unwrap_or(SourceOutputOrdinal::new(0));
            last.entry((edge.from as usize, output))
                .and_modify(|consumer| *consumer = (*consumer).max(edge.to as usize))
                .or_insert(edge.to as usize);
        }
        let mut retain_at = vec![Vec::new(); request.graph.nodes.len()];
        let mut release_at = vec![Vec::new(); request.graph.nodes.len()];
        for ((producer, output), consumer) in last {
            retain_at[producer].push(output);
            release_at[consumer].push((producer, output));
        }
        // A multi-node sink publishes its outputs beyond the graph. Keep
        // those values in the terminal state; they have no graph consumer at
        // which their retention can be released. The single-node whole-work
        // envelope is accounted by its own candidate resource admission.
        if request.graph.nodes.len() > 1 {
            for (node_index, node) in request.graph.nodes.iter().enumerate() {
                if !retain_at[node_index].is_empty() {
                    continue;
                }
                let output_count = node.candidates[0].output_facts.as_ref().map_or(1, Vec::len);
                for output in 0..output_count {
                    retain_at[node_index].push(SourceOutputOrdinal::new(
                        u32::try_from(output).expect("validated output role count fits u32"),
                    ));
                }
            }
        }
        Self {
            retain_at,
            release_at,
        }
    }
}
