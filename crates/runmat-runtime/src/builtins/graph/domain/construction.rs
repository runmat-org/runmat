//! Graph construction routes and bounded representation helpers.

use super::{graph_error, BuiltinResult, Edge, GraphData, Value};

mod adjacency;
mod edge_lists;
mod object;

use adjacency::graph_from_adjacency;
pub(super) use adjacency::is_adjacency_form;
use edge_lists::graph_from_edges;
pub(super) use edge_lists::node_tokens;
use object::graph_object;
pub(super) use object::{graph_data, infer_node_count_from_nodes};

pub(super) fn construct_graph(
    name: &'static str,
    directed: bool,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return graph_object(GraphData {
            directed,
            node_names: None,
            edges: Vec::new(),
            node_count: 0,
        });
    }
    if is_adjacency_form(args) && args.len() > 2 {
        return Err(graph_error(
            name,
            format!("{name}: adjacency construction accepts at most one option or name list"),
        ));
    }
    if args.len() == 1 || is_adjacency_form(args) {
        return graph_from_adjacency(name, directed, args);
    }
    graph_from_edges(name, directed, args)
}
