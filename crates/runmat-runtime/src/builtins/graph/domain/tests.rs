use super::entrypoints::{
    adjacency_builtin, digraph_builtin, distances_builtin, graph_builtin, indegree_builtin,
    numedges_builtin, numnodes_builtin, outdegree_builtin, predecessors_builtin,
    shortestpath_builtin, successors_builtin, treelayout_builtin,
};
use super::parsing::{node_index_from_f64, numeric_vector};
use super::{GRAPH_CLASS, NUM_NODES_PROPERTY};
use crate::builtins::table::table_from_columns;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, ObjectInstance, StringArray, Tensor, Value};

fn numeric(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("tensor"))
}

fn int_numeric(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor");
    Value::Tensor(tensor)
}

fn tensor_data(value: Value) -> Vec<f64> {
    match value {
        Value::Tensor(t) => t.materialize_f64(),
        Value::Num(n) => vec![n],
        other => panic!("expected numeric value, got {other:?}"),
    }
}

#[test]
fn graph_constructor_builds_edges_nodes_and_adjacency() {
    let graph = block_on(graph_builtin(vec![
        numeric(vec![1.0, 2.0, 3.0], 3, 1),
        numeric(vec![2.0, 3.0, 1.0], 3, 1),
    ]))
    .expect("graph");
    assert_eq!(
        block_on(numnodes_builtin(graph.clone())).unwrap(),
        Value::Num(3.0)
    );
    assert_eq!(
        block_on(numedges_builtin(graph.clone())).unwrap(),
        Value::Num(3.0)
    );

    let adjacency = block_on(adjacency_builtin(graph)).expect("adjacency");
    assert_eq!(
        tensor_data(adjacency),
        vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn graph_constructor_accepts_typed_integer_node_vectors() {
    let graph = block_on(graph_builtin(vec![
        int_numeric(IntegerStorage::I16(vec![1, 2, 3]), 3, 1),
        int_numeric(IntegerStorage::U16(vec![2, 3, 1]), 3, 1),
        int_numeric(IntegerStorage::U8(vec![1, 2, 3]), 3, 1),
    ]))
    .expect("graph");
    assert_eq!(
        block_on(numnodes_builtin(graph.clone())).unwrap(),
        Value::Num(3.0)
    );
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
        vec![0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 3.0, 2.0, 0.0]
    );
}

#[test]
fn graph_adjacency_reads_typed_integer_storage_exactly() {
    let graph = block_on(graph_builtin(vec![int_numeric(
        IntegerStorage::I16(vec![0, 2, 0, 0, 0, 3, 4, 0, 0]),
        3,
        3,
    )]))
    .expect("graph from adjacency");

    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
        vec![0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 4.0, 3.0, 0.0]
    );
}

#[test]
fn digraph_adjacency_reads_typed_integer_storage_exactly() {
    let graph = block_on(digraph_builtin(vec![int_numeric(
        IntegerStorage::I16(vec![0, 2, 0, 0, 0, 3, 4, 0, 0]),
        3,
        3,
    )]))
    .expect("digraph from adjacency");

    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
        vec![0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
    );
}

#[test]
fn graph_adjacency_triangle_modes_read_typed_integer_storage_exactly() {
    let upper = block_on(graph_builtin(vec![
        int_numeric(IntegerStorage::I16(vec![0, 9, 0, 2, 0, 8, 4, 3, 0]), 3, 3),
        Value::String("upper".into()),
    ]))
    .expect("upper graph");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(upper)).unwrap()),
        vec![0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 4.0, 3.0, 0.0]
    );

    let lower = block_on(graph_builtin(vec![
        int_numeric(IntegerStorage::I16(vec![0, 2, 4, 9, 0, 3, 0, 8, 0]), 3, 3),
        Value::String("lower".into()),
    ]))
    .expect("lower graph");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(lower)).unwrap()),
        vec![0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 4.0, 3.0, 0.0]
    );
}

#[test]
fn digraph_preserves_direction_for_neighbors_and_degree() {
    let graph = block_on(digraph_builtin(vec![
        numeric(vec![1.0, 1.0, 2.0], 3, 1),
        numeric(vec![2.0, 3.0, 3.0], 3, 1),
    ]))
    .expect("digraph");
    assert_eq!(
        tensor_data(block_on(outdegree_builtin(graph.clone())).unwrap()),
        vec![2.0, 1.0, 0.0]
    );
    assert_eq!(
        tensor_data(block_on(indegree_builtin(graph.clone())).unwrap()),
        vec![0.0, 1.0, 2.0]
    );
    assert_eq!(
        tensor_data(block_on(successors_builtin(graph.clone(), Value::Num(1.0))).unwrap()),
        vec![2.0, 3.0]
    );
    assert_eq!(
        tensor_data(block_on(predecessors_builtin(graph, Value::Num(3.0))).unwrap()),
        vec![1.0, 2.0]
    );
}

#[test]
fn named_graph_returns_named_paths() {
    let names = Value::StringArray(
        StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).expect("names"),
    );
    let graph = block_on(graph_builtin(vec![
        numeric(vec![1.0, 2.0], 2, 1),
        numeric(vec![2.0, 3.0], 2, 1),
        numeric(vec![2.0, 1.0], 2, 1),
        names,
    ]))
    .expect("graph");
    let path = block_on(shortestpath_builtin(
        graph,
        Value::CharArray(runmat_value::CharArray::new_row("a")),
        Value::CharArray(runmat_value::CharArray::new_row("c")),
    ))
    .expect("path");
    let Value::StringArray(path) = path else {
        panic!("expected named path");
    };
    assert_eq!(path.data, vec!["a", "b", "c"]);
}

#[test]
fn adjacency_constructor_and_distances_use_weights() {
    let graph = block_on(digraph_builtin(vec![numeric(
        vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 10.0, 1.0, 0.0],
        3,
        3,
    )]))
    .expect("digraph");
    let distances = block_on(distances_builtin(
        graph,
        vec![Value::Num(1.0), Value::Num(3.0)],
    ))
    .expect("distances");
    assert_eq!(distances, Value::Num(3.0));
}

#[test]
fn edge_endnodes_table_accepts_typed_integer_matrix() {
    let endnodes = int_numeric(IntegerStorage::U16(vec![1, 2, 2, 3]), 2, 2);
    let edges =
        table_from_columns(vec!["EndNodes".to_string()], vec![endnodes]).expect("edges table");
    let nodes = table_from_columns(
        vec!["Index".to_string()],
        vec![numeric(vec![1.0, 2.0, 3.0], 3, 1)],
    )
    .expect("nodes table");
    let mut graph = ObjectInstance::new(GRAPH_CLASS.to_string());
    graph
        .properties
        .insert(NUM_NODES_PROPERTY.to_string(), Value::Num(3.0));
    graph.properties.insert("Edges".to_string(), edges);
    graph.properties.insert("Nodes".to_string(), nodes);
    let graph = Value::Object(graph);
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    );
}

#[test]
fn undirected_adjacency_constructor_accepts_lower_triangle() {
    let graph =
        block_on(graph_builtin(vec![numeric(vec![0.0, 5.0, 0.0, 0.0], 2, 2)])).expect("graph");
    assert_eq!(
        block_on(numedges_builtin(graph.clone())).unwrap(),
        Value::Num(1.0)
    );
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
        vec![0.0, 5.0, 5.0, 0.0]
    );
}

#[test]
fn treelayout_returns_single_or_pair_by_output_count() {
    let result =
        block_on(treelayout_builtin(numeric(vec![0.0, 1.0, 1.0, 2.0], 1, 4))).expect("treelayout");
    assert_eq!(tensor_data(result).len(), 4);

    let _guard = crate::output_count::push_output_count(Some(2));
    let result =
        block_on(treelayout_builtin(numeric(vec![0.0, 1.0, 1.0, 2.0], 1, 4))).expect("treelayout");
    let Value::OutputList(values) = result else {
        panic!("expected output list");
    };
    assert_eq!(values.len(), 2);
    assert_eq!(tensor_data(values[0].clone()).len(), 4);
    assert_eq!(tensor_data(values[1].clone()), vec![-0.0, -1.0, -1.0, -2.0]);
}

#[test]
fn treelayout_accepts_typed_integer_parent_vector() {
    let result = block_on(treelayout_builtin(int_numeric(
        IntegerStorage::I16(vec![0, 1, 1, 2]),
        1,
        4,
    )))
    .expect("treelayout");
    assert_eq!(tensor_data(result).len(), 4);
}

#[test]
fn graph_numeric_vector_reads_large_typed_integer_storage_exactly() {
    let wide = 9_007_199_254_740_993_u64;
    let value = int_numeric(IntegerStorage::U64(vec![wide, wide - 1]), 1, 2);
    match usize::try_from(wide) {
        Ok(wide_usize) => {
            assert_eq!(
                numeric_vector(&value, "graph").expect("numeric vector"),
                vec![wide_usize, wide_usize - 1]
            );
        }
        Err(_) => {
            assert!(numeric_vector(&value, "graph").is_err());
        }
    }
}

#[test]
fn graph_numeric_vector_rejects_unrepresentable_double_boundary_before_cast() {
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };

    assert!(numeric_vector(&Value::Num(boundary), "graph").is_err());
    assert!(numeric_vector(&Value::Num(1.5), "graph").is_err());
}

#[test]
fn graph_node_index_rejects_unrepresentable_double_boundary_before_cast() {
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };

    assert!(node_index_from_f64(boundary, usize::MAX, "graph").is_err());
    assert!(node_index_from_f64(1.5, 3, "graph").is_err());
}
