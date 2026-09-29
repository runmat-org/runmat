use super::entrypoints::{
    adjacency_builtin, bfsearch_builtin, conncomp_builtin, degree_builtin, dfsearch_builtin,
    digraph_builtin, distances_builtin, findedge_builtin, graph_builtin, indegree_builtin,
    neighbors_builtin, numedges_builtin, numnodes_builtin, outdegree_builtin, predecessors_builtin,
    shortestpath_builtin, successors_builtin, toposort_builtin, treelayout_builtin,
};
use super::parsing::{node_index_from_f64, numeric_vector};
use super::{GRAPH_CLASS, NUM_NODES_PROPERTY};
use crate::builtins::table::{table_from_columns, table_replace_variables_like, table_variables};
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
fn traversal_and_component_families_have_concrete_results() {
    let graph = block_on(graph_builtin(vec![
        numeric(vec![1.0, 2.0, 4.0], 3, 1),
        numeric(vec![2.0, 3.0, 5.0], 3, 1),
    ]))
    .expect("graph");
    assert_eq!(
        tensor_data(block_on(degree_builtin(graph.clone())).unwrap()),
        vec![1.0, 2.0, 1.0, 1.0, 1.0],
    );
    assert_eq!(
        tensor_data(block_on(conncomp_builtin(graph.clone())).unwrap()),
        vec![1.0, 1.0, 1.0, 2.0, 2.0],
    );
    assert_eq!(
        tensor_data(block_on(neighbors_builtin(graph.clone(), Value::Num(2.0))).unwrap()),
        vec![1.0, 3.0],
    );
    assert_eq!(
        tensor_data(block_on(bfsearch_builtin(graph.clone(), Value::Num(1.0))).unwrap()),
        vec![1.0, 2.0, 3.0],
    );
    assert_eq!(
        tensor_data(block_on(dfsearch_builtin(graph.clone(), Value::Num(1.0))).unwrap()),
        vec![1.0, 2.0, 3.0],
    );
    assert_eq!(
        block_on(findedge_builtin(
            graph.clone(),
            Value::Num(2.0),
            Value::Num(3.0)
        ))
        .unwrap(),
        Value::Num(2.0),
    );
    assert_eq!(
        block_on(findedge_builtin(graph, Value::Num(1.0), Value::Num(5.0))).unwrap(),
        Value::Num(0.0),
    );
}

#[test]
fn undirected_self_loop_counts_twice_across_degree_queries() {
    let graph =
        block_on(graph_builtin(vec![Value::Num(1.0), Value::Num(1.0)])).expect("self-loop graph");
    for result in [
        block_on(degree_builtin(graph.clone())),
        block_on(indegree_builtin(graph.clone())),
        block_on(outdegree_builtin(graph)),
    ] {
        assert_eq!(tensor_data(result.expect("degree")), vec![2.0]);
    }
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
fn undirected_asymmetric_adjacency_uses_upper_then_lower_fallback() {
    let graph = block_on(graph_builtin(vec![numeric(vec![0.0, 5.0, 7.0, 0.0], 2, 2)]))
        .expect("asymmetric graph");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).expect("adjacency")),
        vec![0.0, 7.0, 7.0, 0.0]
    );

    let fallback = block_on(graph_builtin(vec![numeric(vec![0.0, 5.0, 0.0, 0.0], 2, 2)]))
        .expect("lower fallback");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(fallback)).expect("adjacency")),
        vec![0.0, 5.0, 5.0, 0.0]
    );
}

#[test]
fn adjacency_sums_parallel_edge_weights_in_dense_result() {
    let graph = block_on(graph_builtin(vec![
        numeric(vec![1.0, 1.0], 2, 1),
        numeric(vec![2.0, 2.0], 2, 1),
        numeric(vec![2.0, 3.0], 2, 1),
    ]))
    .expect("parallel weighted edges");
    assert_eq!(
        block_on(numedges_builtin(graph.clone())).unwrap(),
        Value::Num(2.0)
    );
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).expect("adjacency")),
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

#[test]
fn graph_rejects_nonfinite_negative_and_inexact_adjacency_weights() {
    for weight in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(block_on(graph_builtin(vec![numeric(vec![weight], 1, 1)])).is_err());
    }
    let inexact = int_numeric(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), 1, 1);
    assert!(block_on(graph_builtin(vec![inexact])).is_err());
    let exact_wide = int_numeric(IntegerStorage::U64(vec![1_u64 << 63]), 1, 1);
    assert!(block_on(graph_builtin(vec![exact_wide])).is_ok());
}

#[test]
fn graph_rejects_fractional_node_ids_and_mismatched_weight_lengths() {
    assert!(block_on(graph_builtin(vec![
        numeric(vec![1.5], 1, 1),
        numeric(vec![2.0], 1, 1),
    ]))
    .is_err());
    assert!(block_on(graph_builtin(vec![
        numeric(vec![1.0, 2.0], 2, 1),
        numeric(vec![2.0, 3.0], 2, 1),
        numeric(vec![1.0], 1, 1),
    ]))
    .is_err());
    let nd = Value::Tensor(Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap());
    assert!(block_on(graph_builtin(vec![nd.clone()])).is_err());
    assert!(block_on(graph_builtin(vec![nd, numeric(vec![1.0; 8], 8, 1)])).is_err());
    assert!(block_on(graph_builtin(vec![
        numeric(vec![0.0, 1.0, 1.0, 0.0], 2, 2),
        Value::String("upper".into()),
        Value::String("ignored".into()),
    ]))
    .is_err());
}

#[test]
fn topological_order_counts_parallel_edges_without_false_cycle() {
    let graph = block_on(digraph_builtin(vec![
        numeric(vec![1.0, 1.0], 2, 1),
        numeric(vec![2.0, 2.0], 2, 1),
    ]))
    .expect("parallel edges");
    assert_eq!(
        tensor_data(block_on(toposort_builtin(graph)).expect("acyclic graph")),
        vec![1.0, 2.0],
    );
}

#[test]
fn malformed_host_controls_precede_provider_downloads() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
            .expect("resident source");
        provider.reset_telemetry();
        let error = block_on(graph_builtin(vec![
            Value::GpuTensor(resident),
            numeric(vec![1.5], 1, 1),
        ]))
        .expect_err("fractional endpoint");
        assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let source = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
            .expect("resident source");
        provider.reset_telemetry();
        let error = block_on(graph_builtin(vec![
            Value::GpuTensor(resident),
            numeric(vec![2.0], 1, 1),
            Value::Num(-1.0),
        ]))
        .expect_err("negative host weight");
        assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn graph_gathers_only_consumed_nested_table_values() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let graph = block_on(graph_builtin(vec![
            numeric(vec![1.0], 1, 1),
            numeric(vec![2.0], 1, 1),
        ]))
        .expect("graph");
        let Value::Object(mut object) = graph else {
            unreachable!()
        };
        let endpoints = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("endpoints");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &endpoints)
            .expect("resident endpoints");
        object
            .properties
            .insert("Unrelated".into(), Value::GpuTensor(resident.clone()));
        provider.reset_telemetry();
        assert_eq!(
            block_on(numedges_builtin(Value::Object(object.clone()))).unwrap(),
            Value::Num(1.0),
        );
        assert_eq!(
            block_on(numnodes_builtin(Value::Object(object.clone()))).unwrap(),
            Value::Num(2.0),
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let Value::Object(edges) = object.properties.get("Edges").unwrap() else {
            unreachable!()
        };
        let mut vars = table_variables(edges).expect("edge variables");
        vars.fields
            .insert("EndNodes".into(), Value::GpuTensor(resident));
        let replacement = table_replace_variables_like(edges, vars).expect("table replacement");
        object.properties.insert("Edges".into(), replacement);
        provider.reset_telemetry();
        assert_eq!(
            tensor_data(block_on(adjacency_builtin(Value::Object(object))).expect("adjacency")),
            vec![0.0, 1.0, 1.0, 0.0],
        );
        assert!(provider.telemetry_snapshot().download_bytes > 0);
    });
}

#[test]
fn malformed_host_graph_property_precedes_nested_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let graph = block_on(graph_builtin(vec![
            numeric(vec![1.0], 1, 1),
            numeric(vec![2.0], 1, 1),
        ]))
        .expect("graph");
        let Value::Object(mut object) = graph else {
            unreachable!()
        };
        let endpoints = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("endpoints");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &endpoints)
            .expect("resident endpoints");
        let Value::Object(edges) = object.properties.get("Edges").unwrap() else {
            unreachable!()
        };
        let mut vars = table_variables(edges).expect("edge variables");
        vars.fields
            .insert("EndNodes".into(), Value::GpuTensor(resident));
        let replacement = table_replace_variables_like(edges, vars).expect("table replacement");
        object.properties.insert("Edges".into(), replacement);
        object
            .properties
            .insert(NUM_NODES_PROPERTY.into(), Value::String("invalid".into()));
        provider.reset_telemetry();
        let error = block_on(adjacency_builtin(Value::Object(object))).expect_err("invalid count");
        assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}
