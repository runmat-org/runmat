use super::entrypoints::{
    adjacency_builtin, bfsearch_builtin, conncomp_builtin, degree_builtin, dfsearch_builtin,
    digraph_builtin, distances_builtin, findedge_builtin, graph_builtin, indegree_builtin,
    neighbors_builtin, numedges_builtin, numnodes_builtin, outdegree_builtin, predecessors_builtin,
    shortestpath_builtin, successors_builtin, toposort_builtin, treelayout_builtin,
};
use super::parsing::{node_index_from_f64, numeric_vector};
use super::{DIGRAPH_CLASS, GRAPH_CLASS, NUM_NODES_PROPERTY};
use crate::builtins::table::{table_from_columns, table_replace_variables_like, table_variables};
use futures::executor::block_on;
use runmat_value::{IntegerStorage, LogicalArray, ObjectInstance, StringArray, Tensor, Value};

trait GraphFixtureEdit {
    fn fixture_set_property(&mut self, name: &str, value: Value) -> Option<Value>;
    fn fixture_property_mut(&mut self, name: &str) -> Option<&mut Value>;
}
impl GraphFixtureEdit for ObjectInstance {
    fn fixture_set_property(&mut self, name: &str, value: Value) -> Option<Value> {
        let context = crate::context::legacy::active();
        let _source = crate::execution::structure::replacement::lease(
            &Value::Object(self.clone()),
            context.as_ref(),
        )
        .expect("fixture source roots");
        let value = crate::execution::structure::replacement::root(value, context.as_ref())
            .expect("fixture replacement roots");
        value
            .with_owned(|value| {
                crate::execution::object::replacement::replace_property(
                    context.as_ref(),
                    self,
                    name,
                    value,
                    |message, source| {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            context.as_ref(),
                            "RunMat:graph:FixtureAllocation",
                            message,
                            source,
                        )
                    },
                    |message| super::graph_error("graph", message),
                )
            })
            .expect("fixture property destination")
    }
    fn fixture_property_mut(&mut self, name: &str) -> Option<&mut Value> {
        let value = self.property(name)?.clone();
        self.fixture_set_property(name, value);
        self.try_property_mut_unique(name)
    }
}

#[test]
fn graph_classes_register_in_each_runtime_context() {
    let first = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let second = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));

    for runtime in [&first, &second] {
        let _selected = runtime.enter();
        block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("graph constructor");
        block_on(digraph_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("digraph constructor");
        for identity in [GRAPH_CLASS, DIGRAPH_CLASS] {
            assert!(
                crate::class_registry::issue_builtin_class_in_context(runtime, &identity.into())
                    .is_some(),
                "{identity} must be registered in the selected runtime"
            );
        }
    }
}

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
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric(vec![1.0, 2.0, 3.0], 3, 1),
            numeric(vec![2.0, 3.0, 1.0], 3, 1),
        ]),
    ))
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
fn logical_adjacency_is_documented_input_in_both_compatibility_modes() {
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let undirected = Value::LogicalArray(
        LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical adjacency"),
    );
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![undirected]),
    ))
    .expect("logical graph");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).expect("adjacency")),
        vec![0.0, 1.0, 1.0, 0.0],
    );

    let directed = Value::LogicalArray(
        LogicalArray::new(vec![0, 0, 1, 0], vec![2, 2]).expect("logical adjacency"),
    );
    let graph = block_on(digraph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![directed.clone()]),
    ))
    .expect("logical digraph");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).expect("adjacency")),
        vec![0.0, 0.0, 1.0, 0.0],
    );
    let lower = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            directed.clone(),
            Value::String("lower".into()),
        ]),
    ))
    .expect("logical lower triangle");
    assert_eq!(block_on(numedges_builtin(lower)).unwrap(), Value::Num(0.0));
    let upper = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            directed.clone(),
            Value::String("upper".into()),
        ]),
    ))
    .expect("logical upper triangle");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(upper)).expect("adjacency")),
        vec![0.0, 1.0, 1.0, 0.0],
    );
    let error = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            directed,
            Value::String("upper".into()),
            Value::String("extra".into()),
        ]),
    ))
    .expect_err("adjacency accepts at most one option");
    assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
}

#[test]
fn traversal_and_component_families_have_concrete_results() {
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric(vec![1.0, 2.0, 4.0], 3, 1),
            numeric(vec![2.0, 3.0, 5.0], 3, 1),
        ]),
    ))
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
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(1.0),
        ]),
    ))
    .expect("self-loop graph");
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
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            int_numeric(IntegerStorage::I16(vec![1, 2, 3]), 3, 1),
            int_numeric(IntegerStorage::U16(vec![2, 3, 1]), 3, 1),
            int_numeric(IntegerStorage::U8(vec![1, 2, 3]), 3, 1),
        ]),
    ))
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
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![int_numeric(
            IntegerStorage::I16(vec![0, 2, 0, 0, 0, 3, 4, 0, 0]),
            3,
            3,
        )]),
    ))
    .expect("graph from adjacency");

    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
        vec![0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 4.0, 3.0, 0.0]
    );
}

#[test]
fn digraph_adjacency_reads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let graph = block_on(digraph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![int_numeric(
            IntegerStorage::I16(vec![0, 2, 0, 0, 0, 3, 4, 0, 0]),
            3,
            3,
        )]),
    ))
    .expect("digraph from adjacency");

    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
        vec![0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
    );
}

#[test]
fn graph_adjacency_triangle_modes_read_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let upper = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            int_numeric(IntegerStorage::I16(vec![0, 9, 0, 2, 0, 8, 4, 3, 0]), 3, 3),
            Value::String("upper".into()),
        ]),
    ))
    .expect("upper graph");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(upper)).unwrap()),
        vec![0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 4.0, 3.0, 0.0]
    );

    let lower = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            int_numeric(IntegerStorage::I16(vec![0, 2, 4, 9, 0, 3, 0, 8, 0]), 3, 3),
            Value::String("lower".into()),
        ]),
    ))
    .expect("lower graph");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(lower)).unwrap()),
        vec![0.0, 2.0, 4.0, 2.0, 0.0, 3.0, 4.0, 3.0, 0.0]
    );
}

#[test]
fn integer_graph_data_forms_are_gated_but_node_indices_remain_structural() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let endpoints = vec![
        int_numeric(IntegerStorage::I16(vec![1]), 1, 1),
        int_numeric(IntegerStorage::U16(vec![2]), 1, 1),
    ];
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(endpoints.clone()),
    ))
    .expect("integer node pairs remain structural");
    assert_eq!(
        tensor_data(
            block_on(neighbors_builtin(
                graph.clone(),
                int_numeric(IntegerStorage::I64(vec![1]), 1, 1),
            ))
            .expect("integer node selector")
        ),
        vec![2.0],
    );
    assert!(block_on(neighbors_builtin(
        graph,
        int_numeric(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), 1, 1),
    ))
    .is_err());
    let typed_adjacency = int_numeric(IntegerStorage::U16(vec![0, 2, 2, 0]), 2, 2);
    let error = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![typed_adjacency.clone()]),
    ))
    .expect_err("typed adjacency requires RunMat mode");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:GraphIntegerAdjacencyExtension")
    );
    let error = block_on(digraph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![typed_adjacency.clone()]),
    ))
    .expect_err("typed digraph adjacency requires RunMat mode");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DigraphIntegerAdjacencyExtension")
    );
    let weighted = [
        endpoints,
        vec![int_numeric(IntegerStorage::U64(vec![2]), 1, 1)],
    ]
    .concat();
    let error = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(weighted.clone()),
    ))
    .expect_err("typed edge weight requires RunMat mode");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:GraphIntegerWeightsExtension")
    );
    let error = block_on(digraph_builtin(
        crate::builtins::common::test_support::builtin_arguments(weighted.clone()),
    ))
    .expect_err("typed directed edge weight requires RunMat mode");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DigraphIntegerWeightsExtension")
    );
    drop(_strict);

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![typed_adjacency])
    ))
    .is_ok());
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(weighted)
    ))
    .is_ok());
}

#[test]
fn resident_integer_adjacency_gate_precedes_owner_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let matrix = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 1, 0]), vec![2, 2])
            .expect("typed adjacency");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &matrix)
            .expect("resident adjacency");
        provider.reset_telemetry();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                resident.clone(),
            )]),
        ))
        .expect_err("compatibility gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:GraphIntegerAdjacencyExtension")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        drop(_strict);

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let graph = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                resident,
            )]),
        ))
        .expect("RunMat integer adjacency");
        assert_eq!(
            tensor_data(block_on(adjacency_builtin(graph)).expect("adjacency")),
            vec![0.0, 1.0, 1.0, 0.0]
        );
        assert!(provider.telemetry_snapshot().download_bytes > 0);
    });
}

#[test]
fn resident_integer_edge_weight_gate_precedes_owner_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let weight = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63]), vec![1, 1])
            .expect("typed weight");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &weight)
            .expect("resident weight");
        let endpoints = vec![numeric(vec![1.0], 1, 1), numeric(vec![2.0], 1, 1)];
        provider.reset_telemetry();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(
                [endpoints.clone(), vec![Value::GpuTensor(resident.clone())]].concat(),
            ),
        ))
        .expect_err("compatibility gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:GraphIntegerWeightsExtension")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        drop(_strict);

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let graph = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(
                [endpoints, vec![Value::GpuTensor(resident)]].concat(),
            ),
        ))
        .expect("RunMat integer weight");
        assert_eq!(
            tensor_data(block_on(adjacency_builtin(graph)).expect("adjacency")),
            vec![0.0, (1_u64 << 63) as f64, (1_u64 << 63) as f64, 0.0]
        );
        assert!(provider.telemetry_snapshot().download_bytes > 0);
    });
}

#[test]
fn digraph_preserves_direction_for_neighbors_and_degree() {
    let graph = block_on(digraph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric(vec![1.0, 1.0, 2.0], 3, 1),
            numeric(vec![2.0, 3.0, 3.0], 3, 1),
        ]),
    ))
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
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric(vec![1.0, 2.0], 2, 1),
            numeric(vec![2.0, 3.0], 2, 1),
            numeric(vec![2.0, 1.0], 2, 1),
            names,
        ]),
    ))
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
    assert_eq!(path.data(), vec!["a", "b", "c"]);
}

#[test]
fn adjacency_constructor_and_distances_use_weights() {
    let graph = block_on(digraph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![numeric(
            vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 10.0, 1.0, 0.0],
            3,
            3,
        )]),
    ))
    .expect("digraph");
    let distances = block_on(distances_builtin(
        graph,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(3.0),
        ]),
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
    graph.fixture_set_property(NUM_NODES_PROPERTY, Value::Num(3.0));
    graph.fixture_set_property("Edges", edges);
    graph.fixture_set_property("Nodes", nodes);
    let graph = Value::Object(graph);
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).unwrap()),
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    );
}

#[test]
fn undirected_adjacency_constructor_accepts_lower_triangle() {
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![numeric(
            vec![0.0, 5.0, 0.0, 0.0],
            2,
            2,
        )]),
    ))
    .expect("graph");
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
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![numeric(
            vec![0.0, 5.0, 7.0, 0.0],
            2,
            2,
        )]),
    ))
    .expect("asymmetric graph");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(graph)).expect("adjacency")),
        vec![0.0, 7.0, 7.0, 0.0]
    );

    let fallback = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![numeric(
            vec![0.0, 5.0, 0.0, 0.0],
            2,
            2,
        )]),
    ))
    .expect("lower fallback");
    assert_eq!(
        tensor_data(block_on(adjacency_builtin(fallback)).expect("adjacency")),
        vec![0.0, 5.0, 5.0, 0.0]
    );
}

#[test]
fn adjacency_sums_parallel_edge_weights_in_dense_result() {
    let graph = block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric(vec![1.0, 1.0], 2, 1),
            numeric(vec![2.0, 2.0], 2, 1),
            numeric(vec![2.0, 3.0], 2, 1),
        ]),
    ))
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

    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
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
fn empty_treelayout_obeys_requested_output_count() {
    let empty = numeric(Vec::new(), 1, 0);
    let assert_empty_row = |value: &Value| {
        let Value::Tensor(tensor) = value else {
            panic!("expected empty tensor, got {value:?}");
        };
        assert_eq!(tensor.shape(), &[1, 0]);
        assert!(tensor.materialize_f64().is_empty());
    };

    assert_empty_row(&block_on(treelayout_builtin(empty.clone())).expect("default output"));
    for (count, expected) in [(0, 0), (1, 1), (2, 2)] {
        let _outputs =
            crate::output_count::push_output_count(Some(count)).expect("output-count scope");
        let Value::OutputList(values) =
            block_on(treelayout_builtin(empty.clone())).expect("empty tree outputs")
        else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), expected);
        for value in &values {
            assert_empty_row(value);
        }
    }
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let error = block_on(treelayout_builtin(empty)).expect_err("too many outputs");
    assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
}

#[test]
fn impossible_graph_output_counts_precede_resident_downloads() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = Tensor::new(vec![1.0], vec![1, 1]).expect("source");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
            .expect("resident control");
        let graph = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                numeric(vec![1.0], 1, 1),
                numeric(vec![2.0], 1, 1),
            ]),
        ))
        .expect("graph");
        provider.reset_telemetry();
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let error = block_on(treelayout_builtin(Value::GpuTensor(resident.clone())))
            .expect_err("invalid tree output request");
        assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let error = block_on(shortestpath_builtin(
            graph,
            Value::GpuTensor(resident),
            Value::Num(2.0),
        ))
        .expect_err("invalid path output request");
        assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
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
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for weight in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![numeric(
                vec![weight],
                1,
                1
            )])
        ))
        .is_err());
    }
    let inexact = int_numeric(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), 1, 1);
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![inexact])
    ))
    .is_err());
    let exact_wide = int_numeric(IntegerStorage::U64(vec![1_u64 << 63]), 1, 1);
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![exact_wide])
    ))
    .is_ok());

    let endpoints = [numeric(vec![1.0], 1, 1), numeric(vec![2.0], 1, 1)];
    let inexact_weight = int_numeric(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), 1, 1);
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            endpoints[0].clone(),
            endpoints[1].clone(),
            inexact_weight,
        ])
    ))
    .is_err());
    let exact_weight = int_numeric(IntegerStorage::U64(vec![1_u64 << 63]), 1, 1);
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            endpoints[0].clone(),
            endpoints[1].clone(),
            exact_weight,
        ])
    ))
    .is_ok());
}

#[test]
fn graph_rejects_fractional_node_ids_and_mismatched_weight_lengths() {
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric(vec![1.5], 1, 1),
            numeric(vec![2.0], 1, 1),
        ])
    ))
    .is_err());
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric(vec![1.0, 2.0], 2, 1),
            numeric(vec![2.0, 3.0], 2, 1),
            numeric(vec![1.0], 1, 1),
        ])
    ))
    .is_err());
    let nd = Value::Tensor(Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap());
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![nd.clone()])
    ))
    .is_err());
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            nd,
            numeric(vec![1.0; 8], 8, 1)
        ])
    ))
    .is_err());
    assert!(block_on(graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric(vec![0.0, 1.0, 1.0, 0.0], 2, 2),
            Value::String("upper".into()),
            Value::String("ignored".into()),
        ])
    ))
    .is_err());
}

#[test]
fn topological_order_counts_parallel_edges_without_false_cycle() {
    let graph = block_on(digraph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric(vec![1.0, 1.0], 2, 1),
            numeric(vec![2.0, 2.0], 2, 1),
        ]),
    ))
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
        let error = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident),
                numeric(vec![1.5], 1, 1),
            ]),
        ))
        .expect_err("fractional endpoint");
        assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let source = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
            .expect("resident source");
        provider.reset_telemetry();
        let error = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident),
                numeric(vec![2.0], 1, 1),
                Value::Num(-1.0),
            ]),
        ))
        .expect_err("negative host weight");
        assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn graph_gathers_only_consumed_nested_table_values() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let graph = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                numeric(vec![1.0], 1, 1),
                numeric(vec![2.0], 1, 1),
            ]),
        ))
        .expect("graph");
        let Value::Object(mut object) = graph else {
            unreachable!()
        };
        let endpoints = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("endpoints");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &endpoints)
            .expect("resident endpoints");
        object.fixture_set_property("Unrelated", Value::GpuTensor(resident.clone()));
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
        let Value::Object(edges) = object.properties().get("Edges").unwrap() else {
            unreachable!()
        };
        let vars = table_variables(edges).expect("edge variables");
        let context = crate::context::legacy::active();
        let prepared = crate::execution::structure::replacement::record(&vars, context.as_ref())
            .expect("fixture private variables");
        let _roots = prepared.1;
        let mut vars = prepared.0;
        vars.try_replace_unique("EndNodes", Value::GpuTensor(resident))
            .expect("existing fixture column");
        let replacement = table_replace_variables_like(edges, vars).expect("table replacement");
        object.fixture_set_property("Edges", replacement);
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
        let graph = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                numeric(vec![1.0], 1, 1),
                numeric(vec![2.0], 1, 1),
            ]),
        ))
        .expect("graph");
        let Value::Object(mut object) = graph else {
            unreachable!()
        };
        let endpoints = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("endpoints");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &endpoints)
            .expect("resident endpoints");
        let Value::Object(edges) = object.properties().get("Edges").unwrap() else {
            unreachable!()
        };
        let vars = table_variables(edges).expect("edge variables");
        let context = crate::context::legacy::active();
        let prepared = crate::execution::structure::replacement::record(&vars, context.as_ref())
            .expect("fixture private variables");
        let _roots = prepared.1;
        let mut vars = prepared.0;
        vars.try_replace_unique("EndNodes", Value::GpuTensor(resident))
            .expect("existing fixture column");
        let replacement = table_replace_variables_like(edges, vars).expect("table replacement");
        object.fixture_set_property("Edges", replacement);
        object.fixture_set_property(NUM_NODES_PROPERTY, Value::String("invalid".into()));
        provider.reset_telemetry();
        let error = block_on(adjacency_builtin(Value::Object(object))).expect_err("invalid count");
        assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn host_weight_length_uses_resident_endnode_shape_before_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let graph = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                numeric(vec![1.0, 2.0], 2, 1),
                numeric(vec![2.0, 3.0], 2, 1),
                numeric(vec![2.0, 3.0], 2, 1),
            ]),
        ))
        .expect("weighted graph");
        let Value::Object(mut graph) = graph else {
            panic!("expected graph object");
        };
        let endpoints = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("one edge");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &endpoints)
            .expect("resident endpoints");
        let Some(Value::Object(edges)) = graph.fixture_property_mut("Edges") else {
            panic!("expected Edges table");
        };
        // Deliberately forge an inconsistent table: public table construction
        // correctly refuses unequal column heights, but graph admission must
        // still diagnose a malformed object before touching the provider.
        let Some(Value::Struct(variables)) = edges.fixture_property_mut("__table_variables") else {
            panic!("expected table variables");
        };
        let context = crate::context::legacy::active();
        let prepared =
            crate::execution::structure::replacement::record(variables, context.as_ref())
                .expect("fixture private variables");
        let _roots = prepared.1;
        *variables = prepared.0;
        variables
            .try_replace_unique("EndNodes", Value::GpuTensor(resident))
            .expect("existing fixture column");
        provider.reset_telemetry();
        let error = block_on(adjacency_builtin(Value::Object(graph)))
            .expect_err("host Weight length disagrees with resident EndNodes row count");
        assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn invalid_host_node_controls_precede_resident_graph_downloads() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let graph = block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                numeric(vec![1.0], 1, 1),
                numeric(vec![2.0], 1, 1),
            ]),
        ))
        .expect("graph");
        let Value::Object(mut object) = graph else {
            panic!("expected graph object");
        };
        let endpoints = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("endpoints");
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &endpoints)
            .expect("resident endpoints");
        let Value::Object(edges) = object.properties().get("Edges").unwrap() else {
            panic!("expected Edges table");
        };
        let vars = table_variables(edges).expect("edge variables");
        let context = crate::context::legacy::active();
        let prepared = crate::execution::structure::replacement::record(&vars, context.as_ref())
            .expect("fixture private variables");
        let _roots = prepared.1;
        let mut vars = prepared.0;
        vars.try_replace_unique("EndNodes", Value::GpuTensor(resident))
            .expect("existing fixture column");
        object.fixture_set_property(
            "Edges",
            table_replace_variables_like(edges, vars).expect("table replacement"),
        );
        let graph = Value::Object(object);
        let invalid = numeric(vec![1.5], 1, 1);
        provider.reset_telemetry();
        for result in [
            block_on(neighbors_builtin(graph.clone(), invalid.clone())),
            block_on(findedge_builtin(
                graph.clone(),
                Value::Num(1.0),
                invalid.clone(),
            )),
            block_on(shortestpath_builtin(
                graph.clone(),
                invalid.clone(),
                Value::Num(2.0),
            )),
            block_on(bfsearch_builtin(graph.clone(), invalid.clone())),
            block_on(dfsearch_builtin(graph.clone(), invalid.clone())),
            block_on(distances_builtin(
                graph.clone(),
                crate::builtins::common::test_support::builtin_arguments(vec![invalid.clone()]),
            )),
            block_on(distances_builtin(
                graph.clone(),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(1.0),
                    Value::Num(2.0),
                    Value::Num(1.0),
                ]),
            )),
        ] {
            let error = result.expect_err("invalid host control or arity");
            assert!(error.identifier().is_some());
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }
        assert_eq!(
            tensor_data(block_on(neighbors_builtin(graph, Value::Num(1.0))).unwrap()),
            vec![2.0],
        );
        assert!(provider.telemetry_snapshot().download_bytes > 0);
    });
}
