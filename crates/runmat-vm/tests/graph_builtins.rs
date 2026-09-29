#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_undirected_graph_queries_preserve_values_shapes_and_disconnected_paths() {
    execute_source(
        r#"
        G = graph([1 2 4], [2 3 5]);
        assert(numnodes(G) == 5);
        assert(numedges(G) == 3);
        A = adjacency(G);
        assert(isequal(size(A), [5 5]));
        assert(A(1,2) == 1 && A(2,1) == 1);
        assert(A(1,3) == 0 && A(4,5) == 1);
        assert(isequal(degree(G), [1;2;1;1;1]));
        assert(isequal(conncomp(G), [1 1 1 2 2]));
        assert(isequal(neighbors(G, 2), [1;3]));
        assert(isequal(bfsearch(G, 1), [1;2;3]));
        assert(isequal(dfsearch(G, 1), [1;2;3]));
        assert(findedge(G, 2, 3) == 2);
        assert(findedge(G, 1, 5) == 0);
        D = distances(G);
        assert(isequal(size(D), [5 5]));
        assert(D(1,3) == 2 && isinf(D(1,5)));
        selected = distances(G, [1 4], [3 5]);
        assert(isequal(size(selected), [2 2]));
        assert(selected(1,1) == 2 && isinf(selected(1,2)));
        assert(isinf(selected(2,1)) && selected(2,2) == 1);
        [path, distance] = shortestpath(G, 1, 3);
        assert(isequal(path, [1;2;3]) && distance == 2);
        [missing, no_distance] = shortestpath(G, 1, 5);
        assert(isempty(missing) && isinf(no_distance));
        "#,
    )
    .expect("compiled undirected graph construction and queries");
}

#[test]
fn compiled_directed_graph_queries_respect_direction_and_edge_weights() {
    execute_source(
        r#"
        G = digraph([1 1 2], [2 3 3], [2 10 1]);
        assert(numnodes(G) == 3 && numedges(G) == 3);
        A = adjacency(G);
        assert(A(1,2) == 2 && A(2,1) == 0);
        assert(A(1,3) == 10 && A(2,3) == 1);
        assert(isequal(indegree(G), [0;1;2]));
        assert(isequal(outdegree(G), [2;1;0]));
        assert(isequal(degree(G), [2;2;2]));
        assert(isequal(successors(G, 1), [2;3]));
        assert(isequal(predecessors(G, 3), [1;2]));
        assert(isequal(toposort(G), [1;2;3]));
        [path, distance] = shortestpath(G, 1, 3);
        assert(isequal(path, [1;2;3]) && distance == 3);
        D = distances(G, [1 2], 3);
        assert(isequal(size(D), [2 1]));
        assert(D(1) == 3 && D(2) == 1);
        "#,
    )
    .expect("compiled directed graph construction and queries");
}

#[test]
fn compiled_graph_options_parallel_edges_and_empty_forms_keep_documented_behavior() {
    execute_source(
        r#"
        parallel = graph([1 1], [2 2], [2 3]);
        assert(numedges(parallel) == 2);
        P = adjacency(parallel);
        assert(P(1,2) == 5 && P(2,1) == 5);

        upper = graph([0 2; 9 0], 'upper');
        lower = graph([0 2; 9 0], 'lower');
        U = adjacency(upper);
        L = adjacency(lower);
        assert(U(1,2) == 2 && U(2,1) == 2);
        assert(L(1,2) == 9 && L(2,1) == 9);

        directed_parallel = digraph([1 1], [2 2]);
        assert(isequal(toposort(directed_parallel), [1;2]));

        empty_graph = graph();
        assert(numnodes(empty_graph) == 0 && numedges(empty_graph) == 0);
        assert(isequal(size(adjacency(empty_graph)), [0 0]));

        [x, y] = treelayout([0 1 1]);
        assert(isequal(size(x), [1 3]));
        assert(isequal(y, [0 -1 -1]));
        one = treelayout([0 1 1]);
        assert(isequal(one, x));
        [empty_x, empty_y] = treelayout([]);
        assert(isequal(size(empty_x), [1 0]));
        assert(isequal(size(empty_y), [1 0]));
        "#,
    )
    .expect("compiled graph options, parallel edges, and empty results");
}

#[test]
fn compiled_logical_adjacency_works_without_runmat_extensions() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        r#"
        G = graph(logical([0 1; 1 0]));
        assert(numnodes(G) == 2 && numedges(G) == 1);
        A = adjacency(G);
        assert(isequal(A, [0 1; 1 0]));
        D = digraph(logical([0 1; 0 0]));
        B = adjacency(D);
        assert(B(1,2) == 1 && B(2,1) == 0);
        upper = graph(logical([0 1; 0 0]), 'upper');
        U = adjacency(upper);
        assert(U(1,2) == 1);
        "#,
    )
    .expect("compiled logical graph and digraph adjacency in MATLAB mode");
}

#[test]
fn compiled_integer_nodes_are_structural_but_integer_weight_data_is_mode_gated() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        r#"
        G = graph(uint16([1 2]), int32([2 3]));
        assert(numnodes(G) == 3);
        assert(isequal(neighbors(G, uint64(2)), [1;3]));
        assert(isequal(bfsearch(G, int64(1)), [1;2;3]));
        [path, distance] = shortestpath(G, uint64(1), int64(3));
        assert(isequal(path, [1;2;3]) && distance == 2);
        [x, y] = treelayout(uint16([0 1 1]));
        assert(isequal(size(x), [1 3]) && isequal(y, [0 -1 -1]));
        "#,
    )
    .expect("compiled exact integer structural controls in MATLAB mode");

    for (source, identifier) in [
        (
            "G = graph(uint16([0 1; 1 0]));",
            "RunMat:compatibility:GraphIntegerAdjacencyExtension",
        ),
        (
            "G = digraph(uint16([0 1; 0 0]));",
            "RunMat:compatibility:DigraphIntegerAdjacencyExtension",
        ),
        (
            "G = graph([1], [2], uint16([3]));",
            "RunMat:compatibility:GraphIntegerWeightsExtension",
        ),
        (
            "G = digraph([1], [2], uint16([3]));",
            "RunMat:compatibility:DigraphIntegerWeightsExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("typed graph data requires RunMat mode");
        assert_eq!(error.identifier(), Some(identifier), "{source}: {error}");
    }
}

#[test]
fn compiled_runmat_integer_graph_data_keeps_exactness_boundaries() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        r#"
        G = digraph(uint16([0 2; 0 0]));
        A = adjacency(G);
        assert(A(1,2) == 2 && A(2,1) == 0);
        W = graph([1], [2], uint64(9223372036854775808));
        B = adjacency(W);
        assert(B(1,2) == 9223372036854775808);
        "#,
    )
    .expect("compiled RunMat-only integer adjacency and exact wide weights");

    let error =
        execute_source("wide = uint64(9007199254740992) + uint64(1); G = graph([1], [2], wide);")
            .expect_err("integer weight not exactly representable as double");
    assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
}

#[test]
fn compiled_graph_calls_reject_invalid_arity_and_cycles_with_owning_errors() {
    for source in [
        "G = graph(1, 2, 3, 4, 5);",
        "G = graph([1], [2]); [p, d, extra] = shortestpath(G, 1, 2);",
        "[x, y, extra] = treelayout([0 1]);",
        "G = digraph([1 2], [2 1]); order = toposort(G);",
    ] {
        let error = execute_source(source).expect_err("invalid graph call");
        assert_eq!(
            error.identifier(),
            Some("RunMat:graph:InvalidArgument"),
            "{source}: {error}"
        );
    }
    for source in [
        "G = graph([1], [2]); A = adjacency(G, 1);",
        "G = graph([1], [2]); p = shortestpath(G, 1);",
    ] {
        execute_source(source).expect_err("fixed-arity graph binding");
    }
}
