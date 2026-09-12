//! MATLAB-compatible graph and digraph data-type helpers.

mod domain;

pub use domain::{
    DIGRAPH_DESCRIPTOR, GRAPH_DESCRIPTOR, GRAPH_QUERY_DESCRIPTOR, GRAPH_UNARY_DESCRIPTOR,
};

#[cfg(target_arch = "wasm32")]
pub(crate) use domain::{
    __runmat_wasm_register_builtin_adjacency_builtin,
    __runmat_wasm_register_builtin_bfsearch_builtin,
    __runmat_wasm_register_builtin_conncomp_builtin, __runmat_wasm_register_builtin_degree_builtin,
    __runmat_wasm_register_builtin_dfsearch_builtin,
    __runmat_wasm_register_builtin_digraph_builtin,
    __runmat_wasm_register_builtin_distances_builtin,
    __runmat_wasm_register_builtin_findedge_builtin, __runmat_wasm_register_builtin_graph_builtin,
    __runmat_wasm_register_builtin_indegree_builtin,
    __runmat_wasm_register_builtin_neighbors_builtin,
    __runmat_wasm_register_builtin_numedges_builtin,
    __runmat_wasm_register_builtin_numnodes_builtin,
    __runmat_wasm_register_builtin_outdegree_builtin,
    __runmat_wasm_register_builtin_predecessors_builtin,
    __runmat_wasm_register_builtin_shortestpath_builtin,
    __runmat_wasm_register_builtin_successors_builtin,
    __runmat_wasm_register_builtin_toposort_builtin,
    __runmat_wasm_register_builtin_treelayout_builtin,
};
