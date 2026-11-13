//! MATLAB-compatible `symrcm` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};

use log::debug;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::BuiltinErrorDescriptor;
#[cfg(test)]
use runmat_builtins::ResolveContext;
#[cfg(test)]
use runmat_builtins::SYMRCM_DESCRIPTOR;
use runmat_builtins::{SYMRCM_ERROR_INTERNAL, SYMRCM_ERROR_INVALID_INPUT};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, IntegerStorage, LogicalArray, SparseTensor, Tensor, Value};

use super::kernel_error::{StructureKernelError, StructureKernelResult};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(test)]
use crate::builtins::math::linalg::type_resolvers::symrcm_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::structure::symrcm"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "symrcm",
    op_kind: GpuOpKind::Custom("graph-order"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("symrcm")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers return the symmetric reverse Cuthill-McKee permutation; the WGPU implementation currently downloads the matrix and reuses the host algorithm.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::structure::symrcm"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "symrcm",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Structure-analysis builtin; fusion is not applicable.",
};

const BUILTIN_NAME: &str = "symrcm";

fn symrcm_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn symrcm_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    symrcm_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

#[runtime_builtin(
    name = "symrcm",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::structure::symrcm"
)]
async fn symrcm_builtin(matrix: Value) -> crate::BuiltinResult<Value> {
    match matrix {
        Value::SparseTensor(sparse) => {
            let ordering = symrcm_host_sparse(&sparse)?;
            permutation_to_value(&ordering)
        }
        Value::ComplexTensor(ct) => {
            let ordering = symrcm_host_complex_tensor(&ct)?;
            Ok(permutation_to_value(&ordering)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| symrcm_error_with_detail(&SYMRCM_ERROR_INTERNAL, e))?;
            let ordering = symrcm_host_complex_tensor(&tensor)?;
            Ok(permutation_to_value(&ordering)?)
        }
        Value::GpuTensor(handle) => symrcm_gpu(handle).await,
        other => {
            let tensor = value_into_tensor_for(other)?;
            let ordering = symrcm_host_real_tensor(&tensor)?;
            Ok(permutation_to_value(&ordering)?)
        }
    }
}

fn symrcm_host_sparse(sparse: &SparseTensor) -> StructureKernelResult<Vec<usize>> {
    if sparse.rows != sparse.cols {
        return Err(StructureKernelError::detail(
            BUILTIN_NAME,
            &SYMRCM_ERROR_INVALID_INPUT,
            "input matrix must be square",
        ));
    }
    let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); sparse.rows];
    super::sparse_pattern::for_each_nonzero(sparse, |row, col| {
        if row != col {
            adjacency[row].insert(col);
            adjacency[col].insert(row);
        }
    });
    // Building the symmetric graph is expected O(n + nnz). The shared RCM
    // traversal sorts each frontier by (degree, index), so HashSet iteration
    // order cannot affect the published permutation.
    let adjacency: Vec<Vec<usize>> = adjacency
        .into_iter()
        .map(|neighbours| neighbours.into_iter().collect())
        .collect();
    Ok(symmetric_reverse_cuthill_mckee(&adjacency))
}

fn value_into_tensor_for(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::LogicalArray(logical) => logical_to_tensor(&logical),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|e| symrcm_error_with_detail(&SYMRCM_ERROR_INTERNAL, e)),
        Value::Int(i) => Tensor::new_integer(IntegerStorage::from_scalar(i), vec![1, 1])
            .map_err(|e| symrcm_error_with_detail(&SYMRCM_ERROR_INTERNAL, e)),
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| symrcm_error_with_detail(&SYMRCM_ERROR_INTERNAL, e)),
        other => Err(symrcm_error_with_detail(
            &SYMRCM_ERROR_INVALID_INPUT,
            format!(
                "unsupported input type {:?}; expected numeric or logical values",
                other
            ),
        )),
    }
}

fn logical_to_tensor(logical: &LogicalArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = logical
        .data
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, logical.shape.clone())
        .map_err(|e| symrcm_error_with_detail(&SYMRCM_ERROR_INTERNAL, e))
}

async fn symrcm_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    gpu_helpers::require_resident_compute_owner(&handle, || {
        symrcm_error_with_detail(
            &SYMRCM_ERROR_INTERNAL,
            "no acceleration provider owns the input",
        )
    })?;
    if let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) {
        let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
        match provider.sym_rcm(&handle).await {
            Ok(ordering) => return permutation_to_value(&ordering),
            Err(err)
                if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(
                    &err,
                ) =>
            {
                debug!("symrcm: provider hook unavailable, falling back to host: {err}");
            }
            Err(err) => {
                return Err(super::kernel_error::with_provider_source(
                    symrcm_error_with_detail(&SYMRCM_ERROR_INTERNAL, err.to_string()),
                    runmat_accelerate_api::ProviderExecutionError::from(err),
                ))
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let ordering = symrcm_host_real_tensor(&tensor)?;
    permutation_to_value(&ordering)
}

/// Compute the symmetric reverse Cuthill-McKee ordering for a real tensor.
pub fn symrcm_host_real_tensor(tensor: &Tensor) -> StructureKernelResult<Vec<usize>> {
    if let Some(storage) = tensor.integer_storage() {
        return adjacency_from_integer_storage(tensor.shape(), storage)
            .map(|adjacency| symmetric_reverse_cuthill_mckee(&adjacency));
    }
    let values = tensor::tensor_values_f64_cow(tensor);
    symrcm_host_real_data(tensor.shape(), &values)
}

/// Compute the symmetric reverse Cuthill-McKee ordering for a complex tensor.
pub fn symrcm_host_complex_tensor(tensor: &ComplexTensor) -> StructureKernelResult<Vec<usize>> {
    symrcm_host_complex_data(tensor.shape(), &tensor.materialize_f64())
}

/// Host implementation for dense real data.
pub fn symrcm_host_real_data(shape: &[usize], data: &[f64]) -> StructureKernelResult<Vec<usize>> {
    let adjacency = adjacency_from_real_data(shape, data)?;
    Ok(symmetric_reverse_cuthill_mckee(&adjacency))
}

/// Host implementation for dense complex data.
pub fn symrcm_host_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
) -> StructureKernelResult<Vec<usize>> {
    let adjacency = adjacency_from_complex_data(shape, data)?;
    Ok(symmetric_reverse_cuthill_mckee(&adjacency))
}

fn adjacency_from_real_data(
    shape: &[usize],
    data: &[f64],
) -> StructureKernelResult<Vec<Vec<usize>>> {
    let n = ensure_square_matrix_shape(shape)?;
    build_adjacency(n, n, data, |value| *value != 0.0)
}

fn adjacency_from_integer_storage(
    shape: &[usize],
    storage: &IntegerStorage,
) -> StructureKernelResult<Vec<Vec<usize>>> {
    let n = ensure_square_matrix_shape(shape)?;
    build_adjacency_from_integer_storage(n, storage)
}

fn adjacency_from_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
) -> StructureKernelResult<Vec<Vec<usize>>> {
    let n = ensure_square_matrix_shape(shape)?;
    build_adjacency(n, n, data, |(re, im)| !(*re == 0.0 && *im == 0.0))
}

fn ensure_square_matrix_shape(shape: &[usize]) -> StructureKernelResult<usize> {
    let (rows, cols) = super::bandwidth::ensure_matrix_shape(shape)?;
    if rows != cols {
        return Err(StructureKernelError::detail(
            BUILTIN_NAME,
            &SYMRCM_ERROR_INVALID_INPUT,
            "input matrix must be square",
        ));
    }
    Ok(rows)
}

fn build_adjacency<T, F>(
    rows: usize,
    cols: usize,
    data: &[T],
    mut is_nonzero: F,
) -> StructureKernelResult<Vec<Vec<usize>>>
where
    F: FnMut(&T) -> bool,
{
    if rows == 0 {
        return Ok(Vec::new());
    }

    let expected = rows.checked_mul(cols).ok_or_else(|| {
        StructureKernelError::detail(
            BUILTIN_NAME,
            &SYMRCM_ERROR_INTERNAL,
            "matrix dimensions overflow when computing adjacency",
        )
    })?;
    if data.len() < expected {
        return Err(StructureKernelError::detail(
            BUILTIN_NAME,
            &SYMRCM_ERROR_INVALID_INPUT,
            "data does not match matrix dimensions",
        ));
    }

    let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); rows];
    let stride = rows;
    for col in 0..cols {
        for row in 0..rows {
            if row == col {
                continue;
            }
            let idx = row + col * stride;
            if idx >= data.len() {
                continue;
            }
            if is_nonzero(&data[idx]) {
                adjacency[row].insert(col);
                adjacency[col].insert(row);
            }
        }
    }

    Ok(adjacency
        .into_iter()
        .map(|set| {
            let mut neighbours: Vec<usize> = set.into_iter().collect();
            neighbours.sort_unstable();
            neighbours
        })
        .collect())
}

fn build_adjacency_from_integer_storage(
    rows: usize,
    storage: &IntegerStorage,
) -> StructureKernelResult<Vec<Vec<usize>>> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let expected = rows.checked_mul(rows).ok_or_else(|| {
        StructureKernelError::detail(
            BUILTIN_NAME,
            &SYMRCM_ERROR_INTERNAL,
            "matrix dimensions overflow when computing adjacency",
        )
    })?;
    if storage.len() < expected {
        return Err(StructureKernelError::detail(
            BUILTIN_NAME,
            &SYMRCM_ERROR_INVALID_INPUT,
            "data does not match matrix dimensions",
        ));
    }

    let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); rows];
    for col in 0..rows {
        for row in 0..rows {
            if row == col {
                continue;
            }
            let idx = row + col * rows;
            if storage.value_at(idx).is_some_and(|value| !value.is_zero()) {
                adjacency[row].insert(col);
                adjacency[col].insert(row);
            }
        }
    }
    Ok(adjacency
        .into_iter()
        .map(|set| {
            let mut neighbours: Vec<usize> = set.into_iter().collect();
            neighbours.sort_unstable();
            neighbours
        })
        .collect())
}

fn symmetric_reverse_cuthill_mckee(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    if n == 0 {
        return Vec::new();
    }
    let degrees: Vec<usize> = adjacency.iter().map(|nbrs| nbrs.len()).collect();
    let mut visited = vec![false; n];
    let mut seen_in_levels = vec![false; n];
    let mut level_queue = Vec::new();
    let mut starts: Vec<usize> = (0..n).collect();
    starts.sort_unstable_by_key(|&index| (degrees[index], index));
    let mut next_start = 0;
    let mut ordering = Vec::with_capacity(n);

    while ordering.len() < n {
        while visited[starts[next_start]] {
            next_start += 1;
        }
        let start = pseudoperipheral_start(
            starts[next_start],
            adjacency,
            &degrees,
            &visited,
            &mut seen_in_levels,
            &mut level_queue,
        );

        let mut component = cuthill_mckee_component(start, adjacency, &degrees, &mut visited);
        component.reverse();
        ordering.extend(component);
    }

    ordering
}

/// George-Liu style pseudoperipheral search within the next unvisited
/// component. A deeper level structure replaces the candidate; otherwise the
/// current vertex is pseudoperipheral under this deterministic degree/index
/// tie policy. The queue and marks are reused across every component/probe.
fn pseudoperipheral_start(
    initial: usize,
    adjacency: &[Vec<usize>],
    degrees: &[usize],
    visited: &[bool],
    seen: &mut [bool],
    queue: &mut Vec<usize>,
) -> usize {
    let mut candidate = initial;
    let (mut levels, mut farthest) =
        farthest_level_candidate(candidate, adjacency, degrees, visited, seen, queue);
    loop {
        let (next_levels, next_farthest) =
            farthest_level_candidate(farthest, adjacency, degrees, visited, seen, queue);
        if next_levels <= levels {
            return candidate;
        }
        candidate = farthest;
        levels = next_levels;
        farthest = next_farthest;
    }
}

fn farthest_level_candidate(
    start: usize,
    adjacency: &[Vec<usize>],
    degrees: &[usize],
    visited: &[bool],
    seen: &mut [bool],
    queue: &mut Vec<usize>,
) -> (usize, usize) {
    queue.clear();
    queue.push(start);
    seen[start] = true;
    let (mut head, mut levels, mut farthest) = (0, 0, start);
    while head < queue.len() {
        let level_end = queue.len();
        farthest = queue[head..level_end]
            .iter()
            .copied()
            .min_by_key(|&vertex| (degrees[vertex], vertex))
            .expect("nonempty level");
        for position in head..level_end {
            let vertex = queue[position];
            for &neighbour in &adjacency[vertex] {
                if !visited[neighbour] && !seen[neighbour] {
                    seen[neighbour] = true;
                    queue.push(neighbour);
                }
            }
        }
        head = level_end;
        levels += 1;
    }
    for &vertex in queue.iter() {
        seen[vertex] = false;
    }
    (levels, farthest)
}

fn cuthill_mckee_component(
    start: usize,
    adjacency: &[Vec<usize>],
    degrees: &[usize],
    visited: &mut [bool],
) -> Vec<usize> {
    let mut component = Vec::new();
    let mut queue = VecDeque::new();
    visited[start] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        component.push(node);
        let mut neighbours: Vec<usize> = adjacency[node]
            .iter()
            .copied()
            .filter(|nbr| !visited[*nbr])
            .collect();
        neighbours.sort_by(|a, b| match degrees[*a].cmp(&degrees[*b]) {
            Ordering::Equal => a.cmp(b),
            other => other,
        });
        for neighbour in neighbours {
            if !visited[neighbour] {
                visited[neighbour] = true;
                queue.push_back(neighbour);
            }
        }
    }

    component
}

fn permutation_to_value(ordering: &[usize]) -> BuiltinResult<Value> {
    let n = ordering.len();
    let mut data = Vec::with_capacity(n);
    for &idx in ordering {
        data.push((idx + 1) as f64);
    }
    let shape = if n == 0 { vec![1, 0] } else { vec![1, n] };
    let tensor = Tensor::new(data, shape)
        .map_err(|e| symrcm_error_with_detail(&SYMRCM_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::Type;
    use runmat_value::{IntegerStorage, LogicalArray, SparseTensor};

    #[test]
    fn symrcm_sparse_csc_matches_typed_path_graph_and_ignores_stored_zeros() {
        // One-way CSC path: the graph must be symmetrized without densifying.
        let pointers = vec![0, 1, 2, 3, 4, 5];
        let rows = vec![1, 2, 3, 4, 0];
        let double = SparseTensor::new(
            5,
            5,
            pointers.clone(),
            rows.clone(),
            vec![1.0, 1.0, 1.0, 1.0, 0.0],
        )
        .unwrap();
        let expected = symrcm_host_sparse(&double).unwrap();
        assert_eq!(expected, vec![4, 3, 2, 1, 0]);
        let single = SparseTensor::new_f32(
            5,
            5,
            pointers.clone(),
            rows.clone(),
            vec![1.0, 1.0, 1.0, 1.0, -0.0],
        )
        .unwrap();
        assert_eq!(symrcm_host_sparse(&single).unwrap(), expected);
        let wide = SparseTensor::new_integer(
            5,
            5,
            pointers.clone(),
            rows.clone(),
            IntegerStorage::I64(vec![i64::MIN, 1, 1, 1, 0]),
        )
        .unwrap();
        assert_eq!(symrcm_host_sparse(&wide).unwrap(), expected);
        let complex = SparseTensor::new_complex_f32(
            5,
            5,
            pointers.clone(),
            rows.clone(),
            vec![(0.0, 1.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
        )
        .unwrap();
        assert_eq!(symrcm_host_sparse(&complex).unwrap(), expected);
        let logical =
            SparseTensor::new_logical(5, 5, vec![0, 1, 2, 3, 4, 4], vec![1, 2, 3, 4]).unwrap();
        assert_eq!(symrcm_host_sparse(&logical).unwrap(), expected);
        assert!(symrcm_host_sparse(&SparseTensor::zeros(2, 3)).is_err());
        assert!(symrcm_host_sparse(&SparseTensor::zeros(0, 0))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn symrcm_sparse_public_output_is_one_based_row_permutation() {
        let sparse = SparseTensor::new(3, 3, vec![0, 1, 2, 2], vec![1, 2], vec![1.0, 1.0]).unwrap();
        match symrcm_builtin(Value::SparseTensor(sparse)).unwrap() {
            Value::Tensor(permutation) => {
                assert_eq!(permutation.shape(), [1, 3]);
                assert_eq!(permutation.materialize_f64(), vec![3.0, 2.0, 1.0]);
            }
            other => panic!("expected permutation row, got {other:?}"),
        }
    }

    #[test]
    fn symrcm_shared_pseudoperipheral_start_improves_center_leaf_seed() {
        // Vertex zero is a leaf close to the center, not a peripheral leaf.
        // The initial minimum-degree seed has four BFS levels; the endpoint
        // reached through vertex three has five.
        let col_ptrs = vec![0, 1, 3, 4, 5, 5, 5];
        let row_indices = vec![1, 2, 3, 5, 4];
        let sparse =
            SparseTensor::new(6, 6, col_ptrs.clone(), row_indices.clone(), vec![1.0; 5]).unwrap();
        let sparse_ordering = symrcm_host_sparse(&sparse).unwrap();
        assert_eq!(sparse_ordering, vec![5, 2, 0, 1, 3, 4]);
        let mut dense = vec![0.0; 36];
        for col in 0..6 {
            for &row in &row_indices[col_ptrs[col]..col_ptrs[col + 1]] {
                dense[row + 6 * col] = 1.0;
            }
        }
        assert_eq!(
            symrcm_host_real_data(&[6, 6], &dense).unwrap(),
            sparse_ordering
        );
    }

    fn tensor_from_entries(rows: usize, cols: usize, entries: &[(usize, usize, f64)]) -> Tensor {
        let mut data = vec![0.0; rows * cols];
        for &(r, c, v) in entries {
            let idx = r + c * rows;
            data[idx] = v;
        }
        Tensor::new(data, vec![rows, cols]).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_identity_matrix() {
        let tensor = tensor_from_entries(3, 3, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
        let result = symrcm_builtin(Value::Tensor(tensor)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 3]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn symrcm_type_returns_row_vector() {
        let out = symrcm_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn symrcm_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SYMRCM_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"p = symrcm(A)"));
    }

    #[test]
    fn symrcm_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = SYMRCM_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.SYMRCM.INVALID_INPUT"));
        assert!(codes.contains(&"RM.SYMRCM.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_scalar_input() {
        let result = symrcm_builtin(Value::Num(42.0)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 1]);
                assert_eq!(t.materialize_f64(), vec![1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_path_graph() {
        let entries = vec![
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 3, 1.0),
            (3, 2, 1.0),
            (3, 4, 1.0),
            (4, 3, 1.0),
        ];
        let tensor = tensor_from_entries(5, 5, &entries);
        let result = symrcm_builtin(Value::Tensor(tensor)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 5]);
                assert_eq!(t.materialize_f64(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(
            IntegerStorage::I16(vec![
                0, 1, 0, 0, 0, //
                1, 0, 1, 0, 0, //
                0, 1, 0, 1, 0, //
                0, 0, 1, 0, 1, //
                0, 0, 0, 1, 0,
            ]),
            vec![5, 5],
        )
        .expect("integer path graph");

        let result = symrcm_builtin(Value::Tensor(tensor)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 5]);
                assert_eq!(t.materialize_f64(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn symrcm_reads_wide_integer_storage_without_the_float_mirror() {
        let tensor = Tensor::new_integer(
            IntegerStorage::I64(vec![
                0,
                i64::MIN,
                0, //
                i64::MAX,
                0,
                -1, //
                0,
                1,
                0,
            ]),
            vec![3, 3],
        )
        .expect("integer path graph");

        let result = symrcm_builtin(Value::Tensor(tensor)).expect("symrcm");
        match result {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![3.0, 2.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_logical_matrix() {
        let data = vec![
            0, 1, 0, 0, //
            1, 0, 1, 0, //
            0, 1, 0, 1, //
            0, 0, 1, 0,
        ];
        let logical = LogicalArray::new(data, vec![4, 4]).unwrap();
        let result = symrcm_builtin(Value::LogicalArray(logical)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 4]);
                assert_eq!(t.materialize_f64(), vec![4.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_disconnected_components() {
        let entries = vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)];
        let tensor = tensor_from_entries(4, 4, &entries);
        let result = symrcm_builtin(Value::Tensor(tensor)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 4]);
                assert_eq!(t.materialize_f64(), vec![2.0, 1.0, 4.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_unsymmetric_treated_structurally() {
        let entries = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)];
        let tensor = tensor_from_entries(5, 5, &entries);
        let result = symrcm_builtin(Value::Tensor(tensor)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 5]);
                assert_eq!(t.materialize_f64(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_complex_matrix() {
        let data = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (0.0, 0.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = symrcm_builtin(Value::ComplexTensor(tensor)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 2]);
                assert!(
                    t.materialize_f64() == vec![2.0, 1.0] || t.materialize_f64() == vec![1.0, 2.0]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = tensor_from_entries(4, 4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = symrcm_builtin(Value::GpuTensor(handle)).expect("symrcm");
            match result {
                Value::Tensor(t) => {
                    assert_eq!(t.shape(), [1, 4]);
                    assert_eq!(t.materialize_f64(), vec![4.0, 3.0, 2.0, 1.0]);
                }
                other => panic!("expected tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_empty_matrix() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = symrcm_builtin(Value::Tensor(tensor)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_requires_square_matrix() {
        let tensor = tensor_from_entries(2, 3, &[(0, 1, 1.0)]);
        let err = symrcm_builtin(Value::Tensor(tensor)).expect_err("should fail");
        let message = err.to_string();
        assert!(
            message.contains("square"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_vector_is_not_square() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0], vec![3]).unwrap();
        let err = symrcm_builtin(Value::Tensor(tensor)).expect_err("should fail");
        let message = err.to_string();
        assert!(
            message.contains("square"),
            "unexpected error message for non-square input: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_rejects_higher_dimensional_input() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let err = symrcm_builtin(Value::Tensor(tensor)).expect_err("should fail");
        let message = err.to_string();
        assert!(
            message.contains("2-D"),
            "unexpected error message for high-dimensional input: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_rejects_unsupported_type() {
        let err = symrcm_builtin(Value::String("abc".to_string().into())).expect_err("should fail");
        assert_eq!(err.identifier(), SYMRCM_ERROR_INVALID_INPUT.identifier);
        let message = err.to_string();
        assert!(
            message.contains("unsupported"),
            "unexpected error message for unsupported type: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn symrcm_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let tensor = tensor_from_entries(
            5,
            5,
            &[
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (2, 3, 1.0),
                (3, 2, 1.0),
                (3, 4, 1.0),
                (4, 3, 1.0),
            ],
        );
        let expected = symrcm_host_real_tensor(&tensor).expect("host symrcm");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = symrcm_builtin(Value::GpuTensor(handle)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                let expected_f: Vec<f64> =
                    expected.into_iter().map(|idx| (idx + 1) as f64).collect();
                assert_eq!(t.shape(), [1, 5]);
                assert_eq!(t.materialize_f64(), expected_f);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    fn symrcm_builtin(matrix: Value) -> BuiltinResult<Value> {
        block_on(super::symrcm_builtin(matrix))
    }
}
