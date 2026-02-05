//! MATLAB-compatible `symrcm` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};

use log::debug;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, LogicalArray, ResolveContext, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::linalg::type_resolvers::symrcm_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

// NOTE: The `symrcm_type` symbol is referenced from the `#[runtime_builtin]` macro
// attribute arguments, which does not currently count as a "use" for Rust's
// unused-import lint. Keep a small reference so `-D unused-imports` builds.
#[allow(dead_code)]
const _SYMR_CM_TYPE_RESOLVER: fn(&[runmat_builtins::Type], &ResolveContext) ->
    runmat_builtins::Type =
    symrcm_type;
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

fn runtime_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

#[runtime_builtin(
    name = "symrcm",
    category = "math/linalg/structure",
    summary = "Compute the symmetric reverse Cuthill-McKee permutation that reduces matrix bandwidth.",
    keywords = "symrcm,reverse cuthill-mckee,bandwidth reduction,gpu",
    accel = "graph",
    type_resolver(symrcm_type),
    builtin_path = "crate::builtins::math::linalg::structure::symrcm"
)]
async fn symrcm_builtin(matrix: Value) -> crate::BuiltinResult<Value> {
    match matrix {
        Value::ComplexTensor(ct) => {
            let ordering = symrcm_host_complex_tensor(&ct)?;
            Ok(permutation_to_value(&ordering)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| runtime_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {e}")))?;
            let ordering = symrcm_host_complex_tensor(&tensor)?;
            Ok(permutation_to_value(&ordering)?)
        }
        Value::GpuTensor(handle) => symrcm_gpu(handle).await,
        other => {
            let tensor = value_into_tensor_for(BUILTIN_NAME, other)?;
            let ordering = symrcm_host_real_tensor(&tensor)?;
            Ok(permutation_to_value(&ordering)?)
        }
    }
}

fn value_into_tensor_for(name: &str, value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::LogicalArray(logical) => logical_to_tensor(name, &logical),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|e| runtime_error(name, format!("{name}: {e}"))),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1])
            .map_err(|e| runtime_error(name, format!("{name}: {e}"))),
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| runtime_error(name, format!("{name}: {e}"))),
        other => Err(runtime_error(
            name,
            format!(
                "{name}: unsupported input type {:?}; expected numeric or logical values",
                other
            ),
        )),
    }
}

fn logical_to_tensor(name: &str, logical: &LogicalArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = logical
        .data
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, logical.shape.clone())
        .map_err(|e| runtime_error(name, format!("{name}: {e}")))
}

async fn symrcm_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.sym_rcm(&handle).await {
            Ok(ordering) => return permutation_to_value(&ordering),
            Err(err) => {
                debug!("symrcm: provider hook unavailable, falling back to host: {err}");
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let ordering = symrcm_host_real_tensor(&tensor)?;
    permutation_to_value(&ordering)
}

/// Compute the symmetric reverse Cuthill-McKee ordering for a real tensor.
pub fn symrcm_host_real_tensor(tensor: &Tensor) -> BuiltinResult<Vec<usize>> {
    symrcm_host_real_data(&tensor.shape, &tensor.data)
}

/// Compute the symmetric reverse Cuthill-McKee ordering for a complex tensor.
pub fn symrcm_host_complex_tensor(tensor: &ComplexTensor) -> BuiltinResult<Vec<usize>> {
    symrcm_host_complex_data(&tensor.shape, &tensor.data)
}

/// Host implementation for dense real data.
pub fn symrcm_host_real_data(shape: &[usize], data: &[f64]) -> BuiltinResult<Vec<usize>> {
    let adjacency = adjacency_from_real_data(shape, data)?;
    Ok(symmetric_reverse_cuthill_mckee(&adjacency))
}

/// Host implementation for dense complex data.
pub fn symrcm_host_complex_data(shape: &[usize], data: &[(f64, f64)]) -> BuiltinResult<Vec<usize>> {
    let adjacency = adjacency_from_complex_data(shape, data)?;
    Ok(symmetric_reverse_cuthill_mckee(&adjacency))
}

fn adjacency_from_real_data(shape: &[usize], data: &[f64]) -> BuiltinResult<Vec<Vec<usize>>> {
    let n = ensure_square_matrix_shape(shape)?;
    build_adjacency(n, n, data, |value| *value != 0.0)
}

fn adjacency_from_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
) -> BuiltinResult<Vec<Vec<usize>>> {
    let n = ensure_square_matrix_shape(shape)?;
    build_adjacency(n, n, data, |(re, im)| !(*re == 0.0 && *im == 0.0))
}

fn ensure_square_matrix_shape(shape: &[usize]) -> BuiltinResult<usize> {
    let (rows, cols) = super::bandwidth::ensure_matrix_shape(shape)?;
    if rows != cols {
        return Err(runtime_error(
            BUILTIN_NAME,
            "symrcm: input matrix must be square",
        ));
    }
    Ok(rows)
}

fn build_adjacency<T, F>(
    rows: usize,
    cols: usize,
    data: &[T],
    mut is_nonzero: F,
) -> BuiltinResult<Vec<Vec<usize>>>
where
    F: FnMut(&T) -> bool,
{
    if rows == 0 {
        return Ok(Vec::new());
    }

    let expected = rows.checked_mul(cols).ok_or_else(|| {
        runtime_error(
            BUILTIN_NAME,
            "symrcm: matrix dimensions overflow when computing adjacency",
        )
    })?;
    if data.len() < expected {
        return Err(runtime_error(
            BUILTIN_NAME,
            "symrcm: data does not match matrix dimensions",
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

fn symmetric_reverse_cuthill_mckee(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    if n == 0 {
        return Vec::new();
    }
    let degrees: Vec<usize> = adjacency.iter().map(|nbrs| nbrs.len()).collect();
    let mut visited = vec![false; n];
    let mut ordering = Vec::with_capacity(n);

    while ordering.len() < n {
        let start = (0..n)
            .filter(|&idx| !visited[idx])
            .min_by(|&a, &b| {
                let key_a = (degrees[a], a);
                let key_b = (degrees[b], b);
                key_a.cmp(&key_b)
            })
            .expect("symrcm: at least one unvisited vertex remains");

        let mut component = cuthill_mckee_component(start, adjacency, &degrees, &mut visited);
        component.reverse();
        ordering.extend(component);
    }

    ordering
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
        .map_err(|e| runtime_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {e}")))?;
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{LogicalArray, Type};

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
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symrcm_scalar_input() {
        let result = symrcm_builtin(Value::Num(42.0)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(t.data, vec![1.0]);
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
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
            }
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
                assert_eq!(t.shape, vec![1, 4]);
                assert_eq!(t.data, vec![4.0, 3.0, 2.0, 1.0]);
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
                assert_eq!(t.shape, vec![1, 4]);
                assert_eq!(t.data, vec![2.0, 1.0, 4.0, 3.0]);
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
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
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
                assert_eq!(t.shape, vec![1, 2]);
                assert!(t.data == vec![2.0, 1.0] || t.data == vec![1.0, 2.0]);
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
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = symrcm_builtin(Value::GpuTensor(handle)).expect("symrcm");
            match result {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![1, 4]);
                    assert_eq!(t.data, vec![4.0, 3.0, 2.0, 1.0]);
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
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.data.is_empty());
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
        let err = symrcm_builtin(Value::String("abc".to_string())).expect_err("should fail");
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
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = symrcm_builtin(Value::GpuTensor(handle)).expect("symrcm");
        match result {
            Value::Tensor(t) => {
                let expected_f: Vec<f64> =
                    expected.into_iter().map(|idx| (idx + 1) as f64).collect();
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, expected_f);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    fn symrcm_builtin(matrix: Value) -> BuiltinResult<Value> {
        block_on(super::symrcm_builtin(matrix))
    }
}
