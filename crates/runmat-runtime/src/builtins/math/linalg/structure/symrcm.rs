//! MATLAB-compatible `symrcm` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};

use log::debug;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "symrcm",
        builtin_path = "crate::builtins::math::linalg::structure::symrcm"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "symrcm"
category: "math/linalg/structure"
keywords: ["symrcm", "reverse cuthill-mckee", "bandwidth reduction", "symmetric ordering", "permutation", "gpu"]
summary: "Compute the symmetric reverse Cuthill-McKee permutation that reduces matrix bandwidth."
references:
  - "Alan George and Joseph W. Liu, 'Computer Solution of Large Sparse Positive Definite Systems', Prentice Hall, 1981."
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Acceleration providers may implement the `sym_rcm` hook. The WGPU backend currently downloads the matrix, runs the host algorithm, and returns the permutation."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::structure::symrcm::tests"
  integration: "builtins::math::linalg::structure::symrcm::tests::symrcm_gpu_roundtrip"
---

# What does the `symrcm` function do in MATLAB / RunMat?
`r = symrcm(A)` returns a permutation vector that reorders a symmetric (or structurally
symmetric) matrix to reduce its bandwidth. You can use the permutation as `A(r, r)` to
obtain a matrix with a tighter band structure, which improves factorisation and linear
solve performance for sparse matrices.

## How does the `symrcm` function behave in MATLAB / RunMat?
- Treats the input as an undirected graph based on the pattern of `A + A'`. Any value that
  is not exactly zero (including `NaN` or `Inf`) creates an edge; diagonal entries are
  ignored because they do not affect bandwidth.
- Accepts numeric, logical, and complex matrices. Logical and integer inputs are promoted
  to double precision internally, and complex entries are considered nonzero when either
  the real or imaginary component is nonzero.
- Requires a square matrix (scalars and the empty matrix are allowed). Inputs with more
  than two non-singleton dimensions raise an error, matching MATLAB matrix semantics.
- Processes disconnected components independently. Each component starts from a minimum
  degree vertex, runs a Cuthill-McKee breadth-first search, and the component ordering is
  reversed to produce the symmetric variant.
- Returns a row vector of 1-based indices. Fully diagonal or zero matrices produce the
  identity permutation.

## `symrcm` Function GPU Execution Behaviour
RunMat first asks the active acceleration provider whether it implements the `sym_rcm`
hook. The WGPU backend exposes this hook today and downloads the matrix once to reuse the
optimised CPU algorithm before returning the permutation. Providers without support (or
those that report an error) trigger an automatic gather and reuse the host implementation,
so results stay correct regardless of GPU capabilities.

## Examples of using the `symrcm` function in MATLAB / RunMat

### Reducing bandwidth of a near-band matrix

```matlab
A = [4 1 0 0 2;
     1 4 1 0 0;
     0 1 4 1 0;
     0 0 1 4 1;
     2 0 0 1 4];
r = symrcm(A);
B = A(r, r);
bandwidth(A)
bandwidth(B)
```

Expected result:

```matlab
r = [4 3 5 2 1];
bandwidth(A) = [4 4];
bandwidth(B) = [2 2];
```

### Handling disconnected components with `symrcm`

```matlab
A = [1 1 0 0 0;
     1 1 0 0 0;
     0 0 1 0 1;
     0 0 0 1 1;
     0 0 1 1 1];
r = symrcm(A);
B = A(r, r);
```

Expected result (one valid answer):

```matlab
r = [5 4 3 2 1];
bandwidth(B) = [2 2];
```

### Using `symrcm` with logical adjacency matrices

```matlab
adj = logical([0 1 0 0;
               1 0 1 0;
               0 1 0 1;
               0 0 1 0]);
r = symrcm(adj);
```

Expected result:

```matlab
r = [4 3 2 1];
```

### Applying `symrcm` to GPU matrices

```matlab
G = gpuArray([0 1 0 0 2;
              1 0 1 0 0;
              0 1 0 1 0;
              0 0 1 0 1;
              2 0 0 1 0]);
r = symrcm(G);
H = gather(G(r, r));
```

The permutation stays on the host today, so `symrcm` transparently gathers the input matrix
once. When a fully device-resident implementation lands, the same code will run entirely on
the GPU without changes.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. The runtime gathers GPU matrices automatically when computing the permutation today.
When native GPU implementations land, the same builtin will run entirely on the device
without code changes. You can still call `gpuArray` explicitly to mirror MATLAB workflows,
but it is optional in RunMat.

## FAQ

### What class of matrices benefit from `symrcm`?
Any symmetric (or structurally symmetric) sparse matrix whose bandwidth dominates solver
cost, such as discretised PDE operators, circuit matrices, or graph Laplacians.

### Does `symrcm` modify the matrix?
No. It returns a permutation vector. Apply it as `A(r, r)` or reorder vectors with `x(r)`.

### What happens if the matrix is not symmetric?
RunMat mirrors MATLAB and forms the pattern of `A + A'` internally. As long as the matrix
has symmetric nonzero structure, `symrcm` works. Strongly asymmetric inputs may lead to
less effective orderings but still produce a valid permutation.

### Are diagonal entries considered?
Diagonal entries are ignored when building the graph. Only off-diagonal nonzeros contribute
edges, which aligns with MATLAB semantics.

### How are NaNs or Infs handled?
Any value that is not numerically equal to zero is treated as nonzero, including `NaN` or
`Inf`. This matches MATLAB's structural interpretation.

### Can I use `symrcm` on dense matrices?
Yes. Dense inputs are supported. For dense matrices, the result is typically the identity
permutation because all vertices have high degree.

### How do I verify the permutation improves bandwidth?
Compute `bandwidth(A)` and `bandwidth(A(r, r))` and compare the results. Lower lower/upper
bandwidth values indicate a tighter band and more efficient factorizations.

### What output shape should I expect?
The permutation is returned as a row vector (`1 × n`). The empty matrix returns the empty
row vector `[]`.

## See Also
[rcm](https://www.mathworks.com/help/matlab/ref/rcm.html),
[symamd](https://www.mathworks.com/help/matlab/ref/symamd.html),
[bandwidth](./bandwidth),
[issymmetric](./issymmetric),
[gpuArray](../../../acceleration/gpu/gpuArray)

## Source & Feedback
- View the source: [`crates/runmat-runtime/src/builtins/math/linalg/structure/symrcm.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/symrcm.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

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

#[runtime_builtin(
    name = "symrcm",
    category = "math/linalg/structure",
    summary = "Compute the symmetric reverse Cuthill-McKee permutation that reduces matrix bandwidth.",
    keywords = "symrcm,reverse cuthill-mckee,bandwidth reduction,gpu",
    accel = "graph",
    builtin_path = "crate::builtins::math::linalg::structure::symrcm"
)]
fn symrcm_builtin(matrix: Value) -> Result<Value, String> {
    match matrix {
        Value::ComplexTensor(ct) => {
            let ordering = symrcm_host_complex_tensor(&ct)?;
            permutation_to_value(&ordering)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("symrcm: {e}"))?;
            let ordering = symrcm_host_complex_tensor(&tensor)?;
            permutation_to_value(&ordering)
        }
        Value::GpuTensor(handle) => symrcm_gpu(handle),
        other => {
            let tensor = tensor::value_into_tensor_for("symrcm", other)?;
            let ordering = symrcm_host_real_tensor(&tensor)?;
            permutation_to_value(&ordering)
        }
    }
}

fn symrcm_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.sym_rcm(&handle) {
            Ok(ordering) => return permutation_to_value(&ordering),
            Err(err) => {
                debug!("symrcm: provider hook unavailable, falling back to host: {err}");
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let ordering = symrcm_host_real_tensor(&tensor)?;
    permutation_to_value(&ordering)
}

/// Compute the symmetric reverse Cuthill-McKee ordering for a real tensor.
pub fn symrcm_host_real_tensor(tensor: &Tensor) -> Result<Vec<usize>, String> {
    symrcm_host_real_data(&tensor.shape, &tensor.data)
}

/// Compute the symmetric reverse Cuthill-McKee ordering for a complex tensor.
pub fn symrcm_host_complex_tensor(tensor: &ComplexTensor) -> Result<Vec<usize>, String> {
    symrcm_host_complex_data(&tensor.shape, &tensor.data)
}

/// Host implementation for dense real data.
pub fn symrcm_host_real_data(shape: &[usize], data: &[f64]) -> Result<Vec<usize>, String> {
    let adjacency = adjacency_from_real_data(shape, data)?;
    Ok(symmetric_reverse_cuthill_mckee(&adjacency))
}

/// Host implementation for dense complex data.
pub fn symrcm_host_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
) -> Result<Vec<usize>, String> {
    let adjacency = adjacency_from_complex_data(shape, data)?;
    Ok(symmetric_reverse_cuthill_mckee(&adjacency))
}

fn adjacency_from_real_data(shape: &[usize], data: &[f64]) -> Result<Vec<Vec<usize>>, String> {
    let n = ensure_square_matrix_shape(shape)?;
    build_adjacency(n, n, data, |value| *value != 0.0)
}

fn adjacency_from_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
) -> Result<Vec<Vec<usize>>, String> {
    let n = ensure_square_matrix_shape(shape)?;
    build_adjacency(n, n, data, |(re, im)| !(*re == 0.0 && *im == 0.0))
}

fn ensure_square_matrix_shape(shape: &[usize]) -> Result<usize, String> {
    let (rows, cols) = super::bandwidth::ensure_matrix_shape(shape)
        .map_err(|_| "symrcm: input must be a 2-D matrix".to_string())?;
    if rows != cols {
        return Err("symrcm: input matrix must be square".to_string());
    }
    Ok(rows)
}

fn build_adjacency<T, F>(
    rows: usize,
    cols: usize,
    data: &[T],
    mut is_nonzero: F,
) -> Result<Vec<Vec<usize>>, String>
where
    F: FnMut(&T) -> bool,
{
    if rows == 0 {
        return Ok(Vec::new());
    }

    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| "symrcm: matrix dimensions overflow when computing adjacency".to_string())?;
    if data.len() < expected {
        return Err("symrcm: data does not match matrix dimensions".to_string());
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

fn permutation_to_value(ordering: &[usize]) -> Result<Value, String> {
    let n = ordering.len();
    let mut data = Vec::with_capacity(n);
    for &idx in ordering {
        data.push((idx + 1) as f64);
    }
    let shape = if n == 0 { vec![1, 0] } else { vec![1, n] };
    let tensor = Tensor::new(data, shape).map_err(|e| format!("symrcm: {e}"))?;
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::LogicalArray;

    fn tensor_from_entries(rows: usize, cols: usize, entries: &[(usize, usize, f64)]) -> Tensor {
        let mut data = vec![0.0; rows * cols];
        for &(r, c, v) in entries {
            let idx = r + c * rows;
            data[idx] = v;
        }
        Tensor::new(data, vec![rows, cols]).unwrap()
    }

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

    #[test]
    fn symrcm_requires_square_matrix() {
        let tensor = tensor_from_entries(2, 3, &[(0, 1, 1.0)]);
        let err = symrcm_builtin(Value::Tensor(tensor)).expect_err("should fail");
        assert!(err.contains("square"), "unexpected error message: {err}");
    }

    #[test]
    fn symrcm_vector_is_not_square() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0], vec![3]).unwrap();
        let err = symrcm_builtin(Value::Tensor(tensor)).expect_err("should fail");
        assert!(
            err.contains("square"),
            "unexpected error message for non-square input: {err}"
        );
    }

    #[test]
    fn symrcm_rejects_higher_dimensional_input() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let err = symrcm_builtin(Value::Tensor(tensor)).expect_err("should fail");
        assert!(
            err.contains("2-D"),
            "unexpected error message for high-dimensional input: {err}"
        );
    }

    #[test]
    fn symrcm_rejects_unsupported_type() {
        let err = symrcm_builtin(Value::String("abc".to_string())).expect_err("should fail");
        assert!(
            err.contains("unsupported"),
            "unexpected error message for unsupported type: {err}"
        );
    }

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

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
