//! MATLAB-compatible `pagefun` builtin.
//!
//! The `pagefun` builtin applies a MATLAB operator to every 2-D page across the
//! trailing dimensions of the supplied inputs. This mirrors MathWorks MATLAB
//! semantics for GPU arrays while retaining host fallbacks when GPU providers
//! do not expose specialised kernels.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::gather_if_needed;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, PagefunOp, PagefunRequest};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

type ComplexMatrixData = (Vec<(f64, f64)>, usize, usize);

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "pagefun",
        builtin_path = "crate::builtins::acceleration::gpu::pagefun"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "pagefun"
category: "acceleration/gpu"
keywords: ["pagefun", "gpuArray", "batched matrix multiply", "mtimes", "pages"]
summary: "Apply MATLAB operators page-by-page across higher-dimensional arrays."
references:
  - https://www.mathworks.com/help/parallel-computing/pagefun.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "WGPU provider accelerates batched @mtimes; if no provider hook is present the runtime evaluates on the host and re-uploads numeric results."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::acceleration::gpu::pagefun::tests"
  integration: "builtins::acceleration::gpu::pagefun::tests::pagefun_gpu_roundtrip_mtimes"
---

# What does the `pagefun` function do in MATLAB / RunMat?
`pagefun(func, A, B, …)` applies the MATLAB operator referenced by `func` to
every page of the supplied arrays. Pages are the two leading dimensions of the
inputs; the third and higher dimensions index the individual pages. The inputs
must agree in all trailing dimensions, allowing singleton expansion.

## How does the `pagefun` function behave in MATLAB / RunMat?
- Accepts function handles, builtin names (character vectors or string
  scalars), and validates that the requested operator is supported.
- The first two dimensions of each input form the matrix operated on by the
  builtin. Dimensions three and up identify individual pages. Inputs with fewer
  pages are broadcast so long as the page extent is `1`.
- Any combination of empty matrices and empty page extents produces empty
  results with MATLAB-compatible shapes.
- When all operands are `gpuArray` values the builtin gathers data to the host
  if the registered acceleration provider cannot satisfy the request. Uniform
  numeric results are uploaded back to the GPU so later operations retain
  residency.

## `pagefun` GPU Execution Behaviour
RunMat Accelerate exposes a custom `pagefun` provider hook. When the active
provider implements it (the WGPU backend does for `@mtimes`), pages stay on the
device and execute via a tiled compute shader. If the provider does not support
the requested operator the runtime gathers the data to the host, evaluates using
the CPU builtin, and re-uploads numeric outputs so subsequent GPU work can stay
resident.

## Examples of using the `pagefun` function in MATLAB / RunMat

### Batched matrix multiplication across pages
```matlab
A = reshape(1:12, [2 2 3]);
B = reshape(13:24, [2 2 3]);
C = pagefun(@mtimes, A, B)
```
Expected output:
```matlab
C(:,:,1) =
    55    63
    82    94

C(:,:,2) =
   211   235
   246   274

C(:,:,3) =
   431   471
   474   518
```

### Broadcasting a single page against multiple pages
```matlab
A = reshape(1:8, [2 2 2]);
B = eye(2);
C = pagefun(@mtimes, A, B);   % B broadcasts across the page dimension
```

### Working with `gpuArray` inputs
```matlab
G1 = gpuArray(rand(4, 4, 8));
G2 = gpuArray(rand(4, 4, 8));
H = pagefun(@mtimes, G1, G2);
firstPage = gather(H(:, :, 1));   % Inspect the first product page on the host
```

### Handling empty page dimensions
```matlab
Z = zeros(2, 2, 0);
R = pagefun(@mtimes, Z, Z);
size(R)
```
Expected output:
```matlab
ans =
     2     2     0
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. RunMat's auto-offload planner migrates tensors to the GPU when beneficial.
When you explicitly pass `gpuArray` inputs, `pagefun` keeps numeric results on
the device whenever all operands were device-resident and the provider can
accept uploads. Complex outputs remain host-resident until device buffers for
complex doubles ship.

## FAQ

### Which functions does `pagefun` support today?
RunMat currently supports `@mtimes` page-wise. Additional MATLAB page-aware
functions will be added over time as the GPU provider hooks land.

### How are page dimensions inferred?
The first two dimensions represent the matrix operated on by the builtin. Any
remaining dimensions are treated as page indices. Inputs with fewer trailing
dimensions receive implicit singleton expansion to match other operands.

### What happens if the pages are incompatible?
`pagefun` raises a MATLAB-compatible error describing the mismatched page
dimension. Matrix dimension mismatches are forwarded from the underlying
builtin (for example, `Inner matrix dimensions must agree` for `@mtimes`).

### Are results always uploaded back to the GPU?
Numeric results are uploaded when every operand started as a `gpuArray` and an
acceleration provider is registered. Complex results remain on the host today,
matching MATLAB's behaviour when complex GPU buffers are not available.

### Does `pagefun` participate in fusion?
No. Because `pagefun` can invoke arbitrary MATLAB builtins it forms a fusion
barrier. Upstream expressions are evaluated before entering `pagefun`.

## See Also
[gpuArray](./gpuarray), [arrayfun](./arrayfun), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::pagefun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pagefun",
    op_kind: GpuOpKind::Custom("pagefun"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("pagefun")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "WGPU provider accelerates batched @mtimes; runtimes gather to host when no provider hook is available.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::acceleration::gpu::pagefun")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pagefun",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion barrier because pagefun can invoke arbitrary MATLAB operators.",
};

#[runtime_builtin(
    name = "pagefun",
    category = "acceleration/gpu",
    summary = "Apply MATLAB operators page-by-page across higher-dimensional arrays.",
    keywords = "pagefun,gpuArray,mtimes,pages,batch",
    accel = "custom",
    builtin_path = "crate::builtins::acceleration::gpu::pagefun"
)]
fn pagefun_builtin(func: Value, first: Value, rest: Vec<Value>) -> Result<Value, String> {
    let operation = PageOperation::from_callable(func)?;
    let mut operands = Vec::with_capacity(rest.len() + 1);
    operands.push(first);
    operands.extend(rest);
    if operands.is_empty() {
        return Err("pagefun: requires at least one array input".to_string());
    }

    operation.validate_arity(operands.len())?;

    if let Some(value) = try_pagefun_gpu(&operation, &operands)? {
        return Ok(value);
    }

    let all_gpu = operands.iter().all(|v| matches!(v, Value::GpuTensor(_)));
    let mut host_values = Vec::with_capacity(operands.len());
    for value in operands {
        host_values.push(
            gather_if_needed(&value)
                .map_err(|e: runmat_async::RuntimeControlFlow| String::from(e))?,
        );
    }

    let mut page_inputs = Vec::with_capacity(host_values.len());
    for value in host_values {
        page_inputs.push(PageInput::from_value(value)?);
    }

    let rank = page_inputs
        .iter()
        .map(|input| input.page_dims.len())
        .max()
        .unwrap_or(0);

    let mut result_page_dims = if rank == 0 {
        Vec::new()
    } else {
        vec![1usize; rank]
    };

    for dim in 0..rank {
        let mut target = 1usize;
        for input in &page_inputs {
            let size = input.page_dims.get(dim).copied().unwrap_or(1);
            if size == 0 {
                target = 0;
                break;
            }
            if size != 1 {
                if target == 1 {
                    target = size;
                } else if target != size {
                    return Err(format!(
                        "pagefun: page dimension {} mismatch ({} vs {})",
                        dim + 3,
                        target,
                        size
                    ));
                }
            }
        }
        if !result_page_dims.is_empty() {
            result_page_dims[dim] = target;
        }
    }

    let page_volume = if rank == 0 {
        1usize
    } else {
        result_page_dims.iter().copied().product()
    };

    let mut prepared_inputs = Vec::with_capacity(page_inputs.len());
    for input in page_inputs {
        prepared_inputs.push(PreparedInput::new(input, rank));
    }

    operation.validate_shapes(&prepared_inputs)?;
    let output_kind = operation.output_kind(&prepared_inputs);
    let (mut result_rows, mut result_cols) =
        operation.output_matrix_shape(&prepared_inputs, output_kind)?;

    if page_volume == 0 {
        return finalise_empty_output(
            &operation,
            &prepared_inputs,
            &result_page_dims,
            output_kind,
            all_gpu,
        );
    }

    let mut real_data: Option<Vec<f64>> = None;
    let mut complex_data: Option<Vec<(f64, f64)>> = None;
    let mut multi_index = vec![0usize; rank];

    let mut page_counter = 0usize;
    loop {
        let mut page_args = Vec::with_capacity(prepared_inputs.len());
        for input in &prepared_inputs {
            page_args.push(input.page_value(&multi_index)?);
        }

        let mut evaluated = operation.evaluate(&page_args)?;
        evaluated =
            gather_if_needed(&evaluated)
                .map_err(|e: runmat_async::RuntimeControlFlow| String::from(e))?;
        match output_kind {
            OutputKind::Real => {
                let (data, rows, cols) = tensor_matrix_data(evaluated)?;
                if real_data.is_none() {
                    result_rows = rows;
                    result_cols = cols;
                    real_data = Some(Vec::with_capacity(rows * cols * page_volume));
                } else if rows != result_rows || cols != result_cols {
                    return Err("pagefun: result matrices must be the same size".to_string());
                }
                if let Some(vec) = real_data.as_mut() {
                    vec.extend(data);
                }
            }
            OutputKind::Complex => {
                let (data, rows, cols) = complex_matrix_data(evaluated)?;
                if complex_data.is_none() {
                    result_rows = rows;
                    result_cols = cols;
                    complex_data = Some(Vec::with_capacity(rows * cols * page_volume));
                } else if rows != result_rows || cols != result_cols {
                    return Err("pagefun: result matrices must be the same size".to_string());
                }
                if let Some(vec) = complex_data.as_mut() {
                    vec.extend(data);
                }
            }
        }

        page_counter += 1;
        if page_counter == page_volume {
            break;
        }
        increment_multi_index(&mut multi_index, &result_page_dims)?;
    }

    let final_shape = assemble_shape(result_rows, result_cols, &result_page_dims);
    let output = match output_kind {
        OutputKind::Real => {
            let data = real_data.unwrap_or_default();
            let tensor = Tensor::new(data, final_shape)
                .map_err(|e| format!("pagefun: failed to construct result tensor ({e})"))?;
            FinalOutput::Real(tensor)
        }
        OutputKind::Complex => {
            let data = complex_data.unwrap_or_default();
            let tensor = ComplexTensor::new(data, final_shape)
                .map_err(|e| format!("pagefun: failed to construct complex result tensor ({e})"))?;
            FinalOutput::Complex(tensor)
        }
    };

    output.into_value(all_gpu)
}

fn try_pagefun_gpu(operation: &PageOperation, operands: &[Value]) -> Result<Option<Value>, String> {
    if operands.is_empty() {
        return Ok(None);
    }
    if !operands
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }

    #[cfg(all(test, feature = "wgpu"))]
    {
        // Reassert WGPU provider only when operands are WGPU handles (device_id != 0).
        if operands
            .iter()
            .any(|v| matches!(v, Value::GpuTensor(h) if h.device_id != 0))
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };

    let handles: Vec<GpuTensorHandle> = operands
        .iter()
        .map(|value| match value {
            Value::GpuTensor(handle) => handle.clone(),
            _ => unreachable!(),
        })
        .collect();

    let request = match build_pagefun_request(operation, &handles)? {
        Some(request) => request,
        None => return Ok(None),
    };

    match provider.pagefun(&request) {
        Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
        Err(err) => {
            log::debug!("pagefun: provider hook unavailable, falling back to host: {err}");
            Ok(None)
        }
    }
}

fn build_pagefun_request(
    operation: &PageOperation,
    handles: &[GpuTensorHandle],
) -> Result<Option<PagefunRequest>, String> {
    match operation {
        PageOperation::Mtimes => {
            if handles.len() != 2 {
                return Err("pagefun: @mtimes requires exactly two array inputs".to_string());
            }

            let (lhs_rows, lhs_cols, lhs_pages) = handle_matrix_meta(&handles[0])?;
            let (rhs_rows, rhs_cols, rhs_pages) = handle_matrix_meta(&handles[1])?;
            if lhs_cols != rhs_rows {
                return Err(format!(
                    "pagefun: inner matrix dimensions must agree ({}x{} * {}x{})",
                    lhs_rows, lhs_cols, rhs_rows, rhs_cols
                ));
            }

            let rank = lhs_pages.len().max(rhs_pages.len());
            let mut result_page_dims = if rank == 0 {
                Vec::new()
            } else {
                vec![1usize; rank]
            };

            for dim in 0..rank {
                let mut target = 1usize;
                let dims_to_check = [
                    lhs_pages.get(dim).copied().unwrap_or(1),
                    rhs_pages.get(dim).copied().unwrap_or(1),
                ];
                for size in dims_to_check {
                    if size == 0 {
                        target = 0;
                        break;
                    }
                    if size != 1 {
                        if target == 1 {
                            target = size;
                        } else if target != size {
                            return Err(format!(
                                "pagefun: page dimension {} mismatch ({} vs {})",
                                dim + 3,
                                target,
                                size
                            ));
                        }
                    }
                }
                if !result_page_dims.is_empty() {
                    result_page_dims[dim] = target;
                }
            }

            let mut input_page_dims = Vec::with_capacity(2);
            let mut lhs_padded = lhs_pages.clone();
            lhs_padded.resize(rank, 1);
            let mut rhs_padded = rhs_pages.clone();
            rhs_padded.resize(rank, 1);
            input_page_dims.push(lhs_padded);
            input_page_dims.push(rhs_padded);

            let mut output_shape = vec![lhs_rows, rhs_cols];
            output_shape.extend_from_slice(&result_page_dims);

            Ok(Some(PagefunRequest {
                op: PagefunOp::Mtimes,
                inputs: handles.to_vec(),
                output_shape,
                page_dims: result_page_dims,
                input_page_dims,
            }))
        }
    }
}

fn handle_matrix_meta(handle: &GpuTensorHandle) -> Result<(usize, usize, Vec<usize>), String> {
    let canonical = canonical_matrix_shape(&handle.shape);
    if canonical.len() < 2 {
        return Err("pagefun: gpu tensor must be at least 2-D".to_string());
    }
    let rows = canonical[0];
    let cols = canonical[1];
    let pages = if canonical.len() > 2 {
        canonical[2..].to_vec()
    } else {
        Vec::new()
    };
    Ok((rows, cols, pages))
}

fn finalise_empty_output(
    operation: &PageOperation,
    inputs: &[PreparedInput],
    page_dims: &[usize],
    output_kind: OutputKind,
    wants_gpu: bool,
) -> Result<Value, String> {
    let (rows, cols) = operation.output_matrix_shape(inputs, output_kind)?;
    let final_shape = assemble_shape(rows, cols, page_dims);
    let page_factor: usize = if page_dims.is_empty() {
        1
    } else {
        page_dims.iter().copied().product()
    };
    let entries = rows
        .checked_mul(cols)
        .unwrap_or(0)
        .checked_mul(page_factor)
        .unwrap_or(0);
    match output_kind {
        OutputKind::Real => {
            let tensor = Tensor::new(vec![0.0; entries], final_shape)
                .map_err(|e| format!("pagefun: failed to build empty tensor ({e})"))?;
            FinalOutput::Real(tensor).into_value(wants_gpu)
        }
        OutputKind::Complex => {
            let tensor = ComplexTensor::new(vec![(0.0, 0.0); entries], final_shape)
                .map_err(|e| format!("pagefun: failed to build empty complex tensor ({e})"))?;
            FinalOutput::Complex(tensor).into_value(false)
        }
    }
}

fn assemble_shape(rows: usize, cols: usize, page_dims: &[usize]) -> Vec<usize> {
    let mut shape = vec![rows, cols];
    shape.extend_from_slice(page_dims);
    shape
}

fn increment_multi_index(indices: &mut [usize], dims: &[usize]) -> Result<(), String> {
    if dims.contains(&0) {
        return Ok(());
    }
    for (dim, &limit) in dims.iter().enumerate() {
        if limit == 0 {
            continue;
        }
        indices[dim] += 1;
        if indices[dim] < limit {
            return Ok(());
        }
        indices[dim] = 0;
        if dim + 1 == dims.len() {
            break;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputKind {
    Real,
    Complex,
}

enum FinalOutput {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl FinalOutput {
    fn into_value(self, wants_gpu: bool) -> Result<Value, String> {
        match self {
            FinalOutput::Real(tensor) => {
                if wants_gpu {
                    #[cfg(all(test, feature = "wgpu"))]
                    {
                        if runmat_accelerate_api::provider().is_none() {
                            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                            );
                        }
                    }
                    if let Some(provider) = runmat_accelerate_api::provider() {
                        let view = HostTensorView {
                            data: &tensor.data,
                            shape: &tensor.shape,
                        };
                        if let Ok(handle) = provider.upload(&view) {
                            return Ok(Value::GpuTensor(handle));
                        }
                    }
                }
                Ok(Value::Tensor(tensor))
            }
            FinalOutput::Complex(tensor) => Ok(Value::ComplexTensor(tensor)),
        }
    }
}

#[derive(Clone)]
struct PageInput {
    page_dims: Vec<usize>,
    rows: usize,
    cols: usize,
    data: PageData,
}

#[derive(Clone)]
enum PageData {
    Real(Vec<f64>),
    Complex(Vec<(f64, f64)>),
}

impl PageInput {
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::Tensor(t) => Self::from_tensor(t),
            Value::Num(n) => Self::from_tensor(
                Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("pagefun: {e}"))?,
            ),
            Value::Int(i) => Self::from_tensor(
                Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("pagefun: {e}"))?,
            ),
            Value::Bool(flag) => Self::from_tensor(
                Tensor::new(vec![if flag { 1.0 } else { 0.0 }], vec![1, 1])
                    .map_err(|e| format!("pagefun: {e}"))?,
            ),
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| format!("pagefun: {e}"))?;
                Self::from_complex_tensor(tensor)
            }
            Value::ComplexTensor(t) => Self::from_complex_tensor(t),
            other => Err(format!(
                "pagefun: unsupported input type {}",
                other.type_name()
            )),
        }
    }

    fn from_tensor(tensor: Tensor) -> Result<Self, String> {
        let shape = canonical_matrix_shape(&tensor.shape);
        if tensor.data.len() != shape.iter().copied().product::<usize>() {
            return Err("pagefun: tensor data does not match its shape".to_string());
        }
        let rows = shape[0];
        let cols = shape[1];
        let page_dims = if shape.len() > 2 {
            shape[2..].to_vec()
        } else {
            Vec::new()
        };
        Ok(Self {
            page_dims,
            rows,
            cols,
            data: PageData::Real(tensor.data),
        })
    }

    fn from_complex_tensor(tensor: ComplexTensor) -> Result<Self, String> {
        let shape = canonical_matrix_shape(&tensor.shape);
        if tensor.data.len() != shape.iter().copied().product::<usize>() {
            return Err("pagefun: tensor data does not match its shape".to_string());
        }
        let rows = shape[0];
        let cols = shape[1];
        let page_dims = if shape.len() > 2 {
            shape[2..].to_vec()
        } else {
            Vec::new()
        };
        Ok(Self {
            page_dims,
            rows,
            cols,
            data: PageData::Complex(tensor.data),
        })
    }

    fn page_size(&self) -> usize {
        self.rows * self.cols
    }

    fn is_complex(&self) -> bool {
        matches!(self.data, PageData::Complex(_))
    }
}

struct PreparedInput {
    data: PageInput,
    padded_dims: Vec<usize>,
    strides: Vec<usize>,
}

impl PreparedInput {
    fn new(input: PageInput, rank: usize) -> Self {
        let mut padded = input.page_dims.clone();
        padded.resize(rank, 1);
        let strides = compute_strides(&padded);
        Self {
            data: input,
            padded_dims: padded,
            strides,
        }
    }

    fn rows(&self) -> usize {
        self.data.rows
    }

    fn cols(&self) -> usize {
        self.data.cols
    }

    fn is_complex(&self) -> bool {
        self.data.is_complex()
    }

    fn page_value(&self, multi_index: &[usize]) -> Result<Value, String> {
        let mut linear_page = 0usize;
        for (dim, stride) in self.strides.iter().enumerate() {
            let source_extent = self.padded_dims.get(dim).copied().unwrap_or(1);
            let requested = multi_index.get(dim).copied().unwrap_or(0);
            if source_extent == 0 {
                return Err("pagefun: source page extent is zero".to_string());
            }
            if source_extent != 1 && requested >= source_extent {
                return Err("pagefun: page index out of bounds".to_string());
            }
            let actual = if source_extent == 1 { 0 } else { requested };
            linear_page += actual * stride;
        }

        let offset = linear_page * self.data.page_size();
        match &self.data.data {
            PageData::Real(buffer) => {
                let end = offset + self.data.page_size();
                let slice = buffer
                    .get(offset..end)
                    .ok_or_else(|| "pagefun: page slice out of bounds".to_string())?;
                let tensor = Tensor::new(slice.to_vec(), vec![self.data.rows, self.data.cols])
                    .map_err(|e| format!("pagefun: {e}"))?;
                Ok(Value::Tensor(tensor))
            }
            PageData::Complex(buffer) => {
                let end = offset + self.data.page_size();
                let slice = buffer
                    .get(offset..end)
                    .ok_or_else(|| "pagefun: page slice out of bounds".to_string())?;
                let tensor =
                    ComplexTensor::new(slice.to_vec(), vec![self.data.rows, self.data.cols])
                        .map_err(|e| format!("pagefun: {e}"))?;
                Ok(Value::ComplexTensor(tensor))
            }
        }
    }
}

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride = stride.saturating_mul(dim.max(1));
    }
    strides
}

fn tensor_matrix_data(value: Value) -> Result<(Vec<f64>, usize, usize), String> {
    match value {
        Value::Tensor(t) => {
            if t.shape.len() > 2 {
                return Err(
                    "pagefun: operator returned an array with more than two dimensions".to_string(),
                );
            }
            let canonical = canonical_matrix_shape(&t.shape);
            let rows = canonical[0];
            let cols = canonical[1];
            if rows * cols != t.data.len() {
                return Err("pagefun: result size mismatch".to_string());
            }
            Ok((t.data, rows, cols))
        }
        Value::Num(n) => Ok((vec![n], 1, 1)),
        Value::Int(i) => Ok((vec![i.to_f64()], 1, 1)),
        other => Err(format!(
            "pagefun: expected numeric matrix result, received {}",
            other.type_name()
        )),
    }
}

fn complex_matrix_data(value: Value) -> Result<ComplexMatrixData, String> {
    match value {
        Value::ComplexTensor(t) => {
            if t.shape.len() > 2 {
                return Err(
                    "pagefun: operator returned an array with more than two dimensions".to_string(),
                );
            }
            let canonical = canonical_matrix_shape(&t.shape);
            let rows = canonical[0];
            let cols = canonical[1];
            if rows * cols != t.data.len() {
                return Err("pagefun: result size mismatch".to_string());
            }
            Ok((t.data, rows, cols))
        }
        Value::Complex(re, im) => Ok((vec![(re, im)], 1, 1)),
        other => Err(format!(
            "pagefun: expected complex matrix result, received {}",
            other.type_name()
        )),
    }
}

fn canonical_matrix_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => {
            let mut out = shape.to_vec();
            if out.len() == 1 {
                out.push(1);
            }
            out
        }
    }
}

#[derive(Clone, Copy)]
enum PageOperation {
    Mtimes,
}

impl PageOperation {
    fn from_callable(value: Value) -> Result<Self, String> {
        let raw = match value {
            Value::FunctionHandle(func) => func,
            Value::String(s) => s,
            Value::StringArray(sa) => {
                if sa.data.len() != 1 {
                    return Err(
                        "pagefun: function string array must contain exactly one element"
                            .to_string(),
                    );
                }
                sa.data[0].clone()
            }
            Value::CharArray(chars) => {
                if chars.rows != 1 {
                    return Err(
                        "pagefun: function char array must be a single row character vector"
                            .to_string(),
                    );
                }
                chars.data.iter().collect()
            }
            other => {
                return Err(format!(
                    "pagefun: unsupported function handle type {}",
                    other.type_name()
                ))
            }
        };
        let trimmed = raw.trim();
        let lowered = trimmed.trim_start_matches('@').to_ascii_lowercase();
        match lowered.as_str() {
            "mtimes" => Ok(Self::Mtimes),
            _ => Err(format!(
                "pagefun: unsupported function '{}'; currently only @mtimes is implemented",
                trimmed
            )),
        }
    }

    fn validate_arity(&self, arg_count: usize) -> Result<(), String> {
        match self {
            Self::Mtimes => {
                if arg_count != 2 {
                    return Err("pagefun: @mtimes requires exactly two array inputs".to_string());
                }
                Ok(())
            }
        }
    }

    fn validate_shapes(&self, inputs: &[PreparedInput]) -> Result<(), String> {
        match self {
            Self::Mtimes => {
                let lhs = &inputs[0];
                let rhs = &inputs[1];
                if lhs.cols() != rhs.rows() {
                    return Err(format!(
                        "pagefun: inner matrix dimensions must agree ({}x{} * {}x{})",
                        lhs.rows(),
                        lhs.cols(),
                        rhs.rows(),
                        rhs.cols()
                    ));
                }
                Ok(())
            }
        }
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value, String> {
        match self {
            Self::Mtimes => crate::call_builtin("mtimes", args)
                .map_err(|e: runmat_async::RuntimeControlFlow| String::from(e)),
        }
    }

    fn output_kind(&self, inputs: &[PreparedInput]) -> OutputKind {
        match self {
            Self::Mtimes => {
                if inputs.iter().any(|input| input.is_complex()) {
                    OutputKind::Complex
                } else {
                    OutputKind::Real
                }
            }
        }
    }

    fn output_matrix_shape(
        &self,
        inputs: &[PreparedInput],
        kind: OutputKind,
    ) -> Result<(usize, usize), String> {
        match self {
            Self::Mtimes => {
                let lhs = &inputs[0];
                let rhs = &inputs[1];
                let rows = lhs.rows();
                let cols = rhs.cols();
                match kind {
                    OutputKind::Real | OutputKind::Complex => Ok((rows, cols)),
                }
            }
        }
    }
}

trait TypeName {
    fn type_name(&self) -> &'static str;
}

impl TypeName for Value {
    fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::Num(_) => "double",
            Value::Complex(_, _) => "complex double",
            Value::Bool(_) => "logical",
            Value::LogicalArray(_) => "logical array",
            Value::String(_) => "string",
            Value::StringArray(_) => "string array",
            Value::CharArray(_) => "char array",
            Value::Tensor(_) => "double array",
            Value::ComplexTensor(_) => "complex double array",
            Value::Cell(_) => "cell array",
            Value::Struct(_) => "struct",
            Value::GpuTensor(_) => "gpuArray",
            Value::Object(_) => "object",
            Value::HandleObject(_) => "handle object",
            Value::Listener(_) => "listener",
            Value::FunctionHandle(_) => "function handle",
            Value::Closure(_) => "closure",
            Value::ClassRef(_) => "class reference",
            Value::MException(_) => "MException",
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, StringArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_mtimes_single_page() {
        let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result = pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect("pagefun");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![19.0, 43.0, 22.0, 50.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_mtimes_multiple_pages() {
        let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0, 2.0, 1.0, 0.0, 3.0], vec![2, 2, 2]).unwrap();
        let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0, 1.0, 0.0, 2.0, 1.0], vec![2, 2, 2]).unwrap();
        let result = pagefun_builtin(
            Value::from("@mtimes"),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect("pagefun");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2, 2]);
                assert_eq!(t.data, vec![19.0, 43.0, 22.0, 50.0, 2.0, 1.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_mtimes_broadcast_rhs() {
        let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0], vec![2, 2, 2]).unwrap();
        let rhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let result = pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect("pagefun");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2, 2]);
                assert_eq!(
                    t.data,
                    vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
                    "broadcasted identity should preserve pages"
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_mtimes_empty_pages() {
        let lhs = Tensor::new(Vec::new(), vec![2, 2, 0]).unwrap();
        let rhs = Tensor::new(Vec::new(), vec![2, 2, 0]).unwrap();
        let result = pagefun_builtin(
            Value::from("@mtimes"),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect("pagefun");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_mtimes_char_array_handle() {
        let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let func = CharArray::new("@mtimes".chars().collect(), 1, 7).unwrap();
        let result = pagefun_builtin(
            Value::CharArray(func),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect("pagefun char array");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![19.0, 43.0, 22.0, 50.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_mtimes_string_array_handle() {
        let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let strings = StringArray::new(vec!["@mtimes".to_string()], vec![1]).unwrap();
        let result = pagefun_builtin(
            Value::StringArray(strings),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect("pagefun string array");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![19.0, 43.0, 22.0, 50.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_char_array_multirow_error() {
        let chars = CharArray::new("@mtimes@".chars().collect(), 2, 4).unwrap();
        let lhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = pagefun_builtin(
            Value::CharArray(chars),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect_err("expected multi-row char array error");
        assert!(
            err.contains("char array"),
            "unexpected error for multi-row char array: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_string_array_multi_value_error() {
        let strings =
            StringArray::new(vec!["@mtimes".to_string(), "@mtimes".to_string()], vec![2]).unwrap();
        let lhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = pagefun_builtin(
            Value::StringArray(strings),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect_err("expected multi-element string array error");
        assert!(
            err.contains("string array"),
            "unexpected error for string array: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_page_dimension_mismatch() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();
        let rhs = Tensor::new(
            vec![
                1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        )
        .unwrap();
        let err = pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect_err("expected page dimension mismatch");
        assert!(
            err.contains("page dimension"),
            "unexpected mismatch error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_mtimes_dim_mismatch() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::Tensor(lhs),
            vec![Value::Tensor(rhs)],
        )
        .expect_err("expected dimension mismatch");
        assert!(
            err.contains("inner matrix dimensions"),
            "unexpected error message {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pagefun_gpu_roundtrip_mtimes() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0], vec![2, 2, 2]).unwrap();
            let identity = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

            let view_lhs = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let view_rhs = HostTensorView {
                data: &identity.data,
                shape: &identity.shape,
            };
            let lhs = provider.upload(&view_lhs).expect("upload lhs");
            let rhs = provider.upload(&view_rhs).expect("upload rhs");

            let result = pagefun_builtin(
                Value::FunctionHandle("mtimes".into()),
                Value::GpuTensor(lhs),
                vec![Value::GpuTensor(rhs)],
            )
            .expect("pagefun");

            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2, 2]);
            assert_eq!(
                gathered.data,
                vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
                "GPU fallback should match identity broadcast"
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pagefun_wgpu_mtimes_batches() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ =
            register_wgpu_provider(WgpuProviderOptions::default()).expect("register wgpu provider");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let lhs = Tensor::new(
            vec![
                1.0, 4.0, 2.0, 5.0, //
                3.0, 6.0, 4.0, 7.0,
            ],
            vec![2, 2, 2],
        )
        .unwrap();
        let rhs = Tensor::new(
            vec![
                1.0, 0.0, 0.0, 1.0, //
                2.0, 1.0, 3.0, 2.0,
            ],
            vec![2, 2, 2],
        )
        .unwrap();

        let view_lhs = HostTensorView {
            data: &lhs.data,
            shape: &lhs.shape,
        };
        let view_rhs = HostTensorView {
            data: &rhs.data,
            shape: &rhs.shape,
        };

        let lhs_handle = provider.upload(&view_lhs).expect("upload lhs");
        let rhs_handle = provider.upload(&view_rhs).expect("upload rhs");

        let provider_handles = vec![lhs_handle.clone(), rhs_handle.clone()];
        let request = build_pagefun_request(&PageOperation::Mtimes, &provider_handles)
            .expect("build request")
            .expect("request available");

        let provider_result = provider.pagefun(&request).expect("wgpu pagefun execution");
        let provider_tensor =
            test_support::gather(Value::GpuTensor(provider_result)).expect("gather provider");

        let builtin_value = pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::GpuTensor(lhs_handle.clone()),
            vec![Value::GpuTensor(rhs_handle.clone())],
        )
        .expect("pagefun builtin on GPU");
        let builtin_tensor = test_support::gather(builtin_value).expect("gather builtin");

        let expected_value = pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::Tensor(lhs.clone()),
            vec![Value::Tensor(rhs.clone())],
        )
        .expect("pagefun host baseline");
        let expected_tensor = match expected_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        assert_eq!(provider_tensor.shape, expected_tensor.shape);
        assert_eq!(provider_tensor.data, expected_tensor.data);
        assert_eq!(builtin_tensor.shape, expected_tensor.shape);
        assert_eq!(builtin_tensor.data, expected_tensor.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
