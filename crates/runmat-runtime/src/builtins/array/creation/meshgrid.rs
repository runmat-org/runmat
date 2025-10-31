//! MATLAB-compatible `meshgrid` builtin with GPU-aware semantics.

use std::cmp::max;

use log::warn;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, MeshgridAxisView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "meshgrid"
category: "array/creation"
keywords: ["meshgrid", "grid", "surface", "gpu", "like", "3d"]
summary: "Generate MATLAB mesh grids for 2-D and 3-D coordinate vectors with optional GPU residency."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "RunMat materialises grids on the host and uploads them when GPU residency is requested. Providers may supply dedicated meshgrid hooks to avoid the round-trip."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::creation::meshgrid::tests"
  integration: "builtins::array::creation::meshgrid::tests::meshgrid_gpu_inputs_roundtrip"
---

# What does the `meshgrid` function do in MATLAB / RunMat?
`meshgrid` turns one-, two-, or three-dimensional vectors into coordinate arrays that
span a rectangular grid, mirroring MathWorks MATLAB behaviour exactly. The first output
replicates the `x` vector across rows, the second replicates the `y` vector across columns,
and the optional third output expands a `z` vector across pages for volumetric grids.

## How does the `meshgrid` function behave in MATLAB / RunMat?
- `meshgrid(x)` is shorthand for `[X, Y] = meshgrid(x, x)`. It produces square 2-D grids.
- `meshgrid(x, y)` yields `X` of size `length(y) × length(x)` with rows copied from `x`,
  and `Y` of the same size with columns copied from `y`.
- `meshgrid(x, y, z)` returns three outputs sized `length(y) × length(x) × length(z)`,
  enabling 3-D volume visualisation.
- Input vectors may be row or column vectors (or even scalars). Empty vectors propagate to
  empty grids of matching shape.
- Complex inputs produce complex grids where each output shares the input’s complex values.
- Supplying GPU vectors (or a `'like', gpuArray(...)` prototype) keeps the outputs on the GPU
  when an acceleration provider is active. Without provider support, RunMat gathers inputs,
  materialises the grid on the host, and uploads the result transparently.
- `'like', prototype` matches both the residency (host or GPU) and numeric class (real or complex)
  of the prototype. Integer prototypes are promoted to double precision, consistent with MATLAB.

## `meshgrid` Function GPU Execution Behaviour
- When the active acceleration provider implements the custom `meshgrid` hook, RunMat allocates
  every coordinate tensor directly on the device so large grids avoid host round-trips entirely.
- If the hook is missing (or errors), RunMat gathers the 1-D axes, materialises the grids once on
  the host, and uploads the outputs whenever GPU residency is requested, preserving observable
  semantics.
- Complex-valued grids always materialise on the host today; when GPU residency is requested the
  runtime logs a trace warning and returns host complex tensors so callers still receive correct
  MATLAB-compatible results.

## Examples of using the `meshgrid` function in MATLAB / RunMat

### Generating a square 2-D grid from one vector

```matlab
x = -2:2;
[X, Y] = meshgrid(x);
```

Expected output (`X` shown; `Y` mirrors the row/column relationship):

```matlab
X =
    -2    -1     0     1     2
    -2    -1     0     1     2
    -2    -1     0     1     2
    -2    -1     0     1     2
    -2    -1     0     1     2
```

### Building a rectangular grid from two different vectors

```matlab
x = [0 0.5 1.0];
y = [10 20];
[X, Y] = meshgrid(x, y);
```

Expected output:

```matlab
X =
         0    0.5000    1.0000
         0    0.5000    1.0000

Y =
    10    10    10
    20    20    20
```

### Creating a volumetric grid for 3-D plotting

```matlab
u = -1:1;
v = 2:4;
w = linspace(0, 1, 5);
[U, V, W] = meshgrid(u, v, w);
```

Expected output shapes:

```matlab
size(U) == [3 3 5]
size(V) == [3 3 5]
size(W) == [3 3 5]
```

### Matching an existing GPU prototype

```matlab
gx = gpuArray(single(linspace(-pi, pi, 4)));
gy = gpuArray(single([-1 0 1]));
[Xg, Yg] = meshgrid(gx, gy);
```

`Xg` and `Yg` remain `gpuArray` values with `single` precision. Gathering them produces the same
numeric data as the host result.

### Using `'like'` to copy residency from another array

```matlab
proto = gpuArray.zeros(1, 1, 'double');
angles = linspace(0, 2*pi, 8);
radius = [0 1 2];
[X, Y] = meshgrid(angles, radius, 'like', proto);
```

Both `X` and `Y` stay on the GPU because the prototype is a `gpuArray`.

### Complex inputs produce complex grids automatically

```matlab
z = [1+1i, 2+4i];
[Zx, Zy] = meshgrid(z);
```

`Zx` and `Zy` are complex arrays whose imaginary parts match the source vector.

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do **not** need to wrap vectors with `gpuArray` manually. When the active acceleration
provider supports uploads, RunMat automatically constructs the grid on the host and keeps the
outputs on the GPU. Supplying a `'like', gpuArray(...)` prototype produces GPU outputs even when
all inputs are host arrays. Until native provider hooks land, complex-valued grids remain host-side
and emit a warning when GPU residency is requested.

## FAQ

### How many inputs can `meshgrid` accept?

One, two, or three numeric vectors. Use three inputs when you need volumetric (3-D) grids.

### Can I request three outputs with only one or two inputs?

No. RunMat follows MATLAB and requires three input vectors when three outputs are requested.

### Do row or column vectors behave differently?

No. Any vector shape (row, column, or scalar) is accepted. RunMat treats the linearised data
identically and replicates it along the appropriate axes.

### What happens with empty vectors?

Empty inputs propagate to empty outputs. For example, `meshgrid([], 1:3)` returns `0×3`
grids for both outputs.

### Can I use integer vectors?

Yes. Inputs are promoted to double precision internally so the outputs represent the exact same
values as MATLAB.

### Does `meshgrid` support complex numbers?

Absolutely. Any imaginary components propagate into the outputs. Complex grids currently stay
on the host even if GPU residency is requested.

### What does `'like'` do?

It matches the numeric class and residency (host or GPU) of the prototype array. Supply a
`gpuArray` prototype to keep the resulting grids on the GPU.

### How can providers avoid the host fall-back?

Implement the `meshgrid` custom hook in the acceleration provider. RunMat will automatically
dispatch to it once available.

### Is the output always dense?

Yes. `meshgrid` produces dense arrays. Use `ndgrid` when you need permuted axes or higher-dimensional
grids beyond three inputs.

### What error do I get if I omit all inputs?

RunMat raises the MATLAB-compatible error `meshgrid: at least one input vector is required`.

## See Also
[linspace](./linspace), [zeros](./zeros), [ones](./ones), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `meshgrid` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/meshgrid.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/meshgrid.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "meshgrid",
    op_kind: GpuOpKind::Custom("array_construct"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("meshgrid")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may supply a dedicated meshgrid hook; until then the runtime builds grids on the host and uploads them when GPU residency is requested.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "meshgrid",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Meshgrid explicitly materialises dense coordinate arrays and therefore bypasses fusion.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("meshgrid", DOC_MD);

#[runtime_builtin(
    name = "meshgrid",
    category = "array/creation",
    summary = "Generate coordinate matrices for 2-D and 3-D grids.",
    keywords = "meshgrid,grid,gpu,like,3d",
    accel = "array_construct"
)]
fn meshgrid_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&rest)?;
    eval.first()
}

/// Evaluate the `meshgrid` builtin once and reuse the result for multiple outputs.
pub fn evaluate(args: &[Value]) -> Result<MeshgridEval, String> {
    let parsed = ParsedMeshgrid::parse(args)?;
    let (x_axis, y_axis, z_axis) = normalise_axes(&parsed.axes);

    let require_complex = parsed.axes.iter().any(|axis| axis.is_complex);

    let target_class = match &parsed.template {
        OutputTemplate::Default => {
            if require_complex {
                PrototypeClass::Complex
            } else {
                PrototypeClass::Real
            }
        }
        OutputTemplate::Like(spec) => {
            if require_complex {
                PrototypeClass::Complex
            } else {
                spec.class
            }
        }
    };

    let target_residency = match &parsed.template {
        OutputTemplate::Default => {
            if parsed.prefer_gpu {
                DevicePreference::Gpu
            } else {
                DevicePreference::Host
            }
        }
        OutputTemplate::Like(spec) => spec.residency,
    };

    let mut gpu_outputs: Option<Vec<MeshgridOutput>> = None;
    let axes_all_real = !require_complex;

    if axes_all_real
        && matches!(target_class, PrototypeClass::Real)
        && matches!(target_residency, DevicePreference::Gpu)
    {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let x_real = axis_real_values(&x_axis);
            let y_real = axis_real_values(&y_axis);
            let z_real = z_axis.as_ref().map(|axis| axis_real_values(axis));
            let mut axis_views: Vec<MeshgridAxisView<'_>> =
                Vec::with_capacity(if z_real.is_some() { 3 } else { 2 });
            axis_views.push(MeshgridAxisView { data: &x_real });
            axis_views.push(MeshgridAxisView { data: &y_real });
            if let Some(ref data) = z_real {
                axis_views.push(MeshgridAxisView { data });
            }
            match provider.meshgrid(&axis_views) {
                Ok(result) => {
                    let expected = if z_axis.is_some() { 3 } else { 2 };
                    let outputs: Vec<MeshgridOutput> = result
                        .outputs
                        .into_iter()
                        .map(MeshgridOutput::GpuReal)
                        .collect();
                    if outputs.len() == expected {
                        gpu_outputs = Some(outputs);
                    } else {
                        warn!(
                            "meshgrid: provider returned {}/{} outputs; falling back to host",
                            outputs.len(),
                            expected
                        );
                    }
                }
                Err(err) => {
                    warn!("meshgrid: provider meshgrid hook failed, falling back to host: {err}")
                }
            }
        }
    }

    let outputs = gpu_outputs.unwrap_or_else(|| {
        build_outputs(&x_axis, &y_axis, z_axis.as_ref())
            .into_iter()
            .map(MeshgridOutput::Host)
            .collect()
    });

    Ok(MeshgridEval {
        outputs,
        target_class,
        target_residency,
    })
}

#[derive(Clone)]
struct ParsedMeshgrid {
    axes: Vec<AxisData>,
    template: OutputTemplate,
    prefer_gpu: bool,
}

impl ParsedMeshgrid {
    fn parse(args: &[Value]) -> Result<Self, String> {
        if args.is_empty() {
            return Err("meshgrid: at least one input vector is required".to_string());
        }
        let mut axis_values: Vec<Value> = Vec::new();
        let mut like_proto: Option<Value> = None;
        let mut prefer_gpu = false;
        let mut idx = 0;
        while idx < args.len() {
            let value = args[idx].clone();
            if let Some(keyword) = keyword_of(&value) {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(
                                "meshgrid: multiple 'like' specifications are not supported"
                                    .to_string(),
                            );
                        }
                        if axis_values.is_empty() {
                            return Err("meshgrid: 'like' must follow at least one input vector"
                                .to_string());
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("meshgrid: expected prototype after 'like'".to_string());
                        };
                        like_proto = Some(proto);
                        idx += 2;
                        if idx < args.len() {
                            return Err("meshgrid: 'like' must be the final argument".to_string());
                        }
                        break;
                    }
                    other => {
                        return Err(format!("meshgrid: unrecognised option '{other}'"));
                    }
                }
            }

            if let Value::GpuTensor(_) = value {
                prefer_gpu = true;
            }
            axis_values.push(value);
            idx += 1;
        }

        if axis_values.is_empty() {
            return Err("meshgrid: at least one input vector is required".to_string());
        }
        if axis_values.len() > 3 {
            return Err("meshgrid: expected at most three input vectors".to_string());
        }

        let mut axes = Vec::with_capacity(max(axis_values.len(), 2));
        for (i, value) in axis_values.into_iter().enumerate() {
            let mut consumed_gpu = false;
            let data = axis_from_value(value, i, &mut consumed_gpu)?;
            if consumed_gpu {
                prefer_gpu = true;
            }
            axes.push(data);
        }

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(analyse_like_prototype(&proto)?)
        } else {
            OutputTemplate::Default
        };

        Ok(Self {
            axes,
            template,
            prefer_gpu,
        })
    }
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(PrototypeSpec),
}

#[derive(Clone)]
struct PrototypeSpec {
    residency: DevicePreference,
    class: PrototypeClass,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrototypeClass {
    Real,
    Complex,
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

fn analyse_like_prototype(proto: &Value) -> Result<PrototypeSpec, String> {
    match proto {
        Value::GpuTensor(_) => Ok(PrototypeSpec {
            residency: DevicePreference::Gpu,
            class: PrototypeClass::Real,
        }),
        Value::ComplexTensor(_) | Value::Complex(_, _) => Ok(PrototypeSpec {
            residency: DevicePreference::Host,
            class: PrototypeClass::Complex,
        }),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_) => Ok(PrototypeSpec {
            residency: DevicePreference::Host,
            class: PrototypeClass::Real,
        }),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err("meshgrid: prototypes must be numeric or gpuArray values".to_string())
        }
        Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err("meshgrid: prototypes must be numeric arrays".to_string()),
    }
}

#[derive(Clone)]
struct AxisData {
    values: Vec<(f64, f64)>,
    len: usize,
    is_complex: bool,
}

fn axis_from_value(value: Value, index: usize, prefer_gpu: &mut bool) -> Result<AxisData, String> {
    match value {
        Value::Tensor(tensor) => axis_from_tensor(tensor),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            axis_from_tensor(tensor)
        }
        Value::Num(n) => Ok(AxisData {
            values: vec![(n, 0.0)],
            len: 1,
            is_complex: false,
        }),
        Value::Int(i) => {
            let val = i.to_f64();
            Ok(AxisData {
                values: vec![(val, 0.0)],
                len: 1,
                is_complex: false,
            })
        }
        Value::Bool(b) => Ok(AxisData {
            values: vec![(if b { 1.0 } else { 0.0 }, 0.0)],
            len: 1,
            is_complex: false,
        }),
        Value::Complex(re, im) => Ok(AxisData {
            values: vec![(re, im)],
            len: 1,
            is_complex: im != 0.0,
        }),
        Value::ComplexTensor(tensor) => axis_from_complex_tensor(tensor),
        Value::GpuTensor(handle) => {
            *prefer_gpu = true;
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            axis_from_tensor(tensor)
        }
        other => Err(format!(
            "meshgrid: input argument {} must be numeric, got {other:?}",
            index + 1
        )),
    }
}

fn axis_from_tensor(tensor: Tensor) -> Result<AxisData, String> {
    if !is_vector_shape(&tensor.shape) {
        return Err("meshgrid: input vectors must be one-dimensional".to_string());
    }
    let mut values = Vec::with_capacity(tensor.data.len());
    for &v in &tensor.data {
        values.push((v, 0.0));
    }
    Ok(AxisData {
        len: values.len(),
        values,
        is_complex: false,
    })
}

fn axis_from_complex_tensor(tensor: ComplexTensor) -> Result<AxisData, String> {
    if !is_vector_shape(&tensor.shape) {
        return Err("meshgrid: input vectors must be one-dimensional".to_string());
    }
    let is_complex = tensor
        .data
        .iter()
        .any(|&(_, imag)| !imag.is_nan() && imag != 0.0);
    Ok(AxisData {
        len: tensor.data.len(),
        values: tensor.data,
        is_complex,
    })
}

fn axis_real_values(axis: &AxisData) -> Vec<f64> {
    axis.values.iter().map(|(re, _)| *re).collect()
}

fn is_vector_shape(shape: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }
    let mut non_singleton = 0usize;
    for &dim in shape {
        if dim > 1 {
            non_singleton += 1;
        }
    }
    non_singleton <= 1
}

fn normalise_axes(axes: &[AxisData]) -> (AxisData, AxisData, Option<AxisData>) {
    match axes.len() {
        1 => {
            let x = axes[0].clone();
            (x.clone(), x, None)
        }
        2 => {
            let x = axes[0].clone();
            let y = axes[1].clone();
            (x, y, None)
        }
        3 => {
            let x = axes[0].clone();
            let y = axes[1].clone();
            let z = axes[2].clone();
            (x, y, Some(z))
        }
        _ => unreachable!(),
    }
}

fn build_outputs(
    x_axis: &AxisData,
    y_axis: &AxisData,
    z_axis: Option<&AxisData>,
) -> Vec<GridOutput> {
    let nx = x_axis.len;
    let ny = y_axis.len;
    let nz = z_axis.map(|axis| axis.len).unwrap_or(1);
    let total = nx * ny * nz;
    let mut x_data = Vec::with_capacity(total);
    let mut y_data = Vec::with_capacity(total);
    let mut z_data = z_axis.map(|_| Vec::with_capacity(total));

    for k in 0..nz {
        let z_value = z_axis.map(|axis| axis.values[k]);
        for col in 0..nx {
            let x_value = x_axis.values[col];
            for row in 0..ny {
                x_data.push(x_value);
                y_data.push(y_axis.values[row]);
                if let Some(ref mut z_vec) = z_data {
                    z_vec.push(z_value.unwrap());
                }
            }
        }
    }

    let mut outputs = Vec::new();
    let base_shape = if nz == 1 {
        vec![ny, nx]
    } else {
        vec![ny, nx, nz]
    };
    outputs.push(GridOutput {
        shape: base_shape.clone(),
        data: x_data,
    });
    outputs.push(GridOutput {
        shape: base_shape.clone(),
        data: y_data,
    });
    if let Some(z_vec) = z_data {
        outputs.push(GridOutput {
            shape: base_shape,
            data: z_vec,
        });
    }
    outputs
}

struct GridOutput {
    shape: Vec<usize>,
    data: Vec<(f64, f64)>,
}

impl GridOutput {
    fn to_value(
        &self,
        class: PrototypeClass,
        residency: DevicePreference,
    ) -> Result<Value, String> {
        match class {
            PrototypeClass::Real => self.to_real_value(residency),
            PrototypeClass::Complex => self.to_complex_value(residency),
        }
    }

    fn to_real_value(&self, residency: DevicePreference) -> Result<Value, String> {
        let mut real = Vec::with_capacity(self.data.len());
        for &(re, im) in &self.data {
            if im != 0.0 {
                return Err(
                    "meshgrid: cannot represent complex values in a real output".to_string()
                );
            }
            real.push(re);
        }
        let tensor = Tensor::new(real, self.shape.clone()).map_err(|e| format!("meshgrid: {e}"))?;
        match residency {
            DevicePreference::Host => Ok(tensor::tensor_into_value(tensor)),
            DevicePreference::Gpu => to_gpu_tensor_value(tensor),
        }
    }

    fn to_complex_value(&self, residency: DevicePreference) -> Result<Value, String> {
        let tensor = ComplexTensor::new(self.data.clone(), self.shape.clone())
            .map_err(|e| format!("meshgrid: {e}"))?;
        match residency {
            DevicePreference::Host => Ok(complex_tensor_into_value(tensor)),
            DevicePreference::Gpu => {
                warn!("meshgrid: complex GPU outputs are not implemented; returning host complex array");
                Ok(complex_tensor_into_value(tensor))
            }
        }
    }
}

fn to_gpu_tensor_value(tensor: Tensor) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        match provider.upload(&view) {
            Ok(handle) => return Ok(Value::GpuTensor(handle)),
            Err(err) => {
                warn!("meshgrid: failed to upload tensor to GPU, returning host array: {err}")
            }
        }
    }
    Ok(tensor::tensor_into_value(tensor))
}

fn tensor_to_complex_value(tensor: Tensor) -> Result<Value, String> {
    let data: Vec<(f64, f64)> = tensor.data.iter().map(|&re| (re, 0.0)).collect();
    let complex =
        ComplexTensor::new(data, tensor.shape.clone()).map_err(|e| format!("meshgrid: {e}"))?;
    Ok(complex_tensor_into_value(complex))
}

enum MeshgridOutput {
    Host(GridOutput),
    GpuReal(GpuTensorHandle),
}

impl MeshgridOutput {
    fn to_value(
        &self,
        class: PrototypeClass,
        residency: DevicePreference,
    ) -> Result<Value, String> {
        match self {
            MeshgridOutput::Host(host) => host.to_value(class, residency),
            MeshgridOutput::GpuReal(handle) => match (class, residency) {
                (PrototypeClass::Real, DevicePreference::Gpu) => {
                    Ok(Value::GpuTensor(handle.clone()))
                }
                (PrototypeClass::Real, DevicePreference::Host) => {
                    let tensor = gpu_helpers::gather_tensor(handle)?;
                    Ok(tensor::tensor_into_value(tensor))
                }
                (PrototypeClass::Complex, DevicePreference::Host) => {
                    let tensor = gpu_helpers::gather_tensor(handle)?;
                    tensor_to_complex_value(tensor)
                }
                (PrototypeClass::Complex, DevicePreference::Gpu) => {
                    warn!("meshgrid: complex GPU outputs are not implemented; returning host complex array");
                    let tensor = gpu_helpers::gather_tensor(handle)?;
                    tensor_to_complex_value(tensor)
                }
            },
        }
    }
}

/// Holds the results of a `meshgrid` evaluation so multiple outputs can be
/// materialised without recomputing the grid.
pub struct MeshgridEval {
    outputs: Vec<MeshgridOutput>,
    target_class: PrototypeClass,
    target_residency: DevicePreference,
}

impl MeshgridEval {
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    pub fn first(&self) -> Result<Value, String> {
        self.outputs[0].to_value(self.target_class, self.target_residency)
    }

    pub fn second(&self) -> Result<Value, String> {
        if self.outputs.len() < 2 {
            Err("meshgrid: second output unavailable".to_string())
        } else {
            self.outputs[1].to_value(self.target_class, self.target_residency)
        }
    }

    pub fn third(&self) -> Result<Value, String> {
        if self.outputs.len() < 3 {
            Err("meshgrid: third output requested but no Z vector was supplied".to_string())
        } else {
            self.outputs[2].to_value(self.target_class, self.target_residency)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;

    fn tensor_from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor::new(data, vec![rows, cols]).unwrap()
    }

    #[test]
    fn meshgrid_single_input_duplicates_axis() {
        let x = tensor_from_vec(vec![-1.0, 0.0, 1.0], 1, 3);
        let eval = evaluate(&[Value::Tensor(x)]).expect("meshgrid");
        assert_eq!(eval.output_count(), 2);
        let x_out = test_support::gather(eval.first().expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![3, 3]);
        assert_eq!(
            x_out.data,
            vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        );
        let y_out = test_support::gather(eval.second().expect("Y")).expect("host");
        assert_eq!(y_out.shape, vec![3, 3]);
        assert_eq!(
            y_out.data,
            vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn meshgrid_rectangular_inputs() {
        let x = tensor_from_vec(vec![0.0, 0.5, 1.0], 1, 3);
        let y = tensor_from_vec(vec![10.0, 20.0], 2, 1);
        let eval = evaluate(&[Value::Tensor(x), Value::Tensor(y)]).expect("meshgrid");
        assert_eq!(eval.output_count(), 2);
        let x_out = test_support::gather(eval.first().expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![2, 3]);
        assert_eq!(x_out.data, vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
        let y_out = test_support::gather(eval.second().expect("Y")).expect("host");
        assert_eq!(y_out.shape, vec![2, 3]);
        assert_eq!(y_out.data, vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn meshgrid_three_inputs_volume() {
        let x = tensor_from_vec(vec![1.0, 2.0], 1, 2);
        let y = tensor_from_vec(vec![5.0, 6.0, 7.0], 3, 1);
        let z = tensor_from_vec(vec![0.0, 1.0], 1, 2);
        let eval =
            evaluate(&[Value::Tensor(x), Value::Tensor(y), Value::Tensor(z)]).expect("meshgrid");
        assert_eq!(eval.output_count(), 3);
        let x_out = test_support::gather(eval.first().expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![3, 2, 2]);
        assert_eq!(
            x_out.data,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        );
        let z_out = test_support::gather(eval.third().expect("Z")).expect("host");
        assert_eq!(z_out.shape, vec![3, 2, 2]);
        assert_eq!(
            z_out.data,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn meshgrid_like_keeps_gpu_residency() {
        test_support::with_test_provider(|provider| {
            let x = tensor_from_vec(vec![-1.0, 0.0, 1.0], 1, 3);
            let y = tensor_from_vec(vec![2.0, 4.0], 2, 1);
            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &proto.data,
                shape: &proto.shape,
            };
            let proto_handle = provider.upload(&proto_view).expect("upload");
            let eval = evaluate(&[
                Value::Tensor(x),
                Value::Tensor(y),
                Value::from("like"),
                Value::GpuTensor(proto_handle),
            ])
            .expect("meshgrid");
            let x_value = eval.first().expect("X");
            assert!(matches!(x_value, Value::GpuTensor(_)));
            let gathered = test_support::gather(x_value).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
        });
    }

    #[test]
    fn meshgrid_gpu_inputs_roundtrip() {
        test_support::with_test_provider(|provider| {
            let x = tensor_from_vec(vec![0.0, 0.5], 1, 2);
            let y = tensor_from_vec(vec![1.0, 2.0], 2, 1);
            let x_view = HostTensorView {
                data: &x.data,
                shape: &x.shape,
            };
            let y_view = HostTensorView {
                data: &y.data,
                shape: &y.shape,
            };
            let x_handle = provider.upload(&x_view).expect("upload");
            let y_handle = provider.upload(&y_view).expect("upload");
            let eval = evaluate(&[Value::GpuTensor(x_handle), Value::GpuTensor(y_handle)])
                .expect("meshgrid");
            assert!(matches!(eval.first().expect("X"), Value::GpuTensor(_)));
            assert!(matches!(eval.second().expect("Y"), Value::GpuTensor(_)));
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn meshgrid_wgpu_matches_cpu() {
        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("wgpu provider");

        let x = tensor_from_vec(vec![-1.0, 0.0, 1.0, 2.0], 1, 4);
        let y = tensor_from_vec(vec![5.0, 6.0], 2, 1);

        let cpu_eval =
            evaluate(&[Value::Tensor(x.clone()), Value::Tensor(y.clone())]).expect("meshgrid cpu");
        let cpu_x = test_support::gather(cpu_eval.first().expect("X cpu")).expect("gather X cpu");
        let cpu_y = test_support::gather(cpu_eval.second().expect("Y cpu")).expect("gather Y cpu");

        let x_view = HostTensorView {
            data: &x.data,
            shape: &x.shape,
        };
        let y_view = HostTensorView {
            data: &y.data,
            shape: &y.shape,
        };
        let x_gpu = provider.upload(&x_view).expect("upload x");
        let y_gpu = provider.upload(&y_view).expect("upload y");

        let gpu_eval =
            evaluate(&[Value::GpuTensor(x_gpu), Value::GpuTensor(y_gpu)]).expect("meshgrid gpu");
        let gpu_x_value = gpu_eval.first().expect("X gpu");
        let gpu_y_value = gpu_eval.second().expect("Y gpu");

        assert!(matches!(gpu_x_value, Value::GpuTensor(_)));
        assert!(matches!(gpu_y_value, Value::GpuTensor(_)));

        let gathered_x = test_support::gather(gpu_x_value).expect("gather X gpu");
        let gathered_y = test_support::gather(gpu_y_value).expect("gather Y gpu");

        assert_eq!(gathered_x.shape, cpu_x.shape);
        assert_eq!(gathered_x.data, cpu_x.data);
        assert_eq!(gathered_y.shape, cpu_y.shape);
        assert_eq!(gathered_y.data, cpu_y.data);
    }

    #[test]
    fn meshgrid_complex_inputs_produce_complex_outputs() {
        let complex = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let eval = evaluate(&[Value::ComplexTensor(complex)]).expect("meshgrid");
        let x_value = eval.first().expect("X");
        match x_value {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
            }
            Value::Complex(_, _) => {}
            other => panic!("expected complex output, got {other:?}"),
        }
    }

    #[test]
    fn meshgrid_like_host_prototype() {
        let x = tensor_from_vec(vec![1.0, 2.0], 1, 2);
        let eval =
            evaluate(&[Value::Tensor(x), Value::from("like"), Value::Num(0.0)]).expect("meshgrid");
        let x_out = eval.first().expect("X");
        assert!(matches!(x_out, Value::Tensor(_) | Value::Num(_)));
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
