//! MATLAB-compatible `meshgrid` builtin with GPU-aware semantics.

use std::cmp::max;

use log::warn;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, ResolveContext, Tensor, Type, Value};

use crate::builtins::array::type_resolvers::size_vector_len;
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::meshgrid")]
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

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("meshgrid")
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::meshgrid")]
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

fn meshgrid_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Unknown;
    }
    let mut axis_count = args.len();
    if axis_count >= 2 && matches!(args[axis_count - 2], Type::String) {
        axis_count = axis_count.saturating_sub(2);
    }
    if axis_count == 0 {
        return Type::Unknown;
    }
    let axis_args = &args[..axis_count];
    let len_x = axis_args.get(0).and_then(size_vector_len);
    let len_y = axis_args.get(1).and_then(size_vector_len).or(len_x);
    let len_z = axis_args.get(2).and_then(size_vector_len);
    let shape = if axis_count >= 3 {
        vec![len_y, len_x, len_z]
    } else {
        vec![len_y, len_x]
    };
    Type::Tensor { shape: Some(shape) }
}

#[runtime_builtin(
    name = "meshgrid",
    category = "array/creation",
    summary = "Generate coordinate matrices for 2-D and 3-D grids.",
    keywords = "meshgrid,grid,gpu,like,3d",
    accel = "array_construct",
    type_resolver(meshgrid_type),
    builtin_path = "crate::builtins::array::creation::meshgrid"
)]
async fn meshgrid_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let available = eval.output_count();
        if out_count > available {
            let msg = if available == 2 {
                "meshgrid with two inputs supports at most two outputs"
            } else {
                "meshgrid supports at most three outputs"
            };
            return Err(builtin_error(msg));
        }
        let mut outputs = Vec::with_capacity(out_count);
        let first = eval.first().await?;
        outputs.push(first);
        if out_count >= 2 {
            outputs.push(eval.second().await?);
        }
        if out_count >= 3 {
            outputs.push(eval.third().await?);
        }
        return Ok(Value::OutputList(outputs));
    }
    eval.first().await
}

/// Evaluate the `meshgrid` builtin once and reuse the result for multiple outputs.
pub async fn evaluate(args: &[Value]) -> crate::BuiltinResult<MeshgridEval> {
    let parsed = ParsedMeshgrid::parse(args).await?;
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

    let axes_all_real = !require_complex;
    let mut outputs: Vec<MeshgridOutput> = Vec::new();

    if axes_all_real
        && matches!(target_class, PrototypeClass::Real)
        && matches!(target_residency, DevicePreference::Gpu)
    {
        if let Some(gpu) = try_meshgrid_gpu_from_vector_axes(&x_axis, &y_axis, z_axis.as_ref())? {
            outputs = gpu;
        }
    }

    if outputs.is_empty() {
        // Host fallback: ensure we have host axis values materialized.
        let x_host = axis_to_host_async(&x_axis).await?;
        let y_host = axis_to_host_async(&y_axis).await?;
        let z_host = match z_axis.as_ref() {
            Some(axis) => Some(axis_to_host_async(axis).await?),
            None => None,
        };
        outputs = build_outputs(&x_host, &y_host, z_host.as_ref())
            .into_iter()
            .map(MeshgridOutput::Host)
            .collect();
    }

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
    async fn parse(args: &[Value]) -> crate::BuiltinResult<Self> {
        if args.is_empty() {
            return Err(builtin_error(
                "meshgrid: at least one input vector is required",
            ));
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
                            return Err(builtin_error(
                                "meshgrid: multiple 'like' specifications are not supported",
                            ));
                        }
                        if axis_values.is_empty() {
                            return Err(builtin_error(
                                "meshgrid: 'like' must follow at least one input vector",
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("meshgrid: expected prototype after 'like'"));
                        };
                        like_proto = Some(proto);
                        idx += 2;
                        if idx < args.len() {
                            return Err(builtin_error(
                                "meshgrid: 'like' must be the final argument",
                            ));
                        }
                        break;
                    }
                    other => {
                        return Err(builtin_error(format!(
                            "meshgrid: unrecognised option '{other}'"
                        )));
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
            return Err(builtin_error(
                "meshgrid: at least one input vector is required",
            ));
        }
        if axis_values.len() > 3 {
            return Err(builtin_error(
                "meshgrid: expected at most three input vectors",
            ));
        }

        let mut axes = Vec::with_capacity(max(axis_values.len(), 2));
        for (i, value) in axis_values.into_iter().enumerate() {
            let mut consumed_gpu = false;
            let data = axis_from_value(value, i, &mut consumed_gpu).await?;
            if consumed_gpu {
                prefer_gpu = true;
            }
            axes.push(data);
        }

        if !prefer_gpu {
            if let Some(max_len) = axes.iter().map(|axis| axis.len).max() {
                if max_len > 0
                    && sequence_gpu_preference(max_len, SequenceIntent::MeshAxis, false).prefer_gpu
                {
                    prefer_gpu = true;
                }
            }
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

fn analyse_like_prototype(proto: &Value) -> crate::BuiltinResult<PrototypeSpec> {
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
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => Err(builtin_error(
            "meshgrid: prototypes must be numeric or gpuArray values",
        )),
        Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(builtin_error("meshgrid: prototypes must be numeric arrays")),
    }
}

#[derive(Clone)]
struct AxisData {
    values: Vec<(f64, f64)>,
    len: usize,
    is_complex: bool,
    gpu_real: Option<GpuTensorHandle>,
}

async fn axis_from_value(
    value: Value,
    index: usize,
    prefer_gpu: &mut bool,
) -> crate::BuiltinResult<AxisData> {
    match value {
        Value::Tensor(tensor) => axis_from_tensor(tensor, index),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            axis_from_tensor(tensor, index)
        }
        Value::Num(n) => Ok(AxisData {
            values: vec![(n, 0.0)],
            len: 1,
            is_complex: false,
            gpu_real: None,
        }),
        Value::Int(i) => {
            let val = i.to_f64();
            Ok(AxisData {
                values: vec![(val, 0.0)],
                len: 1,
                is_complex: false,
                gpu_real: None,
            })
        }
        Value::Bool(b) => Ok(AxisData {
            values: vec![(if b { 1.0 } else { 0.0 }, 0.0)],
            len: 1,
            is_complex: false,
            gpu_real: None,
        }),
        Value::Complex(re, im) => Ok(AxisData {
            values: vec![(re, im)],
            len: 1,
            is_complex: im != 0.0,
            gpu_real: None,
        }),
        Value::ComplexTensor(tensor) => axis_from_complex_tensor(tensor, index),
        Value::GpuTensor(handle) => {
            // Fast path: if the gpuArray is vector-like, keep it on-device and avoid a download.
            // We'll validate any non-vector shapes by gathering below.
            if is_vector_shape(&handle.shape) {
                *prefer_gpu = true;
                return Ok(AxisData {
                    values: Vec::new(),
                    len: vector_len_from_shape(&handle.shape),
                    is_complex: false,
                    gpu_real: Some(handle),
                });
            }

            // Fallback: gather to validate / recover axes from meshgrid matrices.
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            if is_vector_shape(&tensor.shape) {
                *prefer_gpu = true;
            }
            axis_from_tensor(tensor, index)
        }
        other => Err(builtin_error(format!(
            "meshgrid: input argument {} must be numeric, got {other:?}",
            index + 1
        ))),
    }
}

fn axis_from_tensor(tensor: Tensor, index: usize) -> crate::BuiltinResult<AxisData> {
    if is_vector_shape(&tensor.shape) {
        let mut values = Vec::with_capacity(tensor.data.len());
        for &v in &tensor.data {
            values.push((v, 0.0));
        }
        return Ok(AxisData {
            len: values.len(),
            values,
            is_complex: false,
            gpu_real: None,
        });
    }

    // Be slightly more permissive than MATLAB: if the input is already a meshgrid-style
    // coordinate matrix, accept it and recover the original axis vector.
    //
    // This is a pragmatic compatibility shim for cases where callers already have
    // coordinate matrices (X/Y) and pass them through `meshgrid` again.
    if let Some(axis) = axis_from_meshgrid_matrix_real(&tensor, index)? {
        return Ok(axis);
    }

    Err(builtin_error(format!(
        "meshgrid: input argument {} must be a vector (1xN or Nx1), got shape {:?}",
        index + 1,
        tensor.shape
    )))
}

fn axis_from_complex_tensor(tensor: ComplexTensor, index: usize) -> crate::BuiltinResult<AxisData> {
    if is_vector_shape(&tensor.shape) {
        let is_complex = tensor
            .data
            .iter()
            .any(|&(_, imag)| !imag.is_nan() && imag != 0.0);
        return Ok(AxisData {
            len: tensor.data.len(),
            values: tensor.data,
            is_complex,
            gpu_real: None,
        });
    }

    if let Some(axis) = axis_from_meshgrid_matrix_complex(&tensor, index)? {
        return Ok(axis);
    }

    Err(builtin_error(format!(
        "meshgrid: input argument {} must be a vector (1xN or Nx1), got shape {:?}",
        index + 1,
        tensor.shape
    )))
}

fn axis_from_meshgrid_matrix_real(
    tensor: &Tensor,
    index: usize,
) -> crate::BuiltinResult<Option<AxisData>> {
    let (rows, cols) = match tensor.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => return Ok(None),
    };
    if rows <= 1 || cols <= 1 {
        return Ok(None);
    }

    // Index 0 is expected to be the X-axis: a meshgrid X matrix has identical rows.
    // Index 1 is expected to be the Y-axis: a meshgrid Y matrix has identical columns.
    let expect_rows_constant = index == 0;

    if expect_rows_constant {
        if !matrix_rows_are_identical_real(tensor, rows, cols) {
            return Ok(None);
        }
        // Extract the first row as the axis vector (length = cols).
        let mut values = Vec::with_capacity(cols);
        for col in 0..cols {
            let idx = rows * col;
            values.push((tensor.data[idx], 0.0));
        }
        return Ok(Some(AxisData {
            len: values.len(),
            values,
            is_complex: false,
            gpu_real: None,
        }));
    }

    if !matrix_cols_are_identical_real(tensor, rows, cols) {
        return Ok(None);
    }
    // Extract the first column as the axis vector (length = rows).
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        values.push((tensor.data[row], 0.0));
    }
    Ok(Some(AxisData {
        len: values.len(),
        values,
        is_complex: false,
        gpu_real: None,
    }))
}

fn axis_from_meshgrid_matrix_complex(
    tensor: &ComplexTensor,
    index: usize,
) -> crate::BuiltinResult<Option<AxisData>> {
    let (rows, cols) = match tensor.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => return Ok(None),
    };
    if rows <= 1 || cols <= 1 {
        return Ok(None);
    }

    let expect_rows_constant = index == 0;
    if expect_rows_constant {
        if !matrix_rows_are_identical_complex(tensor, rows, cols) {
            return Ok(None);
        }
        let mut values = Vec::with_capacity(cols);
        for col in 0..cols {
            let idx = rows * col;
            values.push(tensor.data[idx]);
        }
        let is_complex = values.iter().any(|&(_, im)| !im.is_nan() && im != 0.0);
        return Ok(Some(AxisData {
            len: values.len(),
            values,
            is_complex,
            gpu_real: None,
        }));
    }

    if !matrix_cols_are_identical_complex(tensor, rows, cols) {
        return Ok(None);
    }
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        values.push(tensor.data[row]);
    }
    let is_complex = values.iter().any(|&(_, im)| !im.is_nan() && im != 0.0);
    Ok(Some(AxisData {
        len: values.len(),
        values,
        is_complex,
        gpu_real: None,
    }))
}

fn matrix_rows_are_identical_real(tensor: &Tensor, rows: usize, cols: usize) -> bool {
    for row in 1..rows {
        for col in 0..cols {
            let idx0 = rows * col;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

fn matrix_cols_are_identical_real(tensor: &Tensor, rows: usize, cols: usize) -> bool {
    for col in 1..cols {
        for row in 0..rows {
            let idx0 = row;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

fn matrix_rows_are_identical_complex(tensor: &ComplexTensor, rows: usize, cols: usize) -> bool {
    for row in 1..rows {
        for col in 0..cols {
            let idx0 = rows * col;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

fn matrix_cols_are_identical_complex(tensor: &ComplexTensor, rows: usize, cols: usize) -> bool {
    for col in 1..cols {
        for row in 0..rows {
            let idx0 = row;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
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

fn vector_len_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape.iter().copied().max().unwrap_or(0)
}

async fn axis_to_host_async(axis: &AxisData) -> crate::BuiltinResult<AxisData> {
    if axis.gpu_real.is_none() {
        return Ok(axis.clone());
    }
    let handle = axis.gpu_real.as_ref().expect("checked gpu_real is_some");
    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
    // Index is only used for error messages; tensor came from a validated vector-like handle.
    axis_from_tensor(tensor, 0)
}

fn try_meshgrid_gpu_from_vector_axes(
    x_axis: &AxisData,
    y_axis: &AxisData,
    z_axis: Option<&AxisData>,
) -> crate::BuiltinResult<Option<Vec<MeshgridOutput>>> {
    let Some(x_handle) = x_axis.gpu_real.as_ref() else {
        return Ok(None);
    };
    let Some(y_handle) = y_axis.gpu_real.as_ref() else {
        return Ok(None);
    };

    let z_handle = match z_axis {
        Some(axis) => match axis.gpu_real.as_ref() {
            Some(h) => Some(h),
            None => return Ok(None),
        },
        None => None,
    };

    let Some(provider) = runmat_accelerate_api::provider_for_handle(x_handle) else {
        return Ok(None);
    };
    if runmat_accelerate_api::provider_for_handle(y_handle).is_none() {
        return Ok(None);
    }
    if let Some(z) = z_handle {
        if runmat_accelerate_api::provider_for_handle(z).is_none() {
            return Ok(None);
        }
    }

    let nx = x_axis.len;
    let ny = y_axis.len;
    let nz = z_axis.map(|axis| axis.len).unwrap_or(1);

    // Reshape axis vectors (metadata-only) so repmat can build full grids on-device.
    let x_row = provider
        .reshape(x_handle, &[1, nx])
        .map_err(|e| builtin_error(format!("meshgrid: reshape X failed: {e}")))?;
    let y_col = provider
        .reshape(y_handle, &[ny, 1])
        .map_err(|e| builtin_error(format!("meshgrid: reshape Y failed: {e}")))?;

    let mut outputs = Vec::with_capacity(if z_handle.is_some() { 3 } else { 2 });
    if let Some(z) = z_handle {
        let x_base = provider
            .reshape(&x_row, &[1, nx, 1])
            .map_err(|e| builtin_error(format!("meshgrid: reshape X(3d) failed: {e}")))?;
        let y_base = provider
            .reshape(&y_col, &[ny, 1, 1])
            .map_err(|e| builtin_error(format!("meshgrid: reshape Y(3d) failed: {e}")))?;

        let x_grid = provider
            .repmat(&x_base, &[ny, 1, nz])
            .map_err(|e| builtin_error(format!("meshgrid: repmat X failed: {e}")))?;
        let y_grid = provider
            .repmat(&y_base, &[1, nx, nz])
            .map_err(|e| builtin_error(format!("meshgrid: repmat Y failed: {e}")))?;

        outputs.push(MeshgridOutput::GpuReal(x_grid));
        outputs.push(MeshgridOutput::GpuReal(y_grid));
        let z_axis_row = provider
            .reshape(z, &[1, nz])
            .map_err(|e| builtin_error(format!("meshgrid: reshape Z failed: {e}")))?;
        let z_base = provider
            .reshape(&z_axis_row, &[1, 1, nz])
            .map_err(|e| builtin_error(format!("meshgrid: reshape Z(3d) failed: {e}")))?;
        let z_grid = provider
            .repmat(&z_base, &[ny, nx, 1])
            .map_err(|e| builtin_error(format!("meshgrid: repmat Z failed: {e}")))?;
        outputs.push(MeshgridOutput::GpuReal(z_grid));
    } else {
        let x_grid = provider
            .repmat(&x_row, &[ny, 1])
            .map_err(|e| builtin_error(format!("meshgrid: repmat X failed: {e}")))?;
        let y_grid = provider
            .repmat(&y_col, &[1, nx])
            .map_err(|e| builtin_error(format!("meshgrid: repmat Y failed: {e}")))?;
        outputs.push(MeshgridOutput::GpuReal(x_grid));
        outputs.push(MeshgridOutput::GpuReal(y_grid));
    }

    Ok(Some(outputs))
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
    ) -> crate::BuiltinResult<Value> {
        match class {
            PrototypeClass::Real => self.to_real_value(residency),
            PrototypeClass::Complex => self.to_complex_value(residency),
        }
    }

    fn to_real_value(&self, residency: DevicePreference) -> crate::BuiltinResult<Value> {
        let mut real = Vec::with_capacity(self.data.len());
        for &(re, im) in &self.data {
            if im != 0.0 {
                return Err(builtin_error(
                    "meshgrid: cannot represent complex values in a real output",
                ));
            }
            real.push(re);
        }
        let tensor = Tensor::new(real, self.shape.clone())
            .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
        match residency {
            DevicePreference::Host => Ok(tensor::tensor_into_value(tensor)),
            DevicePreference::Gpu => to_gpu_tensor_value(tensor),
        }
    }

    fn to_complex_value(&self, residency: DevicePreference) -> crate::BuiltinResult<Value> {
        let tensor = ComplexTensor::new(self.data.clone(), self.shape.clone())
            .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
        match residency {
            DevicePreference::Host => Ok(complex_tensor_into_value(tensor)),
            DevicePreference::Gpu => {
                warn!("meshgrid: complex GPU outputs are not implemented; returning host complex array");
                Ok(complex_tensor_into_value(tensor))
            }
        }
    }
}

fn to_gpu_tensor_value(tensor: Tensor) -> crate::BuiltinResult<Value> {
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

fn tensor_to_complex_value(tensor: Tensor) -> crate::BuiltinResult<Value> {
    let data: Vec<(f64, f64)> = tensor.data.iter().map(|&re| (re, 0.0)).collect();
    let complex = ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
    Ok(complex_tensor_into_value(complex))
}

enum MeshgridOutput {
    Host(GridOutput),
    GpuReal(GpuTensorHandle),
}

impl MeshgridOutput {
    async fn to_value(
        &self,
        class: PrototypeClass,
        residency: DevicePreference,
    ) -> crate::BuiltinResult<Value> {
        match self {
            MeshgridOutput::Host(host) => host.to_value(class, residency),
            MeshgridOutput::GpuReal(handle) => match (class, residency) {
                (PrototypeClass::Real, DevicePreference::Gpu) => {
                    Ok(Value::GpuTensor(handle.clone()))
                }
                (PrototypeClass::Real, DevicePreference::Host) => {
                    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
                    Ok(tensor::tensor_into_value(tensor))
                }
                (PrototypeClass::Complex, DevicePreference::Host) => {
                    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
                    tensor_to_complex_value(tensor)
                }
                (PrototypeClass::Complex, DevicePreference::Gpu) => {
                    warn!("meshgrid: complex GPU outputs are not implemented; returning host complex array");
                    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
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

    pub async fn first(&self) -> crate::BuiltinResult<Value> {
        self.outputs[0]
            .to_value(self.target_class, self.target_residency)
            .await
    }

    pub async fn second(&self) -> crate::BuiltinResult<Value> {
        if self.outputs.len() < 2 {
            Err(builtin_error("meshgrid: second output unavailable"))
        } else {
            self.outputs[1]
                .to_value(self.target_class, self.target_residency)
                .await
        }
    }

    pub async fn third(&self) -> crate::BuiltinResult<Value> {
        if self.outputs.len() < 3 {
            Err(builtin_error(
                "meshgrid: third output requested but no Z vector was supplied",
            ))
        } else {
            self.outputs[2]
                .to_value(self.target_class, self.target_residency)
                .await
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;

    use runmat_accelerate_api::HostTensorView;

    fn evaluate(args: &[Value]) -> crate::BuiltinResult<MeshgridEval> {
        block_on(super::evaluate(args))
    }

    fn eval_first(eval: &MeshgridEval) -> crate::BuiltinResult<Value> {
        block_on(eval.first())
    }

    fn eval_second(eval: &MeshgridEval) -> crate::BuiltinResult<Value> {
        block_on(eval.second())
    }

    fn eval_third(eval: &MeshgridEval) -> crate::BuiltinResult<Value> {
        block_on(eval.third())
    }

    fn tensor_from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor::new(data, vec![rows, cols]).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_single_input_duplicates_axis() {
        let x = tensor_from_vec(vec![-1.0, 0.0, 1.0], 1, 3);
        let eval = evaluate(&[Value::Tensor(x)]).expect("meshgrid");
        assert_eq!(eval.output_count(), 2);
        let x_out = test_support::gather(eval_first(&eval).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![3, 3]);
        assert_eq!(
            x_out.data,
            vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        );
        let y_out = test_support::gather(eval_second(&eval).expect("Y")).expect("host");
        assert_eq!(y_out.shape, vec![3, 3]);
        assert_eq!(
            y_out.data,
            vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn meshgrid_type_infers_rank_from_axis_count() {
        let ctx = ResolveContext::new(Vec::new());
        assert_eq!(
            meshgrid_type(&[Type::Num, Type::Num], &ctx),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1)])
            }
        );
        assert_eq!(
            meshgrid_type(&[Type::Num, Type::Num, Type::Num], &ctx),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1), Some(1)])
            }
        );
    }

    #[test]
    fn meshgrid_type_uses_vector_lengths() {
        let ctx = ResolveContext::new(Vec::new());
        assert_eq!(
            meshgrid_type(
                &[
                    Type::Tensor {
                        shape: Some(vec![Some(1), Some(201)]),
                    },
                    Type::Tensor {
                        shape: Some(vec![Some(1), Some(101)]),
                    },
                ],
                &ctx,
            ),
            Type::Tensor {
                shape: Some(vec![Some(101), Some(201)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_rectangular_inputs() {
        let x = tensor_from_vec(vec![0.0, 0.5, 1.0], 1, 3);
        let y = tensor_from_vec(vec![10.0, 20.0], 2, 1);
        let eval = evaluate(&[Value::Tensor(x), Value::Tensor(y)]).expect("meshgrid");
        assert_eq!(eval.output_count(), 2);
        let x_out = test_support::gather(eval_first(&eval).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![2, 3]);
        assert_eq!(x_out.data, vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
        let y_out = test_support::gather(eval_second(&eval).expect("Y")).expect("host");
        assert_eq!(y_out.shape, vec![2, 3]);
        assert_eq!(y_out.data, vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_three_inputs_volume() {
        let x = tensor_from_vec(vec![1.0, 2.0], 1, 2);
        let y = tensor_from_vec(vec![5.0, 6.0, 7.0], 3, 1);
        let z = tensor_from_vec(vec![0.0, 1.0], 1, 2);
        let eval =
            evaluate(&[Value::Tensor(x), Value::Tensor(y), Value::Tensor(z)]).expect("meshgrid");
        assert_eq!(eval.output_count(), 3);
        let x_out = test_support::gather(eval_first(&eval).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![3, 2, 2]);
        assert_eq!(
            x_out.data,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        );
        let z_out = test_support::gather(eval_third(&eval).expect("Z")).expect("host");
        assert_eq!(z_out.shape, vec![3, 2, 2]);
        assert_eq!(
            z_out.data,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
            let x_value = eval_first(&eval).expect("X");
            assert!(matches!(x_value, Value::GpuTensor(_)));
            let gathered = test_support::gather(x_value).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
            assert!(matches!(eval_first(&eval).expect("X"), Value::GpuTensor(_)));
            assert!(matches!(
                eval_second(&eval).expect("Y"),
                Value::GpuTensor(_)
            ));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let cpu_x =
            test_support::gather(eval_first(&cpu_eval).expect("X cpu")).expect("gather X cpu");
        let cpu_y =
            test_support::gather(eval_second(&cpu_eval).expect("Y cpu")).expect("gather Y cpu");

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
        let gpu_x_value = eval_first(&gpu_eval).expect("X gpu");
        let gpu_y_value = eval_second(&gpu_eval).expect("Y gpu");

        assert!(matches!(gpu_x_value, Value::GpuTensor(_)));
        assert!(matches!(gpu_y_value, Value::GpuTensor(_)));

        let gathered_x = test_support::gather(gpu_x_value).expect("gather X gpu");
        let gathered_y = test_support::gather(gpu_y_value).expect("gather Y gpu");

        assert_eq!(gathered_x.shape, cpu_x.shape);
        assert_eq!(gathered_x.data, cpu_x.data);
        assert_eq!(gathered_y.shape, cpu_y.shape);
        assert_eq!(gathered_y.data, cpu_y.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_complex_inputs_produce_complex_outputs() {
        let complex = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let eval = evaluate(&[Value::ComplexTensor(complex)]).expect("meshgrid");
        let x_value = eval_first(&eval).expect("X");
        match x_value {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
            }
            Value::Complex(_, _) => {}
            other => panic!("expected complex output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_like_host_prototype() {
        let x = tensor_from_vec(vec![1.0, 2.0], 1, 2);
        let eval =
            evaluate(&[Value::Tensor(x), Value::from("like"), Value::Num(0.0)]).expect("meshgrid");
        let x_out = eval_first(&eval).expect("X");
        assert!(matches!(x_out, Value::Tensor(_) | Value::Num(_)));
    }
}
