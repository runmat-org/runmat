//! MATLAB-compatible `arrayfun` builtin with GPU-aware semantics.
//!
//! This implementation supports applying a scalar MATLAB function to every element
//! of one or more array inputs. When invoked with `gpuArray` inputs the builtin
//! executes on the host today and uploads the uniform output back to the device so
//! downstream code continues to see GPU residency. Future provider hooks can swap
//! in a device kernel without affecting the public API.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::acceleration::gpu::type_resolvers::arrayfun_type;
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};
use runmat_accelerate_api::{set_handle_logical, GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, Closure, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::arrayfun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "arrayfun",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Unary { name: "unary_sin" },
        ProviderHook::Unary { name: "unary_cos" },
        ProviderHook::Unary { name: "unary_abs" },
        ProviderHook::Unary { name: "unary_exp" },
        ProviderHook::Unary { name: "unary_log" },
        ProviderHook::Unary { name: "unary_sqrt" },
        ProviderHook::Binary {
            name: "elem_add",
            commutative: true,
        },
        ProviderHook::Binary {
            name: "elem_sub",
            commutative: false,
        },
        ProviderHook::Binary {
            name: "elem_mul",
            commutative: true,
        },
        ProviderHook::Binary {
            name: "elem_div",
            commutative: false,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers that implement the listed kernels can run supported callbacks entirely on the GPU; unsupported callbacks fall back to the host path with re-upload.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::acceleration::gpu::arrayfun"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "arrayfun",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion barrier because the callback can run arbitrary MATLAB code.",
};

fn arrayfun_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("arrayfun")
        .build()
}

fn arrayfun_error_with_source(message: impl Into<String>, source: RuntimeError) -> RuntimeError {
    let identifier = source.identifier().map(str::to_string);
    let mut builder = build_runtime_error(message)
        .with_builtin("arrayfun")
        .with_source(source);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn arrayfun_flow(message: impl Into<String>) -> RuntimeError {
    arrayfun_error(message)
}

fn arrayfun_flow_with_source(message: impl Into<String>, source: RuntimeError) -> RuntimeError {
    arrayfun_error_with_source(message, source)
}

fn format_handler_error(err: &RuntimeError) -> String {
    if let Some(identifier) = err.identifier() {
        if err.message().is_empty() {
            return identifier.to_string();
        }
        if err.message().starts_with(identifier) {
            return err.message().to_string();
        }
        return format!("{identifier}: {}", err.message());
    }
    err.message().to_string()
}

#[runtime_builtin(
    name = "arrayfun",
    category = "acceleration/gpu",
    summary = "Apply a function element-wise to array inputs.",
    keywords = "arrayfun,gpu,array,map,functional",
    accel = "host",
    type_resolver(arrayfun_type),
    builtin_path = "crate::builtins::acceleration::gpu::arrayfun"
)]
async fn arrayfun_builtin(func: Value, mut rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let callable = Callable::from_function(func)?;

    let mut uniform_output = true;
    let mut error_handler: Option<Callable> = None;

    while rest.len() >= 2 {
        let key_candidate = rest[rest.len() - 2].clone();
        let Some(name) = extract_string(&key_candidate) else {
            break;
        };
        let value = rest.pop().expect("value present");
        rest.pop();
        match name.trim().to_ascii_lowercase().as_str() {
            "uniformoutput" => uniform_output = parse_uniform_output(value)?,
            "errorhandler" => error_handler = Some(Callable::from_function(value)?),
            other => {
                return Err(arrayfun_flow(format!(
                    "arrayfun: unknown name-value argument '{other}'"
                )))
            }
        }
    }

    if rest.is_empty() {
        return Err(arrayfun_flow("arrayfun: expected at least one input array"));
    }

    let inputs_snapshot = rest.clone();
    let has_gpu_input = inputs_snapshot
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)));
    let gpu_device_id = inputs_snapshot.iter().find_map(|v| {
        if let Value::GpuTensor(h) = v {
            Some(h.device_id)
        } else {
            None
        }
    });

    if uniform_output {
        if let Some(gpu_result) =
            try_gpu_fast_path(&callable, &inputs_snapshot, error_handler.as_ref()).await?
        {
            return Ok(gpu_result);
        }
    }

    let mut inputs: Vec<ArrayInput> = Vec::with_capacity(rest.len());
    let mut base_shape: Vec<usize> = Vec::new();
    let mut base_len: Option<usize> = None;

    for (idx, raw) in rest.into_iter().enumerate() {
        if matches!(raw, Value::Cell(_)) {
            return Err(arrayfun_flow(
                "arrayfun: cell inputs are not supported (use cellfun instead)",
            ));
        }
        if matches!(raw, Value::Struct(_)) {
            return Err(arrayfun_flow("arrayfun: struct inputs are not supported"));
        }

        let host_value = gather_if_needed_async(&raw).await?;
        let data = ArrayData::from_value(host_value)?;
        let len = data.len();
        let is_scalar = len == 1;

        let mut input = ArrayInput { data, is_scalar };

        if let Some(current) = base_len {
            if current == len {
                if len > 1 {
                    let shape = input.shape_vec();
                    if shape != base_shape {
                        return Err(arrayfun_flow(format!(
                            "arrayfun: input {} does not match the size of the first array",
                            idx + 1
                        )));
                    }
                }
            } else if len == 1 {
                input.is_scalar = true;
            } else if current == 1 {
                base_len = Some(len);
                base_shape = input.shape_vec();
                for prior in &mut inputs {
                    let prior_len = prior.len();
                    if prior_len == len {
                        if prior.shape_vec() != base_shape {
                            return Err(arrayfun_flow(format!(
                                "arrayfun: input {} does not match the size of the first array",
                                idx
                            )));
                        }
                    } else if prior_len == 1 {
                        prior.is_scalar = true;
                    } else if prior_len == 0 && len == 0 {
                        continue;
                    } else {
                        return Err(arrayfun_flow(format!(
                            "arrayfun: input {} does not match the size of the first array",
                            idx
                        )));
                    }
                }
            } else if len == 0 && current == 0 {
                let shape = input.shape_vec();
                if shape != base_shape {
                    return Err(arrayfun_flow(format!(
                        "arrayfun: input {} does not match the size of the first array",
                        idx + 1
                    )));
                }
            } else {
                return Err(arrayfun_flow(format!(
                    "arrayfun: input {} does not match the size of the first array",
                    idx + 1
                )));
            }
        } else {
            base_len = Some(len);
            base_shape = input.shape_vec();
        }

        inputs.push(input);
    }

    let total_len = base_len.unwrap_or(0);

    if total_len == 0 {
        if uniform_output {
            return Ok(empty_uniform(&base_shape));
        } else {
            return make_cell_with_shape(Vec::new(), base_shape)
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")));
        }
    }

    let mut collector = if uniform_output {
        Some(UniformCollector::Pending)
    } else {
        None
    };

    let mut cell_outputs: Vec<Value> = Vec::new();
    let mut args: Vec<Value> = Vec::with_capacity(inputs.len());

    for idx in 0..total_len {
        args.clear();
        for input in &inputs {
            args.push(input.value_at(idx)?);
        }

        let result = match callable.call(&args).await {
            Ok(value) => value,
            Err(err) => {
                let handler = match error_handler.as_ref() {
                    Some(handler) => handler,
                    None => {
                        return Err(arrayfun_flow_with_source(
                            format!("arrayfun: {}", err.message()),
                            err,
                        ))
                    }
                };
                let err_message = format_handler_error(&err);
                let err_value = make_error_struct(&err_message, idx, &base_shape)?;
                let mut handler_args = Vec::with_capacity(1 + args.len());
                handler_args.push(err_value);
                handler_args.extend(args.clone());
                handler.call(&handler_args).await?
            }
        };

        let host_result = gather_if_needed_async(&result).await?;

        if let Some(collector) = collector.as_mut() {
            collector.push(&host_result)?;
        } else {
            cell_outputs.push(host_result);
        }
    }

    if let Some(collector) = collector {
        let uniform = collector.finish(&base_shape)?;
        maybe_upload_uniform(uniform, has_gpu_input, gpu_device_id)
    } else {
        make_cell_with_shape(cell_outputs, base_shape)
            .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))
    }
}

fn maybe_upload_uniform(
    value: Value,
    has_gpu_input: bool,
    gpu_device_id: Option<u32>,
) -> BuiltinResult<Value> {
    if !has_gpu_input {
        return Ok(value);
    }
    #[cfg(all(test, feature = "wgpu"))]
    {
        if matches!(gpu_device_id, Some(id) if id != 0) {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let _ = gpu_device_id; // may be used only in cfg(test)
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(value),
    };

    match value {
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::LogicalArray(logical) => {
            let data: Vec<f64> = logical
                .data
                .iter()
                .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                .collect();
            let tensor = Tensor::new(data, logical.shape.clone())
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
            set_handle_logical(&handle, true);
            Ok(Value::GpuTensor(handle))
        }
        other => Ok(other),
    }
}

fn empty_uniform(shape: &[usize]) -> Value {
    if shape.is_empty() {
        return Value::Tensor(Tensor::zeros(vec![0, 0]));
    }
    let total: usize = shape.iter().product();
    let tensor = Tensor::new(vec![0.0; total], shape.to_vec())
        .unwrap_or_else(|_| Tensor::zeros(shape.to_vec()));
    Value::Tensor(tensor)
}

fn parse_uniform_output(value: Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(b),
        Value::Num(n) => Ok(n != 0.0),
        Value::Int(iv) => Ok(iv.to_f64() != 0.0),
        Value::String(s) => parse_bool_string(&s)
            .ok_or_else(|| arrayfun_flow("arrayfun: UniformOutput must be logical true or false")),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            parse_bool_string(&text).ok_or_else(|| {
                arrayfun_flow("arrayfun: UniformOutput must be logical true or false")
            })
        }
        other => Err(arrayfun_flow(format!(
            "arrayfun: UniformOutput must be logical true or false, got {other:?}"
        ))),
    }
}

fn parse_bool_string(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "on" => Some(true),
        "false" | "off" => Some(false),
        _ => None,
    }
}

fn extract_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

struct ArrayInput {
    data: ArrayData,
    is_scalar: bool,
}

impl ArrayInput {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn shape_vec(&self) -> Vec<usize> {
        self.data.shape_vec()
    }

    fn value_at(&self, idx: usize) -> BuiltinResult<Value> {
        if self.is_scalar {
            self.data.value_at(0)
        } else {
            self.data.value_at(idx)
        }
    }
}

enum ArrayData {
    Tensor(Tensor),
    Logical(LogicalArray),
    Complex(ComplexTensor),
    Char(CharArray),
    Scalar(Value),
}

impl ArrayData {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(t) => Ok(ArrayData::Tensor(t)),
            Value::LogicalArray(l) => Ok(ArrayData::Logical(l)),
            Value::ComplexTensor(c) => Ok(ArrayData::Complex(c)),
            Value::CharArray(ca) => Ok(ArrayData::Char(ca)),
            Value::Num(_) | Value::Bool(_) | Value::Int(_) | Value::Complex(_, _) => {
                Ok(ArrayData::Scalar(value))
            }
            other => Err(arrayfun_flow(format!(
                "arrayfun: unsupported input type {other:?} (expected numeric, logical, complex, or char arrays)"
            ))),
        }
    }

    fn len(&self) -> usize {
        match self {
            ArrayData::Tensor(t) => t.data.len(),
            ArrayData::Logical(l) => l.data.len(),
            ArrayData::Complex(c) => c.data.len(),
            ArrayData::Char(ca) => ca.rows * ca.cols,
            ArrayData::Scalar(_) => 1,
        }
    }

    fn shape_vec(&self) -> Vec<usize> {
        match self {
            ArrayData::Tensor(t) => {
                if t.shape.is_empty() {
                    vec![1, 1]
                } else {
                    t.shape.clone()
                }
            }
            ArrayData::Logical(l) => {
                if l.shape.is_empty() {
                    vec![1, 1]
                } else {
                    l.shape.clone()
                }
            }
            ArrayData::Complex(c) => {
                if c.shape.is_empty() {
                    vec![1, 1]
                } else {
                    c.shape.clone()
                }
            }
            ArrayData::Char(ca) => vec![ca.rows, ca.cols],
            ArrayData::Scalar(_) => vec![1, 1],
        }
    }

    fn value_at(&self, idx: usize) -> BuiltinResult<Value> {
        match self {
            ArrayData::Tensor(t) => {
                Ok(Value::Num(*t.data.get(idx).ok_or_else(|| {
                    arrayfun_flow("arrayfun: index out of bounds")
                })?))
            }
            ArrayData::Logical(l) => Ok(Value::Bool(
                *l.data
                    .get(idx)
                    .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?
                    != 0,
            )),
            ArrayData::Complex(c) => {
                let (re, im) = c
                    .data
                    .get(idx)
                    .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?;
                Ok(Value::Complex(*re, *im))
            }
            ArrayData::Char(ca) => {
                if ca.rows == 0 || ca.cols == 0 {
                    return Ok(Value::CharArray(
                        CharArray::new(Vec::new(), 0, 0)
                            .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?,
                    ));
                }
                let rows = ca.rows;
                let cols = ca.cols;
                let row = idx % rows;
                let col = idx / rows;
                let data_idx = row * cols + col;
                let ch = *ca
                    .data
                    .get(data_idx)
                    .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?;
                let char_array = CharArray::new(vec![ch], 1, 1)
                    .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
                Ok(Value::CharArray(char_array))
            }
            ArrayData::Scalar(v) => Ok(v.clone()),
        }
    }
}

#[derive(Clone)]
enum Callable {
    Builtin { name: String },
    Closure(Closure),
}

impl Callable {
    fn from_function(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Self::from_text(&text),
            Value::CharArray(ca) => {
                if ca.rows != 1 {
                    Err(arrayfun_flow(
                        "arrayfun: function name must be a character vector or string scalar",
                    ))
                } else {
                    let text: String = ca.data.iter().collect();
                    Self::from_text(&text)
                }
            }
            Value::StringArray(sa) if sa.data.len() == 1 => Self::from_text(&sa.data[0]),
            Value::FunctionHandle(name) => Ok(Callable::Builtin { name }),
            Value::Closure(closure) => Ok(Callable::Closure(closure)),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => Err(arrayfun_flow(
                "arrayfun: expected function handle or builtin name, not a scalar value",
            )),
            other => Err(arrayfun_flow(format!(
                "arrayfun: expected function handle or builtin name, got {other:?}"
            ))),
        }
    }

    fn from_text(text: &str) -> BuiltinResult<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(arrayfun_flow(
                "arrayfun: expected function handle or builtin name, got empty string",
            ));
        }
        if let Some(rest) = trimmed.strip_prefix('@') {
            let name = rest.trim();
            if name.is_empty() {
                Err(arrayfun_flow("arrayfun: empty function handle"))
            } else {
                Ok(Callable::Builtin {
                    name: name.to_string(),
                })
            }
        } else {
            Ok(Callable::Builtin {
                name: trimmed.to_ascii_lowercase(),
            })
        }
    }

    fn builtin_name(&self) -> Option<&str> {
        match self {
            Callable::Builtin { name } => Some(name.as_str()),
            Callable::Closure(_) => None,
        }
    }

    async fn call(&self, args: &[Value]) -> crate::BuiltinResult<Value> {
        match self {
            Callable::Builtin { name } => crate::call_builtin_async(name, args).await,
            Callable::Closure(c) => {
                let mut merged = c.captures.clone();
                merged.extend_from_slice(args);
                crate::call_builtin_async(&c.function_name, &merged).await
            }
        }
    }
}

async fn try_gpu_fast_path(
    callable: &Callable,
    inputs: &[Value],
    error_handler: Option<&Callable>,
) -> BuiltinResult<Option<Value>> {
    if inputs.is_empty() || error_handler.is_some() {
        return Ok(None);
    }
    if !inputs
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }

    #[cfg(all(test, feature = "wgpu"))]
    {
        if inputs
            .iter()
            .any(|v| matches!(v, Value::GpuTensor(h) if h.device_id != 0))
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let Some(name_raw) = callable.builtin_name() else {
        return Ok(None);
    };
    let name = name_raw.to_ascii_lowercase();

    let mut handles: Vec<GpuTensorHandle> = Vec::with_capacity(inputs.len());
    for value in inputs {
        if let Value::GpuTensor(handle) = value {
            handles.push(handle.clone());
        }
    }

    if handles.len() >= 2 {
        let base_shape = handles[0].shape.clone();
        if handles
            .iter()
            .skip(1)
            .any(|handle| handle.shape != base_shape)
        {
            return Ok(None);
        }
    }

    let result = match name.as_str() {
        "sin" if handles.len() == 1 => provider.unary_sin(&handles[0]).await,
        "cos" if handles.len() == 1 => provider.unary_cos(&handles[0]).await,
        "abs" if handles.len() == 1 => provider.unary_abs(&handles[0]).await,
        "exp" if handles.len() == 1 => provider.unary_exp(&handles[0]).await,
        "log" if handles.len() == 1 => provider.unary_log(&handles[0]).await,
        "sqrt" if handles.len() == 1 => provider.unary_sqrt(&handles[0]).await,
        "plus" if handles.len() == 2 => provider.elem_add(&handles[0], &handles[1]).await,
        "minus" if handles.len() == 2 => provider.elem_sub(&handles[0], &handles[1]).await,
        "times" if handles.len() == 2 => provider.elem_mul(&handles[0], &handles[1]).await,
        "rdivide" if handles.len() == 2 => provider.elem_div(&handles[0], &handles[1]).await,
        "ldivide" if handles.len() == 2 => provider.elem_div(&handles[1], &handles[0]).await,
        _ => return Ok(None),
    };

    match result {
        Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
        Err(_) => Ok(None),
    }
}

enum UniformCollector {
    Pending,
    Double(Vec<f64>),
    Logical(Vec<u8>),
    Complex(Vec<(f64, f64)>),
    Char(Vec<char>),
}

impl UniformCollector {
    fn push(&mut self, value: &Value) -> BuiltinResult<()> {
        match self {
            UniformCollector::Pending => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    *self = UniformCollector::Logical(vec![b as u8]);
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    *self = UniformCollector::Double(vec![d]);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    *self = UniformCollector::Complex(vec![c]);
                    Ok(())
                }
                ClassifiedValue::Char(ch) => {
                    *self = UniformCollector::Char(vec![ch]);
                    Ok(())
                }
            },
            UniformCollector::Logical(bits) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    bits.push(b as u8);
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    let mut data: Vec<f64> = bits
                        .iter()
                        .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                        .collect();
                    data.push(d);
                    *self = UniformCollector::Double(data);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    let mut data: Vec<(f64, f64)> = bits
                        .iter()
                        .map(|&bit| if bit != 0 { (1.0, 0.0) } else { (0.0, 0.0) })
                        .collect();
                    data.push(c);
                    *self = UniformCollector::Complex(data);
                    Ok(())
                }
                ClassifiedValue::Char(ch) => {
                    let mut data: Vec<f64> = bits
                        .iter()
                        .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                        .collect();
                    data.push(ch as u32 as f64);
                    *self = UniformCollector::Double(data);
                    Ok(())
                }
            },
            UniformCollector::Double(data) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    data.push(if b { 1.0 } else { 0.0 });
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    data.push(d);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    let promoted: Vec<(f64, f64)> = data.iter().map(|&v| (v, 0.0)).collect();
                    let mut complex = promoted;
                    complex.push(c);
                    *self = UniformCollector::Complex(complex);
                    Ok(())
                }
                ClassifiedValue::Char(ch) => {
                    data.push(ch as u32 as f64);
                    Ok(())
                }
            },
            UniformCollector::Complex(data) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    data.push((if b { 1.0 } else { 0.0 }, 0.0));
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    data.push((d, 0.0));
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    data.push(c);
                    Ok(())
                }
                ClassifiedValue::Char(ch) => {
                    data.push((ch as u32 as f64, 0.0));
                    Ok(())
                }
            },
            UniformCollector::Char(chars) => match classify_value(value)? {
                ClassifiedValue::Char(ch) => {
                    chars.push(ch);
                    Ok(())
                }
                ClassifiedValue::Logical(b) => {
                    let mut data: Vec<f64> = chars.iter().map(|&ch| ch as u32 as f64).collect();
                    data.push(if b { 1.0 } else { 0.0 });
                    *self = UniformCollector::Double(data);
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    let mut data: Vec<f64> = chars.iter().map(|&ch| ch as u32 as f64).collect();
                    data.push(d);
                    *self = UniformCollector::Double(data);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    let mut promoted: Vec<(f64, f64)> =
                        chars.iter().map(|&ch| (ch as u32 as f64, 0.0)).collect();
                    promoted.push(c);
                    *self = UniformCollector::Complex(promoted);
                    Ok(())
                }
            },
        }
    }

    fn finish(self, shape: &[usize]) -> BuiltinResult<Value> {
        match self {
            UniformCollector::Pending => {
                let total = shape.iter().product();
                let tensor = Tensor::new(vec![0.0; total], shape.to_vec())
                    .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Double(data) => {
                let tensor = Tensor::new(data, shape.to_vec())
                    .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Logical(bits) => {
                let logical = LogicalArray::new(bits, shape.to_vec())
                    .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
                Ok(Value::LogicalArray(logical))
            }
            UniformCollector::Complex(entries) => {
                let tensor = ComplexTensor::new(entries, shape.to_vec())
                    .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
                Ok(Value::ComplexTensor(tensor))
            }
            UniformCollector::Char(chars) => {
                let normalized_shape = if shape.is_empty() {
                    vec![1, 1]
                } else {
                    shape.to_vec()
                };

                if normalized_shape.len() > 2 {
                    return Err(arrayfun_flow(
                        "arrayfun: character outputs with UniformOutput=true must be 2-D",
                    ));
                }

                let rows = normalized_shape.first().copied().unwrap_or(1);
                let cols = normalized_shape.get(1).copied().unwrap_or(1);
                let expected = rows.checked_mul(cols).ok_or_else(|| {
                    arrayfun_flow("arrayfun: character output size exceeds platform limits")
                })?;

                if expected != chars.len() {
                    return Err(arrayfun_flow(
                        "arrayfun: callback returned the wrong number of characters",
                    ));
                }

                let mut row_major = vec!['\0'; expected];
                for col in 0..cols {
                    for row in 0..rows {
                        let col_major_idx = row + col * rows;
                        let row_major_idx = row * cols + col;
                        row_major[row_major_idx] = chars[col_major_idx];
                    }
                }

                let array = CharArray::new(row_major, rows, cols)
                    .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
                Ok(Value::CharArray(array))
            }
        }
    }
}

enum ClassifiedValue {
    Logical(bool),
    Double(f64),
    Complex((f64, f64)),
    Char(char),
}

fn classify_value(value: &Value) -> BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(b) => Ok(ClassifiedValue::Logical(*b)),
        Value::LogicalArray(la) if la.len() == 1 => Ok(ClassifiedValue::Logical(la.data[0] != 0)),
        Value::Int(i) => Ok(ClassifiedValue::Double(i.to_f64())),
        Value::Num(n) => Ok(ClassifiedValue::Double(*n)),
        Value::Tensor(t) if t.data.len() == 1 => Ok(ClassifiedValue::Double(t.data[0])),
        Value::Complex(re, im) => Ok(ClassifiedValue::Complex((*re, *im))),
        Value::ComplexTensor(t) if t.data.len() == 1 => Ok(ClassifiedValue::Complex(t.data[0])),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            let ch = ca.data.first().copied().unwrap_or('\0');
            Ok(ClassifiedValue::Char(ch))
        }
        other => Err(arrayfun_flow(format!(
            "arrayfun: callback must return scalar numeric, logical, character, or complex values for UniformOutput=true (got {other:?})"
        ))),
    }
}

fn make_error_struct(
    raw_error: &str,
    linear_index: usize,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let (identifier, message) = split_error_message(raw_error);
    let mut st = runmat_builtins::StructValue::new();
    st.fields
        .insert("identifier".to_string(), Value::String(identifier));
    st.fields
        .insert("message".to_string(), Value::String(message));
    st.fields
        .insert("index".to_string(), Value::Num((linear_index + 1) as f64));
    let subs = linear_to_indices(linear_index, shape);
    let subs_tensor = dims_to_row_tensor(&subs)?;
    st.fields
        .insert("indices".to_string(), Value::Tensor(subs_tensor));
    Ok(Value::Struct(st))
}

fn split_error_message(raw: &str) -> (String, String) {
    let trimmed = raw.trim();
    let mut indices = trimmed.match_indices(':');
    if let Some((_, _)) = indices.next() {
        if let Some((second_idx, _)) = indices.next() {
            let identifier = trimmed[..second_idx].trim().to_string();
            let message = trimmed[second_idx + 1..].trim().to_string();
            if !identifier.is_empty() && identifier.contains(':') {
                return (
                    identifier,
                    if message.is_empty() {
                        trimmed.to_string()
                    } else {
                        message
                    },
                );
            }
        } else if trimmed.len() >= 7
            && (trimmed[..7].eq_ignore_ascii_case("matlab:")
                || trimmed[..7].eq_ignore_ascii_case("runmat:"))
        {
            return (trimmed.to_string(), String::new());
        }
    }
    (
        "MATLAB:arrayfun:FunctionError".to_string(),
        trimmed.to_string(),
    )
}

fn linear_to_indices(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![1];
    }
    let mut subs = Vec::with_capacity(shape.len());
    for &dim in shape {
        if dim == 0 {
            subs.push(1);
            continue;
        }
        let coord = (index % dim) + 1;
        subs.push(coord);
        index /= dim;
    }
    subs
}

fn dims_to_row_tensor(dims: &[usize]) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = dims.iter().map(|&d| d as f64).collect();
    Tensor::new(data, vec![1, dims.len()]).map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{Tensor, Type};

    fn call(func: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(arrayfun_builtin(func, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_basic_sin() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let expected: Vec<f64> = tensor.data.iter().map(|&x| x.sin()).collect();
        let result = call(
            Value::FunctionHandle("sin".to_string()),
            vec![Value::Tensor(tensor.clone())],
        )
        .expect("arrayfun");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, expected);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn arrayfun_type_tracks_function_returns() {
        let func = Type::Function {
            params: vec![Type::Num],
            returns: Box::new(Type::Num),
        };
        assert_eq!(arrayfun_type(&[func, Type::tensor()]), Type::tensor());
    }

    #[test]
    fn arrayfun_type_uses_logical_returns() {
        let func = Type::Function {
            params: vec![Type::Num],
            returns: Box::new(Type::Bool),
        };
        assert_eq!(arrayfun_type(&[func, Type::tensor()]), Type::logical());
    }

    #[test]
    fn arrayfun_type_with_text_args_stays_unknown() {
        let func = Type::Function {
            params: vec![Type::Num],
            returns: Box::new(Type::Num),
        };
        assert_eq!(
            arrayfun_type(&[func, Type::tensor(), Type::String, Type::Bool]),
            Type::Unknown
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_additional_scalar_argument() {
        let tensor = Tensor::new(vec![0.5, 1.0, -1.0], vec![3, 1]).unwrap();
        let expected: Vec<f64> = tensor.data.iter().map(|&y| y.atan2(1.0)).collect();
        let result = call(
            Value::FunctionHandle("atan2".to_string()),
            vec![Value::Tensor(tensor), Value::Num(1.0)],
        )
        .expect("arrayfun");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, expected);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_uniform_false_returns_cell() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let expected: Vec<Value> = tensor.data.iter().map(|&x| Value::Num(x.sin())).collect();
        let result = call(
            Value::FunctionHandle("sin".to_string()),
            vec![
                Value::Tensor(tensor),
                Value::String("UniformOutput".into()),
                Value::Bool(false),
            ],
        )
        .expect("arrayfun");
        let Value::Cell(cell) = result else {
            panic!("expected cell, got something else");
        };
        assert_eq!(cell.rows, 2);
        assert_eq!(cell.cols, 1);
        for (row, value) in expected.iter().enumerate() {
            assert_eq!(cell.get(row, 0).unwrap(), *value);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_size_mismatch_errors() {
        let taller = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let shorter = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let err = call(
            Value::FunctionHandle("sin".to_string()),
            vec![Value::Tensor(taller), Value::Tensor(shorter)],
        )
        .expect_err("expected size mismatch error");
        let err = err.to_string();
        assert!(
            err.contains("does not match"),
            "expected size mismatch error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_error_handler_recovers() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let handler = Value::Closure(Closure {
            function_name: "__arrayfun_test_handler".into(),
            captures: vec![Value::Num(42.0)],
        });
        let result = call(
            Value::String("@nonexistent_builtin".into()),
            vec![
                Value::Tensor(tensor),
                Value::String("ErrorHandler".into()),
                handler,
            ],
        )
        .expect("arrayfun error handler");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.data, vec![42.0, 42.0, 42.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_error_without_handler_propagates_identifier() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = call(
            Value::String("@nonexistent_builtin".into()),
            vec![Value::Tensor(tensor)],
        )
        .expect_err("expected unresolved function error");
        assert_eq!(
            err.identifier(),
            Some("MATLAB:UndefinedFunction"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_uniform_logical_result() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 0.0, f64::INFINITY], vec![4, 1]).unwrap();
        let result = call(
            Value::FunctionHandle("isfinite".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("arrayfun isfinite");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![4, 1]);
                assert_eq!(la.data, vec![1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_uniform_character_result() {
        let tensor = Tensor::new(vec![65.0, 66.0, 67.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("char".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("arrayfun char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
                assert_eq!(ca.data, vec!['A', 'B', 'C']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_uniform_false_gpu_returns_cell() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call(
                Value::FunctionHandle("sin".to_string()),
                vec![
                    Value::GpuTensor(handle),
                    Value::String("UniformOutput".into()),
                    Value::Bool(false),
                ],
            )
            .expect("arrayfun");
            match result {
                Value::Cell(cell) => {
                    assert_eq!(cell.rows, 2);
                    assert_eq!(cell.cols, 1);
                    let first = cell.get(0, 0).expect("first cell");
                    let second = cell.get(1, 0).expect("second cell");
                    match (first, second) {
                        (Value::Num(a), Value::Num(b)) => {
                            assert!((a - 0.0f64.sin()).abs() < 1e-12);
                            assert!((b - 1.0f64.sin()).abs() < 1e-12);
                        }
                        other => panic!("expected numeric cells, got {other:?}"),
                    }
                }
                other => panic!("expected cell, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call(
                Value::FunctionHandle("sin".to_string()),
                vec![Value::GpuTensor(handle)],
            )
            .expect("arrayfun");
            match result {
                Value::GpuTensor(gpu) => {
                    let gathered = test_support::gather(Value::GpuTensor(gpu.clone())).unwrap();
                    let expected: Vec<f64> = tensor.data.iter().map(|&x| x.sin()).collect();
                    assert_eq!(gathered.data, expected);
                    let _ = provider.free(&gpu);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn arrayfun_wgpu_sin_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = call(
            Value::FunctionHandle("sin".into()),
            vec![Value::GpuTensor(handle.clone())],
        )
        .expect("arrayfun sin gpu");
        let Value::GpuTensor(out_handle) = result else {
            panic!("expected GPU tensor result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out_handle.clone())).unwrap();
        let expected: Vec<f64> = tensor.data.iter().map(|v| v.sin()).collect();
        assert_eq!(gathered.shape, tensor.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (actual, expect) in gathered.data.iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < tol,
                "expected {expect}, got {actual}"
            );
        }
        let _ = provider.free(&handle);
        let _ = provider.free(&out_handle);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn arrayfun_wgpu_plus_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let view_a = HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let handle_a = provider.upload(&view_a).expect("upload a");
        let handle_b = provider.upload(&view_b).expect("upload b");
        let result = call(
            Value::FunctionHandle("plus".into()),
            vec![
                Value::GpuTensor(handle_a.clone()),
                Value::GpuTensor(handle_b.clone()),
            ],
        )
        .expect("arrayfun plus gpu");

        let Value::GpuTensor(out_handle) = result else {
            panic!("expected GPU tensor result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out_handle.clone())).unwrap();
        let expected: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| x + y)
            .collect();
        assert_eq!(gathered.shape, a.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (actual, expect) in gathered.data.iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < tol,
                "expected {expect}, got {actual}"
            );
        }
        let _ = provider.free(&handle_a);
        let _ = provider.free(&handle_b);
        let _ = provider.free(&out_handle);
    }

    #[runmat_macros::runtime_builtin(
        name = "__arrayfun_test_handler",
        type_resolver(arrayfun_type),
        builtin_path = "crate::builtins::acceleration::gpu::arrayfun::tests"
    )]
    async fn arrayfun_test_handler(
        seed: Value,
        _err: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        let _ = rest;
        Ok(seed)
    }
}
