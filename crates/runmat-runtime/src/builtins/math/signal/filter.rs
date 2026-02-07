//! MATLAB-compatible `filter` builtin with GPU-aware semantics for RunMat.
use std::cmp::max;

use log::debug;
use num_complex::Complex;
use runmat_accelerate_api::{GpuTensorHandle, ProviderIirFilterOptions};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::type_resolvers::filter_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::filter")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "filter",
    op_kind: GpuOpKind::Custom("iir-filter"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("iir_filter")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses the provider hook `iir_filter` when available. Complex filters or missing hooks fall back to the host implementation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::filter")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "filter",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Filtering is handled via dedicated runtime logic; fusion does not currently optimise IIR/FIR chains.",
};

const BUILTIN_NAME: &str = "filter";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

async fn parse_dimension_arg(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(_) | Value::Num(_) => {
            tensor::dimension_from_value_async(value, BUILTIN_NAME, false)
                .await
                .map_err(runtime_error_for)?
                .ok_or_else(|| {
                    runtime_error_for(format!(
                        "{BUILTIN_NAME}: dimension must be numeric, got {value:?}"
                    ))
                })
        }
        _ => Err(runtime_error_for(format!(
            "{BUILTIN_NAME}: dimension must be numeric, got {value:?}"
        ))),
    }
}

#[runtime_builtin(
    name = "filter",
    category = "math/signal",
    summary = "Apply an IIR/FIR digital filter to scalars, vectors, or tensors.",
    keywords = "filter,IIR,FIR,difference equation,initial conditions,gpu",
    accel = "custom",
    type_resolver(filter_type),
    builtin_path = "crate::builtins::math::signal::filter"
)]
async fn filter_builtin(
    b: Value,
    a: Value,
    x: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let eval = evaluate(b, a, x, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.into_value()]));
        }
        let (output, final_state) = eval.into_pair();
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![output, final_state],
        ));
    }
    Ok(eval.into_value())
}

/// Evaluate the builtin once and expose both filter output and final state.
pub async fn evaluate(
    b: Value,
    a: Value,
    x: Value,
    rest: &[Value],
) -> BuiltinResult<FilterEvaluation> {
    let args = FilterArgs::parse(b, a, x, rest).await?;
    if let Some(eval) = try_filter_gpu(&args).await? {
        return Ok(eval);
    }
    filter_host(&args)
}

#[derive(Debug, Clone)]
pub struct FilterEvaluation {
    output: Value,
    final_state: Value,
}

impl FilterEvaluation {
    pub fn into_value(self) -> Value {
        self.output
    }

    pub fn into_pair(self) -> (Value, Value) {
        (self.output, self.final_state)
    }

    pub fn final_state_value(&self) -> Value {
        self.final_state.clone()
    }
}

struct FilterArgs {
    coeffs_b: CoeffInput,
    coeffs_a: CoeffInput,
    signal: SignalInput,
    dim_idx: usize,
    shape_ext: Vec<usize>,
    state_shape: Vec<usize>,
    order: usize,
    state_len: usize,
    leading: usize,
    trailing: usize,
    channel_count: usize,
    initial: InitialState,
    is_complex: bool,
}

impl FilterArgs {
    async fn parse(b: Value, a: Value, x: Value, rest: &[Value]) -> BuiltinResult<Self> {
        let (zi_raw, dim_raw) = parse_optional_arguments(rest)?;
        let zi_value = match zi_raw {
            Some(val) if is_empty_placeholder(&val) => None,
            other => other,
        };
        let dim_value = match dim_raw {
            Some(val) if is_empty_placeholder(&val) => None,
            other => other,
        };

        let coeffs_b = CoeffInput::from_value("filter", "numerator", b).await?;
        let coeffs_a = CoeffInput::from_value("filter", "denominator", a).await?;

        if coeffs_a.len == 0 {
            return Err(runtime_error_for(
                "filter: denominator coefficients cannot be empty",
            ));
        }
        if coeffs_b.len == 0 {
            return Err(runtime_error_for(
                "filter: numerator coefficients cannot be empty",
            ));
        }

        let signal = SignalInput::from_value(x).await?;

        let dim = if let Some(dim_val) = dim_value {
            parse_dimension_arg(&dim_val).await?
        } else {
            default_dimension_from_shape(&signal.shape)
        };
        let dim_idx = dim - 1;

        let mut shape_ext = signal.shape.clone();
        if dim > shape_ext.len() {
            shape_ext.extend(std::iter::repeat_n(1, dim - shape_ext.len()));
        }

        let order = max(coeffs_b.len, coeffs_a.len);
        let state_len = order.saturating_sub(1);

        let state_shape = filter_state_shape(shape_ext.clone(), dim_idx, state_len);

        let leading = if dim_idx == 0 {
            1
        } else {
            shape_ext[..dim_idx].iter().copied().product()
        };
        let trailing = if dim_idx + 1 >= shape_ext.len() {
            1
        } else {
            shape_ext[dim_idx + 1..].iter().copied().product()
        };
        let channel_count = leading.saturating_mul(trailing);

        let expected_states = state_len.saturating_mul(channel_count);

        let initial = if let Some(zi_val) = zi_value {
            InitialState::from_value(zi_val, state_len, &state_shape, expected_states, "filter")
                .await?
        } else {
            InitialState::empty(state_shape.clone())
        };

        let is_complex =
            coeffs_b.is_complex || coeffs_a.is_complex || signal.is_complex || initial.is_complex;

        Ok(Self {
            coeffs_b,
            coeffs_a,
            signal,
            dim_idx,
            shape_ext,
            state_shape,
            order,
            state_len,
            leading,
            trailing,
            channel_count,
            initial,
            is_complex,
        })
    }
}

struct CoeffInput {
    data: Vec<Complex<f64>>,
    len: usize,
    is_complex: bool,
}

impl CoeffInput {
    async fn from_value(name: &str, label: &str, value: Value) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                Self::from_tensor(name, label, tensor)
            }
            Value::Tensor(tensor) => Self::from_tensor(name, label, tensor),
            Value::ComplexTensor(tensor) => {
                let len = tensor.data.len();
                if len == 0 {
                    return Err(runtime_error_for(format!(
                        "{}: {} coefficients cannot be empty",
                        name, label
                    )));
                }
                ensure_vector_shape(name, label, &tensor.shape)?;
                let data = tensor
                    .data
                    .into_iter()
                    .map(|(re, im)| Complex::new(re, im))
                    .collect();
                Ok(Self {
                    data,
                    len,
                    is_complex: true,
                })
            }
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical).map_err(|e| {
                    runtime_error_for(format!("{}: {label} coefficients: {e}", name))
                })?;
                Self::from_tensor(name, label, tensor)
            }
            Value::Num(n) => Ok(Self {
                data: vec![Complex::new(n, 0.0)],
                len: 1,
                is_complex: false,
            }),
            Value::Int(i) => Ok(Self {
                data: vec![Complex::new(i.to_f64(), 0.0)],
                len: 1,
                is_complex: false,
            }),
            Value::Bool(b) => Ok(Self {
                data: vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)],
                len: 1,
                is_complex: false,
            }),
            Value::Complex(re, im) => Ok(Self {
                data: vec![Complex::new(re, im)],
                len: 1,
                is_complex: true,
            }),
            other => Err(runtime_error_for(format!(
                "{}: unsupported {} type {:?}; expected numeric or logical values",
                name, label, other
            ))),
        }
    }

    fn from_tensor(name: &str, label: &str, tensor: Tensor) -> BuiltinResult<Self> {
        ensure_vector_shape(name, label, &tensor.shape)?;
        let len = tensor.data.len();
        if len == 0 {
            return Err(runtime_error_for(format!(
                "{}: {} coefficients cannot be empty",
                name, label
            )));
        }
        let data = tensor
            .data
            .into_iter()
            .map(|re| Complex::new(re, 0.0))
            .collect();
        Ok(Self {
            data,
            len,
            is_complex: false,
        })
    }
}

struct SignalInput {
    data: Vec<Complex<f64>>,
    shape: Vec<usize>,
    is_complex: bool,
    gpu_handle: Option<GpuTensorHandle>,
}

impl SignalInput {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                let shape = tensor.shape.clone();
                let data = tensor
                    .data
                    .into_iter()
                    .map(|re| Complex::new(re, 0.0))
                    .collect();
                Ok(Self {
                    data,
                    shape,
                    is_complex: false,
                    gpu_handle: Some(handle),
                })
            }
            Value::Tensor(tensor) => {
                let shape = tensor.shape.clone();
                let data = tensor
                    .data
                    .into_iter()
                    .map(|re| Complex::new(re, 0.0))
                    .collect();
                Ok(Self {
                    data,
                    shape,
                    is_complex: false,
                    gpu_handle: None,
                })
            }
            Value::ComplexTensor(tensor) => {
                let shape = tensor.shape.clone();
                let data = tensor
                    .data
                    .into_iter()
                    .map(|(re, im)| Complex::new(re, im))
                    .collect();
                Ok(Self {
                    data,
                    shape,
                    is_complex: true,
                    gpu_handle: None,
                })
            }
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical)
                    .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
                let shape = tensor.shape.clone();
                let data = tensor
                    .data
                    .into_iter()
                    .map(|re| Complex::new(re, 0.0))
                    .collect();
                Ok(Self {
                    data,
                    shape,
                    is_complex: false,
                    gpu_handle: None,
                })
            }
            Value::Num(n) => Ok(Self {
                data: vec![Complex::new(n, 0.0)],
                shape: vec![1, 1],
                is_complex: false,
                gpu_handle: None,
            }),
            Value::Int(i) => Ok(Self {
                data: vec![Complex::new(i.to_f64(), 0.0)],
                shape: vec![1, 1],
                is_complex: false,
                gpu_handle: None,
            }),
            Value::Bool(b) => Ok(Self {
                data: vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)],
                shape: vec![1, 1],
                is_complex: false,
                gpu_handle: None,
            }),
            Value::Complex(re, im) => Ok(Self {
                data: vec![Complex::new(re, im)],
                shape: vec![1, 1],
                is_complex: true,
                gpu_handle: None,
            }),
            other => Err(runtime_error_for(format!(
                "filter: unsupported signal type {:?}; expected numeric or logical values",
                other
            ))),
        }
    }
}

struct InitialState {
    provided: bool,
    column_major: Vec<Complex<f64>>,
    shape: Vec<usize>,
    is_complex: bool,
    gpu_handle: Option<GpuTensorHandle>,
}

impl InitialState {
    fn empty(shape: Vec<usize>) -> Self {
        Self {
            provided: false,
            column_major: Vec::new(),
            shape,
            is_complex: false,
            gpu_handle: None,
        }
    }

    async fn from_value(
        value: Value,
        state_len: usize,
        expected_shape: &[usize],
        expected_states: usize,
        name: &str,
    ) -> BuiltinResult<Self> {
        if state_len == 0 {
            match value {
                Value::Tensor(tensor) if tensor.data.is_empty() => {
                    return Ok(Self::empty(expected_shape.to_vec()))
                }
                Value::GpuTensor(handle) => {
                    let tensor = gpu_helpers::gather_tensor_async(&handle)
                        .await
                        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                    if !tensor.data.is_empty() {
                        return Err(runtime_error_for(format!(
                            "{}: initial conditions must be empty when the filter order is zero",
                            name
                        )));
                    }
                    return Ok(Self {
                        provided: true,
                        column_major: Vec::new(),
                        shape: expected_shape.to_vec(),
                        is_complex: false,
                        gpu_handle: Some(handle),
                    });
                }
                Value::LogicalArray(logical) if logical.data.is_empty() => {
                    return Ok(Self::empty(expected_shape.to_vec()))
                }
                Value::ComplexTensor(tensor) if tensor.data.is_empty() => {
                    return Ok(Self {
                        provided: true,
                        column_major: Vec::new(),
                        shape: expected_shape.to_vec(),
                        is_complex: true,
                        gpu_handle: None,
                    });
                }
                other => {
                    let msg = format!(
                        "{}: initial conditions must be empty when the filter order is zero",
                        name
                    );
                    let detail = match other {
                        Value::Tensor(t)
                            if t.data.is_empty()
                                && !shapes_compatible(expected_shape, &t.shape) =>
                        {
                            msg.clone()
                        }
                        Value::Tensor(_) | Value::ComplexTensor(_) | Value::GpuTensor(_) => {
                            msg.clone()
                        }
                        _ => format!("{msg}; received {:?}", other),
                    };
                    return Err(runtime_error_for(detail));
                }
            }
        }

        let (column_major, shape, is_complex, gpu_handle) = match value {
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                (
                    tensor
                        .data
                        .iter()
                        .map(|&re| Complex::new(re, 0.0))
                        .collect::<Vec<_>>(),
                    tensor.shape.clone(),
                    false,
                    Some(handle),
                )
            }
            Value::Tensor(tensor) => (
                tensor
                    .data
                    .iter()
                    .map(|&re| Complex::new(re, 0.0))
                    .collect::<Vec<_>>(),
                tensor.shape.clone(),
                false,
                None,
            ),
            Value::ComplexTensor(tensor) => (
                tensor
                    .data
                    .iter()
                    .map(|&(re, im)| Complex::new(re, im))
                    .collect::<Vec<_>>(),
                tensor.shape.clone(),
                true,
                None,
            ),
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical)
                    .map_err(|e| runtime_error_for(format!("{name}: initial conditions: {e}")))?;
                (
                    tensor
                        .data
                        .iter()
                        .map(|&re| Complex::new(re, 0.0))
                        .collect::<Vec<_>>(),
                    tensor.shape.clone(),
                    false,
                    None,
                )
            }
            Value::Num(n) => (vec![Complex::new(n, 0.0)], vec![1, 1], false, None),
            Value::Int(i) => (vec![Complex::new(i.to_f64(), 0.0)], vec![1, 1], false, None),
            Value::Bool(b) => (
                vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)],
                vec![1, 1],
                false,
                None,
            ),
            Value::Complex(re, im) => (vec![Complex::new(re, im)], vec![1, 1], true, None),
            other => {
                return Err(runtime_error_for(format!(
                    "{name}: unsupported initial condition type {:?}; expected numeric values",
                    other
                )))
            }
        };

        if column_major.len() != expected_states {
            return Err(runtime_error_for(format!(
                "{name}: initial conditions have {} elements but {} were expected",
                column_major.len(),
                expected_states
            )));
        }

        if !shapes_compatible(expected_shape, &shape) {
            return Err(runtime_error_for(format!(
                "{name}: initial conditions must have shape {:?}, received {:?}",
                expected_shape, shape
            )));
        }

        Ok(Self {
            provided: true,
            column_major,
            shape: expected_shape.to_vec(),
            is_complex,
            gpu_handle,
        })
    }
}

fn parse_optional_arguments(rest: &[Value]) -> BuiltinResult<(Option<Value>, Option<Value>)> {
    match rest.len() {
        0 => Ok((None, None)),
        1 => Ok((Some(rest[0].clone()), None)),
        2 => Ok((Some(rest[0].clone()), Some(rest[1].clone()))),
        _ => Err(runtime_error_for(
            "filter: expected between three and five input arguments (b, a, x [, zi [, dim]])",
        )),
    }
}

fn is_empty_placeholder(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.data.is_empty(),
        Value::ComplexTensor(t) => t.data.is_empty(),
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::StringArray(sa) => sa.data.is_empty(),
        Value::CharArray(ca) => ca.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::String(s) => s.is_empty(),
        Value::Struct(st) => st.fields.is_empty(),
        _ => false,
    }
}

async fn try_filter_gpu(args: &FilterArgs) -> BuiltinResult<Option<FilterEvaluation>> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    if args.is_complex {
        return Ok(None);
    }

    let signal_handle = match &args.signal.gpu_handle {
        Some(handle) => handle.clone(),
        None => return Ok(None),
    };

    let b_real = match to_real_vec(&args.coeffs_b.data) {
        Some(vec) => vec,
        None => return Ok(None),
    };
    let a_real = match to_real_vec(&args.coeffs_a.data) {
        Some(vec) => vec,
        None => return Ok(None),
    };

    let mut temp_handles: Vec<GpuTensorHandle> = Vec::new();

    let b_shape = vec![args.coeffs_b.len, 1];
    let view_b = runmat_accelerate_api::HostTensorView {
        data: &b_real,
        shape: &b_shape,
    };
    let b_handle = provider.upload(&view_b).map_err(|e| {
        runtime_error_for(format!(
            "filter: failed to upload numerator coefficients: {e}"
        ))
    })?;
    temp_handles.push(b_handle.clone());

    let a_shape = vec![args.coeffs_a.len, 1];
    let view_a = runmat_accelerate_api::HostTensorView {
        data: &a_real,
        shape: &a_shape,
    };
    let a_handle = provider.upload(&view_a).map_err(|e| {
        runtime_error_for(format!(
            "filter: failed to upload denominator coefficients: {e}"
        ))
    })?;
    temp_handles.push(a_handle.clone());

    let (zi_handle_opt, zi_temp) = if args.state_len == 0 || !args.initial.provided {
        (None, None)
    } else if let Some(handle) = &args.initial.gpu_handle {
        (Some(handle.clone()), None)
    } else {
        let zi_real = match to_real_vec(&args.initial.column_major) {
            Some(vec) => vec,
            None => {
                cleanup_temp_handles(provider, temp_handles);
                return Ok(None);
            }
        };
        let view = runmat_accelerate_api::HostTensorView {
            data: &zi_real,
            shape: &args.initial.shape,
        };
        let handle = provider.upload(&view).map_err(|e| {
            cleanup_temp_handles(provider, temp_handles.clone());
            runtime_error_for(format!("filter: failed to upload initial conditions: {e}"))
        })?;
        (Some(handle.clone()), Some(handle))
    };

    if let Some(handle) = &zi_temp {
        temp_handles.push(handle.clone());
    }

    let options = ProviderIirFilterOptions {
        dim: args.dim_idx,
        zi: zi_handle_opt,
    };

    let result = provider
        .iir_filter(&b_handle, &a_handle, &signal_handle, options)
        .await;

    cleanup_temp_handles(provider, temp_handles);

    let result = match result {
        Ok(res) => res,
        Err(err) => {
            debug!(
                "filter: provider iir_filter failed ({err}); falling back to host implementation"
            );
            return Ok(None);
        }
    };

    let output_value = Value::GpuTensor(result.output.clone());
    let final_state_value = match result.final_state {
        Some(handle) => Value::GpuTensor(handle),
        None => {
            let zeros = Tensor::new(
                vec![0.0; tensor::element_count(&args.state_shape)],
                args.state_shape.clone(),
            )
            .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
            tensor::tensor_into_value(zeros)
        }
    };

    Ok(Some(FilterEvaluation {
        output: output_value,
        final_state: final_state_value,
    }))
}

fn cleanup_temp_handles(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    handles: Vec<GpuTensorHandle>,
) {
    for handle in handles {
        let _ = provider.free(&handle);
    }
}

fn filter_host(args: &FilterArgs) -> BuiltinResult<FilterEvaluation> {
    let mut b_norm = args.coeffs_b.data.clone();
    let mut a_norm = args.coeffs_a.data.clone();

    let a0 = a_norm[0];
    if a0 == Complex::new(0.0, 0.0) {
        return Err(runtime_error_for(
            "filter: denominator coefficient a(1) must be non-zero",
        ));
    }

    for coeff in &mut b_norm {
        *coeff /= a0;
    }
    for (idx, coeff) in a_norm.iter_mut().enumerate() {
        if idx == 0 {
            *coeff = Complex::new(1.0, 0.0);
        } else if idx < args.coeffs_a.len {
            *coeff /= a0;
        } else {
            *coeff = Complex::new(0.0, 0.0);
        }
    }
    if b_norm.len() < args.order {
        b_norm.resize(args.order, Complex::new(0.0, 0.0));
    }
    if a_norm.len() < args.order {
        a_norm.resize(args.order, Complex::new(0.0, 0.0));
    }

    let expected_states = args.state_len.saturating_mul(args.channel_count);

    let mut states = if args.state_len == 0 {
        Vec::<Complex<f64>>::new()
    } else if args.initial.provided {
        if args.initial.column_major.len() != expected_states {
            return Err(runtime_error_for(format!(
                "filter: initial conditions have {} elements but {} were expected",
                args.initial.column_major.len(),
                expected_states
            )));
        }
        states_from_column_major_complex(
            &args.initial.column_major,
            args.state_len,
            args.dim_idx,
            &args.shape_ext,
        )
    } else {
        vec![Complex::new(0.0, 0.0); expected_states]
    };

    let mut output = vec![Complex::new(0.0, 0.0); args.signal.data.len()];

    if args.state_len == 0 {
        let gain = b_norm[0];
        for (dst, &src) in output.iter_mut().zip(args.signal.data.iter()) {
            *dst = gain * src;
        }
    } else if args.channel_count == 0 || args.shape_ext[args.dim_idx] == 0 {
        // nothing to do, states already contain initial values.
    } else {
        let dim_len = args.shape_ext[args.dim_idx];
        for t in 0..args.trailing {
            let base = t
                .checked_mul(dim_len)
                .and_then(|v| v.checked_mul(args.leading))
                .ok_or_else(|| runtime_error_for("filter: index overflow during evaluation"))?;
            for l in 0..args.leading {
                let channel_idx = t
                    .checked_mul(args.leading)
                    .and_then(|v| v.checked_add(l))
                    .ok_or_else(|| runtime_error_for("filter: index overflow during evaluation"))?;
                if channel_idx >= args.channel_count {
                    continue;
                }
                let state_base = channel_idx
                    .checked_mul(args.state_len)
                    .ok_or_else(|| runtime_error_for("filter: state index overflow"))?;
                for step in 0..dim_len {
                    let idx = base
                        .checked_add(l)
                        .and_then(|v| v.checked_add(step.saturating_mul(args.leading)))
                        .ok_or_else(|| runtime_error_for("filter: signal index overflow"))?;
                    if idx >= args.signal.data.len() {
                        break;
                    }
                    let x_n = args.signal.data[idx];
                    let y = b_norm[0] * x_n + states[state_base];
                    output[idx] = y;
                    for i in 1..args.order {
                        let next_state = if i < args.state_len {
                            states[state_base + i]
                        } else {
                            Complex::new(0.0, 0.0)
                        };
                        let new_state = b_norm[i] * x_n + next_state - a_norm[i] * y;
                        states[state_base + i - 1] = new_state;
                    }
                }
            }
        }
    }

    let final_states_column = if args.state_len == 0 {
        Vec::<Complex<f64>>::new()
    } else {
        states_to_column_major_complex(&states, args.state_len, args.dim_idx, &args.shape_ext)
    };

    let output_value = if args.is_complex {
        let data: Vec<(f64, f64)> = output.iter().map(|c| (c.re, c.im)).collect();
        if data.len() == 1 {
            Value::Complex(data[0].0, data[0].1)
        } else {
            let tensor = ComplexTensor::new(data, args.signal.shape.clone())
                .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
            Value::ComplexTensor(tensor)
        }
    } else {
        let data: Vec<f64> = output.iter().map(|c| c.re).collect();
        let tensor = Tensor::new(data, args.signal.shape.clone())
            .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
        tensor::tensor_into_value(tensor)
    };

    let final_state_value = if args.state_len == 0 {
        let tensor = Tensor::new(
            vec![0.0; tensor::element_count(&args.state_shape)],
            args.state_shape.clone(),
        )
        .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
        tensor::tensor_into_value(tensor)
    } else if args.is_complex || args.initial.is_complex {
        let data: Vec<(f64, f64)> = final_states_column.iter().map(|c| (c.re, c.im)).collect();
        if data.len() == 1 {
            Value::Complex(data[0].0, data[0].1)
        } else {
            let tensor = ComplexTensor::new(data, args.state_shape.clone())
                .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
            Value::ComplexTensor(tensor)
        }
    } else {
        let data: Vec<f64> = final_states_column.iter().map(|c| c.re).collect();
        let tensor = Tensor::new(data, args.state_shape.clone())
            .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
        tensor::tensor_into_value(tensor)
    };

    Ok(FilterEvaluation {
        output: output_value,
        final_state: final_state_value,
    })
}

fn ensure_vector_shape(name: &str, label: &str, shape: &[usize]) -> BuiltinResult<()> {
    let non_singleton = shape.iter().copied().filter(|&d| d > 1).count();
    if non_singleton > 1 {
        Err(runtime_error_for(format!(
            "{}: {} coefficients must be a row or column vector",
            name, label
        )))
    } else {
        Ok(())
    }
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape
            .iter()
            .position(|&d| d > 1)
            .map(|idx| idx + 1)
            .unwrap_or(1)
    }
}

fn shapes_compatible(expected: &[usize], actual: &[usize]) -> bool {
    let max_len = expected.len().max(actual.len());
    for i in 0..max_len {
        let e = expected.get(i).copied().unwrap_or(1);
        let a = actual.get(i).copied().unwrap_or(1);
        if e != a {
            return false;
        }
    }
    true
}

fn filter_state_shape(mut base: Vec<usize>, dim_idx: usize, state_len: usize) -> Vec<usize> {
    if base.len() <= dim_idx {
        base.extend(std::iter::repeat_n(1, dim_idx + 1 - base.len()));
    }
    if !base.is_empty() {
        base[dim_idx] = state_len;
    }
    base
}

fn decode_indices(mut index: usize, dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut coords = Vec::with_capacity(dims.len());
    for &dim in dims {
        if dim == 0 {
            coords.push(0);
        } else {
            let coord = index % dim;
            coords.push(coord);
            index /= dim.max(1);
        }
    }
    coords
}

fn states_from_column_major_complex(
    data: &[Complex<f64>],
    state_len: usize,
    dim_idx: usize,
    shape_ext: &[usize],
) -> Vec<Complex<f64>> {
    if state_len == 0 {
        return Vec::new();
    }
    let dims_before = &shape_ext[..dim_idx];
    let dims_after = if dim_idx + 1 < shape_ext.len() {
        &shape_ext[dim_idx + 1..]
    } else {
        &[]
    };
    let leading = if dims_before.is_empty() {
        1
    } else {
        dims_before.iter().copied().product()
    };
    let trailing = if dims_after.is_empty() {
        1
    } else {
        dims_after.iter().copied().product()
    };
    let channel_count = leading * trailing;
    let shape = filter_state_shape(shape_ext.to_vec(), dim_idx, state_len);
    let mut states = vec![Complex::new(0.0, 0.0); state_len * channel_count];
    for channel in 0..channel_count {
        let before_idx = if dims_before.is_empty() {
            0
        } else {
            channel % leading
        };
        let after_idx = if dims_after.is_empty() {
            0
        } else {
            channel / leading
        };
        let before_coords = decode_indices(before_idx, dims_before);
        let after_coords = decode_indices(after_idx, dims_after);
        for s in 0..state_len {
            let mut offset = 0usize;
            let mut stride = 1usize;
            for (d, &extent) in shape.iter().enumerate() {
                let coord = if d < dim_idx {
                    before_coords.get(d).copied().unwrap_or(0)
                } else if d == dim_idx {
                    s
                } else {
                    let idx = d - dim_idx - 1;
                    after_coords.get(idx).copied().unwrap_or(0)
                };
                offset += coord * stride;
                stride *= extent;
            }
            states[channel * state_len + s] = data[offset];
        }
    }
    states
}

fn states_to_column_major_complex(
    states: &[Complex<f64>],
    state_len: usize,
    dim_idx: usize,
    shape_ext: &[usize],
) -> Vec<Complex<f64>> {
    if state_len == 0 {
        return Vec::new();
    }
    let dims_before = &shape_ext[..dim_idx];
    let dims_after = if dim_idx + 1 < shape_ext.len() {
        &shape_ext[dim_idx + 1..]
    } else {
        &[]
    };
    let leading = if dims_before.is_empty() {
        1
    } else {
        dims_before.iter().copied().product()
    };
    let trailing = if dims_after.is_empty() {
        1
    } else {
        dims_after.iter().copied().product()
    };
    let channel_count = leading * trailing;
    let shape = filter_state_shape(shape_ext.to_vec(), dim_idx, state_len);
    let mut out = vec![Complex::new(0.0, 0.0); states.len()];
    for channel in 0..channel_count {
        let before_idx = if dims_before.is_empty() {
            0
        } else {
            channel % leading
        };
        let after_idx = if dims_after.is_empty() {
            0
        } else {
            channel / leading
        };
        let before_coords = decode_indices(before_idx, dims_before);
        let after_coords = decode_indices(after_idx, dims_after);
        for s in 0..state_len {
            let mut offset = 0usize;
            let mut stride = 1usize;
            for (d, &extent) in shape.iter().enumerate() {
                let coord = if d < dim_idx {
                    before_coords.get(d).copied().unwrap_or(0)
                } else if d == dim_idx {
                    s
                } else {
                    let idx = d - dim_idx - 1;
                    after_coords.get(idx).copied().unwrap_or(0)
                };
                offset += coord * stride;
                stride *= extent;
            }
            out[offset] = states[channel * state_len + s];
        }
    }
    out
}

fn to_real_vec(data: &[Complex<f64>]) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(data.len());
    for c in data {
        if c.im.abs() > EPS {
            return None;
        }
        out.push(c.re);
    }
    Some(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, ResolveContext, Type};

    fn error_message(error: RuntimeError) -> String {
        error.message().to_string()
    }

    fn evaluate(b: Value, a: Value, x: Value, rest: &[Value]) -> BuiltinResult<FilterEvaluation> {
        block_on(super::evaluate(b, a, x, rest))
    }

    #[test]
    fn filter_type_preserves_signal_shape() {
        let out = filter_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(2)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(2)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(4)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(4)])
            }
        );
    }

    fn approx_eq_slice(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "length mismatch ({} vs {})",
            lhs.len(),
            rhs.len()
        );
        let tol = match runmat_accelerate_api::provider()
            .map(|p| p.precision())
            .unwrap_or(runmat_accelerate_api::ProviderPrecision::F64)
        {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-6,
        };
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < tol,
                "value mismatch at index {idx}: {a} vs {b} (diff {diff})"
            );
        }
    }

    fn approx_eq_complex(lhs: &[(f64, f64)], rhs: &[(f64, f64)]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, ((ar, ai), (br, bi))) in lhs.iter().zip(rhs.iter()).enumerate() {
            let dr = (ar - br).abs();
            let di = (ai - bi).abs();
            assert!(
                dr < 1e-9 && di < 1e-9,
                "complex mismatch at {idx}: ({ar},{ai}) vs ({br},{bi})"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter_fir_basic() {
        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new(vec![1.0, 5.0, 2.0, 0.0, 3.0], vec![1, 5]).unwrap();

        let eval =
            evaluate(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x), &[]).expect("filter");
        let (y, zf) = eval.clone().into_pair();

        let Tensor { data, .. } = match y {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };
        approx_eq_slice(
            &data,
            &[0.3333333333, 2.0, 2.6666666667, 2.3333333333, 1.6666666667],
        );

        let Tensor { data: z_data, .. } = match zf {
            Value::Tensor(t) => t,
            other => panic!("expected tensor final state, got {other:?}"),
        };
        approx_eq_slice(&z_data, &[1.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter_iir_impulse() {
        let alpha = 0.8;
        let b = Tensor::new(vec![1.0 - alpha], vec![1, 1]).unwrap();
        let a = Tensor::new(vec![1.0, -alpha], vec![1, 2]).unwrap();
        let x = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![1, 5]).unwrap();

        let eval =
            evaluate(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x), &[]).expect("filter");
        let (y, _) = eval.into_pair();

        let Tensor { data, .. } = match y {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };
        approx_eq_slice(&data, &[0.2, 0.16, 0.128, 0.1024, 0.08192]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter_with_initial_conditions() {
        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x1 = Tensor::new(vec![1.0, 5.0, 2.0], vec![1, 3]).unwrap();

        let eval1 = evaluate(
            Value::Tensor(b.clone()),
            Value::Tensor(a.clone()),
            Value::Tensor(x1),
            &[],
        )
        .expect("filter");
        let (y1, zf1) = eval1.clone().into_pair();
        let Tensor { data: y1_data, .. } = match y1 {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };
        approx_eq_slice(&y1_data, &[0.3333333333, 2.0, 2.6666666667]);

        let Tensor { data: zf_data, .. } = match zf1.clone() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor final state, got {other:?}"),
        };
        approx_eq_slice(&zf_data, &[2.3333333333, 0.6666666667]);

        let x2 = Tensor::new(vec![0.0, 3.0], vec![1, 2]).unwrap();
        let eval2 = evaluate(
            Value::Tensor(b),
            Value::Tensor(a),
            Value::Tensor(x2),
            &[zf1],
        )
        .expect("filter");
        let (y2, zf2) = eval2.into_pair();

        let Tensor { data: y2_data, .. } = match y2 {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };
        approx_eq_slice(&y2_data, &[2.3333333333, 1.6666666667]);

        let Tensor { data: zf2_data, .. } = match zf2 {
            Value::Tensor(t) => t,
            other => panic!("expected tensor final state, got {other:?}"),
        };
        approx_eq_slice(&zf2_data, &[1.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter_accepts_empty_initial_placeholder() {
        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new(vec![1.0, 5.0, 2.0, 0.0, 3.0], vec![1, 5]).unwrap();
        let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();

        let eval_default = evaluate(
            Value::Tensor(b.clone()),
            Value::Tensor(a.clone()),
            Value::Tensor(x.clone()),
            &[],
        )
        .expect("filter default");
        let eval_placeholder = evaluate(
            Value::Tensor(b),
            Value::Tensor(a),
            Value::Tensor(x),
            &[Value::Tensor(placeholder)],
        )
        .expect("filter with []");

        let (y_default, z_default) = eval_default.into_pair();
        let (y_placeholder, z_placeholder) = eval_placeholder.into_pair();

        let Tensor { data: y_def, .. } = match y_default {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };
        let Tensor { data: y_ph, .. } = match y_placeholder {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };
        approx_eq_slice(&y_def, &y_ph);

        let Tensor { data: z_def, .. } = match z_default {
            Value::Tensor(t) => t,
            other => panic!("expected tensor final state, got {other:?}"),
        };
        let Tensor { data: z_ph, .. } = match z_placeholder {
            Value::Tensor(t) => t,
            other => panic!("expected tensor final state, got {other:?}"),
        };
        approx_eq_slice(&z_def, &z_ph);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter_rows_with_dimension_argument() {
        let b = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new(vec![1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0], vec![2, 4]).unwrap();
        let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();

        let eval = evaluate(
            Value::Tensor(b),
            Value::Tensor(a),
            Value::Tensor(x),
            &[Value::Tensor(placeholder), Value::Int(IntValue::I32(2))],
        )
        .expect("filter");
        let (y, zf) = eval.into_pair();

        let Tensor { data, shape, .. } = match y {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };
        assert_eq!(shape, vec![2, 4]);
        approx_eq_slice(&data, &[1.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0]);

        let Tensor {
            data: state_data,
            shape: state_shape,
            ..
        } = match zf {
            Value::Tensor(t) => t,
            other => panic!("expected tensor final state, got {other:?}"),
        };
        assert_eq!(state_shape, vec![2, 1]);
        approx_eq_slice(&state_data, &[-4.0, -1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter_complex_signal() {
        let b = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 1.0)], vec![1, 2]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let t = [0.0, 1.0, 2.0, 3.0];
        let x: Vec<(f64, f64)> = t
            .iter()
            .map(|&ti| {
                let re = (std::f64::consts::PI * ti / 4.0).cos();
                let im = (std::f64::consts::PI * ti / 4.0).sin();
                (re, im)
            })
            .collect();
        let x_tensor = ComplexTensor::new(x, vec![1, 4]).unwrap();

        let eval = evaluate(
            Value::ComplexTensor(b),
            Value::Tensor(a),
            Value::ComplexTensor(x_tensor),
            &[],
        )
        .expect("filter");
        let (y, _) = eval.into_pair();

        let ComplexTensor { data, .. } = match y {
            Value::ComplexTensor(t) => t,
            other => panic!("expected complex tensor, got {other:?}"),
        };
        let root_half = std::f64::consts::FRAC_1_SQRT_2;
        let one_plus = 1.0 + root_half;
        approx_eq_complex(
            &data,
            &[
                (1.0, 0.0),
                (root_half, one_plus),
                (-root_half, one_plus),
                (-one_plus, root_half),
            ],
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter_invalid_initial_shape() {
        let b = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let zi = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).unwrap();

        let err = error_message(
            evaluate(
                Value::Tensor(b),
                Value::Tensor(a),
                Value::Tensor(x),
                &[Value::Tensor(zi)],
            )
            .unwrap_err(),
        );
        assert!(err.contains("initial conditions"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter_gpu_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
            let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let x = Tensor::new(vec![1.0, 5.0, 2.0, 0.0, 3.0], vec![1, 5]).unwrap();

            let view = runmat_accelerate_api::HostTensorView {
                data: &x.data,
                shape: &x.shape,
            };
            let x_gpu = provider.upload(&view).expect("upload signal");

            let eval_gpu = evaluate(
                Value::Tensor(b.clone()),
                Value::Tensor(a.clone()),
                Value::GpuTensor(x_gpu),
                &[],
            )
            .expect("gpu filter");
            let (y_gpu, _) = eval_gpu.into_pair();
            let gathered = test_support::gather(y_gpu).expect("gather");

            let eval_cpu = evaluate(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x), &[])
                .expect("cpu filter");
            let (y_cpu, _) = eval_cpu.into_pair();
            let Tensor { data: cpu_data, .. } = match y_cpu {
                Value::Tensor(t) => t,
                other => panic!("expected tensor, got {other:?}"),
            };

            approx_eq_slice(&gathered.data, &cpu_data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn filter_wgpu_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };

        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new(vec![1.0, 5.0, 2.0, 0.0, 3.0], vec![1, 5]).unwrap();

        let cpu_eval = evaluate(
            Value::Tensor(b.clone()),
            Value::Tensor(a.clone()),
            Value::Tensor(x.clone()),
            &[],
        )
        .expect("cpu filter");
        let (cpu_value, _) = cpu_eval.into_pair();
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };

        let view = runmat_accelerate_api::HostTensorView {
            data: &x.data,
            shape: &x.shape,
        };
        let x_gpu = provider.upload(&view).expect("upload signal");

        let gpu_eval = evaluate(
            Value::Tensor(b),
            Value::Tensor(a),
            Value::GpuTensor(x_gpu),
            &[],
        )
        .expect("wgpu filter");
        let (gpu_value, _) = gpu_eval.into_pair();
        let gathered = test_support::gather(gpu_value).expect("gather");

        approx_eq_slice(&gathered.data, &cpu_tensor.data);
    }
}
