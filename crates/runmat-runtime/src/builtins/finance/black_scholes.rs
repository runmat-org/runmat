//! Black-Scholes option pricing and implied volatility builtins.

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
    ProviderBlackScholesPriceInput, ProviderBlackScholesPriceRequest,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{broadcast, gpu_helpers, tensor as tensor_utils};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BLSPRICE: &str = "blsprice";
const BLSIMPV: &str = "blsimpv";
const DEFAULT_YIELD: f64 = 0.0;
const DEFAULT_LIMIT: f64 = 10.0;
const DEFAULT_TOLERANCE: f64 = 1e-6;
const MAX_IMPLIED_VOL_ITERATIONS: usize = 100;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

const OUTPUT_CALL_PUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Call",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "European call option price.",
    },
    BuiltinParamDescriptor {
        name: "Put",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "European put option price.",
    },
];

const OUTPUT_VOL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Volatility",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Implied annualized asset volatility.",
}];

const BLSPRICE_INPUTS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "Price",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Current underlying asset price.",
    },
    BuiltinParamDescriptor {
        name: "Strike",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exercise price.",
    },
    BuiltinParamDescriptor {
        name: "Rate",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Continuously compounded risk-free rate.",
    },
    BuiltinParamDescriptor {
        name: "Time",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time to expiration in years.",
    },
    BuiltinParamDescriptor {
        name: "Volatility",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annualized asset volatility.",
    },
    BuiltinParamDescriptor {
        name: "Yield",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Continuously compounded yield.",
    },
];

const BLSIMPV_INPUTS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "Price",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Current underlying asset price.",
    },
    BuiltinParamDescriptor {
        name: "Strike",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exercise price.",
    },
    BuiltinParamDescriptor {
        name: "Rate",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Continuously compounded risk-free rate.",
    },
    BuiltinParamDescriptor {
        name: "Time",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time to expiration in years.",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Observed option value.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options Limit, Yield, Tolerance, Class, and Method.",
    },
];

const BLSPRICE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[Call, Put] = blsprice(Price, Strike, Rate, Time, Volatility)",
        inputs: &BLSPRICE_INPUTS,
        outputs: &OUTPUT_CALL_PUT,
    },
    BuiltinSignatureDescriptor {
        label: "[Call, Put] = blsprice(Price, Strike, Rate, Time, Volatility, Yield)",
        inputs: &BLSPRICE_INPUTS,
        outputs: &OUTPUT_CALL_PUT,
    },
];

const BLSIMPV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Volatility = blsimpv(Price, Strike, Rate, Time, Value)",
        inputs: &BLSIMPV_INPUTS,
        outputs: &OUTPUT_VOL,
    },
    BuiltinSignatureDescriptor {
        label: "Volatility = blsimpv(Price, Strike, Rate, Time, Value, Name, Value)",
        inputs: &BLSIMPV_INPUTS,
        outputs: &OUTPUT_VOL,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACK_SCHOLES.INVALID_ARGUMENT",
    identifier: Some("RunMat:BlackScholes:InvalidArgument"),
    when: "Inputs, option values, or requested output count are invalid.",
    message: "Black-Scholes builtin: invalid argument",
};

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACK_SCHOLES.INVALID_INPUT",
    identifier: Some("RunMat:BlackScholes:InvalidInput"),
    when: "Inputs are nonnumeric, nonfinite, or have incompatible shapes.",
    message: "Black-Scholes builtin: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACK_SCHOLES.INTERNAL",
    identifier: Some("RunMat:BlackScholes:Internal"),
    when: "Runtime cannot materialize the requested output.",
    message: "Black-Scholes builtin: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const BLSPRICE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BLSPRICE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const BLSIMPV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BLSIMPV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::finance::black_scholes")]
pub const BLSPRICE_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BLSPRICE,
    op_kind: GpuOpKind::Custom("black-scholes-price"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("black_scholes_price")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Resident real gpuArray inputs use the provider black_scholes_price hook for vectorized call/put pricing; providers without the hook fall back to the host implementation.",
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::finance::black_scholes")]
pub const BLSIMPV_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BLSIMPV,
    op_kind: GpuOpKind::Custom("black-scholes-implied-volatility"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Implied volatility search is host control-flow and gathers gpuArray inputs before solving.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::finance::black_scholes")]
pub const BLSPRICE_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BLSPRICE,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "blsprice executes eagerly and terminates fusion planning.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::finance::black_scholes")]
pub const BLSIMPV_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BLSIMPV,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "blsimpv executes host-side root finding and terminates fusion planning.",
};

#[runtime_builtin(
    name = "blsprice",
    category = "finance",
    summary = "Compute Black-Scholes European call and put option prices.",
    keywords = "blsprice,Black-Scholes,option,finance,call,put",
    accel = "cpu",
    type_resolver(black_scholes_type),
    descriptor(crate::builtins::finance::black_scholes::BLSPRICE_DESCRIPTOR),
    builtin_path = "crate::builtins::finance::black_scholes"
)]
async fn blsprice_builtin(
    price: Value,
    strike: Value,
    rate: Value,
    time: Value,
    volatility: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if matches!(requested_outputs, Some(0)) {
        return Ok(Value::OutputList(Vec::new()));
    }
    if rest.len() > 1 {
        return Err(error(
            BLSPRICE,
            "blsprice: too many input arguments",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let yield_value = match rest.into_iter().next() {
        Some(value) if !is_empty_value(&value) => value,
        _ => Value::Num(DEFAULT_YIELD),
    };
    let raw_values = vec![price, strike, rate, time, volatility, yield_value];
    if raw_values
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        if let Some(eval) = try_blsprice_gpu(&raw_values).await? {
            return eval.output(requested_outputs);
        }
    }

    let values = gather_values(BLSPRICE, raw_values).await?;
    let inputs = BroadcastInputs::from_values(BLSPRICE, values)?;
    let mut call = Vec::with_capacity(inputs.len);
    let mut put = Vec::with_capacity(inputs.len);
    for idx in 0..inputs.len {
        let params = PriceParams {
            price: inputs.arg(0, idx),
            strike: inputs.arg(1, idx),
            rate: inputs.arg(2, idx),
            time: inputs.arg(3, idx),
            volatility: inputs.arg(4, idx),
            yield_rate: inputs.arg(5, idx),
        };
        let result = price_pair(params);
        call.push(result.call);
        put.push(result.put);
    }

    let call = output_value(call, inputs.shape.clone(), BLSPRICE)?;
    let put = output_value(put, inputs.shape, BLSPRICE)?;
    match requested_outputs {
        Some(1) => Ok(Value::OutputList(vec![call])),
        Some(2) => Ok(Value::OutputList(vec![call, put])),
        Some(_) => Err(error(
            BLSPRICE,
            "blsprice: too many output arguments",
            &ERROR_INVALID_ARGUMENT,
        )),
        None => Ok(call),
    }
}

#[runtime_builtin(
    name = "blsimpv",
    category = "finance",
    summary = "Compute Black-Scholes implied volatility.",
    keywords = "blsimpv,Black-Scholes,implied volatility,option,finance",
    accel = "cpu",
    type_resolver(black_scholes_type),
    descriptor(crate::builtins::finance::black_scholes::BLSIMPV_DESCRIPTOR),
    builtin_path = "crate::builtins::finance::black_scholes"
)]
async fn blsimpv_builtin(
    price: Value,
    strike: Value,
    rate: Value,
    time: Value,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let gathered_rest = gather_values(BLSIMPV, rest).await?;
    let options = ImpvOptions::parse(gathered_rest)?;
    let values = gather_values(
        BLSIMPV,
        vec![
            price,
            strike,
            rate,
            time,
            value,
            options.yield_value.clone(),
            options.class_value.clone(),
        ],
    )
    .await?;
    let mut numeric = values;
    let class_value = numeric.pop().expect("class value");
    let inputs = BroadcastInputs::from_values(BLSIMPV, numeric)?;
    let classes = ClassInput::from_value(class_value, &inputs.shape, inputs.len)?;
    let mut out = Vec::with_capacity(inputs.len);
    for idx in 0..inputs.len {
        let params = ImpvParams {
            price: inputs.arg(0, idx),
            strike: inputs.arg(1, idx),
            rate: inputs.arg(2, idx),
            time: inputs.arg(3, idx),
            option_value: inputs.arg(4, idx),
            yield_rate: inputs.arg(5, idx),
            class: classes.value(idx),
            limit: options.effective_limit(),
            tolerance: options.effective_tolerance(),
        };
        out.push(implied_volatility(params));
    }
    output_value(out, inputs.shape, BLSIMPV)
}

fn black_scholes_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::tensor(),
    }
}

#[derive(Clone)]
struct NumericInput {
    values: Vec<f64>,
    shape: Vec<usize>,
    aligned_shape: Vec<usize>,
    strides: Vec<usize>,
    is_scalar: bool,
}

struct BroadcastInputs {
    inputs: Vec<NumericInput>,
    shape: Vec<usize>,
    len: usize,
}

impl BroadcastInputs {
    fn from_values(builtin: &'static str, values: Vec<Value>) -> BuiltinResult<Self> {
        let mut inputs = values
            .into_iter()
            .map(|value| numeric_input(builtin, value))
            .collect::<BuiltinResult<Vec<_>>>()?;
        let mut shape = vec![1, 1];
        for input in &inputs {
            shape = broadcast::broadcast_shapes(builtin, &shape, &input.shape)
                .map_err(|err| error(builtin, format!("{builtin}: {err}"), &ERROR_INVALID_INPUT))?;
        }
        let len = shape_len_checked(builtin, &shape)?;
        for input in &mut inputs {
            input.aligned_shape = broadcast::align_shape(&input.shape, shape.len());
            input.strides = broadcast::compute_strides(&input.aligned_shape);
        }
        Ok(Self { inputs, shape, len })
    }

    fn arg(&self, input: usize, index: usize) -> f64 {
        let input = &self.inputs[input];
        if input.is_scalar {
            input.values[0]
        } else {
            let source_index = broadcast::broadcast_index(
                index,
                &self.shape,
                &input.aligned_shape,
                &input.strides,
            );
            input.values[source_index]
        }
    }
}

fn numeric_input(builtin: &'static str, value: Value) -> BuiltinResult<NumericInput> {
    let (values, shape) = match value {
        Value::Num(n) => (vec![n], vec![1, 1]),
        Value::Int(i) => (vec![i.to_f64()], vec![1, 1]),
        Value::Bool(flag) => (vec![if flag { 1.0 } else { 0.0 }], vec![1, 1]),
        Value::Tensor(tensor) => {
            let shape = tensor_shape(&tensor.shape, tensor.rows, tensor.cols);
            (tensor_utils::tensor_into_values_f64(tensor), shape)
        }
        Value::LogicalArray(logical) => {
            let shape = logical_shape(&logical.shape);
            let data = logical
                .data
                .into_iter()
                .map(|bit| if bit != 0 { 1.0 } else { 0.0 })
                .collect();
            (data, shape)
        }
        other => {
            return Err(error(
                builtin,
                format!("{builtin}: expected real numeric input, got {other:?}"),
                &ERROR_INVALID_INPUT,
            ));
        }
    };
    let is_scalar = values.len() == 1;
    let expected_len = shape_len_checked(builtin, &shape)?;
    if values.len() != expected_len {
        return Err(error(
            builtin,
            format!("{builtin}: numeric input data length does not match shape"),
            &ERROR_INVALID_INPUT,
        ));
    }
    Ok(NumericInput {
        values,
        shape,
        aligned_shape: Vec::new(),
        strides: Vec::new(),
        is_scalar,
    })
}

struct BlsPriceGpuEval {
    provider: &'static dyn AccelProvider,
    call: GpuTensorHandle,
    put: GpuTensorHandle,
}

impl BlsPriceGpuEval {
    fn output(self, requested_outputs: Option<usize>) -> BuiltinResult<Value> {
        match requested_outputs {
            Some(0) => {
                let _ = self.provider.free(&self.call);
                let _ = self.provider.free(&self.put);
                Ok(Value::OutputList(Vec::new()))
            }
            Some(1) => {
                let _ = self.provider.free(&self.put);
                Ok(Value::OutputList(vec![gpu_helpers::resident_gpu_value(
                    self.call,
                )]))
            }
            Some(2) => Ok(Value::OutputList(vec![
                gpu_helpers::resident_gpu_value(self.call),
                gpu_helpers::resident_gpu_value(self.put),
            ])),
            Some(_) => {
                let _ = self.provider.free(&self.call);
                let _ = self.provider.free(&self.put);
                Err(error(
                    BLSPRICE,
                    "blsprice: too many output arguments",
                    &ERROR_INVALID_ARGUMENT,
                ))
            }
            None => {
                let _ = self.provider.free(&self.put);
                Ok(gpu_helpers::resident_gpu_value(self.call))
            }
        }
    }
}

struct GpuPreparedNumericInput {
    handle: GpuTensorHandle,
    shape: Vec<usize>,
    aligned_shape: Vec<usize>,
    strides: Vec<usize>,
    temporary: bool,
}

impl GpuPreparedNumericInput {
    fn free_if_temporary(&self, provider: &'static dyn AccelProvider) {
        if self.temporary {
            let _ = provider.free(&self.handle);
        }
    }
}

async fn try_blsprice_gpu(values: &[Value]) -> BuiltinResult<Option<BlsPriceGpuEval>> {
    let Some(anchor) = values.iter().find_map(|value| match value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    }) else {
        return Ok(None);
    };

    let Some(provider) = runmat_accelerate_api::provider_for_handle(anchor) else {
        return Ok(None);
    };

    let mut prepared = Vec::with_capacity(values.len());
    for value in values {
        prepared.push(prepare_blsprice_gpu_input(provider, value)?);
    }

    let mut output_shape = vec![1, 1];
    for input in &prepared {
        output_shape = broadcast::broadcast_shapes(BLSPRICE, &output_shape, &input.shape)
            .map_err(|err| error(BLSPRICE, format!("{BLSPRICE}: {err}"), &ERROR_INVALID_INPUT))?;
    }
    let len = shape_len_checked(BLSPRICE, &output_shape)?;
    let rank = output_shape.len();
    for input in &mut prepared {
        input.aligned_shape = broadcast::align_shape(&input.shape, rank);
        input.strides = broadcast::compute_strides(&input.aligned_shape);
    }

    let request_inputs = prepared
        .iter()
        .map(|input| ProviderBlackScholesPriceInput {
            handle: &input.handle,
            shape: &input.aligned_shape,
            strides: &input.strides,
        })
        .collect::<Vec<_>>();
    let result = provider.black_scholes_price(&ProviderBlackScholesPriceRequest {
        inputs: &request_inputs,
        output_shape: &output_shape,
        len,
    });

    for input in &prepared {
        input.free_if_temporary(provider);
    }

    match result {
        Ok(result) => Ok(Some(BlsPriceGpuEval {
            provider,
            call: result.call,
            put: result.put,
        })),
        Err(_) => Ok(None),
    }
}

fn prepare_blsprice_gpu_input(
    provider: &'static dyn AccelProvider,
    value: &Value,
) -> BuiltinResult<GpuPreparedNumericInput> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
            {
                return Err(error(
                    BLSPRICE,
                    "blsprice: complex gpuArray inputs are not supported",
                    &ERROR_INVALID_INPUT,
                ));
            }
            Ok(GpuPreparedNumericInput {
                handle: handle.clone(),
                shape: canonical_shape(&handle.shape),
                aligned_shape: Vec::new(),
                strides: Vec::new(),
                temporary: false,
            })
        }
        other => {
            let numeric = numeric_input(BLSPRICE, other.clone())?;
            upload_blsprice_gpu_input(provider, numeric.values, numeric.shape)
        }
    }
}

fn upload_blsprice_gpu_input(
    provider: &'static dyn AccelProvider,
    data: Vec<f64>,
    shape: Vec<usize>,
) -> BuiltinResult<GpuPreparedNumericInput> {
    let handle = provider
        .upload(&HostTensorView {
            data: &data,
            shape: &shape,
        })
        .map_err(|err| {
            error(
                BLSPRICE,
                format!("blsprice: gpu upload failed: {err}"),
                &ERROR_INTERNAL,
            )
        })?;
    Ok(GpuPreparedNumericInput {
        handle,
        shape,
        aligned_shape: Vec::new(),
        strides: Vec::new(),
        temporary: true,
    })
}

fn tensor_shape(shape: &[usize], rows: usize, cols: usize) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else if shape.len() == 1 {
        vec![shape[0], 1]
    } else {
        let mut out = shape.to_vec();
        out[0] = rows;
        out[1] = cols;
        out
    }
}

fn logical_shape(shape: &[usize]) -> Vec<usize> {
    canonical_shape(shape)
}

fn canonical_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![shape[0], 1],
        _ => shape.to_vec(),
    }
}

fn shape_len_checked(builtin: &'static str, shape: &[usize]) -> BuiltinResult<usize> {
    shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, extent| acc.checked_mul(extent))
        .ok_or_else(|| {
            error(
                builtin,
                format!("{builtin}: output size exceeds runtime limits"),
                &ERROR_INVALID_INPUT,
            )
        })
}

fn output_value(
    values: Vec<f64>,
    shape: Vec<usize>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    if values.len() == 1 {
        Ok(Value::Num(values[0]))
    } else {
        Tensor::new(values, shape)
            .map(Value::Tensor)
            .map_err(|err| error(builtin, format!("{builtin}: {err}"), &ERROR_INTERNAL))
    }
}

async fn gather_values(builtin: &'static str, values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| with_context(builtin, err))?,
        );
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct PriceParams {
    price: f64,
    strike: f64,
    rate: f64,
    time: f64,
    volatility: f64,
    yield_rate: f64,
}

struct PriceResult {
    call: f64,
    put: f64,
}

fn price_pair(params: PriceParams) -> PriceResult {
    if !valid_price_domain(params) {
        return PriceResult {
            call: f64::NAN,
            put: f64::NAN,
        };
    }
    if params.time == 0.0 || params.volatility == 0.0 {
        let forward_price = params.price * (-params.yield_rate * params.time).exp();
        let discounted_strike = params.strike * (-params.rate * params.time).exp();
        return PriceResult {
            call: (forward_price - discounted_strike).max(0.0),
            put: (discounted_strike - forward_price).max(0.0),
        };
    }
    let sqrt_time = params.time.sqrt();
    let d1 = ((params.price / params.strike).ln()
        + (params.rate - params.yield_rate + 0.5 * params.volatility * params.volatility)
            * params.time)
        / (params.volatility * sqrt_time);
    let d2 = d1 - params.volatility * sqrt_time;
    let discounted_price = params.price * (-params.yield_rate * params.time).exp();
    let discounted_strike = params.strike * (-params.rate * params.time).exp();
    PriceResult {
        call: discounted_price * normcdf(d1) - discounted_strike * normcdf(d2),
        put: discounted_strike * normcdf(-d2) - discounted_price * normcdf(-d1),
    }
}

fn valid_price_domain(params: PriceParams) -> bool {
    params.price.is_finite()
        && params.strike.is_finite()
        && params.rate.is_finite()
        && params.time.is_finite()
        && params.volatility.is_finite()
        && params.yield_rate.is_finite()
        && params.price >= 0.0
        && params.strike > 0.0
        && params.time >= 0.0
        && params.volatility >= 0.0
}

#[derive(Clone, Copy)]
enum OptionClass {
    Call,
    Put,
}

struct ClassInput {
    values: Vec<OptionClass>,
    is_scalar: bool,
}

impl ClassInput {
    fn from_value(value: Value, shape: &[usize], len: usize) -> BuiltinResult<Self> {
        match value {
            Value::Bool(flag) => Ok(Self {
                values: vec![class_from_bool(flag)],
                is_scalar: true,
            }),
            Value::LogicalArray(logical) => {
                if logical.len() == 1 {
                    Ok(Self {
                        values: vec![class_from_bool(logical.data[0] != 0)],
                        is_scalar: true,
                    })
                } else if logical_shape(&logical.shape) == shape {
                    Ok(Self {
                        values: logical
                            .data
                            .into_iter()
                            .map(|bit| class_from_bool(bit != 0))
                            .collect(),
                        is_scalar: false,
                    })
                } else {
                    Err(error(
                        BLSIMPV,
                        "blsimpv: Class dimensions must match nonscalar inputs",
                        &ERROR_INVALID_INPUT,
                    ))
                }
            }
            Value::String(text) => Ok(Self {
                values: vec![parse_class_text(&text)?],
                is_scalar: true,
            }),
            Value::StringArray(array) => {
                let text_shape = string_array_shape(&array);
                Self::from_texts(array.data, text_shape, shape, len)
            }
            Value::CharArray(chars) => {
                if chars.rows == 1 {
                    Ok(Self {
                        values: vec![parse_class_text(&chars.data.iter().collect::<String>())?],
                        is_scalar: true,
                    })
                } else {
                    let texts = char_rows(chars);
                    Self::from_texts(texts, vec![len, 1], shape, len)
                }
            }
            Value::Cell(cell) => Self::from_cell(cell, shape, len),
            other => Err(error(
                BLSIMPV,
                format!("blsimpv: unsupported Class value {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            )),
        }
    }

    fn from_cell(cell: CellArray, shape: &[usize], len: usize) -> BuiltinResult<Self> {
        if cell.data.len() == 1 {
            return Ok(Self {
                values: vec![class_from_cell_value(&cell.data[0])?],
                is_scalar: true,
            });
        }
        if cell.shape != shape && tensor_shape(&cell.shape, cell.rows, cell.cols) != shape {
            return Err(error(
                BLSIMPV,
                "blsimpv: Class cell dimensions must match nonscalar inputs",
                &ERROR_INVALID_INPUT,
            ));
        }
        if cell.data.len() != len {
            return Err(error(
                BLSIMPV,
                "blsimpv: Class cell length must match nonscalar inputs",
                &ERROR_INVALID_INPUT,
            ));
        }
        let values = cell
            .data
            .iter()
            .map(class_from_cell_value)
            .collect::<BuiltinResult<Vec<_>>>()?;
        Ok(Self {
            values,
            is_scalar: false,
        })
    }

    fn from_texts(
        texts: Vec<String>,
        text_shape: Vec<usize>,
        shape: &[usize],
        len: usize,
    ) -> BuiltinResult<Self> {
        if texts.len() == 1 {
            return Ok(Self {
                values: vec![parse_class_text(&texts[0])?],
                is_scalar: true,
            });
        }
        if text_shape != shape || texts.len() != len {
            return Err(error(
                BLSIMPV,
                "blsimpv: Class string dimensions must match nonscalar inputs",
                &ERROR_INVALID_INPUT,
            ));
        }
        let values = texts
            .iter()
            .map(|text| parse_class_text(text))
            .collect::<BuiltinResult<Vec<_>>>()?;
        Ok(Self {
            values,
            is_scalar: false,
        })
    }

    fn value(&self, index: usize) -> OptionClass {
        if self.is_scalar {
            self.values[0]
        } else {
            self.values[index]
        }
    }
}

fn class_from_bool(flag: bool) -> OptionClass {
    if flag {
        OptionClass::Call
    } else {
        OptionClass::Put
    }
}

fn class_from_cell_value(value: &Value) -> BuiltinResult<OptionClass> {
    match value {
        Value::String(text) => parse_class_text(text),
        Value::CharArray(chars) if chars.rows == 1 => {
            parse_class_text(&chars.data.iter().collect::<String>())
        }
        Value::Bool(flag) => Ok(class_from_bool(*flag)),
        Value::LogicalArray(logical) if logical.len() == 1 => {
            Ok(class_from_bool(logical.data[0] != 0))
        }
        other => Err(error(
            BLSIMPV,
            format!("blsimpv: invalid Class cell value {other:?}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn parse_class_text(text: &str) -> BuiltinResult<OptionClass> {
    match text.trim().to_ascii_lowercase().as_str() {
        "call" => Ok(OptionClass::Call),
        "put" => Ok(OptionClass::Put),
        other => Err(error(
            BLSIMPV,
            format!("blsimpv: unsupported Class '{other}'"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn string_array_shape(array: &StringArray) -> Vec<usize> {
    tensor_shape(&array.shape, array.rows, array.cols)
}

fn char_rows(chars: CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(chars.rows);
    for row in 0..chars.rows {
        let mut text = String::new();
        for col in 0..chars.cols {
            text.push(chars.data[row + col * chars.rows]);
        }
        rows.push(text);
    }
    rows
}

struct ImpvOptions {
    limit: f64,
    tolerance: f64,
    yield_value: Value,
    class_value: Value,
    method: ImpvMethod,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImpvMethod {
    Jackel2016,
    Search,
}

impl ImpvOptions {
    fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(error(
                BLSIMPV,
                "blsimpv: name-value options must appear in pairs",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let mut options = Self {
            limit: DEFAULT_LIMIT,
            tolerance: DEFAULT_TOLERANCE,
            yield_value: Value::Num(DEFAULT_YIELD),
            class_value: Value::Bool(true),
            method: ImpvMethod::Jackel2016,
        };
        let mut idx = 0usize;
        while idx < args.len() {
            let name = text_scalar(BLSIMPV, &args[idx])?;
            let value = args[idx + 1].clone();
            match name.trim().to_ascii_lowercase().as_str() {
                "limit" => {
                    if is_empty_value(&value) {
                        options.limit = DEFAULT_LIMIT;
                        idx += 2;
                        continue;
                    }
                    options.limit = scalar_numeric(BLSIMPV, &value, "Limit")?;
                    if !options.limit.is_finite() || options.limit <= 0.0 {
                        return Err(error(
                            BLSIMPV,
                            "blsimpv: Limit must be a positive scalar",
                            &ERROR_INVALID_ARGUMENT,
                        ));
                    }
                }
                "tolerance" => {
                    if is_empty_value(&value) {
                        options.tolerance = DEFAULT_TOLERANCE;
                        idx += 2;
                        continue;
                    }
                    options.tolerance = scalar_numeric(BLSIMPV, &value, "Tolerance")?;
                    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
                        return Err(error(
                            BLSIMPV,
                            "blsimpv: Tolerance must be a positive scalar",
                            &ERROR_INVALID_ARGUMENT,
                        ));
                    }
                }
                "yield" => {
                    options.yield_value = if is_empty_value(&value) {
                        Value::Num(DEFAULT_YIELD)
                    } else {
                        value
                    };
                }
                "class" => {
                    options.class_value = if is_empty_value(&value) {
                        Value::Bool(true)
                    } else {
                        value
                    };
                }
                "method" => {
                    if is_empty_value(&value) {
                        options.method = ImpvMethod::Jackel2016;
                        idx += 2;
                        continue;
                    }
                    let method = text_scalar(BLSIMPV, &value)?;
                    match method.trim().to_ascii_lowercase().as_str() {
                        "search" => options.method = ImpvMethod::Search,
                        "jackel2016" => options.method = ImpvMethod::Jackel2016,
                        other => {
                            return Err(error(
                                BLSIMPV,
                                format!("blsimpv: unsupported Method '{other}'"),
                                &ERROR_INVALID_ARGUMENT,
                            ));
                        }
                    }
                }
                other => {
                    return Err(error(
                        BLSIMPV,
                        format!("blsimpv: unsupported option '{other}'"),
                        &ERROR_INVALID_ARGUMENT,
                    ));
                }
            }
            idx += 2;
        }
        Ok(options)
    }

    fn effective_limit(&self) -> f64 {
        match self.method {
            ImpvMethod::Jackel2016 => DEFAULT_LIMIT,
            ImpvMethod::Search => self.limit,
        }
    }

    fn effective_tolerance(&self) -> f64 {
        match self.method {
            ImpvMethod::Jackel2016 => DEFAULT_TOLERANCE,
            ImpvMethod::Search => self.tolerance,
        }
    }
}

#[derive(Clone, Copy)]
struct ImpvParams {
    price: f64,
    strike: f64,
    rate: f64,
    time: f64,
    option_value: f64,
    yield_rate: f64,
    class: OptionClass,
    limit: f64,
    tolerance: f64,
}

fn implied_volatility(params: ImpvParams) -> f64 {
    if !params.price.is_finite()
        || !params.strike.is_finite()
        || !params.rate.is_finite()
        || !params.time.is_finite()
        || !params.option_value.is_finite()
        || !params.yield_rate.is_finite()
        || params.price < 0.0
        || params.strike <= 0.0
        || params.time < 0.0
        || params.option_value < 0.0
    {
        return f64::NAN;
    }
    let target = params.option_value;
    let value_at_zero = option_value(params, 0.0);
    let value_at_limit = option_value(params, params.limit);
    if (value_at_zero - target).abs() <= params.tolerance {
        return 0.0;
    }
    if !value_at_limit.is_finite() || target < value_at_zero || target > value_at_limit {
        return f64::NAN;
    }
    let mut low = 0.0;
    let mut high = params.limit;
    for _ in 0..MAX_IMPLIED_VOL_ITERATIONS {
        let mid = 0.5 * (low + high);
        let value = option_value(params, mid);
        if !value.is_finite() {
            return f64::NAN;
        }
        let diff = value - target;
        if diff.abs() <= params.tolerance || (high - low).abs() <= params.tolerance {
            return mid;
        }
        if diff < 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

fn option_value(params: ImpvParams, volatility: f64) -> f64 {
    let result = price_pair(PriceParams {
        price: params.price,
        strike: params.strike,
        rate: params.rate,
        time: params.time,
        volatility,
        yield_rate: params.yield_rate,
    });
    match params.class {
        OptionClass::Call => result.call,
        OptionClass::Put => result.put,
    }
}

fn scalar_numeric(builtin: &'static str, value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(logical) if logical.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(error(
            builtin,
            format!("{builtin}: {name} must be a scalar numeric value"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor_utils::tensor_element_len(tensor) == 0,
        Value::StringArray(array) => array.data.is_empty(),
        Value::CharArray(chars) => chars.data.is_empty() || chars.rows == 0 || chars.cols == 0,
        Value::LogicalArray(logical) => logical.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        _ => false,
    }
}

fn text_scalar(builtin: &'static str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        _ => Err(error(
            builtin,
            format!("{builtin}: option names and text values must be scalar text"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn normcdf(value: f64) -> f64 {
    0.5 * (1.0 + libm::erf(value / SQRT_2))
}

fn error(
    builtin: &'static str,
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn with_context(builtin: &'static str, mut err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(builtin)
            .build();
    }
    if err.context.builtin.is_none() {
        err.context = err.context.with_builtin(builtin);
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new_2d(data, rows, cols).unwrap())
    }

    fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn empty_numeric() -> Value {
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap())
    }

    fn output_list(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    fn as_scalar(value: &Value) -> f64 {
        match value {
            Value::Num(n) => *n,
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    fn as_tensor(value: &Value) -> &Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn black_scholes_broadcast_indexing_appends_trailing_singletons() {
        let inputs = BroadcastInputs::from_values(
            BLSPRICE,
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                Value::Tensor(
                    Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 1, 3]).unwrap(),
                ),
            ],
        )
        .unwrap();
        assert_eq!(inputs.shape, vec![2, 1, 3]);
        assert_eq!(
            (0..inputs.len)
                .map(|index| inputs.arg(0, index))
                .collect::<Vec<_>>(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn blsprice_matches_documented_stock_option_example() {
        let guard = crate::output_count::push_output_count(Some(2));
        let out = output_list(
            block_on(blsprice_builtin(
                Value::Num(100.0),
                Value::Num(95.0),
                Value::Num(0.10),
                Value::Num(0.25),
                Value::Num(0.50),
                Vec::new(),
            ))
            .unwrap(),
        );
        drop(guard);
        assert!((as_scalar(&out[0]) - 13.6953).abs() < 5e-4);
        assert!((as_scalar(&out[1]) - 6.3497).abs() < 5e-4);
    }

    #[test]
    fn blsprice_vectorizes_scalar_inputs() {
        let guard = crate::output_count::push_output_count(Some(2));
        let out = output_list(
            block_on(blsprice_builtin(
                tensor(vec![100.0, 90.0], 2, 1),
                Value::Num(95.0),
                Value::Num(0.10),
                Value::Num(0.25),
                Value::Num(0.50),
                vec![Value::Num(0.0)],
            ))
            .unwrap(),
        );
        drop(guard);
        let call = as_tensor(&out[0]);
        let put = as_tensor(&out[1]);
        assert_eq!(call.shape, vec![2, 1]);
        assert_eq!(put.shape, vec![2, 1]);
        assert!(call.materialize_f64()[0] > call.materialize_f64()[1]);
        assert!(put.materialize_f64()[0] < put.materialize_f64()[1]);
    }

    #[test]
    fn blsprice_broadcast_inputs_read_typed_integer_storage_exactly() {
        let guard = crate::output_count::push_output_count(Some(2));
        let out = output_list(
            block_on(blsprice_builtin(
                poisoned_integer_tensor(IntegerStorage::U16(vec![100, 90]), vec![2, 1]),
                poisoned_integer_tensor(IntegerStorage::U16(vec![95]), vec![1, 1]),
                Value::Num(0.10),
                Value::Num(0.25),
                Value::Num(0.50),
                vec![poisoned_integer_tensor(
                    IntegerStorage::U8(vec![0]),
                    vec![1, 1],
                )],
            ))
            .unwrap(),
        );
        drop(guard);
        let call = as_tensor(&out[0]);
        let put = as_tensor(&out[1]);
        assert_eq!(call.shape, vec![2, 1]);
        assert_eq!(put.shape, vec![2, 1]);
        assert!(call.integer_storage().is_none());
        assert!(put.integer_storage().is_none());
        assert!(call.materialize_f64()[0] > call.materialize_f64()[1]);
        assert!(put.materialize_f64()[0] < put.materialize_f64()[1]);
    }

    #[test]
    fn blsprice_implicitly_expands_nonscalar_inputs() {
        let guard = crate::output_count::push_output_count(Some(2));
        let out = output_list(
            block_on(blsprice_builtin(
                tensor(vec![100.0, 110.0], 2, 1),
                tensor(vec![90.0, 100.0, 110.0], 1, 3),
                Value::Num(0.05),
                Value::Num(0.5),
                Value::Num(0.2),
                Vec::new(),
            ))
            .unwrap(),
        );
        drop(guard);

        let call = as_tensor(&out[0]);
        let put = as_tensor(&out[1]);
        assert_eq!(call.shape, vec![2, 3]);
        assert_eq!(put.shape, vec![2, 3]);
        for idx in 0..call.materialize_f64().len() {
            let price = [100.0, 110.0][idx % 2];
            let strike = [90.0, 100.0, 110.0][idx / 2];
            let expected = price_pair(PriceParams {
                price,
                strike,
                rate: 0.05,
                time: 0.5,
                volatility: 0.2,
                yield_rate: 0.0,
            });
            assert_close(call.materialize_f64()[idx], expected.call, 1e-12);
            assert_close(put.materialize_f64()[idx], expected.put, 1e-12);
        }
    }

    #[test]
    fn blsprice_gpu_resident_inputs_preserve_residency() {
        test_support::with_test_provider(|provider| {
            let price = Tensor::new(vec![100.0, 110.0], vec![2, 1]).unwrap();
            let strike = Tensor::new(vec![90.0, 100.0, 110.0], vec![1, 3]).unwrap();
            let rate = Tensor::new(vec![0.05], vec![1, 1]).unwrap();
            let time = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
            let volatility = Tensor::new(vec![0.2], vec![1, 1]).unwrap();
            let yield_rate = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let upload = |tensor: &Tensor| {
                provider
                    .upload(&HostTensorView {
                        data: &tensor.materialize_f64(),
                        shape: &tensor.shape,
                    })
                    .expect("upload")
            };
            let price_gpu = upload(&price);
            let strike_gpu = upload(&strike);
            let rate_gpu = upload(&rate);
            let time_gpu = upload(&time);
            let volatility_gpu = upload(&volatility);
            let yield_gpu = upload(&yield_rate);

            provider.reset_telemetry();
            let guard = crate::output_count::push_output_count(Some(2));
            let out = output_list(
                block_on(blsprice_builtin(
                    Value::GpuTensor(price_gpu.clone()),
                    Value::GpuTensor(strike_gpu.clone()),
                    Value::GpuTensor(rate_gpu.clone()),
                    Value::GpuTensor(time_gpu.clone()),
                    Value::GpuTensor(volatility_gpu.clone()),
                    vec![Value::GpuTensor(yield_gpu.clone())],
                ))
                .unwrap(),
            );
            drop(guard);

            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.upload_bytes, 0);
            assert_eq!(telemetry.download_bytes, 0);
            assert!(matches!(out[0], Value::GpuTensor(_)));
            assert!(matches!(out[1], Value::GpuTensor(_)));

            let call = test_support::gather(out[0].clone()).expect("gather call");
            let put = test_support::gather(out[1].clone()).expect("gather put");
            assert_eq!(call.shape, vec![2, 3]);
            assert_eq!(put.shape, vec![2, 3]);
            for idx in 0..call.materialize_f64().len() {
                let expected = price_pair(PriceParams {
                    price: price.materialize_f64()[idx % 2],
                    strike: strike.materialize_f64()[idx / 2],
                    rate: rate.materialize_f64()[0],
                    time: time.materialize_f64()[0],
                    volatility: volatility.materialize_f64()[0],
                    yield_rate: yield_rate.materialize_f64()[0],
                });
                assert_close(call.materialize_f64()[idx], expected.call, 1e-12);
                assert_close(put.materialize_f64()[idx], expected.put, 1e-12);
            }
        });
    }

    #[test]
    fn black_scholes_nan_inputs_propagate_elementwise() {
        let guard = crate::output_count::push_output_count(Some(2));
        let prices = output_list(
            block_on(blsprice_builtin(
                tensor(vec![100.0, f64::NAN], 2, 1),
                Value::Num(95.0),
                Value::Num(0.10),
                Value::Num(0.25),
                Value::Num(0.50),
                Vec::new(),
            ))
            .unwrap(),
        );
        drop(guard);
        let call = as_tensor(&prices[0]);
        assert!(call.materialize_f64()[0].is_finite());
        assert!(call.materialize_f64()[1].is_nan());

        let vol = block_on(blsimpv_builtin(
            tensor(vec![100.0, f64::NAN], 2, 1),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            tensor(vec![13.6953, 13.6953], 2, 1),
            Vec::new(),
        ))
        .unwrap();
        let vols = as_tensor(&vol);
        assert!(vols.materialize_f64()[0].is_finite());
        assert!(vols.materialize_f64()[1].is_nan());
    }

    #[test]
    fn blsprice_empty_yield_uses_default_zero_yield() {
        let with_empty_yield = block_on(blsprice_builtin(
            Value::Num(100.0),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            vec![empty_numeric()],
        ))
        .unwrap();
        let without_yield = block_on(blsprice_builtin(
            Value::Num(100.0),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            Vec::new(),
        ))
        .unwrap();
        assert!((as_scalar(&with_empty_yield) - as_scalar(&without_yield)).abs() < 1e-12);
    }

    #[test]
    fn blsimpv_inverts_call_price() {
        let call = block_on(blsprice_builtin(
            Value::Num(100.0),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            Vec::new(),
        ))
        .unwrap();
        let vol = block_on(blsimpv_builtin(
            Value::Num(100.0),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            call,
            Vec::new(),
        ))
        .unwrap();
        assert!((as_scalar(&vol) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn blsimpv_supports_put_class_and_yield() {
        let guard = crate::output_count::push_output_count(Some(2));
        let prices = output_list(
            block_on(blsprice_builtin(
                Value::Num(910.0),
                Value::Num(980.0),
                Value::Num(0.02),
                Value::Num(0.25),
                Value::Num(0.25),
                vec![Value::Num(0.025)],
            ))
            .unwrap(),
        );
        drop(guard);
        let vol = block_on(blsimpv_builtin(
            Value::Num(910.0),
            Value::Num(980.0),
            Value::Num(0.02),
            Value::Num(0.25),
            prices[1].clone(),
            vec![
                Value::from("Yield"),
                Value::Num(0.025),
                Value::from("Class"),
                Value::from("put"),
            ],
        ))
        .unwrap();
        assert!((as_scalar(&vol) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn blsimpv_vectorizes_class_strings() {
        let values = tensor(vec![13.6953, 6.3497], 2, 1);
        let classes = Value::StringArray(
            StringArray::new(vec!["call".to_string(), "put".to_string()], vec![2, 1]).unwrap(),
        );
        let out = block_on(blsimpv_builtin(
            tensor(vec![100.0, 100.0], 2, 1),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            values,
            vec![Value::from("Class"), classes],
        ))
        .unwrap();
        let vols = as_tensor(&out);
        assert_eq!(vols.shape, vec![2, 1]);
        assert!((vols.materialize_f64()[0] - 0.5).abs() < 1e-3);
        assert!((vols.materialize_f64()[1] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn blsimpv_returns_nan_when_no_solution_in_limit() {
        let out = block_on(blsimpv_builtin(
            Value::Num(100.0),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(50.0),
            vec![
                Value::from("Limit"),
                Value::Num(0.1),
                Value::from("Method"),
                Value::from("search"),
            ],
        ))
        .unwrap();
        assert!(as_scalar(&out).is_nan());
    }

    #[test]
    fn finance_scalar_numeric_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1])
            .expect("typed finance scalar");

        assert_eq!(
            scalar_numeric(BLSIMPV, &Value::Tensor(tensor), "Limit").expect("scalar numeric"),
            2026.0
        );
    }

    #[test]
    fn blsimpv_jackel_method_uses_default_bound_and_empty_defaults() {
        let call = block_on(blsprice_builtin(
            Value::Num(100.0),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            Vec::new(),
        ))
        .unwrap();
        let vol = block_on(blsimpv_builtin(
            Value::Num(100.0),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            call,
            vec![
                Value::from("Limit"),
                Value::Num(0.1),
                Value::from("Tolerance"),
                empty_numeric(),
                Value::from("Class"),
                empty_numeric(),
                Value::from("Method"),
                Value::from("jackel2016"),
            ],
        ))
        .unwrap();
        assert!((as_scalar(&vol) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn black_scholes_rejects_shape_mismatch_and_bad_options() {
        assert!(block_on(blsprice_builtin(
            tensor(vec![100.0, 90.0], 2, 1),
            tensor(vec![95.0, 96.0, 97.0], 3, 1),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            Vec::new(),
        ))
        .is_err());
        assert!(block_on(blsimpv_builtin(
            Value::Num(100.0),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(10.0),
            vec![Value::from("Method"), Value::from("unknown")],
        ))
        .is_err());
    }
}
