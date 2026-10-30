//! Black-Scholes option pricing and implied volatility builtins.

#[cfg(test)]
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BLACK_SCHOLES_ERROR_INTERNAL as ERROR_INTERNAL,
    BLACK_SCHOLES_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    BLACK_SCHOLES_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT, BLSPRICE_NONDOUBLE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::finance::black_scholes")]
pub const BLSPRICE_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BLSPRICE,
    op_kind: GpuOpKind::Custom("black-scholes-price"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
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
    binding_variant = "default",
    builtin_path = "crate::builtins::finance::black_scholes"
)]
async fn blsprice_builtin(
    price: Value,
    strike: Value,
    rate: Value,
    time: Value,
    volatility: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if matches!(requested_outputs, Some(0)) {
        return Ok(Value::output_list(Vec::new()));
    }
    if matches!(requested_outputs, Some(count) if count > 2) {
        return Err(error(
            BLSPRICE,
            "blsprice: too many output arguments",
            &ERROR_INVALID_ARGUMENT,
        ));
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
    admit_nondouble_pricing_inputs(BLSPRICE, &BLSPRICE_NONDOUBLE_EXTENSION, raw_values.iter())?;
    admission::preflight_numeric_shapes(BLSPRICE, raw_values.iter())?;
    let result_class = precision::select(&raw_values);
    precision::preflight_host_scalars(&raw_values, result_class)?;
    if raw_values
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        if let Some(eval) = try_blsprice_gpu(&raw_values, result_class).await? {
            return eval.output(requested_outputs);
        }
    }

    let values = gather_values(BLSPRICE, raw_values).await?;
    let inputs = BroadcastInputs::from_values(BLSPRICE, values, result_class)?;
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

    let call = output_value(call, inputs.shape.clone(), result_class, BLSPRICE)?;
    let put = output_value(put, inputs.shape, result_class, BLSPRICE)?;
    match requested_outputs {
        Some(1) => Ok(Value::output_list(vec![call])),
        Some(2) => Ok(Value::output_list(vec![call, put])),
        Some(_) => Err(error(
            BLSPRICE,
            "blsprice: too many output arguments",
            &ERROR_INVALID_ARGUMENT,
        )),
        None => Ok(call),
    }
}

pub(in crate::builtins) mod implied_volatility;
#[cfg(test)]
use implied_volatility::blsimpv_builtin_source;

fn admit_nondouble_pricing_inputs<'a>(
    builtin: &'static str,
    extension: &'static BuiltinExtensionDescriptor,
    values: impl IntoIterator<Item = &'a Value>,
) -> BuiltinResult<()> {
    let nondouble = values.into_iter().any(|value| match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => tensor.numeric_dtype() != NumericDType::F64,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    });
    if nondouble {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
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
    fn from_values(
        builtin: &'static str,
        values: impl IntoIterator<Item = Value>,
        class: precision::PricingClass,
    ) -> BuiltinResult<Self> {
        let mut inputs = values
            .into_iter()
            .map(|value| {
                let mut input = numeric_input(builtin, value)?;
                if class == precision::PricingClass::Single {
                    for number in &mut input.values {
                        *number = precision::single_input(*number)?;
                    }
                }
                Ok(input)
            })
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
            let shape = canonical_shape(tensor.shape());
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

mod admission;
mod precision;
mod resident;
use resident::try_blsprice_gpu;

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
    class: precision::PricingClass,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match class {
        precision::PricingClass::Double if values.len() == 1 => Ok(Value::Num(values[0])),
        precision::PricingClass::Double => Tensor::new(values, shape)
            .map(Value::Tensor)
            .map_err(|err| error(builtin, format!("{builtin}: {err}"), &ERROR_INTERNAL)),
        precision::PricingClass::Single => Tensor::from_f32(
            values.into_iter().map(|value| value as f32).collect(),
            shape,
        )
        .map(Value::Tensor)
        .map_err(|err| error(builtin, format!("{builtin}: {err}"), &ERROR_INTERNAL)),
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
        Value::StringArray(array) => array.data().is_empty(),
        Value::CharArray(chars) => chars.data().is_empty() || chars.rows == 0 || chars.cols == 0,
        Value::LogicalArray(logical) => logical.data.is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        _ => false,
    }
}

fn text_scalar(builtin: &'static str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        Value::CharArray(chars) if chars.rows == 1 => Ok(runmat_value::decode_utf16_code_units(
            chars.data(),
        )
        .map_err(|source| {
            let mut error = error(builtin, source.to_string(), &ERROR_INVALID_ARGUMENT);
            error.source = Some(Box::new(source));
            error
        })?),
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
mod tests;
