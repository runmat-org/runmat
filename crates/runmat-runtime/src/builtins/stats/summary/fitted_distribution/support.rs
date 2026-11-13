use super::*;

#[derive(Clone, Copy)]
pub(super) enum FittedOperation {
    Density,
    Cumulative,
    Inverse,
    Sampling,
}

#[derive(Clone, Copy)]
pub(super) enum CdfTail {
    Lower,
    Upper,
}

/// Retain every caller-owned resident evaluation operand until result adoption
/// completes. The first operand selects the existing output owner; the full set
/// is protected against a provider returning any borrowed allocation.
pub(super) struct EvaluationResidency {
    borrowed: Vec<GpuTensorHandle>,
}

impl EvaluationResidency {
    pub(super) fn capture(inputs: &[&Value]) -> Self {
        let borrowed = inputs
            .iter()
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            })
            .collect();
        Self { borrowed }
    }

    pub(super) fn source(&self) -> Option<&GpuTensorHandle> {
        self.borrowed.first()
    }

    pub(super) fn protected(&self) -> Vec<&GpuTensorHandle> {
        self.borrowed.iter().collect()
    }
}

pub(super) enum DistributionSelection {
    Fitted(FittedDistribution),
    Named(DistributionKind),
}

impl DistributionSelection {
    pub(super) fn from_value(operation: FittedOperation, value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::Object(_) => distribution_from_value_for(operation, value).map(Self::Fitted),
            _ => parse_distribution_name_for(operation.name(), value).map(Self::Named),
        }
    }

    pub(super) fn parameter_count(&self) -> usize {
        match self {
            Self::Fitted(_) => 0,
            Self::Named(kind) => kind.parameter_names().len(),
        }
    }
}

impl FittedOperation {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Density => PDF_NAME,
            Self::Cumulative => CDF_NAME,
            Self::Inverse => "icdf",
            Self::Sampling => RANDOM_NAME,
        }
    }
}

pub(super) fn check_fitted_cancellation(
    runtime: Option<&crate::context::RuntimeContext>,
    operation: FittedOperation,
) -> BuiltinResult<()> {
    if let Some(runtime) = runtime.filter(|runtime| runtime.is_cancelled()) {
        let name = operation.name();
        return Err(
            build_runtime_error(format!("{name}: execution was cancelled"))
                .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
                .with_builtin(name)
                .build(),
        );
    }
    Ok(())
}

pub(super) async fn parse_shape_args(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims = Vec::new();
    for arg in rest {
        match extract_dims(arg, RANDOM_NAME).await {
            Ok(Some(values)) => dims.extend(values),
            Ok(None) => return Err(invalid_for(RANDOM_NAME, "random: invalid size argument")),
            Err(err) => return Err(invalid_for(RANDOM_NAME, err)),
        }
    }
    let shape = normalize_dims(dims);
    checked_output_len(&shape)?;
    Ok(shape)
}

pub(super) fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        dims
    }
}

pub(super) fn checked_output_len(shape: &[usize]) -> BuiltinResult<usize> {
    crate::builtins::common::spec::checked_reduction_elements(shape)
        .ok_or_else(|| invalid_for(RANDOM_NAME, "random: output shape exceeds structural range"))
}

pub(super) fn finish_for(
    builtin: &'static str,
    shape: Vec<usize>,
    data: Vec<f64>,
) -> BuiltinResult<Value> {
    if crate::builtins::common::spec::checked_reduction_elements(&shape) == Some(1) {
        return Ok(Value::Num(data.first().copied().unwrap_or(0.0)));
    }
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_for(builtin, format!("{builtin}: {err}")))
}

pub(super) fn two(values: &[f64]) -> [f64; 2] {
    [values[0], values[1]]
}

pub(super) fn invert_positive(
    p: f64,
    initial_hi: f64,
    cdf: impl Fn(f64) -> f64,
    poll: impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<f64> {
    poll()?;
    if p == 0.0 {
        return Ok(0.0);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    let mut lo = 0.0;
    let mut hi = initial_hi.max(1.0);
    let mut iter = 0;
    while cdf(hi) < p {
        poll()?;
        hi *= 2.0;
        iter += 1;
        if !hi.is_finite() || iter > 2048 {
            return Ok(f64::INFINITY);
        }
    }
    for _ in 0..160 {
        poll()?;
        let mid = 0.5 * (lo + hi);
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

pub(super) fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

pub(super) fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 8.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + inv + 0.5 * inv2 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0
}
