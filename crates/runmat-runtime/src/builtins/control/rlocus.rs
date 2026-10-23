//! Root-locus analysis for SISO transfer-function models.

use num_complex::Complex64;
#[cfg(test)]
use runmat_builtins::RLOCUS_DESCRIPTOR;
#[cfg(feature = "plot-core")]
use runmat_builtins::RLOCUS_ERROR_PLOT_FAILED;
use runmat_builtins::{
    BuiltinErrorDescriptor, RLOCUS_ERROR_INTERNAL, RLOCUS_ERROR_INVALID_ARGUMENT,
    RLOCUS_ERROR_INVALID_MODEL, RLOCUS_ERROR_UNSUPPORTED_MODEL, RLOCUS_INTEGER_GAIN_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::{
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::control::model_polynomial::ss_object_to_tf_async;
use crate::builtins::control::tf_model::{
    poly_eval, polynomial_roots, TfModel, EPS, SS_CLASS, TF_CLASS,
};
#[cfg(feature = "plot-core")]
use crate::builtins::plotting::style::{parse_line_style_args, LineStyleParseOptions};
#[cfg(feature = "plot-core")]
use crate::plotting_hooks::{presentation_failed, with_render_cause};
use crate::{BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "rlocus";
const DEFAULT_GAIN_POINTS: usize = 121;
const DEFAULT_GAIN_DECADES: f64 = 4.0;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::rlocus")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rlocus",
    op_kind: GpuOpKind::Custom("control-root-locus"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "rlocus computes closed-loop polynomial roots on the host from transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::rlocus")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rlocus",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "rlocus is model analysis and plotting; it terminates numeric fusion chains.",
};

fn rlocus_error(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "rlocus",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::rlocus"
)]
async fn rlocus_builtin(sys: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&sys)
        || matches!(sys, Value::GpuTensor(_))
    {
        return Err(rlocus_error(
            "rlocus: expected a SISO dynamic system model",
            &RLOCUS_ERROR_INVALID_MODEL,
        ));
    }
    for value in &rest {
        if crate::builtins::common::validation::value_has_native_integer_class(value) {
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                value,
                &RLOCUS_INTEGER_GAIN_EXTENSION,
                BUILTIN_NAME,
                "gain",
            )
            .await?;
        }
    }
    if is_statement_form_call() {
        plot_root_locus_statement(sys, rest).await?;
        return Ok(Value::output_list(Vec::new()));
    }

    if rest.len() > 1 {
        return Err(rlocus_error(
            "rlocus: expected rlocus(sys) or rlocus(sys, k)",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }

    let gains = match rest.first() {
        Some(value) => Some(parse_gain_arg(value).await?),
        None => None,
    };
    let model = DynamicModel::from_value_async(sys).await?;
    let eval = RootLocus::compute(&model.tf, gains)?;

    if crate::output_context::requested_output_count() == Some(0)
        && crate::output_count::current_output_count().is_none()
    {
        render_root_locus_plot(&eval, None).await?;
        return Ok(Value::output_list(Vec::new()));
    }

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            render_root_locus_plot(&eval, None).await?;
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.roots_value()?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs()?,
        ));
    }

    eval.roots_value()
}

fn is_statement_form_call() -> bool {
    matches!(crate::output_count::current_output_count(), Some(0))
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
}

#[cfg(feature = "plot-core")]
async fn plot_root_locus_statement(
    first_sys: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<()> {
    let mut systems = vec![(first_sys, None)];
    let mut gains = None;

    for arg in rest {
        if is_plot_style_arg(&arg) {
            if let Some((_, style)) = systems.last_mut() {
                if style.is_some() {
                    return Err(rlocus_error(
                        "rlocus: only one style argument is supported per system",
                        &RLOCUS_ERROR_INVALID_ARGUMENT,
                    ));
                }
                parse_line_style_args(
                    std::slice::from_ref(&arg),
                    &LineStyleParseOptions::generic(BUILTIN_NAME),
                )
                .map_err(|err| {
                    rlocus_error(err.message().to_string(), &RLOCUS_ERROR_INVALID_ARGUMENT)
                })?;
                *style = Some(arg);
                continue;
            }
        }
        if is_dynamic_model_object(&arg) {
            if gains.is_some() {
                return Err(rlocus_error(
                    "rlocus: gain vector must follow all systems in statement-form plots",
                    &RLOCUS_ERROR_INVALID_ARGUMENT,
                ));
            }
            systems.push((arg, None));
            continue;
        }
        if is_numeric_vector_like(&arg) && gains.is_none() {
            gains = Some(parse_gain_arg(&arg).await?);
            continue;
        }
        return Err(rlocus_error(
            "rlocus: unsupported statement-form plot argument",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }

    // Resolve every model before mutating graphics state. A malformed later
    // system must not leave a partially rendered root-locus plot behind.
    let mut prepared = Vec::with_capacity(systems.len());
    let mut focus = Vec::new();
    for (system, style) in systems {
        let model = DynamicModel::from_value_async(system).await?;
        let eval = RootLocus::compute(&model.tf, gains.clone())?;
        if gains.is_none() {
            focus.extend(eval.focus_points(&model.tf));
        }
        prepared.push((eval, style));
    }
    let mut first = true;
    let mut hold_guard = HoldOffGuard::new();
    for (eval, style) in prepared {
        render_root_locus_plot(&eval, style.as_ref()).await?;
        if first {
            first = false;
            let _ = crate::call_builtin_async("hold", &[Value::from("on")]).await;
            hold_guard.arm();
        }
    }
    // With default gains the locus runs four decades past the interesting region; frame the
    // poles, zeros, breakaway points and imaginary-axis crossings instead, as MATLAB does.
    // A caller-supplied gain vector keeps the full extent of what was asked for.
    if !focus.is_empty() {
        super::frame_complex_plane(&focus, 0.5).await;
    }
    Ok(())
}

#[cfg(not(feature = "plot-core"))]
async fn plot_root_locus_statement(
    _first_sys: Value,
    _rest: crate::call::BuiltinArguments,
) -> BuiltinResult<()> {
    Err(crate::plotting_hooks::unavailable_presentation())
}

#[cfg(feature = "plot-core")]
struct HoldOffGuard {
    armed: bool,
    previous_hold_enabled: bool,
}

#[cfg(feature = "plot-core")]
impl HoldOffGuard {
    fn new() -> Self {
        Self {
            armed: false,
            previous_hold_enabled: crate::builtins::plotting::state::current_hold_enabled(),
        }
    }

    fn arm(&mut self) {
        self.armed = true;
    }
}

#[cfg(feature = "plot-core")]
impl Drop for HoldOffGuard {
    fn drop(&mut self) {
        if self.armed {
            let mode = if self.previous_hold_enabled {
                crate::builtins::plotting::HoldMode::On
            } else {
                crate::builtins::plotting::HoldMode::Off
            };
            crate::builtins::plotting::set_hold(mode);
        }
    }
}

#[cfg(feature = "plot-core")]
fn is_dynamic_model_object(value: &Value) -> bool {
    matches!(value, Value::Object(object) if object.is_class(runmat_types::standard::TRANSFER_FUNCTION) || object.is_class(runmat_types::standard::STATE_SPACE))
}

#[cfg(feature = "plot-core")]
fn is_plot_style_arg(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

#[cfg(feature = "plot-core")]
fn is_numeric_vector_like(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => true,
        Value::Tensor(tensor) => is_vector_shape(tensor.shape()),
        Value::ComplexTensor(tensor) => is_vector_shape(tensor.shape()),
        Value::LogicalArray(logical) => is_vector_shape(&logical.shape),
        Value::GpuTensor(handle) => is_vector_shape(handle.shape()),
        _ => false,
    }
}

async fn parse_gain_arg(value: &Value) -> BuiltinResult<Vec<f64>> {
    let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
    real_gain_vector(gathered)
}

fn real_gain_vector(value: Value) -> BuiltinResult<Vec<f64>> {
    let gains = match value {
        Value::Num(n) => vec![n],
        Value::Int(i) => vec![i.to_f64()],
        Value::Bool(b) => vec![if b { 1.0 } else { 0.0 }],
        Value::Complex(re, 0.0) => vec![re],
        Value::Tensor(tensor) => {
            ensure_vector_shape(tensor.shape())?;
            tensor::tensor_values_f64(&tensor)
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(tensor.shape())?;
            tensor
                .materialize_f64()
                .into_iter()
                .map(|(re, im)| {
                    if im == 0.0 {
                        Ok(re)
                    } else {
                        Err(rlocus_error(
                            "rlocus: gain vector must be real",
                            &RLOCUS_ERROR_INVALID_ARGUMENT,
                        ))
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?
        }
        Value::LogicalArray(logical) => {
            ensure_vector_shape(&logical.shape)?;
            logical
                .data
                .into_iter()
                .map(|value| if value == 0 { 0.0 } else { 1.0 })
                .collect()
        }
        other => {
            return Err(rlocus_error(
                format!("rlocus: k must be a real numeric vector, got {other:?}"),
                &RLOCUS_ERROR_INVALID_ARGUMENT,
            ));
        }
    };
    validate_gains(gains)
}

fn ensure_vector_shape(shape: &[usize]) -> BuiltinResult<()> {
    if is_vector_shape(shape) {
        Ok(())
    } else {
        Err(rlocus_error(
            "rlocus: k must be a vector",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ))
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().copied().filter(|&dim| dim > 1).count() <= 1
}

fn validate_gains(gains: Vec<f64>) -> BuiltinResult<Vec<f64>> {
    if gains.is_empty() {
        return Err(rlocus_error(
            "rlocus: k must not be empty",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }
    if gains.iter().any(|gain| !gain.is_finite()) {
        return Err(rlocus_error(
            "rlocus: k values must be finite",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }
    if gains.iter().any(|gain| *gain < 0.0) {
        return Err(rlocus_error(
            "rlocus: k values must be nonnegative",
            &RLOCUS_ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(gains)
}

#[derive(Clone, Debug)]
struct RootLocus {
    gains: Vec<f64>,
    roots: Vec<Complex64>,
    branches: usize,
    #[cfg(feature = "plot-core")]
    poles: Vec<Complex64>,
    #[cfg(feature = "plot-core")]
    zeros: Vec<Complex64>,
}

#[derive(Clone, Debug)]
struct DynamicModel {
    tf: TfModel,
}

impl DynamicModel {
    async fn from_value_async(value: Value) -> BuiltinResult<Self> {
        let Value::Object(object) = value else {
            return Err(rlocus_error(
                "rlocus: expected a SISO dynamic system model",
                &RLOCUS_ERROR_INVALID_MODEL,
            ));
        };
        if object.is_class(TF_CLASS) {
            return Ok(Self {
                tf: TfModel::from_value_async(Value::Object(object), BUILTIN_NAME).await?,
            });
        }
        if object.is_class(SS_CLASS) {
            return Ok(Self {
                tf: ss_object_to_tf_async(object, BUILTIN_NAME).await?,
            });
        }
        Err(rlocus_error(
            format!(
                "rlocus: unsupported model class '{}'; expected SISO tf or ss",
                object.class_name()
            ),
            &RLOCUS_ERROR_UNSUPPORTED_MODEL,
        ))
    }
}

impl RootLocus {
    fn compute(model: &TfModel, gains: Option<Vec<f64>>) -> BuiltinResult<Self> {
        ensure_supported_model(model)?;
        let physical = model.physical_polynomials();
        let numerator = physical.numerator();
        let denominator = physical.denominator();
        let gains = gains.unwrap_or_else(|| default_gains(numerator, denominator));
        let branches = characteristic_branch_count(numerator, denominator);
        #[cfg(feature = "plot-core")]
        let poles = polynomial_roots(denominator, BUILTIN_NAME)?;
        #[cfg(feature = "plot-core")]
        let zeros = if max_norm(numerator) <= EPS {
            Vec::new()
        } else {
            polynomial_roots(numerator, BUILTIN_NAME)?
        };

        let mut previous: Option<Vec<Complex64>> = None;
        let mut roots = Vec::with_capacity(branches.saturating_mul(gains.len()));
        for gain in &gains {
            let mut column = roots_for_gain(numerator, denominator, *gain, branches)?;
            column = match previous.as_ref() {
                Some(prev) => track_roots(prev, column),
                None => sort_roots(column),
            };
            previous = Some(column.clone());
            roots.extend(column);
        }

        Ok(Self {
            gains,
            roots,
            branches,
            #[cfg(feature = "plot-core")]
            poles,
            #[cfg(feature = "plot-core")]
            zeros,
        })
    }

    /// Open-loop poles and zeros, breakaway points (double roots at the critical gains) and the
    /// points where a branch crosses the imaginary axis.
    #[cfg(feature = "plot-core")]
    fn focus_points(&self, model: &TfModel) -> Vec<Complex64> {
        let physical = model.physical_polynomials();
        let (numerator, denominator) = (physical.numerator(), physical.denominator());
        let mut points: Vec<Complex64> = self
            .poles
            .iter()
            .chain(self.zeros.iter())
            .copied()
            .collect();
        let scale = points.iter().map(|p| p.norm()).fold(1.0_f64, f64::max);
        for gain in critical_gains(numerator, denominator) {
            let Ok(roots) = roots_for_gain(numerator, denominator, gain, self.branches) else {
                continue;
            };
            for (idx, root) in roots.iter().enumerate() {
                let double = roots
                    .iter()
                    .enumerate()
                    .any(|(other, r)| other != idx && (r - root).norm() <= 1.0e-3 * scale);
                if double {
                    points.push(*root);
                }
            }
        }
        for branch in 0..self.branches {
            for gain_idx in 1..self.gains.len() {
                let a = self.root(branch, gain_idx - 1);
                let b = self.root(branch, gain_idx);
                if a.re.is_finite()
                    && b.re.is_finite()
                    && a.re.signum() != b.re.signum()
                    && a.re != b.re
                {
                    let t = a.re / (a.re - b.re);
                    points.push(Complex64::new(0.0, a.im + t * (b.im - a.im)));
                }
            }
        }
        points
    }

    fn outputs(&self) -> BuiltinResult<Vec<Value>> {
        Ok(vec![self.roots_value()?, self.gains_value()?])
    }

    fn roots_value(&self) -> BuiltinResult<Value> {
        let shape = vec![self.branches, self.gains.len()];
        if self.roots.iter().all(|root| root.im.abs() <= EPS) {
            let data = self.roots.iter().map(|root| root.re).collect::<Vec<_>>();
            Tensor::new(data, shape).map(Value::Tensor).map_err(|err| {
                rlocus_error(
                    format!("rlocus: failed to build root matrix: {err}"),
                    &RLOCUS_ERROR_INTERNAL,
                )
            })
        } else {
            let data = self
                .roots
                .iter()
                .map(|root| (root.re, root.im))
                .collect::<Vec<_>>();
            ComplexTensor::new(data, shape)
                .map(Value::ComplexTensor)
                .map_err(|err| {
                    rlocus_error(
                        format!("rlocus: failed to build complex root matrix: {err}"),
                        &RLOCUS_ERROR_INTERNAL,
                    )
                })
        }
    }

    fn gains_value(&self) -> BuiltinResult<Value> {
        Tensor::new(self.gains.clone(), vec![1, self.gains.len()])
            .map(Value::Tensor)
            .map_err(|err| {
                rlocus_error(
                    format!("rlocus: failed to build gain vector: {err}"),
                    &RLOCUS_ERROR_INTERNAL,
                )
            })
    }

    #[cfg(feature = "plot-core")]
    fn root(&self, branch: usize, gain_index: usize) -> Complex64 {
        self.roots[branch + gain_index * self.branches]
    }
}

fn ensure_supported_model(model: &TfModel) -> BuiltinResult<()> {
    if model.input_delay != 0.0 || model.output_delay != 0.0 {
        return Err(rlocus_error(
            "rlocus: transfer functions with input or output delays are not supported",
            &RLOCUS_ERROR_UNSUPPORTED_MODEL,
        ));
    }
    Ok(())
}

fn characteristic_branch_count(numerator: &[Complex64], denominator: &[Complex64]) -> usize {
    denominator.len().max(numerator.len()).saturating_sub(1)
}

fn roots_for_gain(
    numerator: &[Complex64],
    denominator: &[Complex64],
    gain: f64,
    branches: usize,
) -> BuiltinResult<Vec<Complex64>> {
    let polynomial = characteristic_polynomial(numerator, denominator, gain);
    let mut roots = polynomial_roots(&polynomial, BUILTIN_NAME)?;
    if roots.len() > branches {
        return Err(rlocus_error(
            "rlocus: root calculation returned more roots than characteristic branches",
            &RLOCUS_ERROR_INTERNAL,
        ));
    }
    roots.resize(branches, Complex64::new(f64::INFINITY, 0.0));
    Ok(roots)
}

fn characteristic_polynomial(
    numerator: &[Complex64],
    denominator: &[Complex64],
    gain: f64,
) -> Vec<Complex64> {
    let len = denominator.len().max(numerator.len()).max(1);
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    let denominator_offset = len - denominator.len();
    for (idx, coeff) in denominator.iter().enumerate() {
        out[denominator_offset + idx] += *coeff;
    }
    let numerator_offset = len - numerator.len();
    let gain = Complex64::new(gain, 0.0);
    for (idx, coeff) in numerator.iter().enumerate() {
        out[numerator_offset + idx] += gain * *coeff;
    }
    out
}

fn default_gains(numerator: &[Complex64], denominator: &[Complex64]) -> Vec<f64> {
    let numerator_scale = max_norm(numerator);
    if numerator_scale <= EPS {
        return vec![0.0];
    }
    let denominator_scale = max_norm(denominator).max(1.0);
    let center = (denominator_scale / numerator_scale).clamp(1.0e-9, 1.0e9);
    let start = center.log10() - DEFAULT_GAIN_DECADES;
    let stop = center.log10() + DEFAULT_GAIN_DECADES;

    let mut gains = Vec::with_capacity(DEFAULT_GAIN_POINTS + 16);
    gains.push(0.0);
    if DEFAULT_GAIN_POINTS == 1 {
        gains.push(center);
        return gains;
    }
    for idx in 0..DEFAULT_GAIN_POINTS {
        let fraction = idx as f64 / (DEFAULT_GAIN_POINTS - 1) as f64;
        let gain = 10.0_f64.powf(start + fraction * (stop - start));
        if gain.is_finite() && gain > 0.0 {
            gains.push(gain);
        }
    }
    for gain in critical_gains(numerator, denominator) {
        push_gain_window(&mut gains, gain);
    }
    sort_and_dedup_gains(gains)
}

fn max_norm(coeffs: &[Complex64]) -> f64 {
    coeffs
        .iter()
        .map(|coeff| coeff.norm())
        .fold(0.0_f64, f64::max)
}

fn critical_gains(numerator: &[Complex64], denominator: &[Complex64]) -> Vec<f64> {
    let den_derivative = poly_derivative(denominator);
    let num_derivative = poly_derivative(numerator);
    let equation = poly_sub(
        &poly_mul(&den_derivative, numerator),
        &poly_mul(denominator, &num_derivative),
    );
    let Ok(points) = polynomial_roots(&equation, BUILTIN_NAME) else {
        return Vec::new();
    };
    let mut gains = Vec::new();
    for point in points {
        if point.im.abs() > 1.0e-7 {
            continue;
        }
        let numerator_at_point = poly_eval(numerator, point);
        if numerator_at_point.norm() <= EPS {
            continue;
        }
        let gain = -poly_eval(denominator, point) / numerator_at_point;
        if gain.im.abs() <= 1.0e-7 && gain.re.is_finite() && gain.re > 0.0 {
            gains.push(gain.re);
        }
    }
    gains
}

fn push_gain_window(gains: &mut Vec<f64>, gain: f64) {
    for factor in [0.9, 0.99, 1.0, 1.01, 1.1] {
        let candidate = gain * factor;
        if candidate.is_finite() && candidate > 0.0 {
            gains.push(candidate);
        }
    }
}

fn sort_and_dedup_gains(mut gains: Vec<f64>) -> Vec<f64> {
    gains.sort_by(f64::total_cmp);
    gains.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-10 * a.abs().max(b.abs()).max(1.0));
    gains
}

fn poly_derivative(coeffs: &[Complex64]) -> Vec<Complex64> {
    if coeffs.len() <= 1 {
        return vec![Complex64::new(0.0, 0.0)];
    }
    let degree = coeffs.len() - 1;
    coeffs
        .iter()
        .take(degree)
        .enumerate()
        .map(|(idx, coeff)| *coeff * Complex64::new((degree - idx) as f64, 0.0))
        .collect()
}

fn poly_mul(left: &[Complex64], right: &[Complex64]) -> Vec<Complex64> {
    if left.is_empty() || right.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Complex64::new(0.0, 0.0); left.len() + right.len() - 1];
    for (i, lhs) in left.iter().enumerate() {
        for (j, rhs) in right.iter().enumerate() {
            out[i + j] += *lhs * *rhs;
        }
    }
    out
}

fn poly_sub(left: &[Complex64], right: &[Complex64]) -> Vec<Complex64> {
    let len = left.len().max(right.len());
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    let left_offset = len - left.len();
    for (idx, coeff) in left.iter().enumerate() {
        out[left_offset + idx] += *coeff;
    }
    let right_offset = len - right.len();
    for (idx, coeff) in right.iter().enumerate() {
        out[right_offset + idx] -= *coeff;
    }
    out
}

fn sort_roots(mut roots: Vec<Complex64>) -> Vec<Complex64> {
    roots.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    roots
}

fn track_roots(previous: &[Complex64], roots: Vec<Complex64>) -> Vec<Complex64> {
    if previous.len() != roots.len() {
        return sort_roots(roots);
    }
    let mut assigned = vec![false; roots.len()];
    let mut ordered = Vec::with_capacity(roots.len());
    for prev in previous {
        let mut best_idx = None;
        let mut best_distance = f64::INFINITY;
        for (idx, root) in roots.iter().enumerate() {
            if assigned[idx] {
                continue;
            }
            let Some(distance) = root_distance2(*prev, *root) else {
                continue;
            };
            if distance < best_distance {
                best_distance = distance;
                best_idx = Some(idx);
            }
        }
        let idx = best_idx
            .or_else(|| assigned.iter().position(|used| !*used))
            .unwrap_or(0);
        assigned[idx] = true;
        ordered.push(roots[idx]);
    }
    ordered
}

fn root_distance2(a: Complex64, b: Complex64) -> Option<f64> {
    if !a.re.is_finite() || !a.im.is_finite() || !b.re.is_finite() || !b.im.is_finite() {
        return None;
    }
    let dr = a.re - b.re;
    let di = a.im - b.im;
    Some(dr * dr + di * di)
}

#[cfg(feature = "plot-core")]
async fn render_root_locus_plot(eval: &RootLocus, style: Option<&Value>) -> BuiltinResult<()> {
    let mut args = Vec::new();
    for branch in 0..eval.branches {
        let mut x = Vec::with_capacity(eval.gains.len());
        let mut y = Vec::with_capacity(eval.gains.len());
        for gain_idx in 0..eval.gains.len() {
            let root = eval.root(branch, gain_idx);
            if root.re.is_finite() && root.im.is_finite() {
                x.push(root.re);
                y.push(root.im);
            }
        }
        if x.is_empty() {
            continue;
        }
        args.push(column_tensor(x)?);
        args.push(column_tensor(y)?);
        if let Some(style) = style {
            args.push(style.clone());
        }
    }
    push_marker_series(&mut args, &eval.poles, "x")?;
    push_marker_series(&mut args, &eval.zeros, "o")?;

    if args.is_empty() {
        return Ok(());
    }
    if let Err(err) = crate::call_builtin_async("plot", &args).await {
        if presentation_failed(&err) {
            return Ok(());
        }
        return Err(with_render_cause(
            rlocus_error(
                format!("rlocus: plotting failed: {}", err.message()),
                &RLOCUS_ERROR_PLOT_FAILED,
            ),
            err,
        ));
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Root Locus")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Real Axis")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Imaginary Axis")]).await;
    let _ = crate::call_builtin_async("grid", &[Value::from("on")]).await;
    Ok(())
}

#[cfg(not(feature = "plot-core"))]
async fn render_root_locus_plot(_eval: &RootLocus, _style: Option<&Value>) -> BuiltinResult<()> {
    Err(crate::plotting_hooks::unavailable_presentation())
}

#[cfg(feature = "plot-core")]
fn push_marker_series(
    args: &mut Vec<Value>,
    values: &[Complex64],
    marker: &str,
) -> BuiltinResult<()> {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for value in values {
        if value.re.is_finite() && value.im.is_finite() {
            x.push(value.re);
            y.push(value.im);
        }
    }
    if !x.is_empty() {
        args.push(column_tensor(x)?);
        args.push(column_tensor(y)?);
        args.push(Value::from(marker));
    }
    Ok(())
}

#[cfg(feature = "plot-core")]
fn column_tensor(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    Tensor::new(data, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(|err| {
            rlocus_error(
                format!("rlocus: failed to build plot vector: {err}"),
                &RLOCUS_ERROR_INTERNAL,
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::ObjectInstance;

    #[cfg(feature = "plot-core")]
    #[test]
    fn invalid_gain_or_style_rejects_before_model_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let den = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &den).unwrap();
            let mut model = ObjectInstance::new(TF_CLASS.to_string());
            model.replace_property_unbounded("Numerator".into(), Value::Num(1.0));
            model.replace_property_unbounded("Denominator".into(), Value::GpuTensor(resident));
            for name in ["Ts", "InputDelay", "OutputDelay"] {
                model.replace_property_unbounded(name.into(), Value::Num(0.0));
            }
            provider.reset_telemetry();
            let error = block_on(rlocus_builtin(
                Value::Object(model.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                    "not-a-gain",
                )]),
            ))
            .expect_err("invalid gain");
            assert_eq!(error.identifier(), Some("RunMat:rlocus:InvalidArgument"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            provider.reset_telemetry();
            let error = block_on(plot_root_locus_statement(
                Value::Object(model),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                    "not-a-style",
                )]),
            ))
            .expect_err("invalid style");
            assert_eq!(error.identifier(), Some("RunMat:rlocus:InvalidArgument"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn tiny_nonzero_imaginary_gain_is_not_real() {
        let error = real_gain_vector(Value::Complex(1.0, 1.0e-14))
            .expect_err("complex gain must not be rounded to real");
        assert_eq!(error.identifier(), Some("RunMat:rlocus:InvalidArgument"));
    }

    #[test]
    fn unrelated_resident_model_property_is_not_downloaded() {
        let unrelated = Tensor::new(vec![5.0, 6.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &unrelated)
                    .expect("upload unrelated property");
            let mut model = ObjectInstance::new(TF_CLASS.to_string());
            model.replace_property_unbounded("Numerator".into(), Value::Num(1.0));
            model.replace_property_unbounded(
                "Denominator".into(),
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            );
            model.replace_property_unbounded("Ts".into(), Value::Num(0.0));
            model.replace_property_unbounded("InputDelay".into(), Value::Num(0.0));
            model.replace_property_unbounded("OutputDelay".into(), Value::Num(0.0));
            model.replace_property_unbounded("Unrelated".into(), Value::GpuTensor(resident));
            provider.reset_telemetry();
            let Value::Tensor(roots) = block_on(rlocus_builtin(
                Value::Object(model),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
            ))
            .expect("root locus") else {
                panic!("expected root matrix");
            };
            assert_eq!(roots.shape(), &[1, 1]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }
    use runmat_value::IntegerStorage;

    fn tf(num: Vec<f64>, den: Vec<f64>) -> crate::execution::RootedValue {
        block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
                Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
            ],
        ))
        .expect("tf")
    }

    fn discrete_tf(
        num: Vec<f64>,
        den: Vec<f64>,
        sample_time: f64,
    ) -> crate::execution::RootedValue {
        block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
                Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
                Value::Num(sample_time),
            ],
        ))
        .expect("tf")
    }

    fn ss(a: Value, b: Value, c: Value, d: Value) -> crate::execution::RootedValue {
        block_on(crate::call_builtin_async("ss", &[a, b, c, d])).expect("ss")
    }

    fn run_rlocus(sys: crate::execution::RootedValue, rest: Vec<Value>) -> BuiltinResult<Value> {
        sys.with_owned(|sys| {
            block_on(rlocus_builtin(
                sys,
                crate::builtins::common::test_support::builtin_arguments(rest),
            ))
        })
    }

    fn tensor(value: &Value) -> &Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn descriptor_signatures_cover_output_forms() {
        let labels = RLOCUS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"r = rlocus(sys)"));
        assert!(labels.contains(&"r = rlocus(sys, k)"));
        assert!(labels.contains(&"[r,k] = rlocus(sys)"));
        assert!(labels.contains(&"[r,k] = rlocus(sys, k)"));
    }

    #[test]
    fn explicit_gain_returns_closed_loop_roots_and_gains() {
        let sys = tf(vec![1.0], vec![1.0, 1.0]);
        let gains = Value::Tensor(Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = run_rlocus(sys, vec![gains]).expect("rlocus");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);

        let roots = tensor(&outputs[0]);
        assert_eq!(roots.shape(), &[1, 3]);
        assert_eq!(roots.materialize_f64(), vec![-1.0, -2.0, -4.0]);

        let gains = tensor(&outputs[1]);
        assert_eq!(gains.shape(), &[1, 3]);
        assert_eq!(gains.materialize_f64(), vec![0.0, 1.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn explicit_gain_preserves_rank_zero_and_rank_one_vector_admission() {
        let scalar_gain = Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap());
        let scalar_roots =
            run_rlocus(tf(vec![1.0], vec![1.0, 1.0]), vec![scalar_gain]).expect("rank-zero gain");
        let scalar_roots = tensor(&scalar_roots);
        assert_eq!(scalar_roots.shape(), &[1, 1]);
        assert_eq!(scalar_roots.materialize_f64(), vec![-2.0]);

        let vector_gain = Value::Tensor(Tensor::new(vec![0.0, 3.0], vec![2]).unwrap());
        let vector_roots =
            run_rlocus(tf(vec![1.0], vec![1.0, 1.0]), vec![vector_gain]).expect("rank-one gains");
        let vector_roots = tensor(&vector_roots);
        assert_eq!(vector_roots.shape(), &[1, 2]);
        assert_eq!(vector_roots.materialize_f64(), vec![-1.0, -4.0]);
    }

    #[test]
    fn explicit_typed_integer_gain_returns_closed_loop_roots_and_gains() {
        let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
        let sys = tf(vec![1.0], vec![1.0, 1.0]);
        let gains = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3]).unwrap(),
        );
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = run_rlocus(sys, vec![gains]).expect("rlocus");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };

        let roots = tensor(&outputs[0]);
        assert_eq!(roots.shape(), &[1, 3]);
        assert_eq!(roots.materialize_f64(), vec![-1.0, -2.0, -4.0]);

        let gains = tensor(&outputs[1]);
        assert_eq!(gains.shape(), &[1, 3]);
        assert_eq!(gains.materialize_f64(), vec![0.0, 1.0, 3.0]);
        assert!(gains.integer_storage().is_none());
    }

    #[test]
    fn multi_branch_matrix_uses_branch_rows_and_gain_columns() {
        let sys = tf(vec![1.0, 0.0], vec![1.0, 3.0, 2.0]);
        let gains = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap());
        let result = run_rlocus(sys, vec![gains]).expect("rlocus");
        let roots = tensor(&result);
        assert_eq!(roots.shape(), &[2, 3]);

        for (gain_idx, gain) in [0.0_f64, 1.0, 2.0].iter().enumerate() {
            let column = &roots.materialize_f64()[gain_idx * 2..gain_idx * 2 + 2];
            for root in column {
                let residual = root * root + (3.0 + gain) * root + 2.0;
                assert!(
                    residual.abs() < 1.0e-8,
                    "gain={gain} root={root} residual={residual}"
                );
            }
        }
    }

    #[test]
    fn state_space_siso_matches_transfer_function_root_locus() {
        let sys = ss(
            Value::Num(-1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(0.0),
        );
        let gains = Value::Tensor(Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap());
        let result = run_rlocus(sys, vec![gains]).expect("rlocus");
        let roots = tensor(&result);
        assert_eq!(roots.shape(), &[1, 3]);
        for (actual, expected) in roots.materialize_f64().iter().zip([-1.0, -2.0, -4.0]) {
            assert!((actual - expected).abs() < 1.0e-8);
        }
    }

    #[test]
    fn state_space_mimo_is_rejected_with_rlocus_identifier() {
        let sys = ss(
            Value::Num(-1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap()),
        );
        let err = run_rlocus(sys, Vec::new()).expect_err("MIMO ss should fail");
        assert!(err.message().contains("only SISO ss models"));
        assert_eq!(err.identifier(), RLOCUS_ERROR_UNSUPPORTED_MODEL.identifier);
    }

    #[test]
    fn non_model_input_uses_rlocus_invalid_model_identifier() {
        let err = run_rlocus(
            crate::execution::RootedValue::new(Value::Num(1.0)).expect("scalar fixture"),
            Vec::new(),
        )
        .expect_err("should fail");
        assert!(err.message().contains("dynamic system model"));
        assert_eq!(err.identifier(), RLOCUS_ERROR_INVALID_MODEL.identifier);
    }

    #[test]
    fn complex_root_matrix_is_returned_when_branches_are_complex() {
        let sys = tf(vec![1.0], vec![1.0, 2.0, 2.0]);
        let gains = Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
        let result = run_rlocus(sys, vec![gains]).expect("rlocus");
        let Value::ComplexTensor(roots) = result else {
            panic!("expected complex root matrix");
        };
        assert_eq!(roots.shape(), &[2, 1]);
        assert!(roots
            .materialize_f64()
            .iter()
            .any(|(re, im)| (*re + 1.0).abs() < 1.0e-8 && (*im - 1.0).abs() < 1.0e-8));
        assert!(roots
            .materialize_f64()
            .iter()
            .any(|(re, im)| (*re + 1.0).abs() < 1.0e-8 && (*im + 1.0).abs() < 1.0e-8));
    }

    #[test]
    fn discrete_system_uses_closed_loop_polynomial_in_z() {
        let sys = discrete_tf(vec![1.0], vec![1.0, -0.5], 0.1);
        let gains = Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap());
        let roots = run_rlocus(sys, vec![gains]).expect("rlocus");
        let roots = tensor(&roots);
        assert_eq!(roots.shape(), &[1, 1]);
        assert!(roots.materialize_f64()[0].abs() < 1.0e-12);
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn statement_form_plots_without_error() {
        let sys = tf(vec![1.0, 2.0], vec![1.0, 3.0, 4.0]);
        let _guard = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let result = run_rlocus(sys, Vec::new()).expect("rlocus");
        assert!(matches!(result, Value::OutputList(outputs) if outputs.is_empty()));
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn statement_form_turns_hold_off_when_later_system_fails() {
        let _plot_guard = crate::builtins::plotting::tests::lock_plot_registry();
        crate::builtins::plotting::tests::ensure_plot_test_env();
        crate::builtins::plotting::reset_hold_state_for_run();
        let _ = crate::builtins::plotting::clear_figure(None);

        let sys = tf(vec![1.0, 2.0], vec![1.0, 3.0, 4.0]);
        let malformed_tf = Value::Object(ObjectInstance::new("tf".to_string()));
        let _output_guard =
            crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let err = run_rlocus(sys, vec![malformed_tf]).expect_err("second system should fail");
        assert!(err.message().contains("missing"));
        assert!(!crate::builtins::plotting::state::current_hold_enabled());
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn statement_form_restores_existing_hold_on_when_later_system_fails() {
        let _plot_guard = crate::builtins::plotting::tests::lock_plot_registry();
        crate::builtins::plotting::tests::ensure_plot_test_env();
        crate::builtins::plotting::reset_hold_state_for_run();
        let _ = crate::builtins::plotting::clear_figure(None);
        crate::builtins::plotting::set_hold(crate::builtins::plotting::HoldMode::On);

        let sys = tf(vec![1.0, 2.0], vec![1.0, 3.0, 4.0]);
        let malformed_tf = Value::Object(ObjectInstance::new("tf".to_string()));
        let _output_guard =
            crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let err = run_rlocus(sys, vec![malformed_tf]).expect_err("second system should fail");
        assert!(err.message().contains("missing"));
        assert!(crate::builtins::plotting::state::current_hold_enabled());
        crate::builtins::plotting::reset_hold_state_for_run();
    }

    #[test]
    fn rejects_negative_gain() {
        let sys = tf(vec![1.0], vec![1.0, 1.0]);
        let err = run_rlocus(sys, vec![Value::Num(-1.0)]).expect_err("should fail");
        assert!(err.message().contains("nonnegative"));
        assert_eq!(err.identifier(), RLOCUS_ERROR_INVALID_ARGUMENT.identifier);
    }
}
