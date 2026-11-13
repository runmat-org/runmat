//! Two-sample test execution, statistics, output geometry, and typed boundaries.
use super::*;
use std::borrow::Cow;

#[derive(Clone, Copy, Debug)]
pub(super) struct Options {
    pub(super) alpha: f64,
    pub(super) dim: Option<usize>,
    pub(super) tail: Tail,
    pub(super) vartype: VarType,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            dim: None,
            tail: Tail::Both,
            vartype: VarType::Equal,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Evaluation {
    pub(super) h: Value,
    p: Value,
    ci: Value,
    stats: Value,
}

#[derive(Clone, Debug)]
struct TestResult {
    h: bool,
    p: f64,
    ci_low: f64,
    ci_high: f64,
    tstat: f64,
    df: f64,
    sd_equal: f64,
    sd_x: f64,
    sd_y: f64,
}

pub(super) fn outputs_for_count(out_count: usize, eval: Evaluation) -> BuiltinResult<Value> {
    match out_count {
        0 => Ok(Value::output_list(Vec::new())),
        1 => Ok(Value::output_list(vec![eval.h])),
        2 => Ok(Value::output_list(vec![eval.h, eval.p])),
        3 => Ok(crate::output_count::output_list_with_padding(
            3,
            vec![eval.h, eval.p, eval.ci, eval.stats],
        )),
        4 => Ok(Value::output_list(vec![
            eval.h, eval.p, eval.ci, eval.stats,
        ])),
        _ => Err(invalid_argument(
            "ttest2: too many output arguments; maximum is 4",
        )),
    }
}

pub(super) async fn gathered_value(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_in_context_async(&value, runtime)
        .await
        .map_err(|cause| {
            if crate::runtime_error::error_has_control_cancellation_origin(&cause) {
                return cause;
            }
            let mut owner = invalid_argument(format!("ttest2: {cause}"));
            owner.gpu_gather_retry = cause.gpu_gather_retry;
            owner.span = cause.span;
            owner.source = Some(Box::new(cause));
            owner
        })
}

pub(super) async fn gathered_tensor(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Tensor> {
    let value = gathered_value(value, runtime).await?;
    tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(invalid_argument)
}

pub(super) fn ttest2_is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn ensure_ttest2_sample_extension(x: &Value, y: &Value) -> BuiltinResult<()> {
    if ttest2_is_typed_integer(x) || ttest2_is_typed_integer(y) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TTEST2_INTEGER_SAMPLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn ensure_ttest2_extensions(x: &Value, y: &Value, rest: &[Value]) -> BuiltinResult<()> {
    ensure_ttest2_sample_extension(x, y)?;
    Ttest2OptionPlan::admit(crate::builtins::common::test_support::builtin_arguments(
        rest.to_vec(),
    ))
    .map(|_| ())
}

pub(super) fn ensure_exact_ttest2_integer_tensor(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    if let Some(storage) = tensor.integer_storage() {
        for integer in storage.exact_values() {
            if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
                return Err(invalid_argument(format!(
                    "ttest2: integer {role} values must be exactly representable as double"
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn ensure_exact_ttest2_integer_scalar(value: &Value, role: &str) -> BuiltinResult<()> {
    match value {
        Value::Int(integer) => {
            if crate::builtins::common::validation::integer_is_exact_f64(integer) {
                Ok(())
            } else {
                Err(invalid_argument(format!(
                    "ttest2: integer {role} must be exactly representable as double"
                )))
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            ensure_exact_ttest2_integer_tensor(tensor, role)
        }
        _ => Ok(()),
    }
}

pub(super) fn scalar_text<'a>(value: &'a Value, context: &str) -> BuiltinResult<Cow<'a, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(Cow::Borrowed(text.as_str())),
        Value::CharArray(array) if array.rows == 1 => Ok(Cow::Owned(
            crate::builtins::stats::decode_utf16_with_error(array.data(), || {
                invalid_argument(format!("ttest2: {context} is not valid UTF-16 text"))
            })?,
        )),
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .as_str()
            .map(Cow::Borrowed)
            .ok_or_else(|| invalid_argument(format!("ttest2: {context} must be nonmissing text"))),
        Value::Cell(cell) if cell.data().len() == 1 => scalar_text(&cell.data()[0], context),
        other => Err(invalid_argument(format!(
            "ttest2: {context} must be a string scalar, got {other:?}"
        ))),
    }
}

pub(super) fn scalar_number(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_values_f64(tensor).first().copied()
        }
        _ => None,
    }
}

pub(super) fn scalar_usize(value: &Value) -> Option<usize> {
    match value {
        Value::Int(value) => value.try_to_usize(),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                storage.value_at(0)?.try_to_usize()
            } else {
                let value = tensor::tensor_values_f64(tensor).first().copied()?;
                runmat_types::decode_shape_dimension(value).ok()
            }
        }
        Value::Num(value) => runmat_types::decode_shape_dimension(*value).ok(),
        Value::Bool(value) => Some(usize::from(*value)),
        _ => None,
    }
}

pub(super) fn evaluate(x: &Tensor, y: &Tensor, options: Options) -> BuiltinResult<Evaluation> {
    let vector_collapse = options.dim.is_none()
        && is_nonempty_vector_shape(x.shape())
        && is_nonempty_vector_shape(y.shape());
    let dim = if vector_collapse {
        0
    } else {
        options
            .dim
            .unwrap_or_else(|| first_nonsingleton_dim(x.shape()))
    };
    let rank = x
        .shape()
        .len()
        .max(y.shape().len())
        .max(
            dim.checked_add(1)
                .ok_or_else(|| invalid_argument("ttest2: dimension geometry overflow"))?,
        )
        .max(2);
    let x_shape = normalized_shape(x.shape(), rank)?;
    let y_shape = normalized_shape(y.shape(), rank)?;
    let out_shape = if vector_collapse {
        vec![1, 1]
    } else {
        compatible_output_shape(&x_shape, &y_shape, dim)?
    };
    let tests = tensor::element_count(&out_shape);
    let mut h_data = Vec::with_capacity(tests);
    let mut p_data = Vec::with_capacity(tests);
    let mut ci_data = Vec::with_capacity(tests * 2);
    let mut tstat_data = Vec::with_capacity(tests);
    let mut df_data = Vec::with_capacity(tests);
    let mut sd_equal_data = Vec::with_capacity(tests);
    let mut sd_x_data = Vec::with_capacity(tests);
    let mut sd_y_data = Vec::with_capacity(tests);
    let x_values = x.materialize_f64();
    let y_values = y.materialize_f64();

    for linear in 0..tests {
        let result = if vector_collapse {
            evaluate_pair(&x_values, &y_values, options)
        } else {
            let coords = coords_from_linear(linear, &out_shape)?;
            let xs = sample_along_dim(&x_values, &x_shape, &coords, dim)?;
            let ys = sample_along_dim(&y_values, &y_shape, &coords, dim)?;
            evaluate_pair(&xs, &ys, options)
        };
        h_data.push(u8::from(result.h));
        p_data.push(result.p);
        ci_data.push(result.ci_low);
        ci_data.push(result.ci_high);
        tstat_data.push(result.tstat);
        df_data.push(result.df);
        sd_equal_data.push(result.sd_equal);
        sd_x_data.push(result.sd_x);
        sd_y_data.push(result.sd_y);
    }

    let h = logical_output(h_data, out_shape.clone())?;
    let p = numeric_output(p_data, out_shape.clone())?;
    let ci_shape = interval_shape(&out_shape, dim)?;
    let ci = Value::Tensor(Tensor::new(ci_data, ci_shape).map_err(internal_error)?);
    let mut stats = runmat_value::UnboundedStructValueBuilder::new();
    stats.insert("tstat", numeric_output(tstat_data, out_shape.clone())?);
    stats.insert("df", numeric_output(df_data, out_shape.clone())?);
    if options.vartype == VarType::Equal {
        stats.insert("sd", numeric_output(sd_equal_data, out_shape)?);
    } else {
        let sd_shape = interval_shape(&out_shape, dim)?;
        let mut sd_data = Vec::with_capacity(tests * 2);
        for idx in 0..tests {
            sd_data.push(sd_x_data[idx]);
            sd_data.push(sd_y_data[idx]);
        }
        stats.insert(
            "sd",
            Value::Tensor(Tensor::new(sd_data, sd_shape).map_err(internal_error)?),
        );
    }
    Ok(Evaluation {
        h,
        p,
        ci,
        stats: Value::Struct(stats.finish()),
    })
}

#[cfg(test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rank_zero_samples_retain_scalar_vector_collapse() {
    let x = Tensor::new(vec![1.0], Vec::new()).unwrap();
    let y = Tensor::new(vec![2.0], Vec::new()).unwrap();
    let result = evaluate(&x, &y, Options::default()).unwrap();
    assert!(matches!(result.p, Value::Num(value) if value.is_nan()));
    let Value::Tensor(interval) = result.ci else {
        panic!("expected tensor confidence interval");
    };
    assert_eq!(interval.shape(), [2, 1]);
}

fn compatible_output_shape(
    x_shape: &[usize],
    y_shape: &[usize],
    dim: usize,
) -> BuiltinResult<Vec<usize>> {
    let mut out = shape_buffer(x_shape.len())?;
    for axis in 0..x_shape.len() {
        if axis == dim {
            out.push(1);
            continue;
        }
        if x_shape[axis] != y_shape[axis] {
            return Err(invalid_argument(
                "ttest2: X and Y must have the same size along all dimensions except Dim",
            ));
        }
        out.push(x_shape[axis]);
    }
    Ok(out)
}

fn evaluate_pair(x: &[f64], y: &[f64], options: Options) -> TestResult {
    let xs = x
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    let ys = y
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if xs.len() < 2 || ys.len() < 2 {
        return invalid_result();
    }
    let nx = xs.len() as f64;
    let ny = ys.len() as f64;
    let mean_x = mean(&xs);
    let mean_y = mean(&ys);
    let var_x = variance(&xs);
    let var_y = variance(&ys);
    if !var_x.is_finite() || !var_y.is_finite() {
        return invalid_result();
    }
    let diff = mean_x - mean_y;
    let (se, df, sd_equal, sd_x, sd_y) = match options.vartype {
        VarType::Equal => {
            let df = nx + ny - 2.0;
            if df <= 0.0 {
                return invalid_result();
            }
            let pooled_var = ((nx - 1.0) * var_x + (ny - 1.0) * var_y) / df;
            let sd = pooled_var.sqrt();
            (
                sd * (1.0 / nx + 1.0 / ny).sqrt(),
                df,
                sd,
                var_x.sqrt(),
                var_y.sqrt(),
            )
        }
        VarType::Unequal => {
            let ax = var_x / nx;
            let ay = var_y / ny;
            let denom = ax * ax / (nx - 1.0) + ay * ay / (ny - 1.0);
            let df = if denom == 0.0 {
                f64::INFINITY
            } else {
                (ax + ay) * (ax + ay) / denom
            };
            ((ax + ay).sqrt(), df, f64::NAN, var_x.sqrt(), var_y.sqrt())
        }
    };
    let tstat = if se == 0.0 {
        if diff == 0.0 {
            0.0
        } else {
            diff.signum() * f64::INFINITY
        }
    } else {
        diff / se
    };
    let p = p_value(tstat, df, options.tail);
    let (ci_low, ci_high) = confidence_interval(diff, se, df, options);
    TestResult {
        h: p.is_finite() && p <= options.alpha,
        p,
        ci_low,
        ci_high,
        tstat,
        df,
        sd_equal,
        sd_x,
        sd_y,
    }
}

fn invalid_result() -> TestResult {
    TestResult {
        h: false,
        p: f64::NAN,
        ci_low: f64::NAN,
        ci_high: f64::NAN,
        tstat: f64::NAN,
        df: f64::NAN,
        sd_equal: f64::NAN,
        sd_x: f64::NAN,
        sd_y: f64::NAN,
    }
}

fn p_value(tstat: f64, df: f64, tail: Tail) -> f64 {
    if tstat.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    match tail {
        Tail::Both => (2.0
            * student_t_cdf(tstat.abs(), df).min(student_t_cdf_upper(tstat.abs(), df)))
        .min(1.0),
        Tail::Right => student_t_cdf_upper(tstat, df),
        Tail::Left => student_t_cdf(tstat, df),
    }
}

fn confidence_interval(diff: f64, se: f64, df: f64, options: Options) -> (f64, f64) {
    if diff.is_nan() || se.is_nan() || df.is_nan() || df <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    match options.tail {
        Tail::Both => {
            let crit = student_t_inv(1.0 - options.alpha / 2.0, df);
            (diff - crit * se, diff + crit * se)
        }
        Tail::Right => {
            let crit = student_t_inv(1.0 - options.alpha, df);
            (diff - crit * se, f64::INFINITY)
        }
        Tail::Left => {
            let crit = student_t_inv(1.0 - options.alpha, df);
            (f64::NEG_INFINITY, diff + crit * se)
        }
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = mean(values);
    values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64
}

fn normalized_shape(shape: &[usize], rank: usize) -> BuiltinResult<Vec<usize>> {
    let mut out = shape_buffer(rank)?;
    out.extend_from_slice(shape);
    out.resize(rank, 1);
    Ok(out)
}

fn first_nonsingleton_dim(shape: &[usize]) -> usize {
    shape.iter().position(|dim| *dim != 1).unwrap_or(0)
}

fn is_nonempty_vector_shape(shape: &[usize]) -> bool {
    shape.iter().all(|dim| *dim > 0) && shape.iter().filter(|dim| **dim > 1).count() <= 1
}

fn sample_along_dim(
    data: &[f64],
    shape: &[usize],
    out_coords: &[usize],
    dim: usize,
) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::with_capacity(shape[dim]);
    let mut coords = shape_buffer(shape.len())?;
    coords.extend_from_slice(out_coords);
    coords.resize(shape.len(), 0);
    for idx in 0..shape[dim] {
        coords[dim] = idx;
        values.push(data[linear_index(&coords, shape)]);
    }
    Ok(values)
}

fn coords_from_linear(mut linear: usize, shape: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut coords = shape_buffer(shape.len())?;
    for dim in shape {
        coords.push(linear % *dim);
        linear /= *dim;
    }
    Ok(coords)
}

fn linear_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1;
    let mut index = 0;
    for (coord, dim) in coords.iter().zip(shape) {
        index += coord * stride;
        stride *= dim;
    }
    index
}

fn logical_output(data: Vec<u8>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(internal_error)
    }
}

fn numeric_output(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(internal_error)
}

fn interval_shape(out_shape: &[usize], dim: usize) -> BuiltinResult<Vec<usize>> {
    let mut shape = shape_buffer(out_shape.len().max(2))?;
    shape.push(2);
    for (axis, extent) in out_shape.iter().copied().enumerate() {
        if axis != dim {
            shape.push(extent);
        }
    }
    if shape.len() == 1 {
        shape.push(1);
    }
    Ok(shape)
}

fn shape_buffer(rank: usize) -> BuiltinResult<Vec<usize>> {
    runmat_value::exact_capacity_vec(rank).map_err(|source| {
        let mut error = internal_error("ttest2: dimension metadata allocation failed");
        error.source = Some(Box::new(source));
        error
    })
}
