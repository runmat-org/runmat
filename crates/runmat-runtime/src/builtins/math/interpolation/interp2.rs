//! MATLAB-compatible `interp2` builtin for gridded dense real data.

use runmat_builtins::{ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::dispatcher;

use super::pp::{
    interp_error, parse_extrapolation, parse_method, query_points, vector_from_value,
    Extrapolation, InterpMethod,
};

const NAME: &str = "interp2";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::interp2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("interpolation-2d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Initial implementation gathers GPU inputs to the CPU reference path. Bilinear and nearest kernels are good future provider candidates.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::interp2"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "interp2 is currently a runtime sink.",
};

fn interp2_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let query = match args.len() {
        0..=2 => return Type::tensor(),
        3 | 4 => args.get(1),
        _ => args.get(3),
    };
    match query {
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        _ => Type::tensor(),
    }
}

#[runtime_builtin(
    name = "interp2",
    category = "math/interpolation",
    summary = "Two-dimensional interpolation on gridded data.",
    keywords = "interp2,interpolation,bilinear,nearest,grid,meshgrid",
    accel = "sink",
    sink = true,
    type_resolver(interp2_type),
    builtin_path = "crate::builtins::math::interpolation::interp2"
)]
async fn interp2_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedInterp2::parse(args).await?;
    let data = evaluate_grid(&parsed)?;
    if data.len() == 1 {
        return Ok(Value::Num(data[0]));
    }
    let tensor = Tensor::new(data, parsed.output_shape)
        .map_err(|err| interp_error(NAME, format!("{NAME}: {err}")))?;
    Ok(Value::Tensor(tensor))
}

struct ParsedInterp2 {
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    z: Tensor,
    xq: Vec<f64>,
    yq: Vec<f64>,
    output_shape: Vec<usize>,
    method: InterpMethod,
    extrap: Extrapolation,
}

impl ParsedInterp2 {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        if args.len() < 3 {
            return Err(interp_error(
                NAME,
                "interp2: expected Z, Xq, and Yq or X, Y, Z, Xq, and Yq",
            ));
        }

        let mut method = InterpMethod::Linear;
        let mut extrap = Extrapolation::Nan;
        let explicit_axes = args.len() >= 5 && !is_option_arg(&args[3]);
        let (x_axis, y_axis, z, xq_value, yq_value, options) = if explicit_axes {
            let mut iter = args.into_iter();
            let x = iter.next().expect("X");
            let y = iter.next().expect("Y");
            let z_value = iter.next().expect("Z");
            let z = z_tensor(z_value).await?;
            let (x_axis, y_axis) = axes_from_values(x, y, z.rows, z.cols).await?;
            let xq = iter.next().expect("Xq");
            let yq = iter.next().expect("Yq");
            (x_axis, y_axis, z, xq, yq, iter.collect::<Vec<_>>())
        } else {
            let mut iter = args.into_iter();
            let z_value = iter.next().expect("Z");
            let z = z_tensor(z_value).await?;
            let x_axis: Vec<f64> = (1..=z.cols).map(|v| v as f64).collect();
            let y_axis: Vec<f64> = (1..=z.rows).map(|v| v as f64).collect();
            let xq = iter.next().expect("Xq");
            let yq = iter.next().expect("Yq");
            (x_axis, y_axis, z, xq, yq, iter.collect::<Vec<_>>())
        };

        validate_axis(&x_axis, "X")?;
        validate_axis(&y_axis, "Y")?;
        let xq = query_points(xq_value, NAME).await?;
        let yq = query_points(yq_value, NAME).await?;
        let (xq_values, yq_values, output_shape) = align_queries(xq, yq)?;

        for option in &options {
            if let Some(parsed) = parse_extrapolation(option, NAME).await? {
                extrap = parsed;
                continue;
            }
            if let Some(parsed) = parse_method(option, NAME)? {
                match parsed {
                    InterpMethod::Linear | InterpMethod::Nearest => method = parsed,
                    _ => {
                        return Err(interp_error(
                            NAME,
                            "interp2: only linear and nearest methods are supported",
                        ))
                    }
                }
                continue;
            }
            return Err(interp_error(
                NAME,
                "interp2: unsupported interpolation option",
            ));
        }

        Ok(Self {
            x_axis,
            y_axis,
            z,
            xq: xq_values,
            yq: yq_values,
            output_shape,
            method,
            extrap,
        })
    }
}

fn is_option_arg(value: &Value) -> bool {
    crate::builtins::common::random_args::keyword_of(value).is_some()
}

async fn z_tensor(value: Value) -> crate::BuiltinResult<Tensor> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    let z = tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| interp_error(NAME, format!("{NAME}: {err}")))?;
    if z.shape.len() > 2 {
        return Err(interp_error(NAME, "interp2: Z must be a 2-D matrix"));
    }
    if z.rows < 2 || z.cols < 2 {
        return Err(interp_error(
            NAME,
            "interp2: Z must have at least two rows and two columns",
        ));
    }
    Ok(z)
}

async fn axes_from_values(
    x: Value,
    y: Value,
    rows: usize,
    cols: usize,
) -> crate::BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let x_axis = axis_from_value(x, rows, cols, true).await?;
    let y_axis = axis_from_value(y, rows, cols, false).await?;
    Ok((x_axis, y_axis))
}

async fn axis_from_value(
    value: Value,
    rows: usize,
    cols: usize,
    is_x: bool,
) -> crate::BuiltinResult<Vec<f64>> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    let tensor_value = tensor::value_into_tensor_for(NAME, gathered.clone());
    if let Ok(t) = tensor_value {
        if is_vector_shape(&t.shape) {
            let expected = if is_x { cols } else { rows };
            if t.data.len() != expected {
                return Err(interp_error(
                    NAME,
                    format!("{NAME}: axis vector length must match Z dimensions"),
                ));
            }
            return Ok(t.data);
        }
        if t.rows == rows && t.cols == cols {
            return if is_x {
                Ok((0..cols).map(|col| t.data[col * rows]).collect())
            } else {
                Ok((0..rows).map(|row| t.data[row]).collect())
            };
        }
    }
    let label = if is_x { "X" } else { "Y" };
    vector_from_value(gathered, label, NAME).await
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] | [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        dims => dims.iter().filter(|&&dim| dim > 1).count() <= 1,
    }
}

fn validate_axis(axis: &[f64], label: &str) -> crate::BuiltinResult<()> {
    if axis.len() < 2 {
        return Err(interp_error(
            NAME,
            format!("{NAME}: {label} axis must contain at least two points"),
        ));
    }
    if axis.iter().any(|v| !v.is_finite()) {
        return Err(interp_error(
            NAME,
            format!("{NAME}: {label} axis must be finite"),
        ));
    }
    for pair in axis.windows(2) {
        if pair[1] <= pair[0] {
            return Err(interp_error(
                NAME,
                format!("{NAME}: {label} axis must be strictly increasing"),
            ));
        }
    }
    Ok(())
}

fn align_queries(
    xq: super::pp::QueryPoints,
    yq: super::pp::QueryPoints,
) -> crate::BuiltinResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    match (xq.values.len(), yq.values.len()) {
        (1, 1) => Ok((xq.values, yq.values, vec![1, 1])),
        (1, len) => Ok((vec![xq.values[0]; len], yq.values, yq.shape)),
        (len, 1) => Ok((xq.values, vec![yq.values[0]; len], xq.shape)),
        (left, right) if left == right && xq.shape == yq.shape => {
            Ok((xq.values, yq.values, xq.shape))
        }
        _ => Err(interp_error(
            NAME,
            "interp2: Xq and Yq must be scalar or matching-size arrays",
        )),
    }
}

fn evaluate_grid(parsed: &ParsedInterp2) -> crate::BuiltinResult<Vec<f64>> {
    let mut out = Vec::with_capacity(parsed.xq.len());
    for (&xq, &yq) in parsed.xq.iter().zip(parsed.yq.iter()) {
        let value = match parsed.method {
            InterpMethod::Linear => eval_bilinear(parsed, xq, yq),
            InterpMethod::Nearest => eval_nearest(parsed, xq, yq),
            _ => unreachable!("interp2 parse rejects cubic methods"),
        };
        out.push(value);
    }
    Ok(out)
}

fn eval_bilinear(parsed: &ParsedInterp2, xq: f64, yq: f64) -> f64 {
    if !xq.is_finite() || !yq.is_finite() {
        return f64::NAN;
    }
    let allow = matches!(parsed.extrap, Extrapolation::Extrapolate);
    let Some(col) = interval_index(&parsed.x_axis, xq, allow) else {
        return out_of_range(&parsed.extrap);
    };
    let Some(row) = interval_index(&parsed.y_axis, yq, allow) else {
        return out_of_range(&parsed.extrap);
    };
    let x0 = parsed.x_axis[col];
    let x1 = parsed.x_axis[col + 1];
    let y0 = parsed.y_axis[row];
    let y1 = parsed.y_axis[row + 1];
    let tx = (xq - x0) / (x1 - x0);
    let ty = (yq - y0) / (y1 - y0);
    let z00 = z_at(&parsed.z, row, col);
    let z10 = z_at(&parsed.z, row, col + 1);
    let z01 = z_at(&parsed.z, row + 1, col);
    let z11 = z_at(&parsed.z, row + 1, col + 1);
    (1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11
}

fn eval_nearest(parsed: &ParsedInterp2, xq: f64, yq: f64) -> f64 {
    if !xq.is_finite() || !yq.is_finite() {
        return f64::NAN;
    }
    let Some(col) = nearest_index(&parsed.x_axis, xq, &parsed.extrap) else {
        return out_of_range(&parsed.extrap);
    };
    let Some(row) = nearest_index(&parsed.y_axis, yq, &parsed.extrap) else {
        return out_of_range(&parsed.extrap);
    };
    z_at(&parsed.z, row, col)
}

fn z_at(z: &Tensor, row: usize, col: usize) -> f64 {
    z.data[row + col * z.rows]
}

fn interval_index(axis: &[f64], q: f64, allow_extrapolation: bool) -> Option<usize> {
    if q < axis[0] {
        return allow_extrapolation.then_some(0);
    }
    let last = axis.len() - 1;
    if q > axis[last] {
        return allow_extrapolation.then_some(last - 1);
    }
    if q == axis[last] {
        return Some(last - 1);
    }
    match axis.binary_search_by(|probe| probe.partial_cmp(&q).unwrap()) {
        Ok(index) => Some(index.min(last - 1)),
        Err(index) if index > 0 && index < axis.len() => Some(index - 1),
        _ => None,
    }
}

fn nearest_index(axis: &[f64], q: f64, extrap: &Extrapolation) -> Option<usize> {
    if q < axis[0] {
        return matches!(extrap, Extrapolation::Extrapolate).then_some(0);
    }
    let last = axis.len() - 1;
    if q > axis[last] {
        return matches!(extrap, Extrapolation::Extrapolate).then_some(last);
    }
    match axis.binary_search_by(|probe| probe.partial_cmp(&q).unwrap()) {
        Ok(index) => Some(index),
        Err(index) => {
            let left = index.saturating_sub(1);
            let right = index.min(last);
            if (q - axis[left]).abs() <= (axis[right] - q).abs() {
                Some(left)
            } else {
                Some(right)
            }
        }
    }
}

fn out_of_range(extrap: &Extrapolation) -> f64 {
    match extrap {
        Extrapolation::Value(value) => *value,
        Extrapolation::Nan | Extrapolation::Extrapolate => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    #[test]
    fn interp2_implicit_axes_bilinear_scalar() {
        let z = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("tensor"));
        let value =
            block_on(interp2_builtin(vec![z, Value::Num(1.5), Value::Num(1.5)])).expect("interp2");
        let Value::Num(result) = value else {
            panic!("expected scalar");
        };
        assert!((result - 2.5).abs() < 1e-12);
    }

    #[test]
    fn interp2_vector_axes_nearest() {
        let z = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("tensor"));
        let value = block_on(interp2_builtin(vec![
            row(&[10.0, 20.0]),
            row(&[100.0, 200.0]),
            z,
            Value::Num(18.0),
            Value::Num(120.0),
            Value::String("nearest".to_string()),
        ]))
        .expect("interp2");
        assert_eq!(value, Value::Num(2.0));
    }
}
