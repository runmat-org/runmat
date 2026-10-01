use super::*;
use runmat_builtins::OrderStatisticOption;

fn first_non_singleton(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|dim| *dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn parse_dim(name: &str, value: &Value) -> BuiltinResult<usize> {
    tensor::parse_dimension(value, name).map_err(|err| order_error(name, err))
}

#[derive(Clone)]
enum ReductionAxes {
    All,
    Dimensions(Vec<usize>),
}

fn parse_axes(name: &str, value: &Value) -> BuiltinResult<ReductionAxes> {
    let Value::Tensor(array) = value else {
        return Ok(ReductionAxes::Dimensions(vec![parse_dim(name, value)? - 1]));
    };
    if tensor::tensor_element_len(array) <= 1 {
        return Ok(ReductionAxes::Dimensions(vec![parse_dim(name, value)? - 1]));
    }
    if array.shape.iter().filter(|size| **size > 1).count() > 1 {
        return Err(order_error(
            name,
            format!("{name}: vecdim must be a vector"),
        ));
    }
    let mut axes = if let Some(storage) = array.integer_storage() {
        storage
            .exact_values()
            .into_iter()
            .map(|value| parse_dim(name, &Value::Int(value)).map(|dim| dim - 1))
            .collect::<BuiltinResult<Vec<_>>>()?
    } else {
        tensor::tensor_values_f64_cow(array)
            .iter()
            .copied()
            .map(|value| parse_dim(name, &Value::Num(value)).map(|dim| dim - 1))
            .collect::<BuiltinResult<Vec<_>>>()?
    };
    axes.sort_unstable();
    if axes.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(order_error(
            name,
            format!("{name}: vecdim cannot repeat a dimension"),
        ));
    }
    Ok(ReductionAxes::Dimensions(axes))
}

fn parse_probabilities(
    name: &str,
    value: Value,
    unit: QuantileRequestUnit,
) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(name, value)
        .map_err(|err| order_error(name, format!("{name}: {err}")))?;
    let invalid = || {
        order_error(
            name,
            format!(
                "{name}: {} must be in the closed interval [0, {}]",
                unit.label(),
                unit.upper_bound()
            ),
        )
    };
    if unit == QuantileRequestUnit::Probability && tensor::tensor_element_len(&tensor) == 1 {
        if let Some(storage) = tensor.integer_storage() {
            let exact = storage
                .value_at(0)
                .expect("one-element integer request")
                .to_i128();
            if exact > 1 {
                let count = usize::try_from(exact).map_err(|_| {
                    order_error(name, format!("{name}: quantile count is too large"))
                })?;
                return evenly_spaced_probabilities(name, count);
            }
        } else if let Some(value) = tensor::tensor_values_f64_cow(&tensor).first() {
            if *value > 1.0 {
                if !value.is_finite() || value.fract() != 0.0 || *value > usize::MAX as f64 {
                    return Err(order_error(
                        name,
                        format!("{name}: quantile count must be an integer greater than one"),
                    ));
                }
                return evenly_spaced_probabilities(name, *value as usize);
            }
        }
    }
    if let Some(storage) = tensor.integer_storage() {
        if storage
            .exact_values()
            .iter()
            .any(|value| !(0..=unit.upper_bound()).contains(&value.to_i128()))
        {
            return Err(invalid());
        }
    }
    let values = tensor::tensor_into_values_f64(tensor);
    let mut out = Vec::with_capacity(values.len());
    for raw in values {
        let p = raw / unit.upper_bound() as f64;
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return Err(invalid());
        }
        out.push(p);
    }
    Ok(out)
}

fn evenly_spaced_probabilities(name: &str, count: usize) -> BuiltinResult<Vec<f64>> {
    let mut probabilities = Vec::new();
    probabilities
        .try_reserve_exact(count)
        .map_err(|_| order_error(name, format!("{name}: quantile count is too large")))?;
    let denominator = (count as f64) + 1.0;
    for index in 1..=count {
        probabilities.push(index as f64 / denominator);
    }
    Ok(probabilities)
}

#[derive(Clone, Copy)]
enum NanFlag {
    Include,
    Omit,
}

pub(super) struct QuantileArgs {
    input: Tensor,
    probabilities: Vec<f64>,
    axes: ReductionAxes,
    nanflag: NanFlag,
    method: QuantileMethod,
}

#[derive(Clone, Copy)]
enum QuantileMethod {
    Midpoint,
    Inclusive,
    Exclusive,
}

pub(super) async fn parse_quantile_args(
    name: &str,
    input: Value,
    rest: Vec<Value>,
    unit: QuantileRequestUnit,
) -> BuiltinResult<QuantileArgs> {
    if rest.is_empty() {
        return Err(order_error(
            name,
            format!("{name}: probability vector is required"),
        ));
    }
    let mut axes = None;
    let mut nanflag = NanFlag::Omit;
    let mut method = QuantileMethod::Midpoint;
    let mut idx = 1usize;
    while idx < rest.len() {
        let arg = &rest[idx];
        if let Some(keyword) = keyword_of(arg) {
            match OrderStatisticOption::parse(&keyword) {
                Some(OrderStatisticOption::All) => {
                    axes = Some(ReductionAxes::All);
                }
                Some(OrderStatisticOption::IncludeNan) => nanflag = NanFlag::Include,
                Some(OrderStatisticOption::OmitNan) => nanflag = NanFlag::Omit,
                Some(OrderStatisticOption::Midpoint) => method = QuantileMethod::Midpoint,
                Some(OrderStatisticOption::Inclusive) => method = QuantileMethod::Inclusive,
                Some(OrderStatisticOption::Exclusive) => method = QuantileMethod::Exclusive,
                Some(OrderStatisticOption::Approximate) => {
                    return Err(order_error(
                        name,
                        format!(
                            "{name}: approximate quantile method is only supported for tall arrays"
                        ),
                    ));
                }
                Some(OrderStatisticOption::Method) => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(order_error(
                            name,
                            format!("{name}: Method option requires a value"),
                        ));
                    }
                    let method_name = keyword_of(&rest[idx]).ok_or_else(|| {
                        order_error(
                            name,
                            format!("{name}: Method value must be a string scalar"),
                        )
                    })?;
                    match OrderStatisticOption::parse(&method_name) {
                        Some(OrderStatisticOption::Midpoint) => method = QuantileMethod::Midpoint,
                        Some(OrderStatisticOption::Inclusive) => method = QuantileMethod::Inclusive,
                        Some(OrderStatisticOption::Exclusive) => method = QuantileMethod::Exclusive,
                        Some(OrderStatisticOption::Approximate) => {
                            return Err(order_error(
                                name,
                                format!("{name}: approximate quantile method is only supported for tall arrays"),
                            ));
                        }
                        _ => {
                            return Err(order_error(
                                name,
                                format!("{name}: unsupported Method '{method_name}'"),
                            ));
                        }
                    }
                }
                None => {
                    return Err(order_error(
                        name,
                        format!("{name}: unsupported option '{keyword}'"),
                    ));
                }
            }
        } else {
            axes = Some(parse_axes(name, arg)?);
        }
        idx += 1;
    }
    let input = value_to_tensor(name, input).await?;
    let p_value = gather_if_needed_async(&rest[0])
        .await
        .map_err(|err| order_error(name, format!("{name}: {err}")))?;
    let probabilities = parse_probabilities(name, p_value, unit)?;
    let shape = tensor::default_shape_for(&input.shape, tensor::tensor_element_len(&input));
    let axes =
        axes.unwrap_or_else(|| ReductionAxes::Dimensions(vec![first_non_singleton(&shape) - 1]));
    Ok(QuantileArgs {
        input,
        probabilities,
        axes,
        nanflag,
        method,
    })
}

fn sorted_slice(mut values: Vec<OrderedValue>, nanflag: NanFlag) -> Vec<OrderedValue> {
    if values.is_empty() {
        return values;
    }
    match nanflag {
        NanFlag::Include if values.iter().any(OrderedValue::is_nan) => {
            return vec![OrderedValue::Float(f64::NAN)]
        }
        NanFlag::Include => {}
        NanFlag::Omit => values.retain(|value| !value.is_nan()),
    }
    values.sort_by(compare_ordered_values);
    values
}

fn quantile_from_sorted(values: &[OrderedValue], p: f64, method: QuantileMethod) -> f64 {
    if values.is_empty() || values.iter().any(OrderedValue::is_nan) {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0].as_f64();
    }
    let n = values.len() as f64;
    let position = match method {
        QuantileMethod::Midpoint => p * n - 0.5,
        QuantileMethod::Inclusive => p * (n - 1.0),
        QuantileMethod::Exclusive => p * (n + 1.0) - 1.0,
    }
    .clamp(0.0, n - 1.0);
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    if lo == hi {
        values[lo].as_f64()
    } else {
        let weight = position - lo as f64;
        values[lo].as_f64() * (1.0 - weight) + values[hi].as_f64() * weight
    }
}

pub(super) fn quantile_tensor(args: QuantileArgs, name: &str) -> BuiltinResult<Value> {
    let input_values = ordered_input_values(&args.input);
    let shape = tensor::default_shape_for(&args.input.shape, input_values.len());
    if matches!(args.axes, ReductionAxes::All) {
        let values = sorted_slice(input_values, args.nanflag);
        let data = args
            .probabilities
            .iter()
            .map(|p| quantile_from_sorted(&values, *p, args.method))
            .collect::<Vec<_>>();
        let out_shape = if args.probabilities.len() == 1 {
            vec![1, 1]
        } else {
            vec![args.probabilities.len(), 1]
        };
        return Tensor::new(data, out_shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| order_error(name, format!("{name}: {err}")));
    }
    let ReductionAxes::Dimensions(axes) = &args.axes else {
        unreachable!("all handled above")
    };
    if axes.len() > 1 {
        return quantile_multi_axes(&args, input_values, shape, axes, name);
    }
    let axis = axes[0];
    let rank = shape.len().max(axis + 1);
    let mut padded_shape = shape.clone();
    padded_shape.resize(rank, 1);
    let axis_len = padded_shape[axis];
    let p_len = args.probabilities.len();
    let mut out_shape = padded_shape.clone();
    out_shape[axis] = p_len;
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![0.0; out_len];
    let pre: usize = padded_shape[..axis].iter().product();
    let post: usize = padded_shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            let mut slice = Vec::with_capacity(axis_len);
            for idx in 0..axis_len {
                let src = prefix + idx * pre + suffix * pre * axis_len;
                slice.push(input_values[src].clone());
            }
            let slice = sorted_slice(slice, args.nanflag);
            for (p_idx, p) in args.probabilities.iter().enumerate() {
                let dst = prefix + p_idx * pre + suffix * pre * p_len;
                out[dst] = quantile_from_sorted(&slice, *p, args.method);
            }
        }
    }
    Tensor::new(out, out_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| order_error(name, format!("{name}: {err}")))
}

fn quantile_multi_axes(
    args: &QuantileArgs,
    input_values: Vec<OrderedValue>,
    shape: Vec<usize>,
    axes: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    let rank = shape.len().max(axes[axes.len() - 1] + 1);
    let mut input_shape = shape;
    input_shape.resize(rank, 1);
    let mut reduced = vec![false; rank];
    let mut group_shape = input_shape.clone();
    let mut output_shape = input_shape.clone();
    for &axis in axes {
        reduced[axis] = true;
        group_shape[axis] = 1;
        output_shape[axis] = 1;
    }
    output_shape[axes[0]] = args.probabilities.len();
    let mut groups = vec![Vec::new(); tensor::element_count(&group_shape)];
    for (index, value) in input_values.into_iter().enumerate() {
        let mut remaining = index;
        let mut group_index = 0;
        let mut group_stride = 1;
        for axis in 0..rank {
            let coordinate = remaining % input_shape[axis];
            remaining /= input_shape[axis];
            if !reduced[axis] {
                group_index += coordinate * group_stride;
            }
            group_stride *= group_shape[axis];
        }
        groups[group_index].push(value);
    }
    let mut output = vec![0.0; tensor::element_count(&output_shape)];
    let request_stride = output_shape[..axes[0]].iter().product::<usize>();
    for (index, values) in groups.into_iter().enumerate() {
        let sorted = sorted_slice(values, args.nanflag);
        let mut remaining = index;
        let mut base = 0;
        let mut stride = 1;
        for axis in 0..rank {
            let coordinate = remaining % group_shape[axis];
            remaining /= group_shape[axis];
            base += coordinate * stride;
            stride *= output_shape[axis];
        }
        for (request, probability) in args.probabilities.iter().enumerate() {
            output[base + request * request_stride] =
                quantile_from_sorted(&sorted, *probability, args.method);
        }
    }
    Tensor::new(output, output_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| order_error(name, format!("{name}: {err}")))
}
