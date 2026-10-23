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
    if tensor::tensor_element_len(array) == 0 {
        return Err(order_error(name, format!("{name}: vecdim cannot be empty")));
    }
    if tensor::tensor_element_len(array) <= 1 {
        return Ok(ReductionAxes::Dimensions(vec![parse_dim(name, value)? - 1]));
    }
    if array.shape().iter().filter(|size| **size > 1).count() > 1 {
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

fn input_shape_before_gather(name: &str, input: &Value) -> BuiltinResult<Vec<usize>> {
    let (shape, count) = match input {
        Value::Tensor(tensor) => (tensor.shape(), tensor.len()),
        Value::LogicalArray(array) => (&array.shape[..], array.data.len()),
        Value::GpuTensor(handle) => {
            let shape = handle.shape();
            let count = crate::builtins::common::spec::checked_reduction_elements(shape)
                .ok_or_else(|| {
                    order_error(
                        name,
                        format!("{name}: input shape exceeds structural range"),
                    )
                })?;
            (shape, count)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok(vec![1, 1]),
        _ => {
            return Err(order_error(
                name,
                format!("{name}: input must be real numeric data"),
            ))
        }
    };
    Ok(tensor::default_shape_for(shape, count))
}

fn planned_output_shape(
    name: &str,
    input_shape: &[usize],
    axes: &ReductionAxes,
    count: usize,
) -> BuiltinResult<(Vec<usize>, usize)> {
    let output_shape = match axes {
        ReductionAxes::All => vec![count, 1],
        ReductionAxes::Dimensions(dimensions) => {
            let last = dimensions
                .last()
                .ok_or_else(|| order_error(name, format!("{name}: vecdim cannot be empty")))?;
            let rank = last
                .checked_add(1)
                .ok_or_else(|| {
                    order_error(name, format!("{name}: dimension exceeds structural range"))
                })?
                .max(input_shape.len());
            let mut shape = Vec::new();
            shape.try_reserve_exact(rank).map_err(|_| {
                order_error(
                    name,
                    format!("{name}: dimension metadata exceeds host resources"),
                )
            })?;
            shape.extend_from_slice(input_shape);
            shape.resize(rank, 1);
            for &axis in dimensions {
                shape[axis] = 1;
            }
            shape[dimensions[0]] = count;
            shape
        }
    };
    let elements = crate::builtins::common::spec::checked_reduction_elements(&output_shape)
        .ok_or_else(|| {
            order_error(
                name,
                format!("{name}: output shape exceeds structural range"),
            )
        })?;
    Ok((output_shape, elements))
}

fn allocate_output(name: &str, count: usize) -> BuiltinResult<Vec<f64>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| order_error(name, format!("{name}: output exceeds host resources")))?;
    output.resize(count, 0.0);
    Ok(output)
}

struct MultiAxisWorkspace {
    input_shape: Vec<usize>,
    group_shape: Vec<usize>,
    reduced: Vec<bool>,
    groups: Vec<Vec<OrderedValue>>,
}

fn plan_multi_axis_workspace(
    name: &str,
    shape: &[usize],
    axes: &[usize],
    rank: usize,
) -> BuiltinResult<MultiAxisWorkspace> {
    let mut input_shape = Vec::new();
    input_shape.try_reserve_exact(rank).map_err(|_| {
        order_error(
            name,
            format!("{name}: dimension metadata exceeds host resources"),
        )
    })?;
    input_shape.extend_from_slice(shape);
    input_shape.resize(rank, 1);
    let mut group_shape = Vec::new();
    group_shape.try_reserve_exact(rank).map_err(|_| {
        order_error(
            name,
            format!("{name}: group shape metadata exceeds host resources"),
        )
    })?;
    group_shape.extend_from_slice(&input_shape);
    let mut reduced = Vec::new();
    reduced.try_reserve_exact(rank).map_err(|_| {
        order_error(
            name,
            format!("{name}: dimension metadata exceeds host resources"),
        )
    })?;
    reduced.resize(rank, false);
    for &axis in axes {
        reduced[axis] = true;
        group_shape[axis] = 1;
    }
    let group_count = crate::builtins::common::spec::checked_reduction_elements(&group_shape)
        .ok_or_else(|| {
            order_error(
                name,
                format!("{name}: group shape exceeds structural range"),
            )
        })?;
    let mut groups = Vec::new();
    groups.try_reserve_exact(group_count).map_err(|_| {
        order_error(
            name,
            format!("{name}: group metadata exceeds host resources"),
        )
    })?;
    groups.resize_with(group_count, Vec::new);
    Ok(MultiAxisWorkspace {
        input_shape,
        group_shape,
        reduced,
        groups,
    })
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
                if !value.is_finite() || value.fract() != 0.0 || *value >= usize::MAX as f64 {
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
    output_shape: Vec<usize>,
    output: Vec<f64>,
    multi_axis: Option<MultiAxisWorkspace>,
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
    rest: crate::call::BuiltinArguments,
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
        if let Some(keyword) = keyword_of(arg)? {
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
                    let method_name = keyword_of(&rest[idx])?.ok_or_else(|| {
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
    let shape = input_shape_before_gather(name, &input)?;
    let axes =
        axes.unwrap_or_else(|| ReductionAxes::Dimensions(vec![first_non_singleton(&shape) - 1]));
    // Validate structural controls before fetching a resident probability
    // vector or input. A scalar count is not yet known at this stage.
    planned_output_shape(name, &shape, &axes, 1)?;
    let p_value = gather_if_needed_async(&rest[0])
        .await
        .map_err(|err| order_error(name, format!("{name}: {err}")))?;
    let probabilities = parse_probabilities(name, p_value, unit)?;
    let (output_shape, output_len) =
        planned_output_shape(name, &shape, &axes, probabilities.len())?;
    let output = allocate_output(name, output_len)?;
    let multi_axis = match &axes {
        ReductionAxes::Dimensions(dimensions) if dimensions.len() > 1 && output_len > 0 => Some(
            plan_multi_axis_workspace(name, &shape, dimensions, output_shape.len())?,
        ),
        _ => None,
    };
    let input = value_to_tensor(name, input).await?;
    Ok(QuantileArgs {
        input,
        probabilities,
        axes,
        nanflag,
        method,
        output_shape,
        output,
        multi_axis,
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

pub(super) fn quantile_tensor(mut args: QuantileArgs, name: &str) -> BuiltinResult<Value> {
    let input_values = ordered_input_values(&args.input);
    let shape = tensor::default_shape_for(args.input.shape(), input_values.len());
    if matches!(args.axes, ReductionAxes::All) {
        let values = sorted_slice(input_values, args.nanflag);
        for (result, probability) in args.output.iter_mut().zip(&args.probabilities) {
            *result = quantile_from_sorted(&values, *probability, args.method);
        }
        return Tensor::new(args.output, args.output_shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| order_error(name, format!("{name}: {err}")));
    }
    let ReductionAxes::Dimensions(axes) = &args.axes else {
        unreachable!("all handled above")
    };
    if axes.len() > 1 {
        return quantile_multi_axes(args, input_values, name);
    }
    let axis = axes[0];
    // Beyond-rank dimensions are implicit singletons. The preflighted output
    // shape already carries the requested rank; no second rank-sized buffer
    // is needed after a resident input has been gathered.
    let axis_len = shape.get(axis).copied().unwrap_or(1);
    let p_len = args.probabilities.len();
    let pre: usize = shape[..axis.min(shape.len())].iter().product();
    let post: usize = shape.get(axis + 1..).unwrap_or(&[]).iter().product();
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
                args.output[dst] = quantile_from_sorted(&slice, *p, args.method);
            }
        }
    }
    Tensor::new(args.output, args.output_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| order_error(name, format!("{name}: {err}")))
}

fn quantile_multi_axes(
    mut args: QuantileArgs,
    input_values: Vec<OrderedValue>,
    name: &str,
) -> BuiltinResult<Value> {
    if args.output.is_empty() {
        return Tensor::new(args.output, args.output_shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| order_error(name, format!("{name}: {err}")));
    }
    let ReductionAxes::Dimensions(axes) = &args.axes else {
        unreachable!("multi-axis call requires dimensions")
    };
    let MultiAxisWorkspace {
        input_shape,
        group_shape,
        reduced,
        mut groups,
    } = args
        .multi_axis
        .take()
        .expect("multi-axis workspace was preflighted");
    let rank = input_shape.len();
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
        groups[group_index]
            .try_reserve(1)
            .map_err(|_| order_error(name, format!("{name}: group data exceeds host resources")))?;
        groups[group_index].push(value);
    }
    let request_stride = args.output_shape[..axes[0]].iter().product::<usize>();
    for (index, values) in groups.into_iter().enumerate() {
        let sorted = sorted_slice(values, args.nanflag);
        let mut remaining = index;
        let mut base = 0;
        let mut stride = 1;
        for axis in 0..rank {
            let coordinate = remaining % group_shape[axis];
            remaining /= group_shape[axis];
            base += coordinate * stride;
            stride *= args.output_shape[axis];
        }
        for (request, probability) in args.probabilities.iter().enumerate() {
            args.output[base + request * request_stride] =
                quantile_from_sorted(&sorted, *probability, args.method);
        }
    }
    Tensor::new(args.output, args.output_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| order_error(name, format!("{name}: {err}")))
}

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn shape_preflight_rejects_overflow_without_allocating_output() {
        let error = planned_output_shape(
            "quantile",
            &[usize::MAX, 2],
            &ReductionAxes::Dimensions(vec![1]),
            2,
        )
        .expect_err("output cardinality exceeds structural range");
        assert!(error.message().contains("structural range"), "{error}");

        let error = planned_output_shape(
            "quantile",
            &[1, 1],
            &ReductionAxes::Dimensions(vec![usize::MAX]),
            1,
        )
        .expect_err("dimension cannot increase rank beyond usize");
        assert!(error.message().contains("structural range"), "{error}");
    }

    #[test]
    fn empty_vecdim_rejects_without_attempting_shape_extension() {
        let empty = Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap());
        let error = parse_axes("quantile", &empty)
            .err()
            .expect("vecdim requires dimensions");
        assert!(
            error.message().contains("vecdim cannot be empty"),
            "{error}"
        );
    }
}
