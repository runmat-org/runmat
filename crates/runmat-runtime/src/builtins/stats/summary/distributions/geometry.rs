use super::*;

pub(super) fn scalar_tensor(value: f64) -> Tensor {
    Tensor::new(vec![value], vec![1, 1]).expect("scalar tensor shape is valid")
}

pub(super) fn split_upper_tail(
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<(crate::call::BuiltinArguments, DistributionTail)> {
    let tail = rest
        .last()
        .map(crate::builtins::common::random_args::keyword_of)
        .transpose()?
        .flatten()
        .and_then(|keyword| DistributionTail::from_keyword(&keyword))
        .unwrap_or(DistributionTail::Lower);
    let end = rest.len() - usize::from(tail == DistributionTail::Upper);
    Ok((rest.into_range(..end)?, tail))
}

pub(super) async fn normal_args(
    name: &str,
    first: Value,
    rest: crate::call::BuiltinArguments,
    kind: DistributionKind,
) -> BuiltinResult<NormalArgs> {
    let (rest, tail) = if kind.supports_upper_tail() {
        split_upper_tail(rest)?
    } else {
        (rest, DistributionTail::Lower)
    };
    if rest.len() > 2 {
        return Err(normal_error(
            name,
            format!("{name}: expected x, x, mu, or x, mu, sigma"),
        ));
    }
    match rest.len() {
        0 => numeric_inputs(name, &[&first], NumericShapePolicy::Broadcast)?,
        1 => numeric_inputs(name, &[&first, &rest[0]], NumericShapePolicy::Broadcast)?,
        2 => numeric_inputs(
            name,
            &[&first, &rest[0], &rest[1]],
            NumericShapePolicy::Broadcast,
        )?,
        _ => unreachable!(),
    }
    let output = NormalOutputPlan::inspect(name, &first, rest.view())?;
    let x = normal_value_to_tensor(name, "x or p", first).await?;
    let (mu, sigma) = match rest.len() {
        0 => (scalar_tensor(0.0), scalar_tensor(1.0)),
        1 => (
            normal_value_to_tensor(name, "mu", rest[0].clone()).await?,
            scalar_tensor(1.0),
        ),
        2 => (
            normal_value_to_tensor(name, "mu", rest[0].clone()).await?,
            normal_value_to_tensor(name, "sigma", rest[1].clone()).await?,
        ),
        _ => unreachable!(),
    };
    let (mut broadcasted, shape) = broadcast_tensors(name, &[&x, &mu, &sigma])?;
    let x = broadcasted.remove(0);
    let mu = broadcasted.remove(0);
    let sigma = broadcasted.remove(0);
    Ok(NormalArgs {
        x,
        mu,
        sigma,
        shape,
        tail,
        output,
    })
}

pub(super) struct NormalArgs {
    pub(super) x: Vec<f64>,
    pub(super) mu: Vec<f64>,
    pub(super) sigma: Vec<f64>,
    pub(super) shape: Vec<usize>,
    pub(super) tail: DistributionTail,
    pub(super) output: NormalOutputPlan,
}

pub(super) struct TArgs {
    pub(super) x: Vec<f64>,
    pub(super) nu: Vec<f64>,
    pub(super) shape: Vec<usize>,
}

pub(super) struct ThreeArgs {
    pub(super) x: Vec<f64>,
    pub(super) a: Vec<f64>,
    pub(super) b: Vec<f64>,
    pub(super) shape: Vec<usize>,
}

pub(super) fn broadcast_pair(
    name: &str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> BuiltinResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let (mut values, shape) = broadcast_tensors(name, &[lhs, rhs])?;
    let lhs = values.remove(0);
    let rhs = values.remove(0);
    Ok((lhs, rhs, shape))
}

pub(super) fn broadcast_tensors(
    name: &str,
    inputs: &[&Tensor],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let Some(first) = inputs.first() else {
        return Ok((Vec::new(), vec![1, 1]));
    };
    let mut shape = first.shape().to_vec();
    for tensor in inputs.iter().skip(1) {
        shape = broadcast::broadcast_shapes(name, &shape, tensor.shape())
            .map_err(|err| normal_error(name, err))?;
    }
    let mut values = Vec::new();
    values.try_reserve_exact(inputs.len()).map_err(|_| {
        normal_error(
            name,
            format!("{name}: input metadata exceeds host resources"),
        )
    })?;
    for tensor in inputs {
        values.push(broadcast_tensor_to(name, tensor, &shape)?);
    }
    Ok((values, shape))
}

pub(super) fn broadcast_tensor_to(
    name: &str,
    tensor: &Tensor,
    out_shape: &[usize],
) -> BuiltinResult<Vec<f64>> {
    let len =
        crate::builtins::common::spec::checked_reduction_elements(out_shape).ok_or_else(|| {
            normal_error(
                name,
                format!("{name}: output shape exceeds structural range"),
            )
        })?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let in_shape = broadcast::align_shape(tensor.shape(), out_shape.len());
    let strides = broadcast::compute_strides(&in_shape);
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| normal_error(name, format!("{name}: output exceeds host resources")))?;
    let input_len = tensor::tensor_element_len(tensor);
    for idx in 0..len {
        let source_idx = broadcast::broadcast_index(idx, out_shape, &in_shape, &strides);
        if source_idx >= input_len {
            return Err(normal_error(
                name,
                format!("{name}: tensor data does not match tensor shape"),
            ));
        }
        out.push(tensor::tensor_value_f64(tensor, source_idx));
    }
    Ok(out)
}

pub(super) async fn t_args(
    name: &str,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<TArgs> {
    if rest.len() != 1 {
        return Err(normal_error(
            name,
            format!("{name}: expected x and nu arguments"),
        ));
    }
    numeric_inputs(name, &[&first, &rest[0]], NumericShapePolicy::Broadcast)?;
    let x = value_to_tensor(name, first).await?;
    let nu = value_to_tensor(name, rest[0].clone()).await?;
    let (x, nu, shape) = broadcast_pair(name, &x, &nu)?;
    Ok(TArgs { x, nu, shape })
}

pub(super) async fn three_args(
    name: &str,
    first: Value,
    rest: crate::call::BuiltinArguments,
    defaults: Option<(f64, f64)>,
) -> BuiltinResult<ThreeArgs> {
    if !matches!((rest.len(), defaults.is_some()), (0, true) | (2, _)) {
        return Err(normal_error(
            name,
            format!("{name}: expected one or three numeric arguments"),
        ));
    }
    match rest.len() {
        0 => numeric_inputs(name, &[&first], NumericShapePolicy::Broadcast)?,
        1 => numeric_inputs(name, &[&first, &rest[0]], NumericShapePolicy::Broadcast)?,
        2 => numeric_inputs(
            name,
            &[&first, &rest[0], &rest[1]],
            NumericShapePolicy::Broadcast,
        )?,
        _ => unreachable!(),
    }
    let (a, b) = match (rest.len(), defaults) {
        (0, Some((a, b))) => (scalar_tensor(a), scalar_tensor(b)),
        (2, _) => (
            value_to_tensor(name, rest[0].clone()).await?,
            value_to_tensor(name, rest[1].clone()).await?,
        ),
        _ => {
            return Err(normal_error(
                name,
                format!("{name}: expected one or three numeric arguments"),
            ));
        }
    };
    let x = value_to_tensor(name, first).await?;
    let (mut broadcasted, shape) = broadcast_tensors(name, &[&x, &a, &b])?;
    let x = broadcasted.remove(0);
    let a = broadcasted.remove(0);
    let b = broadcasted.remove(0);
    Ok(ThreeArgs { x, a, b, shape })
}

pub(super) fn finish(shape: Vec<usize>, data: Vec<f64>) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| normal_error("normal", format!("normal distribution: {err}")))
}
