use super::*;

#[runtime_builtin(
    name = "dlarray.plus",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(super) fn dlarray_plus_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    dlarray_binary(
        lhs,
        rhs,
        "plus",
        |a, b| a + b,
        |lhs, rhs, out| {
            push_binary_node(
                out,
                |l, r, ls, rs| NodeKind::Add {
                    lhs: l,
                    rhs: r,
                    lhs_shape: ls,
                    rhs_shape: rs,
                },
                lhs,
                rhs,
            )
        },
    )
}

#[runtime_builtin(
    name = "dlarray.minus",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(super) fn dlarray_minus_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    dlarray_binary(
        lhs,
        rhs,
        "minus",
        |a, b| a - b,
        |lhs, rhs, out| {
            push_binary_node(
                out,
                |l, r, ls, rs| NodeKind::Sub {
                    lhs: l,
                    rhs: r,
                    lhs_shape: ls,
                    rhs_shape: rs,
                },
                lhs,
                rhs,
            )
        },
    )
}

#[runtime_builtin(
    name = "dlarray.times",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(super) fn dlarray_times_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    dlarray_binary(
        lhs,
        rhs,
        "times",
        |a, b| a * b,
        |lhs, rhs, out| {
            let (lhs_values, rhs_values, _) =
                tensor::binary_numeric_tensors(&lhs.tensor, &rhs.tensor, "dlarray.times", "times")?;
            if !tape_is_active() || (lhs.node.is_none() && rhs.node.is_none()) {
                return Ok(None);
            }
            Ok(Some(push_node(
                out,
                NodeKind::Mul {
                    lhs: lhs.node,
                    rhs: rhs.node,
                    lhs_data: lhs_values,
                    rhs_data: rhs_values,
                    lhs_shape: lhs.tensor.shape.clone(),
                    rhs_shape: rhs.tensor.shape.clone(),
                },
            )?))
        },
    )
}

#[runtime_builtin(
    name = "dlarray.rdivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(super) fn dlarray_rdivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    dlarray_binary(
        lhs,
        rhs,
        "rdivide",
        |a, b| a / b,
        |lhs, rhs, out| {
            let (lhs_values, rhs_values, _) = tensor::binary_numeric_tensors(
                &lhs.tensor,
                &rhs.tensor,
                "dlarray.rdivide",
                "rdivide",
            )?;
            if !tape_is_active() || (lhs.node.is_none() && rhs.node.is_none()) {
                return Ok(None);
            }
            Ok(Some(push_node(
                out,
                NodeKind::Div {
                    lhs: lhs.node,
                    rhs: rhs.node,
                    lhs_data: lhs_values,
                    rhs_data: rhs_values,
                    lhs_shape: lhs.tensor.shape.clone(),
                    rhs_shape: rhs.tensor.shape.clone(),
                },
            )?))
        },
    )
}

fn dlarray_binary(
    lhs: Value,
    rhs: Value,
    function: &'static str,
    op: impl Fn(f64, f64) -> f64,
    record: impl FnOnce(&Payload, &Payload, Tensor) -> BuiltinResult<Option<usize>>,
) -> BuiltinResult<Value> {
    let lhs = Payload::parse(lhs, function)?;
    let rhs = Payload::parse(rhs, function)?;
    let (lhs_values, rhs_values, shape) =
        tensor::binary_numeric_tensors(&lhs.tensor, &rhs.tensor, function, function)?;
    let data = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(a, b)| op(*a, *b))
        .collect::<Vec<_>>();
    let dtype = floating_binary_dtype(lhs.dtype, rhs.dtype);
    let out = Tensor::new_with_dtype(data, shape, dtype)
        .map_err(|err| deep_learning_error(function, err))?;
    let node = record(&lhs, &rhs, out.clone())?;
    Ok(wrap_result(result_metadata_owner(&lhs, &rhs), out, node))
}

#[runtime_builtin(
    name = "dlarray.mtimes",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(super) fn dlarray_mtimes_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    let lhs = Payload::parse(lhs, "mtimes")?;
    let rhs = Payload::parse(rhs, "mtimes")?;
    let computed = crate::builtins::common::linalg::matmul_real(&lhs.tensor, &rhs.tensor)
        .map_err(|err| deep_learning_error("mtimes", err))?;
    let dtype = floating_binary_dtype(lhs.dtype, rhs.dtype);
    let out = if dtype == NumericDType::F64 {
        computed
    } else {
        let shape = computed.shape.clone();
        let NumericStorage::F64(values) = computed
            .into_numeric_storage()
            .map_err(|err| deep_learning_error("mtimes", err))?
        else {
            unreachable!("real matrix multiplication returns double storage")
        };
        floating_tensor_from_f64(values, shape, dtype, "mtimes")?
    };
    let node = if tape_is_active() && (lhs.node.is_some() || rhs.node.is_some()) {
        Some(push_node(
            out.clone(),
            NodeKind::MatMul {
                lhs: lhs.node,
                rhs: rhs.node,
                lhs_value: lhs.tensor.clone(),
                rhs_value: rhs.tensor.clone(),
            },
        )?)
    } else {
        None
    };
    Ok(wrap_result(result_metadata_owner(&lhs, &rhs), out, node))
}

#[runtime_builtin(
    name = "dlarray.sum",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(super) fn dlarray_sum_builtin(input: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest
        .iter()
        .any(|value| !matches!(value, Value::String(text) if text.eq_ignore_ascii_case("all")))
    {
        return Err(deep_learning_error(
            "sum",
            "sum: dlarray autodiff supports scalar reductions over all elements",
        ));
    }
    let payload = Payload::parse(input, "sum")?;
    let total = floating_tensor_values(&payload.tensor, "sum")?
        .iter()
        .sum::<f64>();
    let out = floating_tensor_from_f64(vec![total], vec![1, 1], payload.dtype, "sum")?;
    let node = if tape_is_active() {
        payload
            .node
            .map(|id| {
                push_node(
                    out.clone(),
                    NodeKind::SumAll {
                        input: id,
                        input_shape: payload.tensor.shape.clone(),
                    },
                )
            })
            .transpose()?
    } else {
        None
    };
    Ok(wrap_result(&payload, out, node))
}
