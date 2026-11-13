use super::*;

#[runtime_builtin(
    name = "dlarray.plus",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(crate) fn dlarray_plus_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
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
pub(crate) fn dlarray_minus_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
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
pub(crate) fn dlarray_times_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    dlarray_binary(lhs, rhs, "times", |a, b| a * b, record_multiply)
}

fn record_multiply(lhs: &Payload, rhs: &Payload, out: Tensor) -> BuiltinResult<Option<usize>> {
    if !tape_is_active() || (lhs.node.is_none() && rhs.node.is_none()) {
        return Ok(None);
    }
    let (lhs_values, rhs_values, _) =
        tensor::binary_numeric_tensors(&lhs.tensor, &rhs.tensor, "dlarray.times", "times")
            .map_err(|error| method_shape_error("times", error))?;
    Ok(Some(push_node(
        out,
        NodeKind::Mul {
            lhs: lhs.node,
            rhs: rhs.node,
            lhs_data: lhs_values,
            rhs_data: rhs_values,
            lhs_shape: lhs.tensor.shape().to_vec(),
            rhs_shape: rhs.tensor.shape().to_vec(),
        },
    )?))
}

#[runtime_builtin(
    name = "dlarray.rdivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(crate) fn dlarray_rdivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
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
            )
            .map_err(|error| method_shape_error("rdivide", error))?;
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
                    lhs_shape: lhs.tensor.shape().to_vec(),
                    rhs_shape: rhs.tensor.shape().to_vec(),
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
    super::super::errors::admit_single_output(
        function,
        &runmat_builtins::DLARRAY_METHOD_INVALID_INPUT,
    )?;
    let lhs = Payload::parse(lhs, function)?;
    let rhs = Payload::parse(rhs, function)?;
    dlarray_binary_payloads(&lhs, &rhs, function, op, record)
}

fn dlarray_binary_payloads(
    lhs: &Payload,
    rhs: &Payload,
    function: &'static str,
    op: impl Fn(f64, f64) -> f64,
    record: impl FnOnce(&Payload, &Payload, Tensor) -> BuiltinResult<Option<usize>>,
) -> BuiltinResult<Value> {
    let (lhs_values, rhs_values, shape) =
        tensor::binary_numeric_tensors(&lhs.tensor, &rhs.tensor, function, function)
            .map_err(|error| method_shape_error(function, error))?;
    let data = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(a, b)| op(*a, *b))
        .collect::<Vec<_>>();
    let dtype = floating_binary_dtype(lhs.dtype, rhs.dtype);
    let out = Tensor::new_with_dtype(data, shape, dtype)
        .map_err(|err| deep_learning_error(function, err))?;
    let node = record(lhs, rhs, out.clone())?;
    wrap_result(result_metadata_owner(lhs, rhs), out, node)
}

fn method_shape_error(function: &'static str, source: crate::RuntimeError) -> crate::RuntimeError {
    let message = source.message().to_owned();
    let mut error = crate::build_runtime_error(message)
        .with_builtin(function)
        .with_source(source);
    if let Some(identifier) = runmat_builtins::DLARRAY_METHOD_INVALID_INPUT.identifier {
        error = error.with_identifier(identifier);
    }
    error.build()
}

#[runtime_builtin(
    name = "dlarray.mtimes",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(crate) fn dlarray_mtimes_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    super::super::errors::admit_single_output(
        "mtimes",
        &runmat_builtins::DLARRAY_METHOD_INVALID_INPUT,
    )?;
    let lhs = Payload::parse(lhs, "mtimes")?;
    let rhs = Payload::parse(rhs, "mtimes")?;
    if tensor::tensor_element_len(&lhs.tensor) == 1 || tensor::tensor_element_len(&rhs.tensor) == 1
    {
        return dlarray_binary_payloads(&lhs, &rhs, "mtimes", |a, b| a * b, record_multiply);
    }
    if lhs
        .tensor
        .shape()
        .iter()
        .skip(2)
        .any(|dimension| *dimension != 1)
        || rhs
            .tensor
            .shape()
            .iter()
            .skip(2)
            .any(|dimension| *dimension != 1)
    {
        return Err(deep_learning_error(
            "mtimes",
            "mtimes: matrix multiplication requires two-dimensional operands",
        ));
    }
    let computed = crate::builtins::common::linalg::matmul_real(&lhs.tensor, &rhs.tensor)
        .map_err(|err| deep_learning_error("mtimes", err))?;
    let dtype = floating_binary_dtype(lhs.dtype, rhs.dtype);
    let out = if dtype == NumericDType::F64 {
        computed
    } else {
        let shape = computed.shape().to_vec();
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
    wrap_result(result_metadata_owner(&lhs, &rhs), out, node)
}

#[runtime_builtin(
    name = "dlarray.sum",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::autodiff::array_methods"
)]
pub(crate) fn dlarray_sum_builtin(
    input: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::super::errors::admit_single_output(
        "sum",
        &runmat_builtins::DLARRAY_METHOD_INVALID_INPUT,
    )?;
    if rest.iter().any(|value| {
        !super::super::scalar_text(value, "sum").is_ok_and(|text| text.eq_ignore_ascii_case("all"))
    }) {
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
                        input_shape: payload.tensor.shape().to_vec(),
                    },
                )
            })
            .transpose()?
    } else {
        None
    };
    wrap_result(&payload, out, node)
}
