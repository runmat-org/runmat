use super::*;

pub(super) async fn parse_spacings(
    args: crate::call::FunctionArgumentView<'_>,
    available_dims: &[usize],
    dim_lengths: &[usize],
    call: &super::super::numerical_execution::NumericalCall<'_>,
) -> BuiltinResult<Vec<GradientSpacing>> {
    match args.len() {
        0 => Ok(vec![GradientSpacing::Scalar(1.0); available_dims.len()]),
        1 => {
            let spacing = parse_spacing_argument(&args[0], dim_lengths[0], call).await?;
            if spacing.is_scalar() {
                Ok(vec![spacing; available_dims.len()])
            } else if available_dims.len() == 1 {
                Ok(vec![spacing])
            } else {
                Err(gradient_invalid_argument(
                    "gradient: coordinate-vector spacing for arrays requires one spacing argument per gradient dimension",
                ))
            }
        }
        count if count == available_dims.len() => {
            let mut spacings = Vec::with_capacity(args.len());
            for (value, &dim_len) in args.iter().zip(dim_lengths.iter()) {
                spacings.push(parse_spacing_argument(value, dim_len, call).await?);
            }
            Ok(spacings)
        }
        _ => Err(gradient_invalid_argument(format!(
            "gradient: expected 0, 1, or {} scalar/coordinate-vector spacing arguments",
            available_dims.len()
        ))),
    }
}

pub(super) async fn parse_spacing_argument(
    value: &Value,
    dim_len: usize,
    call: &super::super::numerical_execution::NumericalCall<'_>,
) -> BuiltinResult<GradientSpacing> {
    if let Value::GpuTensor(_) = value {
        call.fallback([value])?;
        let Value::GpuTensor(handle) = value else {
            unreachable!()
        };
        let readback =
            runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
                crate::runtime_error::with_provider_execution_source(
                    gradient_internal_error("spacing has no exact owning provider"),
                    source.into(),
                )
            })?;
        let gathered = gpu_helpers::download_readback_value_in_context(call.runtime, &readback)
            .await
            .map_err(super::super::numerical_execution::terminal)?;
        call.check()?;
        return parse_host_spacing_argument(&gathered, dim_len);
    }
    parse_host_spacing_argument(value, dim_len)
}

pub(super) fn parse_host_spacing_argument(
    value: &Value,
    dim_len: usize,
) -> BuiltinResult<GradientSpacing> {
    ensure_exact_integer_boundary(value, "spacing")?;
    let count = match value {
        Value::Tensor(t) => t.len(),
        Value::LogicalArray(a) => a.data.len(),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => 1,
        _ => {
            return Err(gradient_invalid_argument(
                "spacing must be real numeric or logical data",
            ))
        }
    };
    if count == 0 {
        return Err(gradient_invalid_argument(
            "gradient: empty spacing arguments are not supported",
        ));
    }
    let at = |index| match value {
        Value::Tensor(t) => tensor::tensor_value_f64(t, index),
        Value::LogicalArray(a) => {
            if a.data[index] != 0 {
                1.
            } else {
                0.
            }
        }
        Value::Num(n) => *n,
        Value::Int(n) => n.to_f64(),
        Value::Bool(n) => {
            if *n {
                1.
            } else {
                0.
            }
        }
        _ => unreachable!("admitted spacing"),
    };
    if count == 1 {
        let spacing = at(0);
        validate_scalar_spacing(spacing)?;
        return Ok(GradientSpacing::Scalar(spacing));
    }
    let coordinates = (0..count).map(at).collect::<Vec<_>>();
    validate_coordinate_spacing(&coordinates, dim_len)?;
    Ok(GradientSpacing::Coordinates(coordinates))
}

pub(super) fn validate_scalar_spacing(spacing: f64) -> BuiltinResult<()> {
    if !spacing.is_finite() {
        return Err(gradient_invalid_argument(
            "gradient: spacing must be finite",
        ));
    }
    if spacing == 0.0 {
        return Err(gradient_invalid_argument(
            "gradient: spacing must be nonzero",
        ));
    }
    Ok(())
}

pub(super) fn validate_coordinate_spacing(coords: &[f64], dim_len: usize) -> BuiltinResult<()> {
    if coords.len() != dim_len {
        return Err(gradient_invalid_argument(format!(
            "gradient: coordinate-vector spacing length {} does not match dimension length {dim_len}",
            coords.len()
        )));
    }

    if coords.iter().any(|coord| !coord.is_finite()) {
        return Err(gradient_invalid_argument(
            "gradient: coordinate-vector spacing must be finite",
        ));
    }

    if coords.len() <= 1 {
        return Ok(());
    }

    if coords[1] == coords[0] {
        return Err(gradient_invalid_argument(
            "gradient: coordinate-vector spacing points must be distinct",
        ));
    }

    for k in 1..coords.len() {
        if coords[k] == coords[k - 1] {
            return Err(gradient_invalid_argument(
                "gradient: coordinate-vector spacing points must be distinct",
            ));
        }
    }

    for k in 1..coords.len() - 1 {
        if coords[k + 1] == coords[k - 1] {
            return Err(gradient_invalid_argument(
                "gradient: coordinate-vector spacing cannot produce zero finite-difference denominator",
            ));
        }
    }
    Ok(())
}
