use super::*;

pub(super) fn parse_constructor(
    mut args: crate::call::BuiltinArguments,
) -> BuiltinResult<InterpolantSpec> {
    let (method, extrapolation) = parse_trailing_methods(&mut args)?;
    match args.len() {
        0 => Ok(InterpolantSpec {
            grid_vectors: Vec::new(),
            values: Tensor::new(Vec::new(), vec![0, 0]).map_err(internal)?,
            method,
            extrapolation,
        }),
        1 => {
            let values = normalize_default_values(numeric_tensor(
                args.pop_front().expect("one value"),
                "V",
            )?)?;
            let grid_vectors = default_grid_vectors(&values);
            validate_grid_value_shape(&grid_vectors, &values, method, extrapolation)?;
            Ok(InterpolantSpec {
                grid_vectors,
                values,
                method,
                extrapolation,
            })
        }
        2 => {
            let values_arg = args.pop().expect("values");
            let grid_arg = args.pop().expect("grid");
            let values = numeric_tensor(values_arg, "V")?;
            let grid_vectors = match grid_arg {
                Value::Cell(cell) => grid_vectors_from_cell(cell)?,
                other => vec![numeric_vector(&other, "x")?],
            };
            let values = normalize_values_for_grid(&grid_vectors, values)?;
            validate_grid_value_shape(&grid_vectors, &values, method, extrapolation)?;
            Ok(InterpolantSpec {
                grid_vectors,
                values,
                method,
                extrapolation,
            })
        }
        _ => {
            let values = numeric_tensor(args.pop().expect("values"), "V")?;
            let grid_vectors = full_grid_vectors(args, &values)?;
            let values = normalize_values_for_grid(&grid_vectors, values)?;
            validate_grid_value_shape(&grid_vectors, &values, method, extrapolation)?;
            Ok(InterpolantSpec {
                grid_vectors,
                values,
                method,
                extrapolation,
            })
        }
    }
}

pub(super) fn numeric_tensor(value: Value, name: &str) -> BuiltinResult<Tensor> {
    match value {
        Value::Num(x) => Tensor::new(vec![x], vec![1, 1]).map_err(internal),
        Value::Int(value) => {
            ensure_exact_integer_scalar(&value, name)?;
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(internal)?;
            tensor_utils::integer_tensor_to_f64(tensor)
                .map_err(|err| invalid(format!("{name}: {err}")))
        }
        Value::Tensor(tensor) => match tensor.numeric_dtype() {
            NumericDType::F64 | NumericDType::F32 => Ok(tensor),
            _ => {
                ensure_exact_integer_tensor(&tensor, name)?;
                tensor_utils::integer_tensor_to_f64(tensor)
                    .map_err(|err| invalid(format!("{name}: {err}")))
            }
        },
        other => Err(invalid(format!(
            "{name} must be a real numeric array, got {other:?}"
        ))),
    }
}

pub(super) fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(x) => Ok(vec![*x]),
        Value::Int(value) => {
            ensure_exact_integer_scalar(value, name)?;
            Ok(vec![value.to_f64()])
        }
        Value::Tensor(tensor) if is_vector_shape(tensor.shape()) => {
            // Grid coordinates are evaluated in the interpolator's f64 domain.
            ensure_exact_integer_tensor(tensor, name)?;
            Ok(tensor_utils::tensor_values_f64(tensor))
        }
        Value::Tensor(_) => Err(invalid(format!("{name} must be a vector"))),
        other => Err(invalid(format!(
            "{name} must be a real numeric vector, got {other:?}"
        ))),
    }
}

pub(super) fn ensure_exact_integer_scalar(value: &IntValue, name: &str) -> BuiltinResult<()> {
    if crate::builtins::common::validation::integer_is_exact_f64(value) {
        Ok(())
    } else {
        Err(invalid(format!(
            "{name}: integer value must be exactly representable as double"
        )))
    }
}

pub(super) fn ensure_exact_integer_tensor(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    let exact = crate::builtins::common::validation::integer_is_exact_f64;
    if (0..storage.len()).all(|index| storage.value_at(index).as_ref().is_some_and(exact)) {
        Ok(())
    } else {
        Err(invalid(format!(
            "{name}: integer values must be exactly representable as double"
        )))
    }
}

pub(super) fn normalize_default_values(values: Tensor) -> BuiltinResult<Tensor> {
    if is_vector_shape(values.shape()) && values.shape().first().copied().unwrap_or(1) == 1 {
        let len = tensor_utils::tensor_element_len(&values);
        values.reshape(vec![len, 1]).map_err(internal)
    } else {
        Ok(values)
    }
}

pub(super) fn normalize_values_for_grid(
    grid_vectors: &[Vec<f64>],
    values: Tensor,
) -> BuiltinResult<Tensor> {
    if grid_vectors.len() == 1
        && is_vector_shape(values.shape())
        && tensor_utils::tensor_element_len(&values) == grid_vectors[0].len()
        && values.shape().first().copied().unwrap_or(1) == 1
    {
        values
            .reshape(vec![grid_vectors[0].len(), 1])
            .map_err(internal)
    } else {
        Ok(values)
    }
}

pub(super) fn is_vector_shape(shape: &[usize]) -> bool {
    shape.is_empty() || shape.iter().filter(|&&dim| dim > 1).count() <= 1
}

pub(super) fn grid_vectors_from_cell(cell: CellArray) -> BuiltinResult<Vec<Vec<f64>>> {
    if cell.data().is_empty() {
        return Err(invalid("gridVecs must contain at least one grid vector"));
    }
    cell.data()
        .iter()
        .enumerate()
        .map(|(idx, value)| numeric_vector(value, &format!("gridVecs{{{}}}", idx + 1)))
        .collect()
}

pub(super) fn default_grid_vectors(values: &Tensor) -> Vec<Vec<f64>> {
    let dims = effective_grid_rank(values.shape());
    (0..dims)
        .map(|dim| {
            let len = values.shape().get(dim).copied().unwrap_or(1);
            (1..=len).map(|value| value as f64).collect::<Vec<_>>()
        })
        .collect()
}

pub(super) fn effective_grid_rank(shape: &[usize]) -> usize {
    match shape.len() {
        0 => 1,
        1 => 1,
        2 if shape[0] == 1 || shape[1] == 1 => 1,
        _ => shape.len().max(1),
    }
}

pub(super) fn full_grid_vectors(
    grids: crate::call::BuiltinArguments,
    values: &Tensor,
) -> BuiltinResult<Vec<Vec<f64>>> {
    if grids.is_empty() {
        return Err(invalid("full-grid syntax requires at least one grid array"));
    }
    let mut tensors: Vec<Tensor> = Vec::with_capacity(grids.len());
    for (idx, grid) in grids.into_iter().enumerate() {
        let tensor = numeric_tensor(grid, &format!("X{}", idx + 1))?;
        if let Some(first) = tensors.first() {
            if tensor.shape() != first.shape() {
                return Err(invalid("full-grid arrays must have matching shapes"));
            }
        }
        tensors.push(tensor);
    }
    let grid_rank = tensors.len();
    let grid_shape = tensors[0].shape().to_vec();
    if grid_shape.len() != grid_rank {
        return Err(invalid(
            "full-grid arrays must have one dimension per grid variable",
        ));
    }
    if values.shape().len() < grid_rank || values.shape()[..grid_rank] != grid_shape[..] {
        return Err(invalid(
            "full-grid arrays must match the leading shape of V",
        ));
    }
    let mut vectors = Vec::with_capacity(grid_rank);
    for (dim, tensor) in tensors.iter().enumerate() {
        let len = grid_shape[dim];
        let values = tensor_utils::tensor_values_f64_cow(tensor);
        let mut vector = Vec::with_capacity(len);
        for idx in 0..len {
            let mut subs = vec![0; grid_rank];
            subs[dim] = idx;
            vector.push(values[column_major_offset(&subs, tensor.shape())]);
        }
        vectors.push(vector);
    }
    validate_rectilinear_full_grids(&tensors, &vectors, &grid_shape)?;
    Ok(vectors)
}

pub(super) fn validate_rectilinear_full_grids(
    tensors: &[Tensor],
    vectors: &[Vec<f64>],
    shape: &[usize],
) -> BuiltinResult<()> {
    let total = shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, len| acc.checked_mul(len))
        .ok_or_else(|| invalid("full-grid size exceeds platform limits"))?;
    for linear in 0..total {
        let subs = unravel_column_major(linear, shape);
        for (dim, tensor) in tensors.iter().enumerate() {
            let actual = tensor_utils::tensor_value_f64(tensor, linear);
            let expected = vectors[dim][subs[dim]];
            if actual != expected {
                return Err(invalid(
                    "full-grid arrays must be rectilinear ndgrid-style arrays",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_grid_value_shape(
    grid_vectors: &[Vec<f64>],
    values: &Tensor,
    method: InterpMethod,
    extrapolation: ExtrapolationMethod,
) -> BuiltinResult<()> {
    if grid_vectors.is_empty() {
        return Ok(());
    }
    validate_method_support(grid_vectors, method, extrapolation)?;
    if grid_vectors.len() > values.shape().len().max(1) {
        return Err(invalid("number of grid vectors exceeds V rank"));
    }
    for (dim, grid) in grid_vectors.iter().enumerate() {
        if grid.len() < 2 {
            return Err(invalid("each grid vector must contain at least two points"));
        }
        if grid.iter().any(|value| !value.is_finite()) {
            return Err(invalid("grid vectors must contain finite values"));
        }
        if !grid.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(invalid("grid vectors must be strictly increasing"));
        }
        let expected = values.shape().get(dim).copied().unwrap_or(1);
        if grid.len() != expected {
            return Err(invalid(format!(
                "grid vector {} length {} does not match V dimension {} length {}",
                dim + 1,
                grid.len(),
                dim + 1,
                expected
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_method_support(
    grid_vectors: &[Vec<f64>],
    method: InterpMethod,
    extrapolation: ExtrapolationMethod,
) -> BuiltinResult<()> {
    let rank = grid_vectors.len();
    if is_step_method(method) && rank != 1 {
        return Err(unsupported(
            "next and previous griddedInterpolant methods are currently supported for 1-D grids",
        ));
    }
    if is_step_extrapolation(extrapolation) && rank != 1 {
        return Err(unsupported(
            "next and previous extrapolation methods are currently supported for 1-D grids",
        ));
    }
    if matches!(method, InterpMethod::Pchip | InterpMethod::Makima) && rank != 1 {
        return Err(unsupported(
            "pchip and makima griddedInterpolant methods are currently supported for 1-D grids",
        ));
    }
    if matches!(
        extrapolation,
        ExtrapolationMethod::Pchip | ExtrapolationMethod::Makima
    ) && rank != 1
    {
        return Err(unsupported(
            "pchip and makima extrapolation methods are currently supported for 1-D grids",
        ));
    }
    if (method == InterpMethod::Spline || extrapolation == ExtrapolationMethod::Spline) && rank > 2
    {
        return Err(unsupported(
            "spline griddedInterpolant currently supports 1-D and 2-D grids",
        ));
    }
    match extrapolation {
        ExtrapolationMethod::Pchip if method != InterpMethod::Pchip => {
            return Err(unsupported(
                "pchip extrapolation currently requires pchip interpolation",
            ))
        }
        ExtrapolationMethod::Spline if method != InterpMethod::Spline => {
            return Err(unsupported(
                "spline extrapolation currently requires spline interpolation",
            ))
        }
        ExtrapolationMethod::Makima if method != InterpMethod::Makima => {
            return Err(unsupported(
                "makima extrapolation currently requires makima interpolation",
            ))
        }
        _ => {}
    }
    if (method == InterpMethod::Pchip || extrapolation == ExtrapolationMethod::Pchip)
        && grid_vectors[0].len() < 4
    {
        return Err(invalid("pchip requires at least four grid points"));
    }
    Ok(())
}
