use super::*;
#[derive(Clone, Copy)]
pub(super) enum CorrType {
    Pearson,
    Spearman,
    Kendall,
}

#[derive(Clone, Copy)]
pub(super) enum RowsMode {
    All,
    Complete,
    Pairwise,
}

#[derive(Clone, Copy)]
pub(super) enum Tail {
    Both,
    Right,
    Left,
}

pub(super) struct CorrArgs {
    pub(super) left: Tensor,
    pub(super) right: Option<Tensor>,
    pub(super) corr_type: CorrType,
    pub(super) rows: RowsMode,
    pub(super) tail: Tail,
    pub(super) weights: Option<Vec<f64>>,
}

pub(super) fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

pub(super) fn ensure_corr_integer_extensions(
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let second_is_data = match rest.first() {
        Some(candidate) => keyword_of(candidate)?.is_none(),
        None => false,
    };
    if is_typed_integer_value(value)
        || second_is_data && rest.first().is_some_and(is_typed_integer_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(&CORR_INTEGER_DATA_EXTENSION, NAME)?;
    }
    let mut integer_weights = false;
    for (name, value) in rest.iter().zip(rest.iter().skip(1)) {
        if keyword_of(name)?.is_some_and(|name| name.eq_ignore_ascii_case("weights"))
            && is_typed_integer_value(value)
        {
            integer_weights = true;
            break;
        }
    }
    if integer_weights {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CORR_INTEGER_WEIGHTS_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

pub(super) async fn parse_args(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<CorrArgs> {
    let left = value_to_tensor(value).await?;
    let mut right = None;
    let mut corr_type = CorrType::Pearson;
    let mut rows = RowsMode::All;
    let mut tail = Tail::Both;
    let mut weights = None;
    let mut idx = 0usize;
    if let Some(first) = rest.first() {
        if keyword_of(first)?.is_none() {
            right = Some(value_to_tensor(first.clone()).await?);
            idx = 1;
        }
    }
    while idx < rest.len() {
        if idx + 1 >= rest.len() {
            return Err(corr_error(
                "corr: name-value options must be provided in pairs",
            ));
        }
        let name = keyword_of(&rest[idx])?
            .ok_or_else(|| corr_error("corr: option names must be string scalars"))?;
        match name.to_ascii_lowercase().as_str() {
            "type" => {
                let option = keyword_of(&rest[idx + 1])?
                    .ok_or_else(|| corr_error("corr: Type must be a string scalar"))?;
                match option.to_ascii_lowercase().as_str() {
                    "pearson" => corr_type = CorrType::Pearson,
                    "spearman" => corr_type = CorrType::Spearman,
                    "kendall" => corr_type = CorrType::Kendall,
                    other => return Err(corr_error(format!("corr: unsupported Type '{other}'"))),
                }
            }
            "rows" => {
                let option = keyword_of(&rest[idx + 1])?
                    .ok_or_else(|| corr_error("corr: Rows must be a string scalar"))?;
                match option.to_ascii_lowercase().as_str() {
                    "all" => rows = RowsMode::All,
                    "complete" => rows = RowsMode::Complete,
                    "pairwise" => rows = RowsMode::Pairwise,
                    other => return Err(corr_error(format!("corr: unsupported Rows '{other}'"))),
                }
            }
            "tail" => {
                let option = keyword_of(&rest[idx + 1])?
                    .ok_or_else(|| corr_error("corr: Tail must be a string scalar"))?;
                match option.to_ascii_lowercase().as_str() {
                    "both" => tail = Tail::Both,
                    "right" => tail = Tail::Right,
                    "left" => tail = Tail::Left,
                    other => return Err(corr_error(format!("corr: unsupported Tail '{other}'"))),
                }
            }
            "weights" => {
                if weights.is_some() {
                    return Err(corr_error("corr: Weights can be specified only once"));
                }
                weights =
                    Some(value_to_weights(rest[idx + 1].clone(), matrix_shape(&left).0).await?);
            }
            other => return Err(corr_error(format!("corr: unsupported option '{other}'"))),
        }
        idx += 2;
    }
    Ok(CorrArgs {
        left,
        right,
        corr_type,
        rows,
        tail,
        weights,
    })
}

pub(super) async fn value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| corr_error(format!("corr: {err}")))?;
    let tensor = tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| corr_error(format!("corr: {err}")))?;
    if tensor
        .shape()
        .iter()
        .skip(2)
        .any(|dimension| *dimension != 1)
    {
        return Err(corr_error(
            "corr: observations must be a two-dimensional matrix",
        ));
    }
    Ok(tensor)
}

pub(super) async fn value_to_weights(
    value: Value,
    expected_rows: usize,
) -> BuiltinResult<Vec<f64>> {
    let tensor = value_to_tensor(value).await?;
    let (rows, cols) = matrix_shape(&tensor);
    if rows != expected_rows || cols != 1 {
        return Err(corr_error(format!(
            "corr: Weights must be an {expected_rows}-by-1 column vector"
        )));
    }
    let weights = tensor::tensor_into_values_f64(tensor);
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(corr_error(
            "corr: Weights must contain finite nonnegative values",
        ));
    }
    Ok(weights)
}

pub(super) fn matrix_shape(tensor: &Tensor) -> (usize, usize) {
    let len = tensor.len();
    match tensor.shape() {
        [] => (1, 1),
        [_] => (len, 1),
        [rows, cols, ..] => (*rows, *cols),
    }
}
