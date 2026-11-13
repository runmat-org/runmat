use super::{
    admission::{BorrowedOperands, OptionInput, StartInput, StatsetInput},
    gather, invalid, tensor_rows_cols, Options, StartSpec, EPS, MAX_WORK_CELLS, NAME,
};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) async fn parse_options(
    args: Vec<OptionInput>,
    borrowed: &mut BorrowedOperands,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Options> {
    let mut options = Options::default();
    for arg in args {
        match arg {
            OptionInput::Distance(distance) => options.distance = distance,
            OptionInput::EmptyAction(action) => options.empty_action = action,
            OptionInput::MaxIter(value) => {
                let value = materialize(value, borrowed).await?;
                options.max_iter = parse_positive_usize(&value, "MaxIter")?;
            }
            OptionInput::Replicates(value) => {
                let value = materialize(value, borrowed).await?;
                options.replicates = parse_positive_usize(&value, "Replicates")?;
            }
            OptionInput::Start(StartInput::Named(start)) => options.start = StartSpec::Named(start),
            OptionInput::Start(StartInput::Prepared(start)) => options.start = start,
            OptionInput::Start(StartInput::Resident(value)) => {
                let value = materialize(value, borrowed).await?;
                if !super::admission::numeric_value_is_exact_cooperative(&value, cooperation)
                    .await?
                {
                    return Err(invalid("kmeans: typed-integer Start values must be exactly representable as double"));
                }
                options.start = parse_numeric_start_cooperative(value, cooperation).await?;
            }
            OptionInput::Display => {}
            OptionInput::OnlinePhase(toggle) => {
                options.online_phase = matches!(toggle, super::KmeansToggle::On)
            }
            OptionInput::Statset(fields) => {
                apply_options_struct(&mut options, fields, borrowed, cooperation).await?
            }
            OptionInput::EmptyStatset => {}
        }
        cooperation.completed(1).await?;
    }
    Ok(options)
}

async fn materialize(value: Value, borrowed: &mut BorrowedOperands) -> BuiltinResult<Value> {
    borrowed.record_consumed(&value)?;
    gather(value).await
}

pub(super) async fn parse_numeric_start_cooperative(
    value: Value,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<StartSpec> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|err| invalid(format!("kmeans: {err}")))?;
    let shape = tensor.shape();
    if shape.len() > 3 {
        return Err(invalid(
            "kmeans: numeric Start must be a matrix or 3-D array",
        ));
    }
    let cells = shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| invalid("kmeans: numeric Start size overflow"))?;
    if cells > MAX_WORK_CELLS {
        return Err(invalid("kmeans: requested clustering work is too large"));
    }
    if shape.len() == 3 {
        let k = shape[0];
        let cols = shape[1];
        let reps = shape[2];
        let mut pages = Vec::new();
        pages.try_reserve_exact(reps).map_err(|source| {
            super::internal_with_source(
                "kmeans: numeric Start pages exceed available memory",
                source,
            )
        })?;
        for page in 0..reps {
            let mut centers = start_centers(k, cols, cooperation).await?;
            for col in 0..cols {
                for (row, center_row) in centers.iter_mut().enumerate().take(k) {
                    let offset = row + col * k + page * k * cols;
                    let value = tensor::tensor_value_f64(&tensor, offset);
                    if !value.is_finite() {
                        return Err(invalid("kmeans: numeric Start values must be finite"));
                    }
                    center_row[col] = value;
                    cooperation.completed(1).await?;
                }
            }
            pages.push(centers);
        }
        Ok(StartSpec::NumericPages(pages))
    } else {
        let (rows, cols) = tensor_rows_cols(&tensor);
        let mut centers = start_centers(rows, cols, cooperation).await?;
        for col in 0..cols {
            for (row, center_row) in centers.iter_mut().enumerate().take(rows) {
                let value = tensor::tensor_value_f64(&tensor, col * rows + row);
                if !value.is_finite() {
                    return Err(invalid("kmeans: numeric Start values must be finite"));
                }
                center_row[col] = value;
                cooperation.completed(1).await?;
            }
        }
        Ok(StartSpec::Numeric(centers))
    }
}

async fn start_centers(
    rows: usize,
    cols: usize,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    let mut centers = Vec::new();
    centers.try_reserve_exact(rows).map_err(|source| {
        super::internal_with_source("kmeans: numeric Start exceeds available memory", source)
    })?;
    for _ in 0..rows {
        let mut row = Vec::new();
        row.try_reserve_exact(cols).map_err(|source| {
            super::internal_with_source("kmeans: numeric Start exceeds available memory", source)
        })?;
        for _ in 0..cols {
            row.push(0.0);
            cooperation.completed(1).await?;
        }
        centers.push(row);
    }
    Ok(centers)
}

pub(super) fn parse_positive_usize(value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid(format!("kmeans: {label} must be a positive integer scalar")));
    }
    let raw = scalar_number(value)
        .ok_or_else(|| invalid(format!("kmeans: {label} must be a positive integer scalar")))?;
    if !raw.is_finite() || raw < 1.0 || raw.fract().abs() > EPS || raw >= usize::MAX as f64 {
        return Err(invalid(format!(
            "kmeans: {label} must be a positive integer scalar"
        )));
    }
    Ok(raw as usize)
}

pub(super) fn parse_k(value: &Value, start: &StartSpec) -> BuiltinResult<usize> {
    if is_empty_numeric(value) {
        return infer_k_from_start(start).ok_or_else(|| {
            invalid("kmeans: k can be [] only when Start is a numeric start matrix or array")
        });
    }
    parse_positive_usize(value, "k")
}

fn infer_k_from_start(start: &StartSpec) -> Option<usize> {
    match start {
        StartSpec::Numeric(centers) => Some(centers.len()),
        StartSpec::NumericPages(pages) => pages.first().map(Vec::len),
        _ => None,
    }
}

fn scalar_number(value: &Value) -> Option<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return Some(integer.to_f64());
    }
    match value {
        Value::Num(value) => Some(*value),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

pub(super) use super::admission::is_empty_numeric;

async fn apply_options_struct(
    options: &mut Options,
    fields: Vec<StatsetInput>,
    borrowed: &mut BorrowedOperands,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<()> {
    for field in fields {
        match field {
            StatsetInput::MaxIter(value) => {
                let value = materialize(value, borrowed).await?;
                options.max_iter = parse_positive_usize(&value, "Options.MaxIter")?;
            }
            StatsetInput::Display => {}
            StatsetInput::UseParallel(value) => {
                let value = materialize(value, borrowed).await?;
                if bool_option(&value)? {
                    return Err(invalid(
                        "kmeans: Options.UseParallel is not supported by the CPU runtime",
                    ));
                }
            }
            StatsetInput::UseSubstreams(value) => {
                let value = materialize(value, borrowed).await?;
                if bool_option(&value)? {
                    return Err(invalid(
                        "kmeans: Options.UseSubstreams is not supported by the CPU runtime",
                    ));
                }
            }
            StatsetInput::Streams => {}
            StatsetInput::TolX(value) => {
                let value = materialize(value, borrowed).await?;
                validate_tolerance(&value, "Options.TolX")?;
            }
            StatsetInput::TolFun(value) => {
                let value = materialize(value, borrowed).await?;
                validate_tolerance(&value, "Options.TolFun")?;
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}

pub(super) fn validate_tolerance(value: &Value, label: &str) -> BuiltinResult<()> {
    let number = scalar_number(value)
        .ok_or_else(|| invalid(format!("kmeans: {label} must be a numeric scalar")))?;
    if !number.is_finite() || number < 0.0 {
        return Err(invalid(format!(
            "kmeans: {label} must be a nonnegative finite scalar"
        )));
    }
    Ok(())
}

pub(super) fn bool_option(value: &Value) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return Ok(!integer.is_zero());
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_value_f64(tensor, 0) != 0.0)
        }
        other => Err(invalid(format!(
            "kmeans: option value must be a logical scalar, got {other:?}"
        ))),
    }
}
