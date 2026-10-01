use super::{
    invalid, tensor_rows_cols, Distance, EmptyAction, KmeansDisplay, KmeansOption, KmeansStart,
    KmeansStatsetField, KmeansToggle, Options, StartSpec, EPS, MAX_WORK_CELLS, NAME,
};
use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::BuiltinResult;
use runmat_value::{StructValue, Value};

/// Validate the public option structure before any provider access. Values
/// remain unexamined here because a numeric Start may itself be resident.
pub(super) fn parse_option_headers(args: &[Value]) -> BuiltinResult<()> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid("kmeans: options must be name-value pairs"));
    }
    for pair in args.chunks_exact(2) {
        let name =
            keyword_of(&pair[0]).ok_or_else(|| invalid("kmeans: option name must be text"))?;
        KmeansOption::from_keyword(&name)
            .ok_or_else(|| invalid(format!("kmeans: unsupported option '{name}'")))?;
    }
    Ok(())
}

pub(super) fn parse_options(args: Vec<Value>) -> BuiltinResult<Options> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid("kmeans: options must be name-value pairs"));
    }
    let mut options = Options::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let name =
            keyword_of(&args[idx]).ok_or_else(|| invalid("kmeans: option name must be text"))?;
        let option = KmeansOption::from_keyword(&name)
            .ok_or_else(|| invalid(format!("kmeans: unsupported option '{name}'")))?;
        let value = &args[idx + 1];
        match option {
            KmeansOption::Distance => options.distance = parse_distance(value)?,
            KmeansOption::EmptyAction => options.empty_action = parse_empty_action(value)?,
            KmeansOption::MaxIter => options.max_iter = parse_positive_usize(value, "MaxIter")?,
            KmeansOption::Replicates => {
                options.replicates = parse_positive_usize(value, "Replicates")?
            }
            KmeansOption::Start => options.start = parse_start(value)?,
            KmeansOption::Display => {
                let display = keyword_of(value)
                    .ok_or_else(|| invalid("kmeans: Display must be 'off', 'final', or 'iter'"))?;
                if KmeansDisplay::from_keyword(&display).is_none() {
                    return Err(invalid(format!("kmeans: unsupported Display '{display}'")));
                }
            }
            KmeansOption::OnlinePhase => {
                let flag = keyword_of(value)
                    .ok_or_else(|| invalid("kmeans: OnlinePhase must be 'off' or 'on'"))?;
                options.online_phase = match KmeansToggle::from_keyword(&flag) {
                    Some(KmeansToggle::Off) => false,
                    Some(KmeansToggle::On) => true,
                    None => {
                        return Err(invalid(format!("kmeans: unsupported OnlinePhase '{flag}'")))
                    }
                };
            }
            KmeansOption::Options => match value {
                Value::Struct(options_struct) => {
                    apply_options_struct(&mut options, options_struct)?
                }
                _ if is_empty_numeric(value) => {}
                _ => {
                    return Err(invalid(
                        "kmeans: Options must be [] or a statset options struct",
                    ))
                }
            },
        }
        idx += 2;
    }
    Ok(options)
}

fn parse_distance(value: &Value) -> BuiltinResult<Distance> {
    let text = keyword_of(value).ok_or_else(|| invalid("kmeans: Distance must be text"))?;
    Distance::from_keyword(&text)
        .ok_or_else(|| invalid(format!("kmeans: unsupported Distance '{text}'")))
}

fn parse_empty_action(value: &Value) -> BuiltinResult<EmptyAction> {
    let text = keyword_of(value).ok_or_else(|| invalid("kmeans: EmptyAction must be text"))?;
    EmptyAction::from_keyword(&text)
        .ok_or_else(|| invalid(format!("kmeans: unsupported EmptyAction '{text}'")))
}

fn parse_start(value: &Value) -> BuiltinResult<StartSpec> {
    if let Some(text) = keyword_of(value) {
        return KmeansStart::from_keyword(&text)
            .map(StartSpec::Named)
            .ok_or_else(|| invalid(format!("kmeans: unsupported Start '{text}'")));
    }
    let tensor = tensor::value_into_tensor_for(NAME, value.clone())
        .map_err(|err| invalid(format!("kmeans: {err}")))?;
    if tensor.shape.len() > 3 {
        return Err(invalid(
            "kmeans: numeric Start must be a matrix or 3-D array",
        ));
    }
    let cells = tensor
        .shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| invalid("kmeans: numeric Start size overflow"))?;
    if cells > MAX_WORK_CELLS {
        return Err(invalid("kmeans: requested clustering work is too large"));
    }
    let values = tensor::tensor_values_f64(&tensor);
    if tensor.shape.len() == 3 {
        let k = tensor.shape[0];
        let cols = tensor.shape[1];
        let reps = tensor.shape[2];
        let mut pages = Vec::with_capacity(reps);
        for page in 0..reps {
            let mut centers = vec![vec![0.0; cols]; k];
            for col in 0..cols {
                for (row, center_row) in centers.iter_mut().enumerate().take(k) {
                    let offset = row + col * k + page * k * cols;
                    center_row[col] = values[offset];
                }
            }
            pages.push(centers);
        }
        Ok(StartSpec::NumericPages(pages))
    } else {
        let (rows, cols) = tensor_rows_cols(&tensor);
        let mut centers = vec![vec![0.0; cols]; rows];
        for col in 0..cols {
            for (row, center_row) in centers.iter_mut().enumerate().take(rows) {
                center_row[col] = values[col * rows + row];
            }
        }
        Ok(StartSpec::Numeric(centers))
    }
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

pub(super) fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 0)
}

fn apply_options_struct(options: &mut Options, value: &StructValue) -> BuiltinResult<()> {
    for (name, field) in &value.fields {
        if is_empty_option_value(field) {
            continue;
        }
        match KmeansStatsetField::from_keyword(name) {
            KmeansStatsetField::MaxIter => {
                options.max_iter = parse_positive_usize(field, "Options.MaxIter")?
            }
            KmeansStatsetField::Display => {
                let display = keyword_of(field).ok_or_else(|| {
                    invalid("kmeans: Options.Display must be 'off', 'final', or 'iter'")
                })?;
                if KmeansDisplay::from_keyword(&display).is_none() {
                    return Err(invalid(format!(
                        "kmeans: unsupported Options.Display '{display}'"
                    )));
                }
            }
            KmeansStatsetField::UseParallel | KmeansStatsetField::UseSubstreams => {
                if bool_option(field)? {
                    return Err(invalid(format!(
                        "kmeans: Options.{name} is not supported by the CPU runtime"
                    )));
                }
            }
            KmeansStatsetField::Streams => {
                if !is_empty_numeric(field) {
                    return Err(invalid(
                        "kmeans: Options.Streams is not supported by the CPU runtime",
                    ));
                }
            }
            KmeansStatsetField::TolX | KmeansStatsetField::TolFun => {
                let value = scalar_number(field).ok_or_else(|| {
                    invalid(format!("kmeans: Options.{name} must be a numeric scalar"))
                })?;
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid(format!(
                        "kmeans: Options.{name} must be a nonnegative finite scalar"
                    )));
                }
            }
            KmeansStatsetField::Other => {}
        }
    }
    Ok(())
}

pub(super) fn is_empty_option_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 0,
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::StringArray(array) => array.data.is_empty(),
        Value::CharArray(array) => array.data.is_empty(),
        _ => false,
    }
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
