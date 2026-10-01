use super::{
    invalid, Algorithm, CacheSize, DistanceMetric, Options, TsneCacheSizeKeyword, TsneDisplay,
    TsneOption, TsneStatsetField, MAX_EMBEDDING_DIMS, MAX_ITERATIONS, MAX_NUM_PRINT, NAME,
};
use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) fn parse_options(values: Vec<Value>) -> BuiltinResult<Options> {
    if !values.len().is_multiple_of(2) {
        return Err(invalid("tsne: name-value options must be paired"));
    }
    let mut options = Options::default();
    let mut index = 0usize;
    while index < values.len() {
        let name =
            keyword_of(&values[index]).ok_or_else(|| invalid("tsne: option name must be text"))?;
        let option = TsneOption::from_keyword(name)
            .ok_or_else(|| invalid(format!("tsne: unknown option '{name}'")))?;
        let value = &values[index + 1];
        match option {
            TsneOption::Algorithm => {
                let text = text_scalar(value, "Algorithm")?;
                options.algorithm = Algorithm::from_keyword(&text)
                    .ok_or_else(|| invalid(format!("tsne: unsupported Algorithm '{text}'")))?;
            }
            TsneOption::CacheSize => {
                options.cache_size = Some(if let Some(text) = keyword_of(value) {
                    if TsneCacheSizeKeyword::from_keyword(text)
                        == Some(TsneCacheSizeKeyword::Maximal)
                    {
                        CacheSize::Maximal
                    } else {
                        return Err(invalid("tsne: CacheSize must be positive or 'maximal'"));
                    }
                } else {
                    CacheSize::Megabytes(positive_scalar(value, "CacheSize")?)
                });
            }
            TsneOption::Distance => {
                if matches!(
                    value,
                    Value::FunctionHandle(_)
                        | Value::ExternalFunctionHandle(_)
                        | Value::MethodFunctionHandle(_)
                        | Value::BoundFunctionHandle { .. }
                ) {
                    return Err(invalid(
                        "tsne: custom distance function handles are not supported yet",
                    ));
                }
                let text = text_scalar(value, "Distance")?;
                options.distance = DistanceMetric::from_keyword(&text)
                    .ok_or_else(|| invalid(format!("tsne: unsupported Distance '{text}'")))?;
            }
            TsneOption::Exaggeration => {
                options.exaggeration = bounded_scalar(value, "Exaggeration", 1.0, f64::INFINITY)?;
            }
            TsneOption::NumDimensions => {
                options.num_dimensions =
                    bounded_positive_integer(value, "NumDimensions", MAX_EMBEDDING_DIMS)?;
            }
            TsneOption::NumPcaComponents => {
                options.num_pca_components = nonnegative_integer(value, "NumPCAComponents")?;
            }
            TsneOption::Perplexity => options.perplexity = positive_scalar(value, "Perplexity")?,
            TsneOption::Standardize => options.standardize = bool_scalar(value, "Standardize")?,
            TsneOption::InitialY => {
                let tensor = tensor::value_into_tensor_for(NAME, value.clone())
                    .map_err(|err| invalid(format!("tsne: InitialY {err}")))?;
                options.initial_y = Some(tensor);
            }
            TsneOption::LearnRate => options.learn_rate = positive_scalar(value, "LearnRate")?,
            TsneOption::NumPrint => {
                options.num_print = bounded_positive_integer(value, "NumPrint", MAX_NUM_PRINT)?
            }
            TsneOption::Options => apply_options_struct(&mut options, value)?,
            TsneOption::Theta => options.theta = bounded_scalar(value, "Theta", 0.0, 1.0)?,
            TsneOption::Verbose => {
                let value = nonnegative_integer(value, "Verbose")?;
                if value > 2 {
                    return Err(invalid("tsne: Verbose must be 0, 1, or 2"));
                }
                options.verbose = value;
            }
        }
        index += 2;
    }
    Ok(options)
}

fn apply_options_struct(options: &mut Options, value: &Value) -> BuiltinResult<()> {
    let Value::Struct(fields) = value else {
        if is_empty_numeric(value) {
            return Ok(());
        }
        return Err(invalid("tsne: Options must be a struct"));
    };
    for (name, field) in &fields.fields {
        if is_empty_option_value(field) {
            continue;
        }
        let field_kind = TsneStatsetField::from_keyword(name)
            .ok_or_else(|| invalid(format!("tsne: unsupported Options field '{name}'")))?;
        match field_kind {
            TsneStatsetField::MaxIter => {
                options.max_iter =
                    bounded_positive_integer(field, "Options.MaxIter", MAX_ITERATIONS)?
            }
            TsneStatsetField::TolFun => options.tol_fun = positive_scalar(field, "Options.TolFun")?,
            TsneStatsetField::Display => {
                let display = text_scalar(field, "Options.Display")?;
                if TsneDisplay::from_keyword(&display).is_none() {
                    return Err(invalid(format!(
                        "tsne: unsupported Options.Display '{display}'"
                    )));
                }
            }
            TsneStatsetField::OutputFcn => {
                if !is_empty_numeric(field) {
                    return Err(invalid(
                        "tsne: Options.OutputFcn is not supported by the CPU runtime",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn is_empty_option_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 0,
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::StringArray(array) => array.data.is_empty(),
        Value::CharArray(array) => array.data.is_empty(),
        _ => false,
    }
}

fn text_scalar(value: &Value, name: &str) -> BuiltinResult<String> {
    keyword_of(value).ok_or_else(|| invalid(format!("tsne: {name} must be text")))
}

fn positive_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    let value = numeric_scalar(value, name)?;
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!("tsne: {name} must be a positive scalar")));
    }
    Ok(value)
}

fn bounded_scalar(value: &Value, name: &str, min: f64, max: f64) -> BuiltinResult<f64> {
    let value = numeric_scalar(value, name)?;
    if !value.is_finite() || value < min || value > max {
        return Err(invalid(format!(
            "tsne: {name} must be in the supported numeric range"
        )));
    }
    Ok(value)
}

fn bounded_positive_integer(value: &Value, name: &str, max: usize) -> BuiltinResult<usize> {
    integer_in_range(value, name, 1, max)
}

fn bounded_nonnegative_integer(value: &Value, name: &str, max: usize) -> BuiltinResult<usize> {
    integer_in_range(value, name, 0, max)
}

fn integer_in_range(value: &Value, name: &str, min: usize, max: usize) -> BuiltinResult<usize> {
    let integer = match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    };
    if let Some(value) = integer {
        if let Some(value) = value
            .try_to_usize()
            .filter(|value| *value >= min && *value <= max)
        {
            return Ok(value);
        }
        let kind = if min == 0 {
            "a nonnegative integer"
        } else {
            "a positive integer"
        };
        return Err(invalid(format!(
            "tsne: {name} must be {kind} no greater than {max}"
        )));
    }
    let value = numeric_scalar(value, name)?;
    // `usize::MAX as f64` rounds to 2^N. That endpoint is outside usize,
    // even though a saturating float-to-integer cast would appear to accept it.
    let exceeds_max = if max == usize::MAX {
        value >= max as f64
    } else {
        value > max as f64
    };
    if !value.is_finite() || value.fract() != 0.0 || value < min as f64 || exceeds_max {
        let kind = if min == 0 {
            "a nonnegative integer"
        } else {
            "a positive integer"
        };
        return Err(invalid(format!(
            "tsne: {name} must be {kind} no greater than {max}"
        )));
    }
    Ok(value as usize)
}

fn nonnegative_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    bounded_nonnegative_integer(value, name, usize::MAX)
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_value_f64(tensor, 0))
        }
        _ => Err(invalid(format!("tsne: {name} must be a numeric scalar"))),
    }
}

fn bool_scalar(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let raw = tensor::tensor_value_f64(tensor, 0);
            if raw == 0.0 || raw == 1.0 {
                Ok(raw != 0.0)
            } else {
                Err(invalid(format!("tsne: {name} must be logical scalar")))
            }
        }
        _ => Err(invalid(format!("tsne: {name} must be logical scalar"))),
    }
}

pub(super) fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 0)
}
