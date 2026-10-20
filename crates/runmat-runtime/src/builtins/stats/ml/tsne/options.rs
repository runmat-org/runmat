use super::{
    admission::{AdmittedOption, AdmittedStatsetField},
    invalid, CacheSize, Options, MAX_EMBEDDING_DIMS, MAX_ITERATIONS, MAX_NUM_PRINT, NAME,
};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) fn parse_options(values: Vec<AdmittedOption>) -> BuiltinResult<Options> {
    let mut options = Options::default();
    for option in values {
        match option {
            AdmittedOption::Algorithm(value) => options.algorithm = value,
            AdmittedOption::CacheSizeMaximal => options.cache_size = Some(CacheSize::Maximal),
            AdmittedOption::CacheSizeNumeric(value) => {
                options.cache_size =
                    Some(CacheSize::Megabytes(positive_scalar(&value, "CacheSize")?));
            }
            AdmittedOption::Distance(value) => options.distance = value,
            AdmittedOption::Exaggeration(value) => {
                options.exaggeration = bounded_scalar(&value, "Exaggeration", 1.0, f64::INFINITY)?;
            }
            AdmittedOption::NumDimensions(value) => {
                options.num_dimensions =
                    bounded_positive_integer(&value, "NumDimensions", MAX_EMBEDDING_DIMS)?;
            }
            AdmittedOption::NumPcaComponents(value) => {
                options.num_pca_components = nonnegative_integer(&value, "NumPCAComponents")?;
            }
            AdmittedOption::Perplexity(value) => {
                options.perplexity = positive_scalar(&value, "Perplexity")?
            }
            AdmittedOption::Standardize(value) => {
                options.standardize = bool_scalar(&value, "Standardize")?
            }
            AdmittedOption::InitialY(value) => {
                let tensor = tensor::value_into_tensor_for(NAME, value)
                    .map_err(|err| invalid(format!("tsne: InitialY {err}")))?;
                options.initial_y = Some(tensor);
            }
            AdmittedOption::LearnRate(value) => {
                options.learn_rate = positive_scalar(&value, "LearnRate")?
            }
            AdmittedOption::NumPrint(value) => {
                options.num_print = bounded_positive_integer(&value, "NumPrint", MAX_NUM_PRINT)?
            }
            AdmittedOption::Options(fields) => apply_options_struct(&mut options, fields)?,
            AdmittedOption::OptionsEmptyResident(_) => {
                return Err(invalid(
                    "tsne: Options must be materialized before execution",
                ));
            }
            AdmittedOption::Theta(value) => {
                options.theta = bounded_scalar(&value, "Theta", 0.0, 1.0)?
            }
            AdmittedOption::Verbose(value) => {
                let value = nonnegative_integer(&value, "Verbose")?;
                if value > 2 {
                    return Err(invalid("tsne: Verbose must be 0, 1, or 2"));
                }
                options.verbose = value;
            }
        }
    }
    Ok(options)
}

fn apply_options_struct(
    options: &mut Options,
    fields: Vec<AdmittedStatsetField>,
) -> BuiltinResult<()> {
    for field in fields {
        match field {
            AdmittedStatsetField::MaxIter(value) => {
                options.max_iter =
                    bounded_positive_integer(&value, "Options.MaxIter", MAX_ITERATIONS)?
            }
            AdmittedStatsetField::TolFun(value) => {
                options.tol_fun = positive_scalar(&value, "Options.TolFun")?
            }
            AdmittedStatsetField::Display | AdmittedStatsetField::EmptyOutputFcn => {}
            AdmittedStatsetField::EmptyOutputFcnResident(_) => {
                return Err(invalid(
                    "tsne: Options.OutputFcn must be materialized before execution",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn is_empty_option_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 0,
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        Value::StringArray(array) => array.data().is_empty(),
        Value::CharArray(array) => array.data().is_empty(),
        _ => false,
    }
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
