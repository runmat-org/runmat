use crate::builtins::common::tensor;
use crate::call::FunctionArgumentView;
use runmat_builtins::{CumulativeControl, CumulativeMissingMode, CumulativeScanDirection};
use runmat_value::Value;

pub(in crate::builtins::math::reduction) struct Options {
    pub dimension: Option<usize>,
    pub direction: CumulativeScanDirection,
    pub missing: CumulativeMissingMode,
    pub explicit_missing: bool,
}
pub(in crate::builtins::math::reduction) fn parse(
    args: FunctionArgumentView<'_>,
) -> Result<Options, String> {
    let args = match args.split_first() {
        Some((Value::Tensor(tensor), rest)) if tensor.is_empty() => rest,
        Some((Value::LogicalArray(array), rest)) if array.data.is_empty() => rest,
        _ => args,
    };
    if args.len() > 3 {
        return Err("expected at most three cumulative controls".into());
    }
    let mut result = Options {
        dimension: None,
        direction: CumulativeScanDirection::Forward,
        missing: CumulativeMissingMode::Include,
        explicit_missing: false,
    };
    let mut direction = false;
    for value in args {
        let numeric = matches!(value, Value::Num(_) | Value::Int(_))
            || matches!(value,Value::Tensor(t) if tensor::is_scalar_tensor(t));
        if numeric {
            if result.dimension.is_some() {
                return Err("dimension specified more than once".into());
            }
            result.dimension = Some(tensor::parse_dimension(value, "cumulative scan")?);
            continue;
        }
        if matches!(value,Value::Tensor(t) if t.is_empty())
            || matches!(value,Value::LogicalArray(t) if t.data.is_empty())
        {
            continue;
        }
        let control = match value {
            Value::String(runmat_value::StringElement::Text(text)) => {
                CumulativeControl::from_public_option(text.as_str())
            }
            Value::StringArray(array)
                if array.data().len() == 1 && !array.data()[0].is_missing() =>
            {
                CumulativeControl::from_public_option(
                    array.data()[0]
                        .text_ref()
                        .expect("checked nonmissing scalar string")
                        .as_str(),
                )
            }
            Value::CharArray(array) if array.rows == 1 && array.shape.len() <= 2 => {
                array.validate_layout()?;
                CumulativeControl::from_public_code_units(
                    array.data().iter().map(|unit| unit.raw()),
                )
            }
            _ => return Err("unsupported cumulative control type".into()),
        };
        match control {
            Some(CumulativeControl::Direction(value)) => {
                if direction {
                    return Err("direction specified more than once".into());
                }
                result.direction = value;
                direction = true;
            }
            Some(CumulativeControl::Missing(value)) => {
                if result.explicit_missing {
                    return Err("missing-value handling specified more than once".into());
                }
                result.missing = value;
                result.explicit_missing = true;
            }
            None => return Err("unrecognised option".into()),
        }
    }
    Ok(result)
}
