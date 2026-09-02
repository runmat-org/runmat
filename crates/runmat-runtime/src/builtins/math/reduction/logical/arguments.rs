use crate::builtins::common::{arg_tokens::ArgToken, spec::ReductionNaN, tensor};
use crate::{BuiltinResult, RuntimeError};
use runmat_value::Value;

use super::LogicalReductionConfig;

#[derive(Debug, Clone)]
pub(in crate::builtins::math::reduction) enum ReductionSpec {
    Default,
    Dimension(usize),
    Dimensions(Vec<usize>),
    All,
}

pub(super) async fn parse(
    config: &LogicalReductionConfig,
    arguments: &[Value],
) -> BuiltinResult<(ReductionSpec, ReductionNaN)> {
    let mut spec = ReductionSpec::Default;
    let mut nan_mode = config.default_nan_mode();
    let mut nan_mode_seen = false;
    let tokens = crate::builtins::common::arg_tokens::tokens_from_values(arguments);

    for (argument, token) in arguments.iter().zip(&tokens) {
        if matches!(token, ArgToken::String(text) if text == "all") {
            if !matches!(spec, ReductionSpec::Default) {
                return Err(config.error(
                    config.invalid_argument,
                    "the `all` selector cannot be combined with dimensions",
                ));
            }
            spec = ReductionSpec::All;
            continue;
        }
        if let Some(mode) = parse_nan_mode(config, token)? {
            if nan_mode_seen {
                return Err(config.error(
                    config.invalid_argument,
                    "only one NaN handling option may be specified",
                ));
            }
            crate::compatibility::ensure_builtin_extension_enabled(
                config.nanflag_extension,
                config.name(),
            )?;
            nan_mode = mode;
            nan_mode_seen = true;
            continue;
        }

        let dimensions = parse_dimensions(config, argument).await?;
        if dimensions.is_empty() {
            continue;
        }
        if !matches!(spec, ReductionSpec::Default) {
            return Err(config.error(
                config.invalid_argument,
                "only one dimension specification may be supplied",
            ));
        }
        spec = if dimensions.len() == 1 {
            ReductionSpec::Dimension(dimensions[0])
        } else {
            ReductionSpec::Dimensions(dimensions)
        };
    }
    Ok((spec, nan_mode))
}

fn parse_nan_mode(
    config: &LogicalReductionConfig,
    token: &ArgToken,
) -> BuiltinResult<Option<ReductionNaN>> {
    match token {
        ArgToken::String(text) => match text.as_str() {
            "omitnan" => Ok(Some(ReductionNaN::Omit)),
            "includenan" => Ok(Some(ReductionNaN::Include)),
            _ => Err(config.error(config.invalid_argument, format!("unknown option `{text}`"))),
        },
        _ => Ok(None),
    }
}

async fn parse_dimensions(
    config: &LogicalReductionConfig,
    value: &Value,
) -> BuiltinResult<Vec<usize>> {
    let dimensions = tensor::dims_from_value_async(value)
        .await
        .map_err(|message| map_dimension_error(config, message))?;
    let dimensions = match dimensions {
        Some(dimensions) => dimensions,
        None => match tensor::dimension_from_value_async(value, config.name(), false)
            .await
            .map_err(|message| map_dimension_error(config, message))?
        {
            Some(dimension) => vec![dimension],
            None => return Ok(Vec::new()),
        },
    };
    let mut unique = Vec::with_capacity(dimensions.len());
    for dimension in dimensions {
        if dimension < 1 {
            return Err(config.error(config.invalid_argument, "dimensions must be positive"));
        }
        if !unique.contains(&dimension) {
            unique.push(dimension);
        }
    }
    Ok(unique)
}

fn map_dimension_error(config: &LogicalReductionConfig, message: String) -> RuntimeError {
    let detail = if message.contains("finite") {
        "dimensions must be finite".to_string()
    } else if message.contains("integer") {
        "dimensions must be integers".to_string()
    } else if message.contains("non-negative") {
        "dimensions must be positive".to_string()
    } else {
        message
    };
    config.error(config.invalid_argument, detail)
}
