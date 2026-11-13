//! Borrowed controls after the rooted caller has gathered resident controls.
use super::keywords::{keyword, Keyword};
use super::*;
mod counts;
mod samples;
pub(super) use counts::{parse_variance_normalization, parse_window};
use samples::{parse_sample_points, parse_sample_window};

pub(super) async fn parse_moving_args(
    name: &'static str,
    op: MovingOp,
    k: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ParsedMoving> {
    let mut dim = None;
    let mut nan_mode = op.default_nan_mode();
    let mut endpoints = Endpoints::Shrink;
    let mut sample_points: Option<Vec<f64>> = None;
    let mut normalization = VarianceNormalization::Sample;
    let mut normalization_seen = false;
    let mut idx = 0;
    while idx < rest.len() {
        if let Some(keyword) = keyword(&rest[idx]) {
            match keyword {
                Keyword::Omit => {
                    nan_mode = NanMode::Omit;
                    idx += 1;
                    continue;
                }
                Keyword::Include => {
                    nan_mode = NanMode::Include;
                    idx += 1;
                    continue;
                }
                Keyword::Endpoints => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(invalid_argument(
                            name,
                            format!("{name}: Endpoints requires a value"),
                        ));
                    };
                    endpoints = parse_endpoints(name, value)?;
                    idx += 2;
                    continue;
                }
                Keyword::SamplePoints => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(invalid_argument(
                            name,
                            format!("{name}: SamplePoints requires a value"),
                        ));
                    };
                    if sample_points.is_some() {
                        return Err(invalid_argument(
                            name,
                            format!("{name}: multiple SamplePoints options provided"),
                        ));
                    }
                    sample_points = Some(parse_sample_points(name, value).await?);
                    idx += 2;
                    continue;
                }
                _ => {
                    if rest.get(idx + 1).is_some() && !looks_like_dimension(&rest[idx]).await? {
                        return Err(invalid_argument(
                            name,
                            format!(
                                "{name}: unrecognised option '{}'",
                                value_as_string(name, &rest[idx])?.unwrap_or_default()
                            ),
                        ));
                    }
                }
            }
        }

        if op.uses_variance_normalization() && !normalization_seen {
            if let Some(parsed_normalization) = parse_variance_normalization(name, &rest[idx])? {
                normalization = parsed_normalization;
                normalization_seen = true;
                idx += 1;
                continue;
            }
        }

        if dim.is_none() {
            if let Some(parsed_dim) = tensor::dimension_from_value_async(&rest[idx], name, false)
                .await
                .map_err(|err| invalid_argument(name, err))?
            {
                dim = Some(parsed_dim);
                idx += 1;
                continue;
            }
        }

        return Err(invalid_argument(
            name,
            format!("{name}: unrecognised argument {:?}", rest[idx]),
        ));
    }
    if sample_points.is_some() && !matches!(endpoints, Endpoints::Shrink) {
        return Err(invalid_argument(
            name,
            format!("{name}: SamplePoints currently requires Endpoints 'shrink'"),
        ));
    }
    let window = if sample_points.is_some() {
        parse_sample_window(name, k).await?
    } else {
        parse_window(name, k)?
    };
    Ok(ParsedMoving {
        window,
        dim,
        nan_mode,
        endpoints,
        sample_points,
        normalization,
    })
}

async fn looks_like_dimension(value: &Value) -> BuiltinResult<bool> {
    tensor::dimension_from_value_async(value, "moving", false)
        .await
        .map(|dim| dim.is_some())
        .map_err(|err| invalid_argument("moving", err))
}

fn parse_endpoints(name: &'static str, value: &Value) -> BuiltinResult<Endpoints> {
    if let Some(text) = keyword(value) {
        return match text {
            Keyword::Shrink => Ok(Endpoints::Shrink),
            Keyword::Discard => Ok(Endpoints::Discard),
            Keyword::Fill => Ok(Endpoints::FillDefault),
            _ => Err(invalid_argument(
                name,
                format!(
                    "{name}: unsupported Endpoints value '{}'",
                    value_as_string(name, value)?
                        .unwrap_or_default()
                        .to_ascii_lowercase()
                ),
            )),
        };
    }
    let scalar = scalar_numeric(value).ok_or_else(|| {
        invalid_argument(
            name,
            format!("{name}: Endpoints must be a keyword or scalar fill value"),
        )
    })?;
    Ok(Endpoints::Fill(scalar))
}
