//! Runtime admission and typed normalization option parsing.
use super::*;
use crate::builtins::common::random_args::keyword_of;
use crate::gather_if_needed_async;
use runmat_builtins::NormalizeKeyword;

mod input;
use input::ensure_exact_integer_value;
pub(super) use input::value_to_input;

pub(super) async fn parse_args(
    input: &NumericInput,
    rest: Vec<Value>,
) -> BuiltinResult<ParsedArgs> {
    let rest = gather_rest(rest).await?;
    let keywords = rest
        .iter()
        .map(|value| keyword_of(value).map(|text| NormalizeKeyword::from_public_text(&text)))
        .collect::<Vec<_>>();
    let mut dim = None;
    let mut idx = 0usize;
    if let Some(first) = rest.first() {
        if keywords[0].is_none() {
            if let Some(parsed_dim) = tensor::dimension_from_value_async(first, "normalize", false)
                .await
                .map_err(normalize_error)?
            {
                if parsed_dim > MAX_NORMALIZE_DIM {
                    return Err(normalize_error(format!(
                        "normalize: dimension must be <= {MAX_NORMALIZE_DIM}"
                    )));
                }
                dim = Some(parsed_dim);
                idx = 1;
            }
        }
    }

    let mut real_plan = RealPlan::CenterScale {
        center: CenterSpec::Mean,
        scale: ScaleSpec::Std,
    };
    let mut complex_plan = ComplexPlan::CenterScale {
        center: ComplexCenterSpec::Mean,
        scale: ScaleSpec::Std,
    };
    if idx < rest.len() {
        let keyword = keywords[idx].as_ref().ok_or_else(|| {
            normalize_error(format!(
                "normalize: expected method string at argument {}",
                idx + 2
            ))
        })?;
        idx += 1;
        match keyword {
            NormalizeKeyword::ZScore => {
                let method_type = optional_keyword(&keywords, &mut idx);
                match method_type.as_ref().unwrap_or(&NormalizeKeyword::Std) {
                    NormalizeKeyword::Std => {
                        real_plan = RealPlan::CenterScale {
                            center: CenterSpec::Mean,
                            scale: ScaleSpec::Std,
                        };
                        complex_plan = ComplexPlan::CenterScale {
                            center: ComplexCenterSpec::Mean,
                            scale: ScaleSpec::Std,
                        };
                    }
                    NormalizeKeyword::Robust => {
                        real_plan = RealPlan::CenterScale {
                            center: CenterSpec::Median,
                            scale: ScaleSpec::Mad,
                        };
                        complex_plan = ComplexPlan::CenterScale {
                            center: ComplexCenterSpec::Median,
                            scale: ScaleSpec::Mad,
                        };
                    }
                    other => {
                        return Err(normalize_error(format!(
                            "normalize: unsupported zscore method type '{}'",
                            other.public_name()
                        )));
                    }
                }
            }
            NormalizeKeyword::Norm => {
                let p = if idx < rest.len() && keywords[idx].is_none() {
                    ensure_exact_integer_value(&rest[idx], "numeric parameter")?;
                    let p = scalar_f64(&rest[idx]).await?.ok_or_else(|| {
                        normalize_error("normalize: norm method type must be a scalar")
                    })?;
                    idx += 1;
                    p
                } else {
                    2.0
                };
                if p.is_nan() || p <= 0.0 {
                    return Err(normalize_error("normalize: norm order must be positive"));
                }
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::None,
                    scale: ScaleSpec::Norm(p),
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::None,
                    scale: ScaleSpec::Norm(p),
                };
            }
            NormalizeKeyword::Scale => {
                let spec = parse_scale_spec(&rest, &keywords, &mut idx).await?;
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::None,
                    scale: spec.clone(),
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::None,
                    scale: spec,
                };
            }
            NormalizeKeyword::Center => {
                let (real_center, complex_center) = parse_center_spec(
                    &rest,
                    &keywords,
                    &mut idx,
                    matches!(input, NumericInput::Complex { .. }),
                )
                .await?;
                real_plan = RealPlan::CenterScale {
                    center: real_center.clone(),
                    scale: ScaleSpec::One,
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: complex_center.clone(),
                    scale: ScaleSpec::One,
                };
                if idx < rest.len() && keywords[idx] == Some(NormalizeKeyword::Scale) {
                    idx += 1;
                    let scale = parse_scale_spec(&rest, &keywords, &mut idx).await?;
                    real_plan = RealPlan::CenterScale {
                        center: real_center,
                        scale: scale.clone(),
                    };
                    complex_plan = ComplexPlan::CenterScale {
                        center: complex_center,
                        scale,
                    };
                }
            }
            NormalizeKeyword::MedianIqr => {
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::Median,
                    scale: ScaleSpec::Iqr,
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::Median,
                    scale: ScaleSpec::Iqr,
                };
            }
            NormalizeKeyword::Range => {
                if matches!(input, NumericInput::Complex { .. }) {
                    return Err(normalize_error(
                        "normalize: range normalization is not supported for complex inputs",
                    ));
                }
                let bounds = parse_range_bounds(&rest, &keywords, &mut idx).await?;
                real_plan = RealPlan::Range { bounds };
            }
            NormalizeKeyword::DataVariables | NormalizeKeyword::ReplaceValues => {
                return Err(normalize_error(
                    "normalize: table name-value options are not supported for array inputs",
                ));
            }
            other => {
                return Err(normalize_error(format!(
                    "normalize: unsupported method '{}'",
                    other.public_name()
                )));
            }
        }
        if idx < rest.len() {
            return Err(normalize_error(
                "normalize: unexpected arguments after normalization method",
            ));
        }
    }

    Ok(ParsedArgs {
        dim,
        real_plan,
        complex_plan,
    })
}

async fn gather_rest(rest: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(rest.len());
    for value in rest {
        let gathered = gather_if_needed_async(&value)
            .await
            .map_err(|err| normalize_internal(format!("normalize: {err}")))?;
        out.push(gathered);
    }
    Ok(out)
}

fn optional_keyword(
    keywords: &[Option<NormalizeKeyword>],
    idx: &mut usize,
) -> Option<NormalizeKeyword> {
    let keyword = keywords.get(*idx)?.clone()?;
    *idx += 1;
    Some(keyword)
}

async fn parse_center_spec(
    rest: &[Value],
    keywords: &[Option<NormalizeKeyword>],
    idx: &mut usize,
    input_is_complex: bool,
) -> BuiltinResult<(CenterSpec, ComplexCenterSpec)> {
    if *idx >= rest.len() || keywords[*idx] == Some(NormalizeKeyword::Scale) {
        return Ok((CenterSpec::Mean, ComplexCenterSpec::Mean));
    }
    if let Some(keyword) = &keywords[*idx] {
        *idx += 1;
        return match keyword {
            NormalizeKeyword::Mean => Ok((CenterSpec::Mean, ComplexCenterSpec::Mean)),
            NormalizeKeyword::Median => Ok((CenterSpec::Median, ComplexCenterSpec::Median)),
            other => Err(normalize_error(format!(
                "normalize: unsupported center type '{}'",
                other.public_name()
            ))),
        };
    }
    let value = rest[*idx].clone();
    *idx += 1;
    ensure_exact_integer_value(&value, "numeric parameter")?;
    match value {
        Value::Complex(re, im) => {
            if !input_is_complex {
                return Err(normalize_error(
                    "normalize: complex explicit center is not supported for real inputs",
                ));
            }
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|err| normalize_internal(format!("normalize: {err}")))?;
            Ok((
                CenterSpec::Explicit(Tensor::new(vec![re], vec![1, 1]).unwrap()),
                ComplexCenterSpec::ExplicitComplex(tensor),
            ))
        }
        Value::ComplexTensor(tensor) => {
            if !input_is_complex {
                return Err(normalize_error(
                    "normalize: complex explicit center is not supported for real inputs",
                ));
            }
            Ok((
                CenterSpec::Explicit(real_part_tensor(&tensor)?),
                ComplexCenterSpec::ExplicitComplex(tensor),
            ))
        }
        other => {
            let tensor = tensor::value_into_tensor_for("normalize", other)
                .map_err(|err| normalize_error(format!("normalize: {err}")))?;
            Ok((
                CenterSpec::Explicit(tensor.clone()),
                ComplexCenterSpec::ExplicitReal(tensor),
            ))
        }
    }
}

async fn parse_scale_spec(
    rest: &[Value],
    keywords: &[Option<NormalizeKeyword>],
    idx: &mut usize,
) -> BuiltinResult<ScaleSpec> {
    if *idx >= rest.len() {
        return Ok(ScaleSpec::Std);
    }
    if let Some(keyword) = &keywords[*idx] {
        *idx += 1;
        return match keyword {
            NormalizeKeyword::Std => Ok(ScaleSpec::Std),
            NormalizeKeyword::Mad => Ok(ScaleSpec::Mad),
            NormalizeKeyword::Iqr => Ok(ScaleSpec::Iqr),
            NormalizeKeyword::First => Ok(ScaleSpec::First),
            other => Err(normalize_error(format!(
                "normalize: unsupported scale type '{}'",
                other.public_name()
            ))),
        };
    }
    ensure_exact_integer_value(&rest[*idx], "numeric parameter")?;
    let tensor = tensor::value_into_tensor_for("normalize", rest[*idx].clone())
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    *idx += 1;
    Ok(ScaleSpec::Explicit(tensor))
}

async fn parse_range_bounds(
    rest: &[Value],
    keywords: &[Option<NormalizeKeyword>],
    idx: &mut usize,
) -> BuiltinResult<RangeBounds> {
    if *idx >= rest.len() || keywords[*idx].is_some() {
        return Ok(RangeBounds {
            lower: 0.0,
            upper: 1.0,
        });
    }
    ensure_exact_integer_value(&rest[*idx], "numeric parameter")?;
    let tensor = tensor::value_into_tensor_for("normalize", rest[*idx].clone())
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    *idx += 1;
    let values = tensor::tensor_values_f64(&tensor);
    match values.as_slice() {
        [upper] => Ok(RangeBounds {
            lower: 0.0,
            upper: *upper,
        }),
        [lower, upper] if lower < upper => Ok(RangeBounds {
            lower: *lower,
            upper: *upper,
        }),
        _ => Err(normalize_error(
            "normalize: range target must be a scalar or increasing two-element vector",
        )),
    }
}

async fn scalar_f64(value: &Value) -> BuiltinResult<Option<f64>> {
    tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| normalize_error(format!("normalize: {err}")))
}
