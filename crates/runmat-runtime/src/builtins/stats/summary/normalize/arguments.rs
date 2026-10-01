//! Runtime admission and typed normalization option parsing.
use super::*;
use crate::builtins::common::random_args::keyword_of;
use crate::gather_if_needed_async;
use runmat_value::{IntValue, NumericScalar};

pub(super) async fn value_to_input(value: Value) -> BuiltinResult<NumericInput> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| normalize_internal(format!("normalize: {err}")))?;
    match value {
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(&tensor, "input data")?;
            let shape = normalize_shape_for(&tensor.shape, tensor::tensor_element_len(&tensor));
            Ok(NumericInput::Real {
                shape,
                data: tensor::tensor_into_values_f64(tensor),
            })
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| normalize_internal(format!("normalize: {err}")))?;
            let len = tensor.len();
            let shape = normalize_shape_for(&tensor.shape, len);
            Ok(NumericInput::Real {
                shape,
                data: tensor::tensor_into_values_f64(tensor),
            })
        }
        Value::Num(n) => Ok(NumericInput::Real {
            data: vec![n],
            shape: vec![1, 1],
        }),
        Value::Int(i) => Ok(NumericInput::Real {
            data: vec![exact_integer_as_f64(&i, "input data")?],
            shape: vec![1, 1],
        }),
        Value::Bool(b) => Ok(NumericInput::Real {
            data: vec![if b { 1.0 } else { 0.0 }],
            shape: vec![1, 1],
        }),
        Value::Complex(re, im) => Ok(NumericInput::Complex {
            data: vec![(re, im)],
            shape: vec![1, 1],
        }),
        Value::ComplexTensor(tensor) => Ok(NumericInput::Complex {
            shape: {
                ensure_exact_complex_integer_tensor(&tensor, "input data")?;
                normalize_shape_for(&tensor.shape, tensor::complex_tensor_element_len(&tensor))
            },
            data: tensor::complex_tensor_into_values_complex64(tensor)
                .into_iter()
                .map(|value| (value.re, value.im))
                .collect(),
        }),
        other => Err(normalize_error(format!(
            "normalize: unsupported input type {other:?}"
        ))),
    }
}

pub(super) async fn parse_args(
    input: &NumericInput,
    rest: Vec<Value>,
) -> BuiltinResult<ParsedArgs> {
    let rest = gather_rest(rest).await?;
    let mut dim = None;
    let mut idx = 0usize;
    if let Some(first) = rest.first() {
        if keyword_of(first).is_none() {
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
        let keyword = keyword_of(&rest[idx]).ok_or_else(|| {
            normalize_error(format!(
                "normalize: expected method string at argument {}",
                idx + 2
            ))
        })?;
        idx += 1;
        match keyword.as_str() {
            "zscore" => {
                let method_type = optional_keyword(&rest, &mut idx);
                match method_type.as_deref().unwrap_or("std") {
                    "std" => {
                        real_plan = RealPlan::CenterScale {
                            center: CenterSpec::Mean,
                            scale: ScaleSpec::Std,
                        };
                        complex_plan = ComplexPlan::CenterScale {
                            center: ComplexCenterSpec::Mean,
                            scale: ScaleSpec::Std,
                        };
                    }
                    "robust" => {
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
                            "normalize: unsupported zscore method type '{other}'"
                        )));
                    }
                }
            }
            "norm" => {
                let p = if idx < rest.len() && keyword_of(&rest[idx]).is_none() {
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
            "scale" => {
                let spec = parse_scale_spec(&rest, &mut idx).await?;
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::None,
                    scale: spec.clone(),
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::None,
                    scale: spec,
                };
            }
            "center" => {
                let (real_center, complex_center) = parse_center_spec(
                    &rest,
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
                if idx < rest.len() && keyword_of(&rest[idx]).as_deref() == Some("scale") {
                    idx += 1;
                    let scale = parse_scale_spec(&rest, &mut idx).await?;
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
            "medianiqr" => {
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::Median,
                    scale: ScaleSpec::Iqr,
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::Median,
                    scale: ScaleSpec::Iqr,
                };
            }
            "range" => {
                if matches!(input, NumericInput::Complex { .. }) {
                    return Err(normalize_error(
                        "normalize: range normalization is not supported for complex inputs",
                    ));
                }
                let bounds = parse_range_bounds(&rest, &mut idx).await?;
                real_plan = RealPlan::Range { bounds };
            }
            "datavariables" | "replacevalues" => {
                return Err(normalize_error(
                    "normalize: table name-value options are not supported for array inputs",
                ));
            }
            other => {
                return Err(normalize_error(format!(
                    "normalize: unsupported method '{other}'"
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
        ensure_exact_integer_value(&gathered, "numeric parameter")?;
        out.push(gathered);
    }
    Ok(out)
}

fn ensure_exact_integer_value(value: &Value, role: &str) -> BuiltinResult<()> {
    match value {
        Value::Int(value) => {
            exact_integer_as_f64(value, role)?;
            Ok(())
        }
        Value::Tensor(tensor) => ensure_exact_integer_tensor(tensor, role),
        Value::ComplexTensor(tensor) => ensure_exact_complex_integer_tensor(tensor, role),
        _ => Ok(()),
    }
}

fn ensure_exact_complex_integer_tensor(tensor: &ComplexTensor, role: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    for index in 0..tensor.len() {
        let Some((real, imag)) = tensor.numeric_value_at(index) else {
            continue;
        };
        if !numeric_scalar_is_exact_f64(real) || !numeric_scalar_is_exact_f64(imag) {
            return Err(normalize_error(format!(
                "normalize: complex integer {role} must be exactly representable as double"
            )));
        }
    }
    Ok(())
}

fn numeric_scalar_is_exact_f64(value: NumericScalar) -> bool {
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    match value {
        NumericScalar::I8(value) => {
            (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&i128::from(value))
        }
        NumericScalar::I16(value) => {
            (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&i128::from(value))
        }
        NumericScalar::I32(value) => {
            (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&i128::from(value))
        }
        NumericScalar::I64(value) => {
            (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&i128::from(value))
        }
        NumericScalar::U8(value) => i128::from(value) <= MAX_EXACT_INTEGER,
        NumericScalar::U16(value) => i128::from(value) <= MAX_EXACT_INTEGER,
        NumericScalar::U32(value) => i128::from(value) <= MAX_EXACT_INTEGER,
        NumericScalar::U64(value) => i128::from(value) <= MAX_EXACT_INTEGER,
        NumericScalar::F32(_) | NumericScalar::F64(_) => true,
    }
}

fn exact_integer_as_f64(value: &IntValue, role: &str) -> BuiltinResult<f64> {
    const MAX_EXACT_INTEGER: u64 = 1 << 53;
    match value {
        IntValue::I8(value) => Ok(*value as f64),
        IntValue::I16(value) => Ok(*value as f64),
        IntValue::I32(value) => Ok(*value as f64),
        IntValue::I64(value) if value.unsigned_abs() <= MAX_EXACT_INTEGER => Ok(*value as f64),
        IntValue::U8(value) => Ok(*value as f64),
        IntValue::U16(value) => Ok(*value as f64),
        IntValue::U32(value) => Ok(*value as f64),
        IntValue::U64(value) if *value <= MAX_EXACT_INTEGER => Ok(*value as f64),
        _ => Err(normalize_error(format!(
            "normalize: integer {role} must be exactly representable as double"
        ))),
    }
}

fn ensure_exact_integer_tensor(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    for index in 0..tensor.len() {
        let exact = match tensor.numeric_value_at(index) {
            Some(NumericScalar::I8(value)) => i128::from(value),
            Some(NumericScalar::I16(value)) => i128::from(value),
            Some(NumericScalar::I32(value)) => i128::from(value),
            Some(NumericScalar::I64(value)) => i128::from(value),
            Some(NumericScalar::U8(value)) => i128::from(value),
            Some(NumericScalar::U16(value)) => i128::from(value),
            Some(NumericScalar::U32(value)) => i128::from(value),
            Some(NumericScalar::U64(value)) => i128::from(value),
            _ => continue,
        };
        if !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            return Err(normalize_error(format!(
                "normalize: integer {role} must be exactly representable as double"
            )));
        }
    }
    Ok(())
}

fn optional_keyword(rest: &[Value], idx: &mut usize) -> Option<String> {
    let keyword = rest.get(*idx).and_then(keyword_of)?;
    *idx += 1;
    Some(keyword)
}

async fn parse_center_spec(
    rest: &[Value],
    idx: &mut usize,
    input_is_complex: bool,
) -> BuiltinResult<(CenterSpec, ComplexCenterSpec)> {
    if *idx >= rest.len() || keyword_of(&rest[*idx]).as_deref() == Some("scale") {
        return Ok((CenterSpec::Mean, ComplexCenterSpec::Mean));
    }
    if let Some(keyword) = keyword_of(&rest[*idx]) {
        *idx += 1;
        return match keyword.as_str() {
            "mean" => Ok((CenterSpec::Mean, ComplexCenterSpec::Mean)),
            "median" => Ok((CenterSpec::Median, ComplexCenterSpec::Median)),
            other => Err(normalize_error(format!(
                "normalize: unsupported center type '{other}'"
            ))),
        };
    }
    let value = rest[*idx].clone();
    *idx += 1;
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

async fn parse_scale_spec(rest: &[Value], idx: &mut usize) -> BuiltinResult<ScaleSpec> {
    if *idx >= rest.len() {
        return Ok(ScaleSpec::Std);
    }
    if let Some(keyword) = keyword_of(&rest[*idx]) {
        *idx += 1;
        return match keyword.as_str() {
            "std" => Ok(ScaleSpec::Std),
            "mad" => Ok(ScaleSpec::Mad),
            "iqr" => Ok(ScaleSpec::Iqr),
            "first" => Ok(ScaleSpec::First),
            other => Err(normalize_error(format!(
                "normalize: unsupported scale type '{other}'"
            ))),
        };
    }
    let tensor = tensor::value_into_tensor_for("normalize", rest[*idx].clone())
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    *idx += 1;
    Ok(ScaleSpec::Explicit(tensor))
}

async fn parse_range_bounds(rest: &[Value], idx: &mut usize) -> BuiltinResult<RangeBounds> {
    if *idx >= rest.len() || keyword_of(&rest[*idx]).is_some() {
        return Ok(RangeBounds {
            lower: 0.0,
            upper: 1.0,
        });
    }
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
