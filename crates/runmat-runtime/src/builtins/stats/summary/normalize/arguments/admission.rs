//! Assign public argument roles without reading provider-owned payloads.

use super::*;

/// Keywords are decoded once, and only selected numerical operands are retained.
pub(in crate::builtins::stats::summary::normalize) fn admit_args(
    domain: InputDomain,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<AdmittedArgs> {
    let keywords = rest
        .iter()
        .map(|value| {
            keyword_of(value).map(|text| text.map(|text| NormalizeKeyword::from_public_text(&text)))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let mut cursor = Cursor {
        rest,
        keywords,
        idx: 0,
    };
    let dimension = if cursor.peek_keyword().is_none() && cursor.peek().is_some_and(is_real_numeric)
    {
        let value = cursor.take().expect("peeked dimension");
        require_scalar(&value, "dimension")?;
        Some(value)
    } else {
        None
    };
    let method = if cursor.peek().is_none() {
        Method::ZScore { robust: false }
    } else {
        let keyword = cursor.take_keyword().ok_or_else(|| {
            normalize_error(format!(
                "normalize: expected method string at argument {}",
                cursor.idx + 2
            ))
        })?;
        match keyword {
            NormalizeKeyword::ZScore => {
                let robust = match cursor.take_keyword().unwrap_or(NormalizeKeyword::Std) {
                    NormalizeKeyword::Std => false,
                    NormalizeKeyword::Robust => true,
                    other => {
                        return Err(normalize_error(format!(
                            "normalize: unsupported zscore method type '{}'",
                            other.public_name()
                        )))
                    }
                };
                Method::ZScore { robust }
            }
            NormalizeKeyword::Norm => {
                let order = if cursor.peek().is_some() && cursor.peek_keyword().is_none() {
                    let value = cursor.take().expect("peeked norm order");
                    require_real_numeric(&value, "norm method type")?;
                    require_scalar(&value, "norm method type")?;
                    Some(value)
                } else {
                    None
                };
                Method::Norm(order)
            }
            NormalizeKeyword::Scale => Method::Scale(cursor.take_scale()?),
            NormalizeKeyword::Center => {
                let center = cursor.take_center(domain)?;
                let scale = if cursor.peek_keyword() == Some(&NormalizeKeyword::Scale) {
                    cursor.take();
                    Some(cursor.take_scale()?)
                } else {
                    None
                };
                Method::Center { center, scale }
            }
            NormalizeKeyword::MedianIqr => Method::MedianIqr,
            NormalizeKeyword::Range => {
                if domain == InputDomain::Complex {
                    return Err(normalize_error(
                        "normalize: range normalization is not supported for complex inputs",
                    ));
                }
                let bounds = if cursor.peek().is_some() && cursor.peek_keyword().is_none() {
                    let value = cursor.take().expect("peeked range bounds");
                    require_real_numeric(&value, "range target")?;
                    if let Some(len) = numeric_len(&value) {
                        if len != 1 && len != 2 {
                            return Err(normalize_error(
                                "normalize: range target must be a scalar or increasing two-element vector",
                            ));
                        }
                    }
                    Some(value)
                } else {
                    None
                };
                Method::Range(bounds)
            }
            NormalizeKeyword::DataVariables | NormalizeKeyword::ReplaceValues => {
                return Err(normalize_error(
                    "normalize: table name-value options are not supported for array inputs",
                ))
            }
            other => {
                return Err(normalize_error(format!(
                    "normalize: unsupported method '{}'",
                    other.public_name()
                )))
            }
        }
    };
    if cursor.peek().is_some() {
        return Err(normalize_error(
            "normalize: unexpected arguments after normalization method",
        ));
    }
    Ok(AdmittedArgs { dimension, method })
}

struct Cursor {
    rest: crate::call::BuiltinArguments,
    keywords: Vec<Option<NormalizeKeyword>>,
    idx: usize,
}

impl Cursor {
    fn peek(&self) -> Option<&Value> {
        self.rest.get(self.idx)
    }

    fn peek_keyword(&self) -> Option<&NormalizeKeyword> {
        self.keywords.get(self.idx)?.as_ref()
    }

    fn take(&mut self) -> Option<Value> {
        let value = self.peek()?.clone();
        self.idx += 1;
        Some(value)
    }

    fn take_keyword(&mut self) -> Option<NormalizeKeyword> {
        let keyword = self.peek_keyword()?.clone();
        self.idx += 1;
        Some(keyword)
    }

    fn take_scale(&mut self) -> BuiltinResult<ScaleArg> {
        if let Some(keyword) = self.take_keyword() {
            return match keyword {
                NormalizeKeyword::Std => Ok(ScaleArg::Std),
                NormalizeKeyword::Mad => Ok(ScaleArg::Mad),
                NormalizeKeyword::Iqr => Ok(ScaleArg::Iqr),
                NormalizeKeyword::First => Ok(ScaleArg::First),
                other => Err(normalize_error(format!(
                    "normalize: unsupported scale type '{}'",
                    other.public_name()
                ))),
            };
        }
        let Some(value) = self.take() else {
            return Ok(ScaleArg::Std);
        };
        require_real_numeric(&value, "scale type")?;
        Ok(ScaleArg::Explicit(value))
    }

    fn take_center(&mut self, domain: InputDomain) -> BuiltinResult<CenterArg> {
        if self.peek().is_none() || self.peek_keyword() == Some(&NormalizeKeyword::Scale) {
            return Ok(CenterArg::Mean);
        }
        if let Some(keyword) = self.take_keyword() {
            return match keyword {
                NormalizeKeyword::Mean => Ok(CenterArg::Mean),
                NormalizeKeyword::Median => Ok(CenterArg::Median),
                other => Err(normalize_error(format!(
                    "normalize: unsupported center type '{}'",
                    other.public_name()
                ))),
            };
        }
        let value = self.take().expect("peeked center");
        let complex = matches!(&value, Value::Complex(..) | Value::ComplexTensor(_))
            || matches!(&value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved);
        if complex {
            if domain == InputDomain::Real {
                return Err(normalize_error(
                    "normalize: complex explicit center is not supported for real inputs",
                ));
            }
            ensure_exact_integer_value(&value, "numeric parameter")?;
        } else {
            require_real_numeric(&value, "center type")?;
        }
        Ok(CenterArg::Explicit(value))
    }
}

fn is_real_numeric(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Tensor(_) | Value::LogicalArray(_)
    ) || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real)
}

fn require_real_numeric(value: &Value, role: &str) -> BuiltinResult<()> {
    if is_real_numeric(value) {
        ensure_exact_integer_value(value, "numeric parameter")?;
        Ok(())
    } else {
        Err(normalize_error(format!(
            "normalize: unsupported {role} input type {value:?}"
        )))
    }
}

fn numeric_len(value: &Value) -> Option<usize> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(..) => Some(1),
        Value::Tensor(value) => Some(value.len()),
        Value::ComplexTensor(value) => Some(value.len()),
        Value::LogicalArray(value) => Some(value.data.len()),
        Value::GpuTensor(handle) => handle
            .shape()
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim)),
        _ => None,
    }
}

fn require_scalar(value: &Value, role: &str) -> BuiltinResult<()> {
    if numeric_len(value) == Some(1) {
        Ok(())
    } else {
        Err(normalize_error(format!(
            "normalize: {role} must be a scalar"
        )))
    }
}
