//! Read only the admitted numerical controls, retaining provider failures.

use super::*;

impl AdmittedArgs {
    pub(in crate::builtins::stats::summary::normalize) async fn materialize(
        self,
    ) -> BuiltinResult<ParsedArgs> {
        let dim = if let Some(value) = self.dimension {
            let value = gather(value).await?;
            let dim = tensor::dimension_from_value_async(&value, "normalize", false)
                .await
                .map_err(normalize_error)?;
            if let Some(parsed) = dim {
                if parsed > MAX_NORMALIZE_DIM {
                    return Err(normalize_error(format!(
                        "normalize: dimension must be <= {MAX_NORMALIZE_DIM}"
                    )));
                }
            }
            dim
        } else {
            None
        };
        let mut real_plan = RealPlan::CenterScale {
            center: CenterSpec::Mean,
            scale: ScaleSpec::Std,
        };
        let mut complex_plan = ComplexPlan::CenterScale {
            center: ComplexCenterSpec::Mean,
            scale: ScaleSpec::Std,
        };
        match self.method {
            Method::ZScore { robust } => {
                if robust {
                    real_plan = RealPlan::CenterScale {
                        center: CenterSpec::Median,
                        scale: ScaleSpec::Mad,
                    };
                    complex_plan = ComplexPlan::CenterScale {
                        center: ComplexCenterSpec::Median,
                        scale: ScaleSpec::Mad,
                    };
                }
            }
            Method::Norm(order) => {
                let p = if let Some(value) = order {
                    let value = gather(value).await?;
                    ensure_exact_integer_value(&value, "numeric parameter")?;
                    tensor::scalar_f64_from_value_async(&value)
                        .await
                        .map_err(normalize_error)?
                        .ok_or_else(|| {
                            normalize_error("normalize: norm method type must be a scalar")
                        })?
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
            Method::Scale(spec) => {
                let scale = spec.materialize().await?;
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::None,
                    scale: scale.clone(),
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::None,
                    scale,
                };
            }
            Method::Center { center, scale } => {
                let (real_center, complex_center) = center.materialize().await?;
                let scale = match scale {
                    Some(scale) => scale.materialize().await?,
                    None => ScaleSpec::One,
                };
                real_plan = RealPlan::CenterScale {
                    center: real_center,
                    scale: scale.clone(),
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: complex_center,
                    scale,
                };
            }
            Method::MedianIqr => {
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::Median,
                    scale: ScaleSpec::Iqr,
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::Median,
                    scale: ScaleSpec::Iqr,
                };
            }
            Method::Range(bounds) => {
                let bounds = match bounds {
                    Some(value) => range_bounds(gather(value).await?)?,
                    None => RangeBounds {
                        lower: 0.0,
                        upper: 1.0,
                    },
                };
                real_plan = RealPlan::Range { bounds };
            }
        }
        Ok(ParsedArgs {
            dim,
            real_plan,
            complex_plan,
        })
    }
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(super::super::normalize_transfer_error)
}

impl CenterArg {
    async fn materialize(self) -> BuiltinResult<(CenterSpec, ComplexCenterSpec)> {
        match self {
            Self::Mean => Ok((CenterSpec::Mean, ComplexCenterSpec::Mean)),
            Self::Median => Ok((CenterSpec::Median, ComplexCenterSpec::Median)),
            Self::Explicit(value) => {
                let value = gather(value).await?;
                ensure_exact_integer_value(&value, "numeric parameter")?;
                match value {
                    Value::Complex(re, im) => {
                        let complex = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                            .map_err(|error| normalize_internal(format!("normalize: {error}")))?;
                        let real = Tensor::new(vec![re], vec![1, 1])
                            .map_err(|error| normalize_internal(format!("normalize: {error}")))?;
                        Ok((
                            CenterSpec::Explicit(real),
                            ComplexCenterSpec::ExplicitComplex(complex),
                        ))
                    }
                    Value::ComplexTensor(tensor) => Ok((
                        CenterSpec::Explicit(real_part_tensor(&tensor)?),
                        ComplexCenterSpec::ExplicitComplex(tensor),
                    )),
                    other => {
                        let tensor = tensor::value_into_tensor_for("normalize", other)
                            .map_err(normalize_error)?;
                        Ok((
                            CenterSpec::Explicit(tensor.clone()),
                            ComplexCenterSpec::ExplicitReal(tensor),
                        ))
                    }
                }
            }
        }
    }
}

impl ScaleArg {
    async fn materialize(self) -> BuiltinResult<ScaleSpec> {
        match self {
            Self::Std => Ok(ScaleSpec::Std),
            Self::Mad => Ok(ScaleSpec::Mad),
            Self::Iqr => Ok(ScaleSpec::Iqr),
            Self::First => Ok(ScaleSpec::First),
            Self::Explicit(value) => {
                let value = gather(value).await?;
                ensure_exact_integer_value(&value, "numeric parameter")?;
                tensor::value_into_tensor_for("normalize", value)
                    .map(ScaleSpec::Explicit)
                    .map_err(normalize_error)
            }
        }
    }
}

fn range_bounds(value: Value) -> BuiltinResult<RangeBounds> {
    ensure_exact_integer_value(&value, "numeric parameter")?;
    let tensor = tensor::value_into_tensor_for("normalize", value).map_err(normalize_error)?;
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
