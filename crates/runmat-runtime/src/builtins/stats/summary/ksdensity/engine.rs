use super::*;
use runmat_builtins::KsdensityFunction;
pub(super) async fn grid(
    data: &[f64],
    width: f64,
    count: usize,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<Tensor> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for chunk in data.chunks(1024) {
        for &value in chunk {
            lo = lo.min(value);
            hi = hi.max(value);
        }
        cooperation.completed(chunk.len()).await?;
    }
    let lower = lo - 3.0 * width;
    let upper = hi + 3.0 * width;
    if !lower.is_finite() || !upper.is_finite() {
        return Err(invalid(
            NAME,
            "automatic grid endpoints are not finite; supply explicit evaluation points",
        ));
    }
    let mut values = buffer(count, runtime)?;
    for index in 0..count {
        let fraction = if count == 1 {
            1.0
        } else {
            index as f64 / (count - 1) as f64
        };
        values.push(if index == 0 && count > 1 {
            lower
        } else if index + 1 == count {
            upper
        } else {
            lower * (1.0 - fraction) + upper * fraction
        });
        if index % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    cooperation.completed(count % 1024).await?;
    Tensor::new(values, vec![1, count]).map_err(|error| invalid(NAME, error))
}
pub(super) async fn evaluate(
    data: &[f64],
    points: &Tensor,
    width: f64,
    function: KsdensityFunction,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<Vec<f64>> {
    let mut values = buffer(points.len(), runtime)?;
    let log_norm = -width.ln() - (data.len() as f64).ln();
    for index in 0..points.len() {
        let point = crate::builtins::common::tensor::tensor_value_f64(points, index);
        let mut value = if function == KsdensityFunction::Pdf {
            f64::NEG_INFINITY
        } else {
            0.0
        };
        let mut correction = 0.0;
        for chunk in data.chunks(1024) {
            for &sample in chunk {
                let difference = point - sample;
                let z = if point.is_infinite() {
                    // Infinite evaluation coordinates have exact kernel limits,
                    // even when the finite sample divided by width overflows.
                    point
                } else if difference.is_finite() {
                    difference / width
                } else {
                    point / width - sample / width
                };
                let contribution = match function {
                    KsdensityFunction::Pdf => {
                        super::super::distribution_math::standard_normal_log_pdf(z)
                    }
                    KsdensityFunction::Cdf => {
                        super::super::distribution_math::standard_normal_cdf(z)
                    }
                    KsdensityFunction::Survivor => {
                        super::super::distribution_math::standard_normal_cdf_upper(z)
                    }
                };
                if function == KsdensityFunction::Pdf {
                    value = log_add(value, contribution);
                    continue;
                }
                // Preserve true infinite/NaN estimates rather than producing
                // a spurious Inf-Inf compensation or clamping a PDF to one.
                if contribution.is_finite() && value.is_finite() {
                    let adjusted = contribution - correction;
                    let sum = value + adjusted;
                    correction = (sum - value) - adjusted;
                    value = sum;
                } else {
                    value += contribution;
                    correction = 0.0;
                }
            }
            cooperation.completed(chunk.len()).await?;
        }
        values.push(if function == KsdensityFunction::Pdf {
            (value + log_norm).exp()
        } else {
            value / data.len() as f64
        });
    }
    Ok(values)
}

fn log_add(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        return f64::NAN;
    }
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let high = left.max(right);
    let low = left.min(right);
    high + (low - high).exp().ln_1p()
}
