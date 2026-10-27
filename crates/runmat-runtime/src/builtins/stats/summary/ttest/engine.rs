use super::super::distribution_math::{
    student_t_cdf, student_t_cdf_upper, student_t_inv_upper_tail,
};
use super::*;
use runmat_builtins::TtestTail;

pub(super) async fn evaluate(
    x: &Tensor,
    y: Option<&Tensor>,
    options: options::Options,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<Vec<Value>> {
    let scalar_mean = y.is_some_and(|value| value.shape().iter().product::<usize>() == 1);
    if let Some(y) = y {
        if !scalar_mean
            && (0..x.shape().len().max(y.shape().len())).any(|axis| {
                x.shape().get(axis).copied().unwrap_or(1)
                    != y.shape().get(axis).copied().unwrap_or(1)
            })
        {
            return Err(invalid(
                NAME,
                "paired samples must have exactly the same size",
            ));
        }
    }
    let null_mean = if scalar_mean {
        crate::builtins::common::tensor::tensor_value_f64(y.expect("scalar mean"), 0)
    } else {
        0.0
    };
    if !null_mean.is_finite() {
        return Err(invalid(NAME, "hypothesized mean must be finite"));
    }
    let dim = options
        .dim
        .unwrap_or_else(|| x.shape().iter().position(|&n| n != 1).unwrap_or(0));
    let rank = x
        .shape()
        .len()
        .max(
            dim.checked_add(1)
                .ok_or_else(|| invalid(NAME, "dimension geometry overflow"))?,
        )
        .max(2);
    let mut shape = shape_buffer(rank)?;
    shape.extend_from_slice(x.shape());
    shape.resize(rank, 1);
    let extent = shape[dim];
    let stride = shape[..dim]
        .iter()
        .try_fold(1usize, |size, &extent| size.checked_mul(extent))
        .ok_or_else(|| invalid(NAME, "sample stride geometry overflow"))?;
    shape[dim] = 1;
    let lanes = shape
        .iter()
        .try_fold(1usize, |n, &next| n.checked_mul(next))
        .ok_or_else(|| invalid(NAME, "output geometry overflow"))?;
    let mut h = Vec::with_capacity(lanes);
    let mut p = Vec::with_capacity(lanes);
    let mut low = Vec::with_capacity(lanes);
    let mut high = Vec::with_capacity(lanes);
    let mut t = Vec::with_capacity(lanes);
    let mut df = Vec::with_capacity(lanes);
    let mut sd = Vec::with_capacity(lanes);
    let mut samples = Vec::with_capacity(extent);
    for lane in 0..lanes {
        samples.clear();
        let mut finite_difference_overflow = false;
        let mut all_pairs_finite = true;
        let mut pair_scale = 0.0f64;
        let base = (lane / stride) * stride * extent + lane % stride;
        for offset in 0..extent {
            let index = base + offset * stride;
            let a = crate::builtins::common::tensor::tensor_value_f64(x, index);
            let b = if scalar_mean {
                0.0
            } else {
                y.map(|y| crate::builtins::common::tensor::tensor_value_f64(y, index))
                    .unwrap_or(0.0)
            };
            if !a.is_nan() && !b.is_nan() {
                let difference = a - b;
                finite_difference_overflow |=
                    difference.is_infinite() && a.is_finite() && b.is_finite();
                all_pairs_finite &= a.is_finite() && b.is_finite();
                pair_scale = pair_scale.max(a.abs()).max(b.abs());
                samples.push(difference);
            }
            if offset % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(extent % 1024).await?;
        // Only exceptional finite paired subtraction needs a second pass.
        // Reuse the sample workspace in a shared coordinate system: t and
        // probabilities are scale-free; physical sd/CI are rescaled last.
        let coordinate_scale = if finite_difference_overflow && all_pairs_finite {
            samples.clear();
            for offset in 0..extent {
                let index = base + offset * stride;
                let a = crate::builtins::common::tensor::tensor_value_f64(x, index);
                let b = crate::builtins::common::tensor::tensor_value_f64(
                    y.expect("paired overflow"),
                    index,
                );
                if !a.is_nan() && !b.is_nan() {
                    let difference = a - b;
                    samples.push(if difference.is_finite() {
                        // Preserve cancellation already resolved accurately
                        // by the original subtraction, then change units.
                        difference / pair_scale
                    } else {
                        a / pair_scale - b / pair_scale
                    });
                }
                if offset % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
            cooperation.completed(extent % 1024).await?;
            pair_scale
        } else {
            1.0
        };
        let m = inferential::moments(&samples, cooperation).await?;
        let degrees = m.count as f64 - 1.0;
        let deviation = if m.count >= 2 {
            m.scale * (m.normalized_squares / degrees).sqrt()
        } else {
            f64::NAN
        };
        // The standard deviation can overflow while its standard error is
        // still representable. Do the count divisions before rescaling.
        let se = if m.count >= 2 {
            m.scale * ((m.normalized_squares / degrees) / m.count as f64).sqrt()
        } else {
            f64::NAN
        };
        let difference = m.mean - null_mean;
        let statistic = if difference.is_infinite() && m.mean.is_finite() && se.is_finite() {
            m.mean / se - null_mean / se
        } else {
            difference / se
        };
        let probability = if degrees <= 0.0 || statistic.is_nan() {
            f64::NAN
        } else {
            match options.tail {
                TtestTail::Both => (2.0 * student_t_cdf_upper(statistic.abs(), degrees)).min(1.0),
                TtestTail::Right => student_t_cdf_upper(statistic, degrees),
                TtestTail::Left => student_t_cdf(statistic, degrees),
            }
        };
        let critical = signed_upper_quantile(
            if options.tail == TtestTail::Both {
                options.alpha / 2.0
            } else {
                options.alpha
            },
            degrees,
        );
        h.push(u8::from(probability < options.alpha));
        p.push(probability);
        t.push(statistic);
        df.push(degrees);
        sd.push(deviation * coordinate_scale);
        low.push(if options.tail == TtestTail::Left {
            f64::NEG_INFINITY
        } else {
            confidence_bound(m.mean, se, -critical) * coordinate_scale
        });
        high.push(if options.tail == TtestTail::Right {
            f64::INFINITY
        } else {
            confidence_bound(m.mean, se, critical) * coordinate_scale
        });
    }
    let decision = if lanes == 1 {
        Value::Bool(h[0] != 0)
    } else {
        Value::LogicalArray(
            LogicalArray::new(h, shape.clone()).map_err(|error| invalid(NAME, error))?,
        )
    };
    let probability = inferential::numeric(NAME, p, shape.clone())?;
    // Confidence limits occupy the tested dimension, including row vectors.
    let mut ci_shape = shape.clone();
    ci_shape[dim] = 2;
    let mut ci = vec![
        0.0;
        lanes
            .checked_mul(2)
            .ok_or_else(|| invalid(NAME, "confidence geometry overflow"))?
    ];
    for lane in 0..lanes {
        let base = (lane / stride) * stride * 2 + lane % stride;
        ci[base] = low[lane];
        ci[base + stride] = high[lane];
    }
    let mut stats = runmat_value::UnboundedStructValueBuilder::new();
    stats.insert("tstat", inferential::numeric(NAME, t, shape.clone())?);
    stats.insert("df", inferential::numeric(NAME, df, shape.clone())?);
    stats.insert("sd", inferential::numeric(NAME, sd, shape)?);
    Ok(vec![
        decision,
        probability,
        Value::Tensor(Tensor::new(ci, ci_shape).map_err(|error| invalid(NAME, error))?),
        Value::Struct(stats.finish()),
    ])
}

fn confidence_bound(mean: f64, se: f64, critical: f64) -> f64 {
    let margin = critical * se;
    if margin.is_infinite() && mean.is_finite() && se.is_finite() && critical.is_finite() {
        (mean / se + critical) * se
    } else {
        mean + margin
    }
}

fn signed_upper_quantile(probability: f64, degrees: f64) -> f64 {
    if probability > 0.5 {
        -student_t_inv_upper_tail(1.0 - probability, degrees)
    } else {
        student_t_inv_upper_tail(probability, degrees)
    }
}

fn shape_buffer(rank: usize) -> BuiltinResult<Vec<usize>> {
    runmat_value::exact_capacity_vec(rank).map_err(|source| {
        crate::build_runtime_error("ttest: dimension metadata allocation failed")
            .with_builtin(NAME)
            .with_identifier("RunMat:ttest:AllocationFailed")
            .with_source(source)
            .build()
    })
}
