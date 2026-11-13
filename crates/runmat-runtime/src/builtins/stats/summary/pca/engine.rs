use super::*;
pub(super) async fn evaluate(
    prepared: prepare::Prepared,
    options: &options::Options,
    count: usize,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<Vec<Value>> {
    let prepare::Prepared {
        normalized,
        original_rows,
        kept_rows,
        columns,
        degrees,
        means,
        scale,
        dtype,
    } = prepared;
    let n = kept_rows.len();
    cooperation.completed(0).await?;
    // This child owns only normalized numeric storage and text controls.
    // Reuse the exact lazy continuation owner instead of embedding SVD's
    // general resident-input future into PCA's caller poll frame.
    let make = move || async move {
        let svd_options = [Value::from("econ")];
        crate::builtins::math::linalg::factor::svd::evaluate(
            Value::Tensor(normalized),
            crate::call::FunctionArgumentView::from_slice(&svd_options),
        )
        .await
    };
    let factors = match runtime {
        Some(runtime) => {
            crate::execution::rooted_continuation::admit_native_hostcall_with_contextual_error(
                runtime, make,
            )?
            .await?
        }
        None => {
            crate::execution::rooted_continuation::admit_optional_hostcall_with(
                None,
                |source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        None,
                        "RunMat:pca:Continuation",
                        "PCA factorization continuation admission failed",
                        source,
                    )
                },
                make,
            )?
            .await?
        }
    };
    cooperation.completed(0).await?;
    let u = crate::builtins::common::tensor::value_into_tensor_for(NAME, factors.u())
        .map_err(|error| invalid(NAME, error))?;
    let v = crate::builtins::common::tensor::value_into_tensor_for(NAME, factors.v())
        .map_err(|error| invalid(NAME, error))?;
    let singular =
        crate::builtins::common::tensor::value_into_tensor_for(NAME, factors.singular_values())
            .map_err(|error| invalid(NAME, error))?;
    drop(factors);
    let all = degrees.min(columns);
    if singular.len() < all
        || u.rows() != n
        || u.cols() < all
        || v.rows() != columns
        || v.cols() != columns
    {
        return Err(invalid(
            NAME,
            "SVD factor geometry does not match the PCA request",
        ));
    }
    let singular = singular.materialize_f64();
    for chunk in singular.chunks(1024) {
        if chunk.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(invalid(
                NAME,
                "SVD did not produce finite nonnegative singular values",
            ));
        }
        cooperation.completed(chunk.len()).await?;
    }
    let tolerance = crate::builtins::common::linalg::svd_default_tolerance(&singular, n, columns);
    let k = match options.components {
        None => {
            if options.economy {
                all
            } else {
                columns
            }
        }
        Some(k) => {
            if options.economy {
                k.min(all)
            } else if k < degrees {
                k
            } else {
                columns
            }
        }
    };
    let variance_count = if options.economy { all } else { columns };
    let mut coefficients = if count >= 1 {
        Vec::with_capacity(
            columns
                .checked_mul(k)
                .ok_or_else(|| invalid(NAME, "coefficient geometry overflow"))?,
        )
    } else {
        Vec::new()
    };
    let mut scores = if count >= 2 {
        vec![
            f64::NAN;
            original_rows
                .checked_mul(k)
                .ok_or_else(|| invalid(NAME, "score geometry overflow"))?
        ]
    } else {
        Vec::new()
    };
    let mut tsquared = if count >= 4 {
        vec![f64::NAN; original_rows]
    } else {
        Vec::new()
    };
    if count >= 4 {
        for &row in &kept_rows {
            tsquared[row] = 0.0;
        }
    }
    let largest = singular.first().copied().unwrap_or(0.0);
    let mut total = 0.0;
    if count >= 5 && largest > 0.0 {
        for chunk in singular[..all].chunks(1024) {
            total += chunk
                .iter()
                .map(|&value| (value / largest).powi(2))
                .sum::<f64>();
            cooperation.completed(chunk.len()).await?;
        }
    }
    let mut latent = if count >= 3 {
        Vec::with_capacity(variance_count)
    } else {
        Vec::new()
    };
    let mut explained = if count >= 5 {
        Vec::with_capacity(variance_count)
    } else {
        Vec::new()
    };
    for component in 0..if count >= 3 { variance_count } else { 0 } {
        if component < all {
            let value = singular[component];
            latent.push((2.0 * scale.ln() + 2.0 * value.ln() - (degrees as f64).ln()).exp());
            let relative = if largest > 0.0 {
                value / largest
            } else {
                f64::NAN
            };
            if count >= 5 {
                explained.push((100.0 * relative) * relative / total);
            }
        } else {
            latent.push(0.0);
            if count >= 5 {
                explained.push(if total > 0.0 { 0.0 } else { f64::NAN });
            }
        }
        cooperation.completed(1).await?;
    }
    // T² uses every degrees-of-freedom-admissible PC before NumComponents.
    // Within that space, numerical nullspace
    // exclusion uses the same SVD tolerance owner as rank/pinv, not a clamp
    // of covariance eigenvalues or a fabricated positive variance.
    for (component, &value) in singular
        .iter()
        .take(if count >= 4 { all } else { 0 })
        .enumerate()
    {
        if value > tolerance {
            for (index, &row) in kept_rows.iter().enumerate() {
                let coordinate =
                    crate::builtins::common::tensor::tensor_value_f64(&u, component * n + index);
                tsquared[row] += (degrees as f64 * coordinate) * coordinate;
                if index % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
            cooperation.completed(n % 1024).await?;
        }
    }
    for component in 0..if count >= 1 { k } else { 0 } {
        let mut pivot = 0.0f64;
        for row in 0..columns {
            let value =
                crate::builtins::common::tensor::tensor_value_f64(&v, component * columns + row);
            if value.abs() > pivot.abs() {
                pivot = value;
            }
            if row % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(columns % 1024).await?;
        let sign = if pivot < 0.0 { -1.0 } else { 1.0 };
        for row in 0..columns {
            coefficients.push(
                sign * crate::builtins::common::tensor::tensor_value_f64(
                    &v,
                    component * columns + row,
                ),
            );
            if row % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(columns % 1024).await?;
        for (index, &row) in kept_rows
            .iter()
            .take(if count >= 2 { n } else { 0 })
            .enumerate()
        {
            scores[component * original_rows + row] = if component < all {
                product(
                    scale,
                    sign * crate::builtins::common::tensor::tensor_value_f64(
                        &u,
                        component * n + index,
                    ),
                    singular[component],
                )
            } else {
                0.0
            };
            if index % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(n % 1024).await?;
    }
    let mut outputs = Vec::with_capacity(count);
    if count >= 1 {
        outputs.push(output(coefficients, vec![columns, k], dtype)?);
    }
    if count >= 2 {
        outputs.push(output(scores, vec![original_rows, k], dtype)?);
    }
    if count >= 3 {
        outputs.push(output(latent, vec![variance_count, 1], dtype)?);
    }
    if count >= 4 {
        outputs.push(output(tsquared, vec![original_rows, 1], dtype)?);
    }
    if count >= 5 {
        outputs.push(output(explained, vec![variance_count, 1], dtype)?);
    }
    if count >= 6 {
        outputs.push(output(means, vec![1, columns], dtype)?);
    }
    Ok(outputs)
}
fn product(scale: f64, left: f64, right: f64) -> f64 {
    let value = scale * (left * right);
    if (value == 0.0 || !value.is_finite()) && left != 0.0 && right != 0.0 {
        (scale.ln() + left.abs().ln() + right.ln())
            .exp()
            .copysign(left)
    } else {
        value
    }
}
