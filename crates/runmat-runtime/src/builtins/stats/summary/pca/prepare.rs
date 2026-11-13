use super::*;
pub(super) struct Prepared {
    pub normalized: Tensor,
    pub original_rows: usize,
    pub kept_rows: Vec<usize>,
    pub columns: usize,
    pub degrees: usize,
    pub means: Vec<f64>,
    pub scale: f64,
    pub dtype: NumericDType,
}
pub(super) async fn prepare(
    input: &Tensor,
    options: &options::Options,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<Prepared> {
    if input.shape().iter().skip(2).any(|&extent| extent != 1) || input.is_empty() {
        return Err(invalid(NAME, "X must be a nonempty real matrix"));
    }
    let rows = input.shape().first().copied().unwrap_or(1);
    let columns = input.shape().get(1).copied().unwrap_or(1);
    if options.components.is_some_and(|k| k > columns) {
        return Err(invalid(
            NAME,
            "NumComponents cannot exceed the number of variables",
        ));
    }
    let mut kept_rows = Vec::with_capacity(rows);
    let mut scale = 0.0f64;
    for row in 0..rows {
        let mut missing = false;
        let mut infinite = false;
        let mut row_scale = 0.0f64;
        for column in 0..columns {
            let value =
                crate::builtins::common::tensor::tensor_value_f64(input, column * rows + row);
            missing |= value.is_nan();
            infinite |= value.is_infinite();
            row_scale = row_scale.max(value.abs());
            if column % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(columns % 1024).await?;
        if missing {
            if options.rows == runmat_builtins::PcaRows::All {
                return Err(invalid(
                    NAME,
                    "Rows='all' requires observations without NaNs",
                ));
            }
            continue;
        }
        if infinite {
            return Err(invalid(NAME, "PCA requires finite retained observations"));
        }
        kept_rows.push(row);
        scale = scale.max(row_scale);
    }
    let n = kept_rows.len();
    let degrees = n
        .checked_sub(usize::from(options.centered))
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            invalid(
                NAME,
                "PCA requires positive degrees of freedom after removing incomplete rows",
            )
        })?;
    let count = n
        .checked_mul(columns)
        .ok_or_else(|| invalid(NAME, "PCA workspace geometry overflow"))?;
    let mut normalized = Vec::with_capacity(count);
    let mut means = vec![0.0; columns];
    let mut column_values = Vec::with_capacity(if options.centered { rows } else { 0 });
    if options.centered {
        // Means use all available entries per column, including a finite
        // entry in a row omitted because another variable is NaN.
        for (column, mean) in means.iter_mut().enumerate() {
            column_values.clear();
            for row in 0..rows {
                let value =
                    crate::builtins::common::tensor::tensor_value_f64(input, column * rows + row);
                if !value.is_nan() {
                    if !value.is_finite() {
                        return Err(invalid(
                            NAME,
                            "centering requires finite available column observations",
                        ));
                    }
                    column_values.push(value);
                }
                if row % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
            cooperation.completed(rows % 1024).await?;
            *mean = inferential::moments(&column_values, cooperation)
                .await?
                .mean;
            if !mean.is_finite() {
                return Err(invalid(NAME, "available column means are not finite"));
            }
            scale = scale.max(mean.abs());
        }
    }
    // Unit rescaling of an all-zero matrix is an identity transformation,
    // not a fabricated variance, eigenvalue, or smoothing parameter.
    if scale == 0.0 {
        scale = 1.0;
    }
    for (column, &mean) in means.iter().enumerate() {
        for chunk in kept_rows.chunks(1024) {
            for &row in chunk {
                let value =
                    crate::builtins::common::tensor::tensor_value_f64(input, column * rows + row);
                let difference = value - mean;
                normalized.push(if difference.is_finite() {
                    difference / scale
                } else {
                    value / scale - mean / scale
                });
            }
            cooperation.completed(chunk.len()).await?;
        }
    }
    let normalized =
        Tensor::new(normalized, vec![n, columns]).map_err(|error| invalid(NAME, error))?;
    Ok(Prepared {
        normalized,
        original_rows: rows,
        kept_rows,
        columns,
        degrees,
        means,
        scale,
        dtype: input.numeric_dtype(),
    })
}
