use super::*;
pub(super) fn corr_compute(args: CorrArgs) -> BuiltinResult<Value> {
    let (left_rows, left_cols) = matrix_shape(&args.left);
    let right = args.right.as_ref().unwrap_or(&args.left);
    let (right_rows, right_cols) = matrix_shape(right);
    if left_rows != right_rows {
        return Err(corr_error(
            "corr: X and Y must have the same number of observations",
        ));
    }
    let result_len = left_cols
        .checked_mul(right_cols)
        .ok_or_else(|| corr_error("corr: result dimensions exceed platform limits"))?;
    let mut rho_data = Vec::new();
    let mut p_data = Vec::new();
    rho_data
        .try_reserve_exact(result_len)
        .map_err(|_| corr_internal("corr: cannot allocate coefficient output"))?;
    p_data
        .try_reserve_exact(result_len)
        .map_err(|_| corr_internal("corr: cannot allocate p-value output"))?;
    let complete_mask = match args.rows {
        RowsMode::Complete => Some(complete_rows(
            &args.left, right, left_rows, left_cols, right_cols,
        )),
        RowsMode::All | RowsMode::Pairwise => None,
    };
    let all_mask = vec![true; left_rows];
    for right_col in 0..right_cols {
        let y = CorrColumn::from_tensor(right, right_rows, right_col)?;
        for left_col in 0..left_cols {
            let x = CorrColumn::from_tensor(&args.left, left_rows, left_col)?;
            let pair = match complete_mask.as_ref() {
                Some(mask) => corr_pair_masked(
                    &x,
                    &y,
                    mask,
                    args.corr_type,
                    RowsMode::All,
                    args.weights.as_deref(),
                ),
                None => corr_pair_masked(
                    &x,
                    &y,
                    &all_mask,
                    args.corr_type,
                    args.rows,
                    args.weights.as_deref(),
                ),
            };
            rho_data.push(pair.rho);
            p_data.push(if args.weights.is_some() {
                f64::NAN
            } else {
                correlation_pvalue(pair.rho, pair.n, args.corr_type, args.tail)
            });
        }
    }
    let shape = vec![left_cols, right_cols];
    let rho = Tensor::new(rho_data, shape.clone())
        .map(tensor::tensor_into_value)
        .map_err(|err| corr_internal(format!("corr: {err}")))?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![rho]));
        }
        let p = Tensor::new(p_data, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| corr_internal(format!("corr: {err}")))?;
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![rho, p],
        ));
    }
    Ok(rho)
}

pub(super) fn complete_rows(
    left: &Tensor,
    right: &Tensor,
    rows: usize,
    left_cols: usize,
    right_cols: usize,
) -> Vec<bool> {
    let mut mask = vec![true; rows];
    for row in 0..rows {
        for col in 0..left_cols {
            if matches!(
                left.numeric_value_at(col * rows + row),
                Some(runmat_value::NumericScalar::F64(value)) if value.is_nan()
            ) || matches!(
                left.numeric_value_at(col * rows + row),
                Some(runmat_value::NumericScalar::F32(value)) if value.is_nan()
            ) {
                mask[row] = false;
            }
        }
        for col in 0..right_cols {
            if matches!(
                right.numeric_value_at(col * rows + row),
                Some(runmat_value::NumericScalar::F64(value)) if value.is_nan()
            ) || matches!(
                right.numeric_value_at(col * rows + row),
                Some(runmat_value::NumericScalar::F32(value)) if value.is_nan()
            ) {
                mask[row] = false;
            }
        }
    }
    mask
}
