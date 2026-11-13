//! Sparse original-row normalization, including public two-sided inequalities.
use super::*;
use clarabel::algebra::CscMatrix;
struct Sides {
    upper: Option<usize>,
    lower: Option<usize>,
    impossible: bool,
}
pub(super) async fn constraints(
    a: CscMatrix<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    aeq: CscMatrix<f64>,
    beq: Vec<f64>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(CscMatrix<f64>, Vec<f64>, usize)> {
    if a.n != aeq.n || lower.len() != a.m || upper.len() != a.m || beq.len() != aeq.m {
        return Err(invalid("inconsistent constraint geometry"));
    }
    let equalities = beq.len();
    let mut rhs = beq;
    let additional =
        a.m.checked_mul(2)
            .ok_or_else(|| invalid("inequality row count overflow"))?;
    rhs.try_reserve(additional)
        .map_err(|e| reserve_error(e, "inequality rows"))?;
    let mut sides = runmat_value::exact_capacity_vec(a.m).map_err(allocation)?;
    for row in 0..a.m {
        if lower[row].is_nan() || upper[row].is_nan() {
            return Err(invalid("inequality bounds cannot be NaN"));
        }
        let impossible = lower[row] > upper[row]
            || lower[row] == f64::INFINITY
            || upper[row] == f64::NEG_INFINITY;
        let mut side = Sides {
            upper: None,
            lower: None,
            impossible,
        };
        if impossible {
            side.upper = Some(rhs.len());
            rhs.push(-1.0);
        } else {
            if upper[row].is_finite() {
                side.upper = Some(rhs.len());
                rhs.push(upper[row]);
            }
            if lower[row].is_finite() {
                side.lower = Some(rhs.len());
                rhs.push(-lower[row]);
            }
        }
        sides.push(side);
        cooperation.completed(1).await?;
    }
    let capacity = a
        .nzval
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_add(aeq.nzval.len()))
        .ok_or_else(|| invalid("normalized sparse size overflow"))?;
    let mut pointers = runmat_value::exact_capacity_vec(
        a.n.checked_add(1)
            .ok_or_else(|| invalid("column count overflow"))?,
    )
    .map_err(allocation)?;
    let mut rows = runmat_value::exact_capacity_vec(capacity).map_err(allocation)?;
    let mut values = runmat_value::exact_capacity_vec(capacity).map_err(allocation)?;
    pointers.push(0);
    for col in 0..a.n {
        for k in aeq.colptr[col]..aeq.colptr[col + 1] {
            rows.push(aeq.rowval[k]);
            values.push(aeq.nzval[k]);
            if k % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        for k in a.colptr[col]..a.colptr[col + 1] {
            let side = &sides[a.rowval[k]];
            if !side.impossible {
                if let Some(row) = side.upper {
                    rows.push(row);
                    values.push(a.nzval[k]);
                }
                if let Some(row) = side.lower {
                    rows.push(row);
                    values.push(-a.nzval[k]);
                }
            }
            if k % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        pointers.push(values.len());
        cooperation.completed(1).await?;
    }
    Ok((
        CscMatrix::new(rhs.len(), a.n, pointers, rows, values),
        rhs,
        equalities,
    ))
}
