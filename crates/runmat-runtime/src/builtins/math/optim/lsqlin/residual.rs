//! Borrowed original residual arithmetic shared by actual least-squares algorithms.
use super::*;
#[derive(Clone, Copy)]
pub(super) struct View<'a> {
    pub c: &'a Matrix,
    pub d: &'a [f64],
    pub weights: &'a [f64],
    pub units: f64,
}
pub(super) struct Residual {
    pub values: Vec<f64>,
    pub value: f64,
}
pub(super) struct Measurements {
    pub residual: Vec<f64>,
    pub value: f64,
    pub gradient: Vec<f64>,
    pub measure: f64,
}
pub(super) struct Evaluated {
    pub x: Vec<f64>,
    pub residual: Vec<f64>,
    pub value: f64,
    pub gradient: Vec<f64>,
    pub measure: f64,
}
async fn zeros(
    n: usize,
    allocation: fn(runmat_value::ExactHostNumericAllocationError) -> RuntimeError,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = runmat_value::exact_capacity_vec(n).map_err(allocation)?;
    while values.len() < n {
        let end = values.len().saturating_add(1024).min(n);
        let work = end - values.len();
        values.resize(end, 0.);
        c.completed(work as u64).await?;
    }
    Ok(values)
}
pub(super) async fn multiply(
    model: View<'_>,
    input: &[f64],
    output: &mut [f64],
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<()>> {
    if input.len() != model.c.n || output.len() != model.c.m {
        return Err(invalid("reflective residual product geometry mismatch"));
    }
    for chunk in output.chunks_mut(1024) {
        chunk.fill(0.);
        c.completed(chunk.len() as u64).await?;
    }
    for (column, &x) in input.iter().enumerate() {
        for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
            let row = model.c.rowval[entry];
            output[row] = model.c.nzval[entry].mul_add(x, output[row]);
            if !output[row].is_finite() {
                return Ok(None);
            }
            c.completed(1).await?;
        }
        c.completed(1).await?;
    }
    Ok(Some(()))
}

pub(super) async fn measure(
    model: View<'_>,
    x: &[f64],
    allocation: fn(runmat_value::ExactHostNumericAllocationError) -> RuntimeError,
    mut measure: impl FnMut(usize, f64, f64) -> Option<f64>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Measurements>> {
    let Some(Residual {
        values: residual,
        value,
    }) = residual(model, x, allocation, c).await?
    else {
        return Ok(None);
    };
    let mut gradient = zeros(model.c.n, allocation, c).await?;
    let mut optimality = 0f64;
    for (column, to) in gradient.iter_mut().enumerate() {
        for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
            let row = model.c.rowval[entry];
            let root = model.weights[row].sqrt();
            *to = (model.c.nzval[entry] * root).mul_add(root * residual[row], *to);
            c.completed(1).await?;
        }
        *to *= model.units;
        if !to.is_finite() {
            return Ok(None);
        }
        let Some(first) = measure(column, *to, x[column]) else {
            return Ok(None);
        };
        optimality = optimality.max(first);
        c.completed(1).await?;
    }
    Ok(Some(Measurements {
        residual,
        value,
        gradient,
        measure: optimality,
    }))
}

pub(super) async fn evaluate(
    model: View<'_>,
    x: Vec<f64>,
    allocation: fn(runmat_value::ExactHostNumericAllocationError) -> RuntimeError,
    measure_value: impl FnMut(usize, f64, f64) -> Option<f64>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Evaluated>> {
    let result = measure(model, &x, allocation, measure_value, c).await?;
    Ok(result.map(|value| Evaluated {
        x,
        residual: value.residual,
        value: value.value,
        gradient: value.gradient,
        measure: value.measure,
    }))
}
pub(super) async fn residual(
    model: View<'_>,
    x: &[f64],
    allocation: fn(runmat_value::ExactHostNumericAllocationError) -> RuntimeError,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Residual>> {
    let mut residual = zeros(model.c.m, allocation, c).await?;
    if multiply(model, x, &mut residual, c).await?.is_none() {
        return Ok(None);
    }
    let mut length = 0f64;
    for (row, value) in residual.iter_mut().enumerate() {
        *value -= model.d[row];
        let weighted = model.weights[row].sqrt() * *value;
        if !value.is_finite() || !weighted.is_finite() {
            return Ok(None);
        }
        length = length.hypot(weighted);
        c.completed(1).await?;
    }
    let value = length * length;
    if !value.is_finite() {
        return Ok(None);
    }

    Ok(Some(Residual {
        values: residual,
        value,
    }))
}
