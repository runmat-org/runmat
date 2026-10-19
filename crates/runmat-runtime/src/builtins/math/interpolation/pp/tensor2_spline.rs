//! Four knot fields for the tensor product of the existing not-a-knot splines.
//! There is no 16-coefficient allocation per grid cell or query-sized grid copy.

use super::coefficients::{spline_coefs_for_series, SplineCooperation};
use super::{interp_error, validate_breaks};
use crate::context::host_allocation::{HostAllocationReceipt, HostNumericWorkspace};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_value::NumericDType;
use std::ops::{Deref, DerefMut};

enum DerivativeField {
    Ordinary(Vec<f64>),
    Managed(HostNumericWorkspace),
}

impl DerivativeField {
    fn zeroed(
        count: usize,
        name: &'static str,
        allocation: Option<(&HostAllocationReceipt, &RuntimeContext)>,
    ) -> BuiltinResult<Self> {
        if let Some((receipt, runtime)) = allocation {
            let workspace = receipt
                .temporary_numeric(runtime, NumericDType::F64, count)
                .map_err(|error| super::host_allocation_error(name, error))?;
            Ok(Self::Managed(workspace))
        } else {
            Ok(Self::Ordinary(zeroed(count, name)?))
        }
    }
}

impl Deref for DerivativeField {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ordinary(values) => values,
            Self::Managed(values) => values.as_f64_slice().expect("f64 derivative workspace"),
        }
    }
}

impl DerefMut for DerivativeField {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Ordinary(values) => values,
            Self::Managed(values) => values.as_f64_slice_mut().expect("f64 derivative workspace"),
        }
    }
}

pub struct TensorProductSpline2D {
    rows: usize,
    dx: DerivativeField,
    dy: DerivativeField,
    dxy: DerivativeField,
}

impl TensorProductSpline2D {
    pub async fn prepare(
        x: &[f64],
        y: &[f64],
        value_at: impl Fn(usize) -> f64,
        work: &mut impl SplineCooperation,
        name: &'static str,
        allocation: Option<(&HostAllocationReceipt, &RuntimeContext)>,
    ) -> BuiltinResult<Self> {
        validate_breaks(x, name)?;
        validate_breaks(y, name)?;
        let rows = y.len();
        let cols = x.len();
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| interp_error(name, format!("{name}: grid size overflow")))?;
        // All three retained derivative fields and the longest line scratch
        // have checked geometry before any values are read.
        let _retained_bytes = count
            .checked_mul(3)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| interp_error(name, format!("{name}: spline workspace size overflow")))?;
        let mut dx = DerivativeField::zeroed(count, name, allocation)?;
        let mut dy = DerivativeField::zeroed(count, name, allocation)?;
        let mut dxy = DerivativeField::zeroed(count, name, allocation)?;
        let mut line = zeroed(rows.max(cols), name)?;

        // For a fixed Y row, apply the exact existing 1-D spline owner along X.
        for row in 0..rows {
            for col in 0..cols {
                line[col] = value_at(row + col * rows);
                if col > 0 && col.is_multiple_of(4096) {
                    work.checkpoint(4096).await?;
                }
            }
            let coefs = spline_coefs_for_series(x, &line[..cols], None, work).await?;
            write_knot_derivatives(&coefs, x, work, name, |col, derivative| {
                dx[row + col * rows] = derivative;
            })
            .await?;
            work.checkpoint(cols as u64).await?;
        }
        // Apply the same Y operator to f and to its X derivatives. Since
        // not-a-knot spline construction is linear, the latter is dxy.
        for col in 0..cols {
            for row in 0..rows {
                line[row] = value_at(row + col * rows);
                if row > 0 && row.is_multiple_of(4096) {
                    work.checkpoint(4096).await?;
                }
            }
            let coefs = spline_coefs_for_series(y, &line[..rows], None, work).await?;
            write_knot_derivatives(&coefs, y, work, name, |row, derivative| {
                dy[row + col * rows] = derivative;
            })
            .await?;
            for row in 0..rows {
                line[row] = dx[row + col * rows];
                if row > 0 && row.is_multiple_of(4096) {
                    work.checkpoint(4096).await?;
                }
            }
            let coefs = spline_coefs_for_series(y, &line[..rows], None, work).await?;
            write_knot_derivatives(&coefs, y, work, name, |row, derivative| {
                dxy[row + col * rows] = derivative;
            })
            .await?;
            work.checkpoint(rows as u64 * 2).await?;
        }
        Ok(Self { rows, dx, dy, dxy })
    }

    pub fn evaluate(
        &self,
        x: &[f64],
        y: &[f64],
        xq: f64,
        yq: f64,
        value_at: impl Fn(usize) -> f64,
    ) -> f64 {
        if xq.is_nan() || yq.is_nan() {
            return f64::NAN;
        }
        let col = super::interval_index(x, xq, true).expect("validated X grid");
        let row = super::interval_index(y, yq, true).expect("validated Y grid");
        let hx = x[col + 1] - x[col];
        let hy = y[row + 1] - y[row];
        let tx = (xq - x[col]) / hx;
        let ty = (yq - y[row]) / hy;
        let bx = hermite_basis(tx);
        let by = hermite_basis(ty);
        let mut result = 0.0;
        for ix in 0..2 {
            for iy in 0..2 {
                let index = row + iy + (col + ix) * self.rows;
                let f = value_at(index);
                let fx = self.dx[index];
                let fy = self.dy[index];
                let fxy = self.dxy[index];
                result += bx[ix] * by[iy] * f
                    + hx * bx[ix + 2] * by[iy] * fx
                    + hy * bx[ix] * by[iy + 2] * fy
                    + hx * hy * bx[ix + 2] * by[iy + 2] * fxy;
            }
        }
        result
    }
}

fn hermite_basis(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        2.0 * t3 - 3.0 * t2 + 1.0,
        -2.0 * t3 + 3.0 * t2,
        t3 - 2.0 * t2 + t,
        t3 - t2,
    ]
}

async fn write_knot_derivatives(
    coefs: &[[f64; 4]],
    breaks: &[f64],
    work: &mut impl SplineCooperation,
    name: &'static str,
    mut publish: impl FnMut(usize, f64),
) -> BuiltinResult<()> {
    for (index, coef) in coefs.iter().enumerate() {
        if coef.iter().any(|value| !value.is_finite()) {
            return Err(interp_error(
                name,
                format!("{name}: invalid spline coefficients"),
            ));
        }
        publish(index, coef[2]);
        if index > 0 && index.is_multiple_of(4096) {
            work.checkpoint(4096).await?;
        }
    }
    let last = coefs[coefs.len() - 1];
    let h = breaks[breaks.len() - 1] - breaks[breaks.len() - 2];
    let final_derivative = (3.0 * last[0] * h + 2.0 * last[1]) * h + last[2];
    if !final_derivative.is_finite() {
        return Err(interp_error(
            name,
            format!("{name}: invalid spline derivative"),
        ));
    }
    publish(breaks.len() - 1, final_derivative);
    Ok(())
}

fn zeroed(len: usize, name: &'static str) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| interp_error(name, format!("{name}: insufficient spline workspace")))?;
    values.resize(len, 0.0);
    Ok(values)
}
