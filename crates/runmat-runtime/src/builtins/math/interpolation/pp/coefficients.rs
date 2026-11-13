use super::*;
pub(crate) trait SplineCooperation {
    async fn checkpoint(&mut self, completed_work: u64) -> BuiltinResult<()>;
}

/// Keeps the exact allocation authority live across every cooperative
/// coefficient step. The scheduler remains the sole yielding owner.
pub(crate) struct AdmittedSplineWork<'a> {
    budget: Option<crate::execution::CpuCooperationBudget>,
    runtime: Option<&'a crate::context::RuntimeContext>,
    receipt: Option<&'a crate::context::host_allocation::HostAllocationReceipt>,
    name: &'static str,
}

impl<'a> AdmittedSplineWork<'a> {
    pub(crate) fn new(
        runtime: Option<&'a crate::context::RuntimeContext>,
        receipt: Option<&'a crate::context::host_allocation::HostAllocationReceipt>,
        name: &'static str,
    ) -> Self {
        Self {
            budget: runtime.map(crate::context::RuntimeContext::cpu_cooperation),
            runtime,
            receipt,
            name,
        }
    }
}

impl SplineCooperation for AdmittedSplineWork<'_> {
    async fn checkpoint(&mut self, completed_work: u64) -> BuiltinResult<()> {
        if let (Some(receipt), Some(runtime)) = (self.receipt, self.runtime) {
            receipt
                .checkpoint(runtime)
                .map_err(|error| super::host_allocation_error(self.name, error))?;
        }
        self.budget.checkpoint(completed_work).await?;
        if let (Some(receipt), Some(runtime)) = (self.receipt, self.runtime) {
            receipt
                .checkpoint(runtime)
                .map_err(|error| super::host_allocation_error(self.name, error))?;
        }
        Ok(())
    }
}

impl SplineCooperation for Option<crate::execution::CpuCooperationBudget> {
    async fn checkpoint(&mut self, completed_work: u64) -> BuiltinResult<()> {
        if let Some(budget) = self.as_mut() {
            budget.checkpoint(completed_work).await.map_err(|source| {
                let mut error = source.clone().into_runtime_error();
                error.source = Some(Box::new(source));
                error
            })?;
        }
        Ok(())
    }
}

/// Scheduler failures are typed causes, not numerical spline failures. The
/// callers may add builtin context but must not replace their control flow
/// or infrastructure identity with a coefficient-error descriptor.
pub(crate) fn is_execution_failure(error: &RuntimeError) -> bool {
    error.source.as_deref().is_some_and(|source| {
        source
            .downcast_ref::<crate::execution::ExecutionServiceError>()
            .is_some()
    })
}

#[cfg(test)]
impl SplineCooperation for () {
    async fn checkpoint(&mut self, _: u64) -> BuiltinResult<()> {
        Ok(())
    }
}

pub(super) async fn spline_coefs_for_series(
    x: &[f64],
    y: &[f64],
    endpoint_slopes: Option<EndpointSlopes>,
    work: &mut impl SplineCooperation,
) -> BuiltinResult<Vec<[f64; 4]>> {
    let n = x.len();
    if let Some(slopes) = endpoint_slopes {
        return clamped_spline_coefs(x, y, slopes, work).await;
    }
    if n == 2 {
        let h = x[1] - x[0];
        return Ok(vec![[0.0, 0.0, (y[1] - y[0]) / h, y[0]]]);
    }
    if n == 3 {
        return Ok(quadratic_as_piecewise_cubic(x, y));
    }

    let second = not_a_knot_second_derivatives(x, y, work).await?;
    second_derivatives_to_coefs(x, y, &second, work).await
}

async fn cooperate_spline(work: &mut impl SplineCooperation, index: usize) -> BuiltinResult<()> {
    if index.is_multiple_of(4096) {
        work.checkpoint(if index == 0 { 0 } else { 4096 }).await?;
    }
    Ok(())
}

/// Eliminate the two endpoint second derivatives from the not-a-knot
/// equations, then solve the remaining tridiagonal system in O(n) space
/// and time. The endpoint values are recovered from continuity of the
/// third derivative across the first and last interior breaks.
async fn not_a_knot_second_derivatives(
    x: &[f64],
    y: &[f64],
    work: &mut impl SplineCooperation,
) -> BuiltinResult<Vec<f64>> {
    let n = x.len();
    let mut h = Vec::new();
    let mut secants = Vec::new();
    h.try_reserve_exact(n - 1)
        .map_err(|_| interp_error("spline", "spline: insufficient coefficient workspace"))?;
    secants
        .try_reserve_exact(n - 1)
        .map_err(|_| interp_error("spline", "spline: insufficient coefficient workspace"))?;
    for i in 0..n - 1 {
        cooperate_spline(work, i).await?;
        let width = x[i + 1] - x[i];
        h.push(width);
        secants.push((y[i + 1] - y[i]) / width);
    }
    let count = n - 2;
    let mut lower = zeroed_buffer(count, "spline")?;
    let mut diagonal = zeroed_buffer(count, "spline")?;
    let mut upper = zeroed_buffer(count, "spline")?;
    let mut rhs = zeroed_buffer(count, "spline")?;
    for local in 0..count {
        cooperate_spline(work, local).await?;
        let i = local + 1;
        lower[local] = h[i - 1];
        diagonal[local] = 2.0 * (h[i - 1] + h[i]);
        upper[local] = h[i];
        rhs[local] = 6.0 * (secants[i] - secants[i - 1]);
    }
    diagonal[0] += h[0] * (h[0] + h[1]) / h[1];
    upper[0] -= h[0] * h[0] / h[1];
    lower[0] = 0.0;
    let last = count - 1;
    diagonal[last] += h[n - 2] * (h[n - 3] + h[n - 2]) / h[n - 3];
    lower[last] -= h[n - 2] * h[n - 2] / h[n - 3];
    upper[last] = 0.0;

    for i in 1..count {
        cooperate_spline(work, i).await?;
        if !diagonal[i - 1].is_finite() || diagonal[i - 1] == 0.0 {
            return Err(interp_error(
                "spline",
                "spline: singular interpolation system",
            ));
        }
        let factor = lower[i] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    if !diagonal[last].is_finite() || diagonal[last] == 0.0 {
        return Err(interp_error(
            "spline",
            "spline: singular interpolation system",
        ));
    }
    let mut interior = zeroed_buffer(count, "spline")?;
    interior[last] = rhs[last] / diagonal[last];
    for i in (0..last).rev() {
        cooperate_spline(work, i).await?;
        interior[i] = (rhs[i] - upper[i] * interior[i + 1]) / diagonal[i];
    }
    let mut second = zeroed_buffer(n, "spline")?;
    second[1..n - 1].copy_from_slice(&interior);
    second[0] = ((h[0] + h[1]) * second[1] - h[0] * second[2]) / h[1];
    second[n - 1] = ((h[n - 3] + h[n - 2]) * second[n - 2] - h[n - 2] * second[n - 3]) / h[n - 3];
    for (index, value) in second.iter().enumerate() {
        cooperate_spline(work, index).await?;
        if !value.is_finite() {
            return Err(interp_error(
                "spline",
                "spline: invalid interpolation system",
            ));
        }
    }
    Ok(second)
}

/// Solve the strictly diagonally dominant tridiagonal second-derivative
/// system for a clamped cubic spline in linear space and time.
async fn clamped_spline_coefs(
    x: &[f64],
    y: &[f64],
    slopes: EndpointSlopes,
    work: &mut impl SplineCooperation,
) -> BuiltinResult<Vec<[f64; 4]>> {
    let n = x.len();
    let mut widths = Vec::new();
    let mut secants = Vec::new();
    widths
        .try_reserve_exact(n - 1)
        .map_err(|_| interp_error("spline", "spline: insufficient coefficient workspace"))?;
    secants
        .try_reserve_exact(n - 1)
        .map_err(|_| interp_error("spline", "spline: insufficient coefficient workspace"))?;
    for i in 0..n - 1 {
        cooperate_spline(work, i).await?;
        let width = x[i + 1] - x[i];
        widths.push(width);
        secants.push((y[i + 1] - y[i]) / width);
    }
    let mut diagonal = zeroed_buffer(n, "spline")?;
    let mut upper = zeroed_buffer(n - 1, "spline")?;
    let mut rhs = zeroed_buffer(n, "spline")?;
    diagonal[0] = 2.0 * widths[0];
    upper[0] = widths[0];
    rhs[0] = 6.0 * (secants[0] - slopes.first);
    for i in 1..n - 1 {
        cooperate_spline(work, i).await?;
        diagonal[i] = 2.0 * (widths[i - 1] + widths[i]);
        upper[i] = widths[i];
        rhs[i] = 6.0 * (secants[i] - secants[i - 1]);
    }
    diagonal[n - 1] = 2.0 * widths[n - 2];
    rhs[n - 1] = 6.0 * (slopes.last - secants[n - 2]);

    for i in 1..n {
        cooperate_spline(work, i).await?;
        let lower = widths[i - 1];
        let factor = lower / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    let mut second = zeroed_buffer(n, "spline")?;
    second[n - 1] = rhs[n - 1] / diagonal[n - 1];
    for i in (0..n - 1).rev() {
        cooperate_spline(work, i).await?;
        second[i] = (rhs[i] - upper[i] * second[i + 1]) / diagonal[i];
    }
    for (index, value) in second.iter().enumerate() {
        cooperate_spline(work, index).await?;
        if !value.is_finite() {
            return Err(interp_error(
                "spline",
                "spline: invalid endpoint slope system",
            ));
        }
    }
    second_derivatives_to_coefs(x, y, &second, work).await
}

fn quadratic_as_piecewise_cubic(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let y0 = y[0];
    let y1 = y[1];
    let y2 = y[2];
    let c2 =
        y0 / ((x0 - x1) * (x0 - x2)) + y1 / ((x1 - x0) * (x1 - x2)) + y2 / ((x2 - x0) * (x2 - x1));
    let c1 = -y0 * (x1 + x2) / ((x0 - x1) * (x0 - x2))
        - y1 * (x0 + x2) / ((x1 - x0) * (x1 - x2))
        - y2 * (x0 + x1) / ((x2 - x0) * (x2 - x1));
    let c0 = y0 * x1 * x2 / ((x0 - x1) * (x0 - x2))
        + y1 * x0 * x2 / ((x1 - x0) * (x1 - x2))
        + y2 * x0 * x1 / ((x2 - x0) * (x2 - x1));
    (0..2)
        .map(|i| {
            let xi = x[i];
            let a = c2 * xi * xi + c1 * xi + c0;
            let b = 2.0 * c2 * xi + c1;
            [0.0, c2, b, a]
        })
        .collect()
}

async fn second_derivatives_to_coefs(
    x: &[f64],
    y: &[f64],
    second: &[f64],
    work: &mut impl SplineCooperation,
) -> BuiltinResult<Vec<[f64; 4]>> {
    let mut coefs = Vec::new();
    coefs
        .try_reserve_exact(x.len() - 1)
        .map_err(|_| interp_error("spline", "spline: insufficient coefficient workspace"))?;
    for i in 0..(x.len() - 1) {
        cooperate_spline(work, i).await?;
        let h = x[i + 1] - x[i];
        let a = y[i];
        let b = (y[i + 1] - y[i]) / h - h * (2.0 * second[i] + second[i + 1]) / 6.0;
        let c = second[i] / 2.0;
        let d = (second[i + 1] - second[i]) / (6.0 * h);
        coefs.push([d, c, b, a]);
    }
    Ok(coefs)
}

pub(super) fn pchip_coefs_for_series(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    let n = x.len();
    let h: Vec<f64> = x.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let delta: Vec<f64> = y
        .windows(2)
        .zip(h.iter())
        .map(|(pair, &width)| (pair[1] - pair[0]) / width)
        .collect();

    let mut slopes = vec![0.0; n];
    if n == 2 {
        slopes[0] = delta[0];
        slopes[1] = delta[0];
    } else {
        slopes[0] = endpoint_slope(h[0], h[1], delta[0], delta[1]);
        slopes[n - 1] = endpoint_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        for i in 1..(n - 1) {
            if delta[i - 1] == 0.0 || delta[i] == 0.0 || delta[i - 1].signum() != delta[i].signum()
            {
                slopes[i] = 0.0;
            } else {
                let w1 = 2.0 * h[i] + h[i - 1];
                let w2 = h[i] + 2.0 * h[i - 1];
                slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }
    }

    let mut coefs = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let c0 = y[i];
        let c1 = slopes[i];
        let c2 = (3.0 * delta[i] - 2.0 * slopes[i] - slopes[i + 1]) / h[i];
        let c3 = (slopes[i] + slopes[i + 1] - 2.0 * delta[i]) / (h[i] * h[i]);
        coefs.push([c3, c2, c1, c0]);
    }
    coefs
}

/// Modified Akima slopes use the adjacent secants and two linearly extended
/// secants at each boundary. The Hermite coefficients share the same local pp
/// representation as pchip, but the derivatives are deliberately different.
pub(super) fn makima_coefs_for_series(
    x: &[f64],
    y: &[f64],
    name: &'static str,
) -> BuiltinResult<Vec<[f64; 4]>> {
    let n = x.len();
    if n == 2 {
        return Ok(vec![[0.0, 0.0, (y[1] - y[0]) / (x[1] - x[0]), y[0]]]);
    }
    let secant_count = n
        .checked_add(3)
        .ok_or_else(|| interp_error(name, format!("{name}: interpolation size overflow")))?;
    let mut secants = zeroed_buffer(secant_count, name)?;
    for i in 0..n - 1 {
        secants[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }
    // Express linear extension as a secant plus its finite difference. The
    // algebraically equivalent 2*a-b can overflow before cancellation.
    secants[1] = secants[2] + (secants[2] - secants[3]);
    secants[0] = secants[1] + (secants[1] - secants[2]);
    secants[n + 1] = secants[n] + (secants[n] - secants[n - 1]);
    secants[n + 2] = secants[n + 1] + (secants[n + 1] - secants[n]);

    let mut slopes = zeroed_buffer(n, name)?;
    for i in 0..n {
        let before = secants[i + 1];
        let after = secants[i + 2];
        let next = secants[i + 3];
        let previous = secants[i];
        let mut right_weight = (next - after).abs() + (0.5 * next + 0.5 * after).abs();
        let mut left_weight = (before - previous).abs() + (0.5 * before + 0.5 * previous).abs();
        if !right_weight.is_finite() || !left_weight.is_finite() {
            // Preserve the weight ratio when finite secants have a difference
            // too large to represent directly.
            let scale = previous
                .abs()
                .max(before.abs())
                .max(after.abs())
                .max(next.abs());
            right_weight = ((next / scale) - (after / scale)).abs()
                + (0.5 * (next / scale) + 0.5 * (after / scale)).abs();
            left_weight = ((before / scale) - (previous / scale)).abs()
                + (0.5 * (before / scale) + 0.5 * (previous / scale)).abs();
        }
        let weight_scale = left_weight.max(right_weight);
        slopes[i] = if weight_scale == 0.0 {
            0.5 * before + 0.5 * after
        } else {
            let right_fraction = right_weight / weight_scale;
            let left_fraction = left_weight / weight_scale;
            (right_fraction / (right_fraction + left_fraction)) * before
                + (left_fraction / (right_fraction + left_fraction)) * after
        };
    }

    let mut coefs = Vec::new();
    coefs.try_reserve_exact(n - 1).map_err(|_| {
        interp_error(
            name,
            format!("{name}: insufficient memory for interpolation data"),
        )
    })?;
    for i in 0..n - 1 {
        let h = x[i + 1] - x[i];
        let delta = secants[i + 2];
        let c0 = y[i];
        let c1 = slopes[i];
        // Form Hermite numerators from deviations from the local secant, so
        // a large finite linear slope does not overflow in 3*delta or 2*m.
        let left_deviation = delta - slopes[i];
        let right_deviation = delta - slopes[i + 1];
        let c2 = (left_deviation + (left_deviation + right_deviation)) / h;
        let c3 = -(left_deviation + right_deviation) / h / h;
        coefs.push([c3, c2, c1, c0]);
    }
    Ok(coefs)
}

fn endpoint_slope(h0: f64, h1: f64, del0: f64, del1: f64) -> f64 {
    let mut slope = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1);
    if slope.signum() != del0.signum() {
        slope = 0.0;
    } else if del0.signum() != del1.signum() && slope.abs() > (3.0 * del0).abs() {
        slope = 3.0 * del0;
    }
    slope
}

pub(super) fn validate_finite_coefs(coefs: &[f64], name: &'static str) -> BuiltinResult<()> {
    if coefs.iter().any(|value| !value.is_finite()) {
        return Err(interp_error(
            name,
            format!("{name}: interpolation coefficients must be finite"),
        ));
    }
    Ok(())
}
