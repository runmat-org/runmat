//! For a common-metric edge star A B P_i P_(i+1), quality q implies
//! |r_i-r_(i+1)| <= sqrt(2) |AB| sin(delta_i) / q. Summing the
//! positive wedge angles gives q <= sqrt(2) |AB| theta / |r_0-r_n|.
//! In an acute wedge theta <= tan(theta), avoiding a libm angle bound.
use super::interval::Interval as I;
use super::*;
type Vector = [I; 3];
fn delta(point: [f64; 3], origin: [f64; 3]) -> Option<Vector> {
    Some([
        I::exact(point[0]).sub(I::exact(origin[0]))?,
        I::exact(point[1]).sub(I::exact(origin[1]))?,
        I::exact(point[2]).sub(I::exact(origin[2]))?,
    ])
}
fn matrix(metric: MetricTensor3) -> [[I; 3]; 3] {
    [
        [metric.xx, metric.xy, metric.xz],
        [metric.xy, metric.yy, metric.yz],
        [metric.xz, metric.yz, metric.zz],
    ]
    .map(|row| row.map(I::exact))
}
fn dot(a: Vector, metric: [[I; 3]; 3], b: Vector) -> Option<I> {
    let mut value = I::exact(0.);
    for (i, ai) in a.into_iter().enumerate() {
        for (j, bj) in b.into_iter().enumerate() {
            value = value.add(ai.mul(metric[i][j])?.mul(bj)?)?;
        }
    }
    Some(value)
}
fn determinant(a: [[I; 3]; 3]) -> Option<I> {
    a[0][0]
        .mul(a[1][1].mul(a[2][2])?.sub(a[1][2].mul(a[2][1])?)?)?
        .sub(a[0][1].mul(a[1][0].mul(a[2][2])?.sub(a[1][2].mul(a[2][0])?)?)?)?
        .add(a[0][2].mul(a[1][0].mul(a[2][1])?.sub(a[1][1].mul(a[2][0])?)?)?)
}
pub(super) fn edge_star_upper_bound(points: [[f64; 3]; 4], metric: MetricTensor3) -> Option<f64> {
    let m = matrix(metric);
    let edge = delta(points[1], points[0])?;
    let a = delta(points[2], points[0])?;
    let b = delta(points[3], points[0])?;
    let length_squared = dot(edge, m, edge)?;
    if length_squared.lo <= 0. {
        return None;
    }
    let ea = dot(edge, m, a)?;
    let eb = dot(edge, m, b)?;
    let ra = dot(a, m, a)?.sub(ea.square()?.div(length_squared)?)?;
    let rb = dot(b, m, b)?.sub(eb.square()?.div(length_squared)?)?;
    let product = dot(a, m, b)?.sub(ea.mul(eb)?.div(length_squared)?)?;
    if product.lo <= 0. {
        return None;
    }
    let sine_numerator = ra.mul(rb)?.sub(product.square()?)?.sqrt()?;
    let radial_difference = ra.sqrt()?.sub(rb.sqrt()?)?.abs();
    if radial_difference.lo <= 0. {
        return None;
    }
    let upper = I::exact(2.)
        .sqrt()?
        .mul(length_squared.sqrt()?)?
        .mul(sine_numerator.div(product)?)?
        .div(radial_difference)?
        .hi;
    upper.is_finite().then_some(upper)
}
pub(super) fn tetrahedron_jacobian_upper_bound(
    points: [[f64; 3]; 4],
    metric: MetricTensor3,
) -> Option<f64> {
    let m = matrix(metric);
    let physical = [
        delta(points[1], points[0])?,
        delta(points[2], points[0])?,
        delta(points[3], points[0])?,
    ];
    let numerator = I::exact(2.)
        .sqrt()?
        .mul(determinant(m)?.sqrt()?)?
        .mul(determinant(physical)?.abs())?;
    let mut upper = f64::INFINITY;
    for corner in 0..4 {
        let mut denominator_squared = I::exact(1.);
        for other in 0..4 {
            if other != corner {
                let edge = delta(points[other], points[corner])?;
                denominator_squared = denominator_squared.mul(dot(edge, m, edge)?)?;
            }
        }
        let denominator = denominator_squared.sqrt()?;
        if denominator.lo <= 0. {
            return None;
        }
        upper = upper.min(numerator.div(denominator)?.hi);
    }
    upper.is_finite().then_some(upper)
}
#[cfg(test)]
mod tests;
