//! Conservative tensor intersection without repeated-constraint density growth.
//! Commuting coordinate-diagonal tensors have the exact directional maximum.
//! Noncommuting tensors share one positive-subspace envelope. Never repeatedly
//! sum full tensors: tiny rotation differences must not multiply source density.
use super::{MetricContractError, MetricFieldRequest, MetricTensor3};
mod envelope;
mod spectral;
#[cfg(test)]
mod tests;

pub(super) fn combine(
    request: &MetricFieldRequest,
    indices: &[usize],
) -> Result<MetricTensor3, MetricContractError> {
    let base = request.global_metric;
    // Retain an original dominating tensor exactly. A singleton/dominant
    // anisotropic constraint must not become an isotropic subspace bound.
    let mut candidate = base;
    for &i in indices {
        let metric = request.contributions[i].metric;
        if dominates(metric, candidate) {
            candidate = metric;
        }
    }
    if dominates(candidate, base)
        && indices
            .iter()
            .all(|&i| dominates(candidate, request.contributions[i].metric))
    {
        return Ok(candidate);
    }
    if indices.iter().all(|&i| {
        let m = request.contributions[i].metric;
        [m.xy, m.xz, m.yz] == [base.xy, base.xz, base.yz]
    }) {
        return indices
            .iter()
            .try_fold(base, |a, &i| intersect(a, request.contributions[i].metric));
    }
    let mut envelope = envelope::TensorEnvelope::new(base)?;
    for &i in indices {
        envelope.include(request.contributions[i].metric)?;
    }
    envelope.metric()
}

/// All principal minors characterize a symmetric positive-semidefinite
/// difference. Scaling keeps these fixed-size tests away from overflow.
pub(crate) fn dominates(a: MetricTensor3, b: MetricTensor3) -> bool {
    let [xx, yy, zz, xy, xz, yz] = [
        a.xx - b.xx,
        a.yy - b.yy,
        a.zz - b.zz,
        a.xy - b.xy,
        a.xz - b.xz,
        a.yz - b.yz,
    ];
    let scale = [xx, yy, zz, xy, xz, yz]
        .into_iter()
        .map(f64::abs)
        .fold(0., f64::max);
    if scale == 0. {
        return true;
    }
    if !scale.is_finite() {
        return false;
    }
    let [x, y, z, u, v, w] = [xx, yy, zz, xy, xz, yz].map(|q| q / scale);
    x >= 0.
        && y >= 0.
        && z >= 0.
        && x * y - u * u >= 0.
        && x * z - v * v >= 0.
        && y * z - w * w >= 0.
        && x * y * z + 2. * u * v * w - x * w * w - y * v * v - z * u * u >= 0.
}

/// Deduplicate physical constraints in the existing local index buffer, then
/// restore their first canonical request order. No per-query identity trees or
/// tensor inventories are retained; all source provenance is counted beforehand.
pub(super) fn deduplicate(request: &MetricFieldRequest, indices: &mut Vec<usize>) {
    indices.sort_unstable_by_key(|i| (key(request.contributions[*i].metric), *i));
    indices.dedup_by(|a, b| {
        key(request.contributions[*a].metric) == key(request.contributions[*b].metric)
    });
    indices.retain(|i| key(request.contributions[*i].metric) != key(request.global_metric));
    indices.sort_unstable();
}
fn key(m: MetricTensor3) -> [u64; 6] {
    [m.xx, m.yy, m.zz, m.xy, m.xz, m.yz].map(|v| if v == 0.0 { 0 } else { v.to_bits() })
}

/// Preserves every directional constraint. Equal off-diagonal entries cancel
/// in both differences, so a componentwise diagonal maximum dominates both
/// operands exactly. General callers also use a bounded two-tensor envelope.
pub(crate) fn intersect(
    a: MetricTensor3,
    b: MetricTensor3,
) -> Result<MetricTensor3, MetricContractError> {
    if dominates(a, b) {
        return Ok(a);
    }
    if dominates(b, a) {
        return Ok(b);
    }
    let metric = if [a.xy, a.xz, a.yz] == [b.xy, b.xz, b.yz] {
        MetricTensor3 {
            xx: a.xx.max(b.xx),
            yy: a.yy.max(b.yy),
            zz: a.zz.max(b.zz),
            ..a
        }
    } else {
        let mut envelope = envelope::TensorEnvelope::new(a)?;
        envelope.include(b)?;
        return envelope.metric();
    };
    metric.validate()?;
    Ok(metric)
}
