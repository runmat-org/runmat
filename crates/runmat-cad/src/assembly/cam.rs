//! A cam mate's pitch curve: the cam design's published closed path, seen in the cam
//! connector's frame and sampled as station radii every [`CAM_STEP_DEG`] degrees from the
//! connector's X towards its Y. The solver passes a smooth periodic curve through the stations.

use crate::assembly::resolve::{NodeId, Resolver};
use crate::assembly::source::PublishedGeometry;
use crate::ids::PublishName;
use crate::regen::kernel::Transform;

/// The stations' angular spacing.
pub const CAM_STEP_DEG: usize = 1;

impl Resolver<'_, '_> {
    /// The station radii of `profile`, published by `host`'s design, in the connector frame
    /// `frame` (host coordinates).
    pub(super) fn cam_stations(&mut self, host: NodeId, frame: &Transform, profile: &PublishName) -> Result<Vec<f64>, String> {
        let points = self.published_path(host, frame, profile, "a cam's profile is a closed curve")?;
        let polar: Vec<(f64, f64)> = points
            .iter()
            .map(|q| (q[1].atan2(q[0]).rem_euclid(std::f64::consts::TAU), q[0].hypot(q[1])))
            .collect();
        stations(&polar).ok_or_else(|| {
            let n = &self.nodes[host as usize];
            format!("`{profile}` in `{}` does not go once round the cam connector's axis", n.definition)
        })
    }

    /// The points of `name`, a path `host`'s design publishes, in the connector frame `frame`
    /// (host coordinates), its chains one after another. `why` says what the path is for when
    /// `name` is something else.
    pub(super) fn published_path(&mut self, host: NodeId, frame: &Transform, name: &PublishName, why: &str) -> Result<Vec<[f64; 3]>, String> {
        let n = self.nodes[host as usize].clone();
        let summary = self
            .configured_summary(&n.definition, n.revision.as_ref(), &n.configuration)
            .map_err(|e| e.to_string())?;
        let points = match summary.publishes.get(name) {
            Some(PublishedGeometry::Path { chains }) => chains.iter().flat_map(|c| c.points.iter().copied()).collect::<Vec<_>>(),
            Some(_) => return Err(format!("`{name}` in `{}` is not a path; {why}", n.definition)),
            None => return Err(format!("`{}` publishes no `{name}`", n.definition)),
        };
        let to_local = frame.inverse();
        Ok(points.iter().map(|p| to_local.apply(*p)).collect())
    }
}

/// Radii every [`CAM_STEP_DEG`] degrees, read off the path (angle, radius) where it crosses each
/// station's ray, between its points in angle order. `None` when the path doesn't go round the
/// axis (an open arc, or a gap wider than a quarter turn).
pub fn stations(polar: &[(f64, f64)]) -> Option<Vec<f64>> {
    let mut sorted: Vec<(f64, f64)> = polar.iter().copied().filter(|(_, r)| *r > 0.0).collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    sorted.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);
    if sorted.len() < 3 {
        return None;
    }
    let tau = std::f64::consts::TAU;
    let widest = sorted
        .windows(2)
        .map(|w| w[1].0 - w[0].0)
        .chain(std::iter::once(sorted[0].0 + tau - sorted[sorted.len() - 1].0))
        .fold(0.0, f64::max);
    if widest > tau / 4.0 {
        return None;
    }
    let count = 360 / CAM_STEP_DEG;
    let mut out = Vec::with_capacity(count);
    let mut j = 0;
    for k in 0..count {
        let phi = (k * CAM_STEP_DEG) as f64 * tau / 360.0;
        while j < sorted.len() && sorted[j].0 <= phi {
            j += 1;
        }
        // The points either side of the ray, wrapping round.
        let (lo, hi) = match j {
            0 => ((sorted[sorted.len() - 1].0 - tau, sorted[sorted.len() - 1].1), sorted[0]),
            j if j == sorted.len() => (sorted[j - 1], (sorted[0].0 + tau, sorted[0].1)),
            j => (sorted[j - 1], sorted[j]),
        };
        let t = if hi.0 > lo.0 { (phi - lo.0) / (hi.0 - lo.0) } else { 0.0 };
        out.push(lo.1 + (hi.1 - lo.1) * t);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::stations;

    #[test]
    fn a_circle_off_its_points_reads_its_radius_and_an_arc_is_refused() {
        let circle: Vec<(f64, f64)> = (0..36).map(|k| ((k as f64 * 10.0 + 5.0).to_radians(), 0.03)).collect();
        let r = stations(&circle).expect("goes round");
        assert_eq!(r.len(), 360);
        assert!(r.iter().all(|v| (v - 0.03).abs() < 1e-12));
        let arc: Vec<(f64, f64)> = (0..10).map(|k| ((k as f64 * 10.0).to_radians(), 0.03)).collect();
        assert!(stations(&arc).is_none());
    }
}
