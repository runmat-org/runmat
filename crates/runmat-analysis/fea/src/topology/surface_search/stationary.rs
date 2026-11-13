//! Bounded smooth-branch refinement of a numerical global closest candidate.
//! Monotone distance retains its original distance envelope; no within-face uniqueness proof.
use super::*;
use differential::geometry::ProjectionGeometry;
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryStationaryPolicy {
    pub differential: BoundaryProjectionDifferentialPolicy,
    pub maximum_iterations: u32,
    pub maximum_backtracks: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct StationaryBoundaryProjection {
    pub closest: ClosestBoundaryPoint,
    pub refinement_steps: u32,
    pub backtracks: u64,
    pub normalized_stationarity_residual_m: f64,
}
impl BoundaryStationaryPolicy {
    pub fn validate(self) -> Result<(), BoundarySearchError> {
        self.differential.validate()?;
        if self.maximum_iterations == 0 || self.maximum_backtracks == 0 {
            return Err(BoundarySearchError::InvalidInput);
        }
        Ok(())
    }
}
impl BoundarySurface {
    pub fn closest_point_stationary(
        &self,
        query: [f64; 3],
        search: BoundarySearchPolicy,
        refinement: BoundaryStationaryPolicy,
    ) -> Result<StationaryBoundaryProjection, BoundarySearchError> {
        refinement.validate()?;
        let candidate = self.closest_point(query, search)?;
        self.refine_candidate(query, candidate, refinement)
    }
    pub(in crate::topology::surface_search) fn refine_candidate(
        &self,
        query: [f64; 3],
        mut candidate: ClosestBoundaryPoint,
        policy: BoundaryStationaryPolicy,
    ) -> Result<StationaryBoundaryProjection, BoundarySearchError> {
        policy.validate()?;
        let mut backtracks = 0u64;
        for steps in 0..=policy.maximum_iterations {
            if crate::is_fea_cancelled() {
                return Err(BoundarySearchError::Cancelled);
            }
            let g = ProjectionGeometry::derive_local(
                self,
                query,
                candidate.barycentric,
                policy.differential,
                false,
            )?;
            let residual =
                g.t.iter()
                    .map(|t| dot(g.d, *t).abs() / norm(*t))
                    .fold(0.0_f64, f64::max);
            if residual <= policy.differential.stationarity_tolerance_m {
                candidate.point = g.point;
                candidate.distance_m = norm(g.d);
                candidate.signed_normal_offset_m = dot(g.d, g.point.unit_normal);
                if crate::is_fea_cancelled() {
                    return Err(BoundarySearchError::Cancelled);
                }
                return Ok(StationaryBoundaryProjection {
                    closest: candidate,
                    refinement_steps: steps,
                    backtracks,
                    normalized_stationarity_residual_m: residual,
                });
            }
            if steps == policy.maximum_iterations {
                return Err(BoundarySearchError::RefinementExhausted);
            }
            let delta = g.solve([dot(g.d, g.t[0]), dot(g.d, g.t[1])]);
            if delta.iter().any(|x| !x.is_finite()) {
                return Err(BoundarySearchError::Resolution);
            }
            let old_distance = norm(g.d);
            let mut scale = 1.0;
            let mut next = None;
            for _ in 0..policy.maximum_backtracks {
                if crate::is_fea_cancelled() {
                    return Err(BoundarySearchError::Cancelled);
                }
                let u = candidate.barycentric[1] + scale * delta[0];
                let v = candidate.barycentric[2] + scale * delta[1];
                let barycentric = [1.0 - u - v, u, v];
                if barycentric
                    .iter()
                    .all(|b| *b > policy.differential.minimum_barycentric)
                {
                    match ProjectionGeometry::derive_local(
                        self,
                        query,
                        barycentric,
                        policy.differential,
                        false,
                    ) {
                        Ok(trial) => {
                            let distance = norm(trial.d);
                            let trial_residual = trial
                                .t
                                .iter()
                                .map(|t| dot(trial.d, *t).abs() / norm(*t))
                                .fold(0.0_f64, f64::max);
                            if distance <= old_distance && trial_residual < residual {
                                next = Some((barycentric, trial.point, distance));
                                break;
                            }
                        }
                        Err(BoundarySearchError::DegenerateProjection) => {}
                        Err(error) => return Err(error),
                    }
                }
                backtracks += 1;
                scale *= 0.5;
            }
            let (barycentric, point, distance) =
                next.ok_or(BoundarySearchError::RefinementExhausted)?;
            candidate.barycentric = barycentric;
            candidate.point = point;
            candidate.distance_m = distance;
        }
        Err(BoundarySearchError::RefinementExhausted)
    }
}
#[cfg(test)]
mod tests;
