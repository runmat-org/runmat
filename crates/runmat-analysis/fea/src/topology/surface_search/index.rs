//! Immutable current-configuration inventory, stable face order, numerical broad/narrow search.
//! Distance separation between faces is not proof of uniqueness within a curved face.
use super::*;
use crate::topology::AnalysisProjection;
use runmat_meshing_core::StableDigest;
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryPairingPolicy {
    pub narrow: BoundarySearchPolicy,
    pub maximum_distance_m: f64,
    pub distance_separation_m: f64,
    pub maximum_faces: u64,
    pub maximum_total_patches: u64,
}
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum BoundaryPairingError {
    #[error(transparent)]
    Geometry(#[from] BoundarySearchError),
    #[error("empty, duplicate or invalid boundary surface inventory")]
    InvalidInventory,
    #[error("invalid boundary pairing controls")]
    InvalidPolicy,
    #[error("boundary pairing exceeds face or patch budget")]
    Exhausted,
    #[error("no surface lies within the declared search range")]
    Unmatched,
    #[error("closest surface distance intersects the declared search range")]
    RangeUncertain,
    #[error("closest surface distances are not separated by the declared margin")]
    Ambiguous,
}
#[derive(Clone, Copy, Debug)]
struct Entry {
    face_index: usize,
    surface: OrientedBoundarySurface,
    bounds: BoundarySurfaceBounds,
}
pub struct BoundarySurfaceIndex {
    entries: Vec<Entry>,
}
#[derive(Clone, Copy, Debug)]
pub struct BoundarySurfaceMatch {
    pub face_index: usize,
    pub face_identity: StableDigest,
    pub closest: ClosestBoundaryPoint,
    pub tested_faces: u64,
    pub visited_patches: u64,
}
impl BoundaryPairingPolicy {
    pub fn validate(self) -> Result<(), BoundaryPairingError> {
        self.narrow.resident_bytes()?;
        if !self.maximum_distance_m.is_finite()
            || self.maximum_distance_m <= 0.0
            || !self.distance_separation_m.is_finite()
            || self.distance_separation_m < 0.0
            || self.maximum_faces == 0
            || self.maximum_total_patches == 0
        {
            return Err(BoundaryPairingError::InvalidPolicy);
        }
        Ok(())
    }
}
impl BoundarySurfaceIndex {
    /// Requested inventory heap; excludes borrowed source, narrow scratch and allocator overhead.
    pub fn resident_bytes(face_count: usize) -> Result<u64, BoundaryPairingError> {
        crate::topology::storage::slots::<Entry>(face_count)
            .map_err(|_| BoundaryPairingError::Geometry(BoundarySearchError::Capacity))
    }
    pub fn derive(
        projection: &AnalysisProjection,
        face_indices: &[usize],
        positions_m: &[[f64; 3]],
        maximum_faces: u64,
    ) -> Result<Self, BoundaryPairingError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if face_indices.is_empty() {
            return Err(BoundaryPairingError::InvalidInventory);
        }
        if u64::try_from(face_indices.len()).map_err(|_| BoundaryPairingError::Exhausted)?
            > maximum_faces
        {
            return Err(BoundaryPairingError::Exhausted);
        }
        Self::resident_bytes(face_indices.len())?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(face_indices.len())
            .map_err(|_| BoundarySearchError::Capacity)?;
        for &face_index in face_indices {
            if crate::is_fea_cancelled() {
                return Err(BoundarySearchError::Cancelled.into());
            }
            let surface = projection.boundary_surface_snapshot(face_index, positions_m)?;
            entries.push(Entry {
                face_index,
                surface,
                bounds: surface.bounds()?,
            });
        }
        entries.sort_unstable_by_key(|e| e.surface.face_identity());
        if entries
            .windows(2)
            .any(|w| w[0].surface.face_identity() == w[1].surface.face_identity())
        {
            return Err(BoundaryPairingError::InvalidInventory);
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        Ok(Self { entries })
    }
    pub fn closest_point_stationary(
        &self,
        query: [f64; 3],
        pairing: BoundaryPairingPolicy,
        refinement: BoundaryStationaryPolicy,
    ) -> Result<(BoundarySurfaceMatch, StationaryBoundaryProjection), BoundaryPairingError> {
        refinement.validate()?;
        let mut matched = self.closest_point(query, pairing)?;
        let entry = self
            .entries
            .iter()
            .find(|e| e.surface.face_identity() == matched.face_identity)
            .ok_or(BoundaryPairingError::InvalidInventory)?;
        let refined = entry
            .surface
            .refine_candidate(query, matched.closest, refinement)?;
        matched.closest = refined.closest;
        Ok((matched, refined))
    }
    pub fn closest_point(
        &self,
        query: [f64; 3],
        policy: BoundaryPairingPolicy,
    ) -> Result<BoundarySurfaceMatch, BoundaryPairingError> {
        policy.validate()?;
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(BoundarySearchError::InvalidInput.into());
        }
        if self.entries.len() as u64 > policy.maximum_faces {
            return Err(BoundaryPairingError::Exhausted);
        }
        let mut best = None::<BoundarySurfaceMatch>;
        let mut other_lower = f64::INFINITY;
        let mut tested_faces = 0_u64;
        let mut patches = 0_u64;
        for entry in &self.entries {
            if crate::is_fea_cancelled() {
                return Err(BoundarySearchError::Cancelled.into());
            }
            let lower = entry.bounds.distance_lower_bound_m(query)?;
            // Even pruned entries remain in the separation test through their lower bound.
            let cutoff = best.as_ref().map_or(policy.maximum_distance_m, |b| {
                b.closest.distance_m.min(policy.maximum_distance_m)
            });
            if lower > cutoff + policy.distance_separation_m {
                other_lower = other_lower.min(lower);
                continue;
            }
            let remaining = policy
                .maximum_total_patches
                .checked_sub(patches)
                .filter(|n| *n > 0)
                .ok_or(BoundaryPairingError::Exhausted)?;
            let narrow = BoundarySearchPolicy {
                maximum_patches: remaining.min(policy.narrow.maximum_patches),
                ..policy.narrow
            };
            let closest = entry.surface.closest_point(query, narrow)?;
            patches = patches
                .checked_add(closest.visited_patches)
                .ok_or(BoundaryPairingError::Exhausted)?;
            tested_faces += 1;
            let candidate_lower = (closest.distance_m - closest.distance_error_m).max(0.0);
            if best
                .as_ref()
                .is_none_or(|b| closest.distance_m < b.closest.distance_m)
            {
                if let Some(previous) = best {
                    other_lower = other_lower.min(
                        (previous.closest.distance_m - previous.closest.distance_error_m).max(0.0),
                    );
                }
                best = Some(BoundarySurfaceMatch {
                    face_index: entry.face_index,
                    face_identity: entry.surface.face_identity(),
                    closest,
                    tested_faces: 0,
                    visited_patches: 0,
                });
            } else {
                other_lower = other_lower.min(candidate_lower);
            }
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        let mut best = best.ok_or(BoundaryPairingError::Unmatched)?;
        if best.closest.distance_m - best.closest.distance_error_m > policy.maximum_distance_m {
            return Err(BoundaryPairingError::Unmatched);
        }
        if best.closest.distance_m > policy.maximum_distance_m {
            return Err(BoundaryPairingError::RangeUncertain);
        }
        if other_lower <= best.closest.distance_m + policy.distance_separation_m {
            return Err(BoundaryPairingError::Ambiguous);
        }
        best.tested_faces = tested_faces;
        best.visited_patches = patches;
        Ok(best)
    }
}
#[cfg(test)]
mod tests;
