//! One immutable point index per candidate-selection inventory. Predicates and
//! canonical topology retain authority; this index only removes duplicate scans.
use super::super::{
    error, DelaunayVolumeRefinementCandidateError, DelaunayVolumeRefinementCandidateErrorKind,
};
use crate::cdt::DelaunayVolumeTopology;
#[cfg(test)]
use runmat_meshing_core::Aabb3;
use runmat_meshing_core::{MeshingCancellationSignal, PointSpatialIndex, SpatialPoint};

pub(in crate::cdt::volume_refinement) struct CandidateQueries {
    locations: PointSpatialIndex<()>,
}
impl CandidateQueries {
    pub(in crate::cdt::volume_refinement) fn topology<'a>(
        &'a self,
        topology: &'a DelaunayVolumeTopology,
    ) -> crate::cdt::topology::IndexedCanonicalTopology<'a> {
        crate::cdt::topology::IndexedCanonicalTopology {
            topology,
            locations: &self.locations,
        }
    }

    pub(in crate::cdt::volume_refinement) fn new(
        topology: &DelaunayVolumeTopology,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, DelaunayVolumeRefinementCandidateError> {
        let mut checkpoint = || checkpoint(cancellation);
        let entries = topology
            .nodes
            .iter()
            .map(|node| {
                checkpoint()?;
                SpatialPoint::new(node.coordinates_m.canonical_value(), ()).ok_or_else(|| {
                    error(
                        DelaunayVolumeRefinementCandidateErrorKind::InvalidTopology,
                        "candidate lookup requires finite admitted coordinates",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let locations = PointSpatialIndex::from_entries_with_checkpoint(entries, &mut checkpoint)?;
        Ok(Self { locations })
    }

    #[cfg(test)]
    pub(in crate::cdt::volume_refinement) fn contains_location(
        &self,
        coordinates: [f64; 3],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<bool, DelaunayVolumeRefinementCandidateError> {
        let mut found = false;
        self.locations.visit_bounds_with_checkpoint(
            Aabb3 {
                min_m: coordinates,
                max_m: coordinates,
            },
            &mut |point| {
                // Preserve IEEE coordinate equality, including signed zero.
                found |= point.position() == coordinates;
            },
            &mut || checkpoint(cancellation),
        )?;
        Ok(found)
    }

    /// Accepted insertion never moves old coordinates. Append only the new
    /// admitted points, preserving shared pages/branches across batch retries.
    pub(in crate::cdt::volume_refinement) fn append(
        &mut self,
        nodes: &[crate::cdt::DelaunayVolumeNode],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), DelaunayVolumeRefinementCandidateError> {
        let mut updated = self.locations.clone();
        for node in nodes {
            checkpoint(cancellation)?;
            let entry =
                SpatialPoint::new(node.coordinates_m.canonical_value(), ()).ok_or_else(|| {
                    error(
                        DelaunayVolumeRefinementCandidateErrorKind::InvalidTopology,
                        "accepted candidate requires finite coordinates",
                    )
                })?;
            updated.insert_with_checkpoint(entry, &mut || checkpoint(cancellation))?;
        }
        self.locations = updated;
        Ok(())
    }
}

fn checkpoint(
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeRefinementCandidateError> {
    if cancellation.is_cancelled() {
        Err(error(
            DelaunayVolumeRefinementCandidateErrorKind::Cancelled,
            "cancelled",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
