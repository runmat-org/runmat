//! Volume construction charges the heap it holds against the run's memory ceiling.
//!
//! Quality convergence grows the working mesh one cavity at a time. Before each step the charge
//! covers the working inventory (`WorkingState::resident_bytes`) plus a growth window: one eighth
//! more of the working inventory and one doubling of each large vector, whose reallocation briefly
//! holds the old and the new buffer. When the mesh grows past the window it measures the inventory
//! again and reserves the next window first, so a mesh that does not fit stops with a memory
//! refusal before it grows into memory it was not admitted. The growth within one window is
//! charged at the window's starting bytes per tetrahedron.
use super::{error, DelaunayVolumeMesh, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind};
use super::{DelaunayVolumeMeshStage, DelaunayVolumeQualityMutation};
use crate::cdt::volume_quality::working::quality_payload_bytes;
use crate::cdt::volume_state::{SearchState, WorkingState};
use crate::cdt::{
    DelaunayBoundaryFacet, DelaunayRegionIncidence, DelaunayTetrahedronQuality, DelaunayVolumeNode,
    DelaunayVolumeQuality, DelaunayVolumeSliverMutation, DelaunayVolumeTetrahedron,
    DelaunayVolumeTopology,
};
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};
use runmat_meshing_core::resident_bytes::vec_bytes;

/// Growth window: the charge covers the working mesh until its tetrahedron slots or nodes grow by
/// this fraction of their count at the last inventory.
const WINDOW_DIVISOR: u64 = 8;

/// The construction's charge on the run's memory authority. Without an authority (tests and
/// diagnostic replay) it charges nothing.
pub(super) struct ResidentCharge<'a> {
    control: Option<&'a dyn GeometryEvaluationControl>,
    reservation: Option<GeometryAllocationReservation<'a>>,
    covered_tetrahedra: usize,
    covered_nodes: usize,
}

impl<'a> ResidentCharge<'a> {
    pub(super) fn new(control: Option<&'a dyn GeometryEvaluationControl>) -> Self {
        Self {
            control,
            reservation: None,
            covered_tetrahedra: 0,
            covered_nodes: 0,
        }
    }

    /// Hold exactly `bytes`: reserve the increase before the storage grows, or release the part
    /// that is no longer held.
    pub(super) fn hold(&mut self, bytes: u64) -> Result<(), DelaunayVolumeMeshError> {
        let Some(control) = self.control else {
            return Ok(());
        };
        let reservation = match &mut self.reservation {
            Some(reservation) => reservation,
            empty => empty.insert(
                GeometryAllocationReservation::new(control, 0)
                    .map_err(|failure| refusal(failure, bytes))?,
            ),
        };
        let held = reservation.bytes();
        if bytes > held {
            reservation
                .grow(bytes - held)
                .map_err(|failure| refusal(failure, bytes))?;
        } else {
            reservation.release(held - bytes);
        }
        Ok(())
    }

    /// Cover the working mesh and its next growth window before a convergence step.
    pub(super) fn cover_working(
        &mut self,
        state: &WorkingState<'_>,
        mutations: &Vec<DelaunayVolumeQualityMutation>,
        retained: u64,
    ) -> Result<(), DelaunayVolumeMeshError> {
        if self.control.is_none() {
            return Ok(());
        }
        let topology = state.topology();
        let tetrahedra = topology.tetrahedra.len();
        let nodes = topology.nodes.len();
        if self.reservation.is_some()
            && tetrahedra < self.covered_tetrahedra
            && nodes < self.covered_nodes
        {
            return Ok(());
        }
        let working = state
            .resident_bytes()
            .saturating_add(mutation_log_bytes(mutations));
        let doublings = [
            vec_bytes::<Option<crate::cdt::mutation::MutableTetrahedron>>(
                topology.tetrahedra.capacity(),
            ),
            vec_bytes::<DelaunayVolumeNode>(topology.nodes.capacity()),
            vec_bytes::<std::collections::BTreeSet<u32>>(topology.vertex_stars.capacity()),
            vec_bytes::<DelaunayVolumeQualityMutation>(mutations.capacity()),
        ]
        .into_iter()
        .fold(0, u64::saturating_add)
        .saturating_mul(2);
        let required = retained
            .saturating_add(working)
            .saturating_add(working / WINDOW_DIVISOR)
            .saturating_add(doublings);
        self.hold(required)?;
        self.covered_tetrahedra = tetrahedra + tetrahedra / WINDOW_DIVISOR as usize + 1;
        self.covered_nodes = nodes + nodes / WINDOW_DIVISOR as usize + 1;
        Ok(())
    }

    /// Reserve the canonical topology and quality that the working state hands off, on top of
    /// the working state that is still live while they are built.
    pub(super) fn cover_handoff(
        &mut self,
        state: &WorkingState<'_>,
        mutations: &Vec<DelaunayVolumeQualityMutation>,
        retained: u64,
    ) -> Result<(), DelaunayVolumeMeshError> {
        if self.control.is_none() {
            return Ok(());
        }
        let working = state
            .resident_bytes()
            .saturating_add(mutation_log_bytes(mutations));
        let handoff = state.handoff_bytes();
        self.hold(retained.saturating_add(working).saturating_add(handoff))?;
        self.covered_tetrahedra = 0;
        self.covered_nodes = 0;
        Ok(())
    }
}

fn mutation_log_bytes(mutations: &Vec<DelaunayVolumeQualityMutation>) -> u64 {
    mutations.iter().fold(
        vec_bytes::<DelaunayVolumeQualityMutation>(mutations.capacity()),
        |sum, mutation| match mutation {
            DelaunayVolumeQualityMutation::Refined(_) => sum,
            DelaunayVolumeQualityMutation::Optimized(mutation) => {
                sum.saturating_add(sliver_mutation_bytes(mutation))
            }
        },
    )
}

fn sliver_mutation_bytes(mutation: &DelaunayVolumeSliverMutation) -> u64 {
    match mutation {
        DelaunayVolumeSliverMutation::Reconnected(reconnection) => {
            vec_bytes::<[runmat_meshing_core::StableDigest; 4]>(
                reconnection.removed_tetrahedron_node_identities.capacity(),
            )
            .saturating_add(vec_bytes::<[runmat_meshing_core::StableDigest; 4]>(
                reconnection.created_tetrahedron_node_identities.capacity(),
            ))
        }
        DelaunayVolumeSliverMutation::Relocated(_)
        | DelaunayVolumeSliverMutation::Deleted(_)
        | DelaunayVolumeSliverMutation::Inserted(_) => 0,
    }
}

/// Heap bytes of a canonical topology's vectors.
pub(in crate::cdt) fn topology_bytes(topology: &DelaunayVolumeTopology) -> u64 {
    let incidence = &topology.incidence;
    let stars = incidence.vertex_stars.iter().fold(
        vec_bytes::<Vec<u32>>(incidence.vertex_stars.capacity()),
        |sum, star| sum.saturating_add(vec_bytes::<u32>(star.capacity())),
    );
    let regions = incidence.regions.iter().fold(
        vec_bytes::<DelaunayRegionIncidence>(incidence.regions.capacity()),
        |sum, region| sum.saturating_add(vec_bytes::<u32>(region.tetrahedron_indices.capacity())),
    );
    [
        vec_bytes::<DelaunayVolumeNode>(topology.nodes.capacity()),
        vec_bytes::<DelaunayVolumeTetrahedron>(topology.tetrahedra.capacity()),
        stars,
        vec_bytes::<DelaunayBoundaryFacet>(incidence.boundary_facets.capacity()),
        regions,
        vec_bytes::<u32>(incidence.unassigned_tetrahedron_indices.capacity()),
    ]
    .into_iter()
    .fold(0, u64::saturating_add)
}

/// Heap bytes of a canonical quality record set.
pub(in crate::cdt) fn quality_bytes(quality: &DelaunayVolumeQuality) -> u64 {
    quality.tetrahedra.iter().fold(
        vec_bytes::<DelaunayTetrahedronQuality>(quality.tetrahedra.capacity()),
        |sum, tetrahedron| sum.saturating_add(quality_payload_bytes(tetrahedron)),
    )
}

/// Heap bytes of a provenance record set.
pub(in crate::cdt) fn provenance_bytes(provenance: &crate::cdt::DelaunayVolumeProvenance) -> u64 {
    use crate::cdt::{DelaunayFacetProvenance, DelaunayNodeProvenance, DelaunaySegmentProvenance};
    use runmat_geometry_core::PersistentEntityId;
    let nodes = provenance.nodes.iter().fold(
        vec_bytes::<DelaunayNodeProvenance>(provenance.nodes.capacity()),
        |sum, node| sum.saturating_add(vec_bytes::<PersistentEntityId>(node.entity_ids.capacity())),
    );
    let segments = provenance.segments.iter().fold(
        vec_bytes::<DelaunaySegmentProvenance>(provenance.segments.capacity()),
        |sum, segment| {
            sum.saturating_add(vec_bytes::<PersistentEntityId>(
                segment.entity_ids.capacity(),
            ))
        },
    );
    provenance.facets.iter().fold(
        nodes
            .saturating_add(segments)
            .saturating_add(vec_bytes::<DelaunayFacetProvenance>(
                provenance.facets.capacity(),
            )),
        |sum, facet| {
            sum.saturating_add(vec_bytes::<PersistentEntityId>(facet.entity_ids.capacity()))
                .saturating_add(vec_bytes::<PersistentEntityId>(facet.region_ids.capacity()))
        },
    )
}

impl DelaunayVolumeMesh {
    /// An upper bound on the heap bytes this mesh holds: topology, quality, provenance and the
    /// facet and quality mutation logs. Shared persistent identities are not counted.
    pub fn resident_bytes(&self) -> u64 {
        [
            topology_bytes(&self.topology),
            quality_bytes(&self.quality),
            provenance_bytes(&self.provenance),
            vec_bytes::<crate::cdt::DelaunayFacetSteinerInsertion>(
                self.facet_recovery_insertions.capacity(),
            ),
            mutation_log_bytes(&self.quality_mutations),
        ]
        .into_iter()
        .fold(0, u64::saturating_add)
    }
}

fn refusal(
    failure: runmat_geometry_core::GeometryEvaluationError,
    required_bytes: u64,
) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        runmat_geometry_core::GeometryEvaluationErrorKind::AllocationBudgetExceeded => {
            DelaunayVolumeMeshErrorKind::MemoryBudgetExceeded { required_bytes }
        }
        runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled => {
            DelaunayVolumeMeshErrorKind::Cancelled
        }
        _ => DelaunayVolumeMeshErrorKind::ResourceLimit,
    };
    error(
        DelaunayVolumeMeshStage::Refinement,
        kind,
        format!("volume construction needs {required_bytes} resident bytes: {failure}"),
    )
}
