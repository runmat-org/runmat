//! Borrow exact region/solid lineage once; no owned identity tree per projection.
use super::*;
use runmat_geometry_core::{ExactRegion, GeometryAllocationReservation};

pub(super) struct RegionSources<'a> {
    regions: Vec<&'a ExactRegion>,
    _storage: Option<GeometryAllocationReservation<'a>>,
}
impl<'a> RegionSources<'a> {
    pub fn prepare(
        input: &'a ProjectionInput<'_>,
        options: DelaunaySolverTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let count = input.exact_topology.regions.len();
        let bytes = (count as u64)
            .checked_mul(std::mem::size_of::<&ExactRegion>() as u64)
            .ok_or_else(|| {
                error::failure(
                    DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded,
                    "region lookup storage overflow",
                )
            })?;
        let control = input.exact_evaluation.map(|evaluation| evaluation.control);
        let storage = control
            .map(|control| GeometryAllocationReservation::new(control, bytes))
            .transpose()
            .map_err(error::geometry)?;
        let mut regions = input.exact_topology.regions.iter().collect::<Vec<_>>();
        let scratch_storage = control
            .map(|control| GeometryAllocationReservation::new(control, bytes))
            .transpose()
            .map_err(error::geometry)?;
        let mut scratch = regions.clone();
        runmat_cpu_execution::sort_copy_with_checkpoint(
            &mut regions,
            &mut scratch,
            |a: &&ExactRegion, b: &&ExactRegion| a.id.cmp(&b.id),
            || checkpoint(0, options, cancellation),
        )?;
        drop(scratch);
        drop(scratch_storage);
        let mesh_regions = &input.volume_mesh.topology.incidence.regions;
        if regions.len() != mesh_regions.len()
            || regions
                .iter()
                .zip(mesh_regions)
                .any(|(exact, mesh)| exact.id != mesh.region_id)
        {
            return Err(invalid_mesh(
                "solver projection requires one nonempty mesh region for every exact region",
            ));
        }
        Ok(Self {
            regions,
            _storage: storage,
        })
    }
    pub fn solid(&self, id: &PersistentEntityId) -> Option<&PersistentEntityId> {
        self.regions
            .binary_search_by(|region| region.id.cmp(id))
            .ok()
            .map(|index| &self.regions[index].solid_id)
    }
}
