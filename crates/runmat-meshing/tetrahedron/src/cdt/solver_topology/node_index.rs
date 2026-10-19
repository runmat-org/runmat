//! Flat identity lookup; deterministic storage and no tree node per volume vertex.
use super::*;
use runmat_meshing_core::StableDigest;
pub(super) struct NodeIndex<'a> {
    entries: Vec<(StableDigest, u32)>,
    _storage: Option<runmat_geometry_core::GeometryAllocationReservation<'a>>,
}
impl<'a> NodeIndex<'a> {
    pub fn new(
        input: &ProjectionInput<'a>,
        options: DelaunaySolverTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let count = input.volume_mesh.topology.nodes.len();
        let bytes = (count as u64)
            .checked_mul(std::mem::size_of::<(StableDigest, u32)>() as u64)
            .ok_or_else(|| {
                error::failure(
                    DelaunaySolverTopologyErrorKind::ResourceLimit,
                    "node lookup storage overflow",
                )
            })?;
        let control = input.exact_evaluation.map(|evaluation| evaluation.control);
        let storage = control
            .map(|control| runmat_geometry_core::GeometryAllocationReservation::new(control, bytes))
            .transpose()
            .map_err(error::geometry)?;
        let mut entries = Vec::with_capacity(count);
        for (index, node) in input.volume_mesh.topology.nodes.iter().enumerate() {
            checkpoint(index as u64, options, cancellation)?;
            let index = u32::try_from(index).map_err(|_| {
                error::failure(
                    DelaunaySolverTopologyErrorKind::ResourceLimit,
                    "volume node index exceeds u32",
                )
            })?;
            entries.push((node.identity, index));
        }
        // Shared cooperative sort retains bounded checkpoint intervals.
        let scratch_storage = control
            .map(|control| runmat_geometry_core::GeometryAllocationReservation::new(control, bytes))
            .transpose()
            .map_err(error::geometry)?;
        let mut scratch = entries.clone();
        runmat_cpu_execution::sort_copy_with_checkpoint(
            &mut entries,
            &mut scratch,
            |a: &(StableDigest, u32), b: &(StableDigest, u32)| a.0.cmp(&b.0),
            || checkpoint(0, options, cancellation),
        )?;
        drop(scratch);
        drop(scratch_storage);
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(error::failure(
                DelaunaySolverTopologyErrorKind::InvalidMesh,
                "volume nodes repeat a stable identity",
            ));
        }
        Ok(Self {
            entries,
            _storage: storage,
        })
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn get(&self, identity: &StableDigest) -> Option<&u32> {
        self.entries
            .binary_search_by_key(identity, |entry| entry.0)
            .ok()
            .map(|index| &self.entries[index].1)
    }
}
