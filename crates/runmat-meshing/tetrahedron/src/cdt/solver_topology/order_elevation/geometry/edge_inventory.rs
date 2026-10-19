//! Flat borrowed owner preparation. Repeated tetrahedron visits allocate no entity strings.
use super::*;
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};
use std::ops::Range;
type Row<'a> = ([u64; 2], &'a PersistentEntityId);
pub(super) struct EdgeOwners<'source, 'control> {
    pub groups: Vec<([u64; 2], Range<usize>)>,
    rows: Vec<Row<'source>>,
    _rows_storage: GeometryAllocationReservation<'control>,
    _groups_storage: GeometryAllocationReservation<'control>,
}
impl<'source, 'control> EdgeOwners<'source, 'control> {
    pub fn prepare(
        topology: &SolverMeshTopology,
        owners: &[&'source PersistentEntityId],
        control: &'control dyn GeometryEvaluationControl,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let count = topology.volume_elements.iter().enumerate().try_fold(
            0usize,
            |count, (work, element)| {
                if work.is_multiple_of(128) {
                    control.checkpoint().map_err(error::geometry)?;
                }
                let owners = element.provenance.len()
                    + usize::from(!element.provenance.contains(&element.region_id));
                count
                    .checked_add(
                        owners
                            .checked_mul(6)
                            .ok_or_else(|| resource("edge owner inventory overflow"))?,
                    )
                    .ok_or_else(|| resource("edge owner inventory overflow"))
            },
        )?;
        let rows_storage = GeometryAllocationReservation::new(control, bytes::<Row<'_>>(count)?)
            .map_err(error::geometry)?;
        let mut rows = Vec::with_capacity(count);
        for element in &topology.volume_elements {
            control.checkpoint().map_err(error::geometry)?;
            let corners: [u64; 4] = element
                .node_ids
                .as_slice()
                .try_into()
                .map_err(|_| invalid("edge preparation requires linear volume elements"))?;
            for owner in element.provenance.iter().chain(
                (!element.provenance.contains(&element.region_id)).then_some(&element.region_id),
            ) {
                let index = owners
                    .binary_search_by(|candidate| (*candidate).cmp(owner))
                    .map_err(|_| invalid("solver volume owner is absent from its exact source"))?;
                for edge in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
                    control.consume_search_work(1).map_err(error::geometry)?;
                    rows.push((
                        sorted_edge([corners[edge[0]], corners[edge[1]]]),
                        owners[index],
                    ));
                }
            }
        }
        let scratch_storage = GeometryAllocationReservation::new(control, bytes::<Row<'_>>(count)?)
            .map_err(error::geometry)?;
        let mut scratch = rows.clone();
        runmat_cpu_execution::sort_copy_with_checkpoint(
            &mut rows,
            &mut scratch,
            |a: &Row<'_>, b: &Row<'_>| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)),
            || control.checkpoint().map_err(error::geometry),
        )?;
        drop(scratch);
        drop(scratch_storage);
        let mut retained = 0;
        for index in 0..rows.len() {
            if index.is_multiple_of(128) {
                control.checkpoint().map_err(error::geometry)?;
            }
            let row = rows[index];
            if retained == 0 || rows[retained - 1] != row {
                rows[retained] = row;
                retained += 1;
            }
        }
        rows.truncate(retained);
        let mut groups_count = 0usize;
        let mut prior = None;
        for (work, row) in rows.iter().enumerate() {
            if work.is_multiple_of(128) {
                control.checkpoint().map_err(error::geometry)?;
            }
            if prior != Some(row.0) {
                groups_count += 1;
                prior = Some(row.0);
            }
        }
        let groups_storage = GeometryAllocationReservation::new(
            control,
            bytes::<([u64; 2], Range<usize>)>(groups_count)?,
        )
        .map_err(error::geometry)?;
        let mut groups = Vec::with_capacity(groups_count);
        let mut start = 0;
        while start < rows.len() {
            control.checkpoint().map_err(error::geometry)?;
            let mut end = start + 1;
            while end < rows.len() && rows[end].0 == rows[start].0 {
                end += 1;
            }
            groups.push((rows[start].0, start..end));
            start = end;
        }
        Ok(Self {
            groups,
            rows,
            _rows_storage: rows_storage,
            _groups_storage: groups_storage,
        })
    }
    pub fn owners(
        &self,
        range: Range<usize>,
    ) -> impl Iterator<Item = &'source PersistentEntityId> + '_ {
        self.rows[range].iter().map(|entry| entry.1)
    }
}
fn bytes<T>(count: usize) -> Result<u64, DelaunaySolverTopologyError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| resource("flat edge storage inventory overflow"))
}

#[cfg(test)]
mod tests;
