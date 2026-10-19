//! Immutable face/chart broad phase with flat admitted buffers and packed bounds.
use super::{bytes, invalid, sort, work, NodeSources};
use crate::cdt::solver_topology::order_elevation::geometry::SurfaceUse;
use crate::cdt::solver_topology::{storage::Storage, DelaunaySolverTopologyError};
use runmat_geometry_core::{
    GeometryAllocationReservation, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::{
    quality::spatial_index::{Aabb3, PackedBoundsSpatialIndex, SpatialEntry},
    StableDigest,
};
use runmat_meshing_surface::ExactSurfaceMesh;
type SurfaceKey<'a> = (&'a PersistentEntityId, StableDigest);
type Row<'a> = (SurfaceKey<'a>, Aabb3, usize);
type SurfaceIndex<'a> = (SurfaceKey<'a>, PackedBoundsSpatialIndex<usize>);
pub(super) struct SurfaceLookup<'a> {
    by_surface: Vec<SurfaceIndex<'a>>,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> SurfaceLookup<'a> {
    pub(super) fn prepare(
        surface: &'a ExactSurfaceMesh,
        tolerance: f64,
        nodes: &NodeSources<'a>,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let row_storage =
            GeometryAllocationReservation::new(control, bytes::<Row<'_>>(surface.triangles.len())?)
                .map_err(crate::cdt::solver_topology::error::geometry)?;
        let mut rows = Vec::with_capacity(surface.triangles.len());
        for (index, triangle) in surface.triangles.iter().enumerate() {
            work(control)?;
            let mut points = [[0.0; 3]; 3];
            for (slot, id) in triangle.node_ids.iter().enumerate() {
                points[slot] = nodes
                    .get(id)
                    .ok_or_else(|| invalid("surface triangle references an absent source node"))?
                    .point_m;
            }
            let mut bounds = Aabb3::from_points(&points)
                .ok_or_else(|| invalid("nonfinite source surface triangle"))?;
            let scale = (1..3)
                .map(|corner| {
                    points[corner]
                        .iter()
                        .zip(points[0])
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0_f64, f64::max);
            // Barycentric admission permits tiny negative weights and a normal
            // residual. Include world-coordinate arithmetic roundoff as well as
            // local span; this widens only the broad phase, never physical admission.
            let world_scale = points
                .iter()
                .flatten()
                .map(|value| value.abs())
                .fold(scale, f64::max);
            let padding = 8.0 * tolerance.max(64.0 * f64::EPSILON * world_scale);
            for axis in 0..3 {
                bounds.min_m[axis] -= padding;
                bounds.max_m[axis] += padding;
            }
            rows.push(((&triangle.source_face_id, triangle.chart_id), bounds, index));
        }
        sort(
            &mut rows,
            |a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)),
            control,
        )?;
        let mut groups = 0usize;
        let mut previous = None;
        for (key, _, _) in &rows {
            work(control)?;
            if previous != Some(*key) {
                groups += 1;
                previous = Some(*key);
            }
        }
        let profile = PackedBoundsSpatialIndex::<usize>::storage_profile(rows.len())
            .ok_or_else(|| invalid("surface spatial storage overflow"))?;
        let requested = profile
            .retained_bytes
            .checked_add(bytes::<SurfaceIndex<'_>>(groups)?)
            .ok_or_else(|| invalid("surface spatial storage overflow"))?;
        let storage = GeometryAllocationReservation::new(control, requested)
            .map_err(crate::cdt::solver_topology::error::geometry)?;
        let mut by_surface = Vec::with_capacity(groups);
        let mut start = 0;
        while start < rows.len() {
            work(control)?;
            let key = rows[start].0;
            let end = start + rows[start..].partition_point(|row| row.0 == key);
            let profile = PackedBoundsSpatialIndex::<usize>::storage_profile(end - start)
                .ok_or_else(|| invalid("surface spatial storage overflow"))?;
            let input_storage = GeometryAllocationReservation::new(
                control,
                bytes::<SpatialEntry<usize>>(end - start)?,
            )
            .map_err(crate::cdt::solver_topology::error::geometry)?;
            let mut entries = Vec::with_capacity(end - start);
            entries.extend(rows[start..end].iter().map(|row| SpatialEntry {
                bounds: row.1,
                payload: row.2,
            }));
            let construction_storage =
                GeometryAllocationReservation::new(control, profile.construction_scratch_bytes)
                    .map_err(crate::cdt::solver_topology::error::geometry)?;
            let index =
                PackedBoundsSpatialIndex::from_entries_with_checkpoint(entries, &mut || {
                    work(control)
                })?
                .ok_or_else(|| invalid("invalid source surface bounds"))?;
            drop(construction_storage);
            drop(input_storage);
            by_surface.push((key, index));
            start = end;
        }
        drop(rows);
        drop(row_storage);
        Ok(Self {
            by_surface,
            _storage: storage,
        })
    }
    pub(super) fn candidates_into(
        &self,
        surface: &SurfaceUse,
        point: [f64; 3],
        control: &dyn GeometryEvaluationControl,
        result: &mut Vec<usize>,
    ) -> Result<(), DelaunaySolverTopologyError> {
        work(control)?;
        result.clear();
        let key = (surface.source_face_id, surface.chart_id);
        if let Ok(index) = self.by_surface.binary_search_by(|entry| entry.0.cmp(&key)) {
            let failure = std::cell::RefCell::new(None::<DelaunaySolverTopologyError>);
            self.by_surface[index].1.visit_bounds_with_checkpoint(
                Aabb3 {
                    min_m: point,
                    max_m: point,
                },
                &mut |entry| {
                    if failure.borrow().is_some() {
                        return;
                    }
                    if result.len() == result.capacity() {
                        let capacity = result.capacity().saturating_mul(2).max(4);
                        if let Err(error) = Storage(Some(control)).reserve(result, capacity) {
                            *failure.borrow_mut() = Some(error);
                            return;
                        }
                    }
                    result.push(entry.payload);
                },
                &mut || {
                    if let Some(error) = failure.borrow().as_ref() {
                        return Err(error.clone());
                    }
                    work(control)
                },
            )?;
        }
        Ok(())
    }
    #[cfg(test)]
    fn candidates(
        &self,
        surface: &SurfaceUse,
        point: [f64; 3],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Vec<usize>, DelaunaySolverTopologyError> {
        let mut result = Vec::new();
        self.candidates_into(surface, point, control, &mut result)?;
        Ok(result)
    }
}
#[cfg(test)]
mod tests;
