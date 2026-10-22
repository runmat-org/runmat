//! Exact trim-membership predicates with cooperative progress between triangles.
use super::invalid;
use crate::refinement::{control::checkpoint, ExactFaceRefinementError};
use crate::ExactFaceGeometry;
use runmat_geometry_core::GeometryEvaluationControl;
use runmat_meshing_core::{predicate::orient2d, PredicateSign};

#[cfg(test)]
pub(super) fn lies_in_trimmed_domain(
    geometry: &ExactFaceGeometry,
    uv: [f64; 2],
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<bool, ExactFaceRefinementError> {
    for triangle in &geometry.triangles {
        checkpoint(control, &geometry.source_face_id)?;
        let vertices = triangle
            .triangle
            .vertex_indices
            .map(|index| geometry.vertices[index as usize].evaluation.uv);
        let signs = [
            orient2d([vertices[0], vertices[1], uv]),
            orient2d([vertices[1], vertices[2], uv]),
            orient2d([vertices[2], vertices[0], uv]),
        ]
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid(geometry, "refinement candidate has invalid predicate input"))?;
        let nonnegative = signs.iter().all(|sign| *sign != PredicateSign::Negative);
        let nonpositive = signs.iter().all(|sign| *sign != PredicateSign::Positive);
        if nonnegative || nonpositive {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Immutable retained-domain broad phase; exact orientation signs decide membership.
pub(super) struct DomainQueries<'a> {
    geometry: &'a ExactFaceGeometry,
    index: runmat_meshing_core::quality::spatial_index::PackedBoundsSpatialIndex<usize>,
}
impl<'a> DomainQueries<'a> {
    pub(super) fn prepare(
        geometry: &'a ExactFaceGeometry,
        control: Option<&dyn GeometryEvaluationControl>,
    ) -> Result<Self, ExactFaceRefinementError> {
        use runmat_meshing_core::quality::spatial_index::SpatialEntry;
        let index = crate::refinement::spatial::build(
            geometry.triangles.len(),
            geometry
                .triangles
                .iter()
                .enumerate()
                .map(|(payload, triangle)| SpatialEntry {
                    bounds: crate::refinement::spatial::bounds(
                        triangle
                            .triangle
                            .vertex_indices
                            .map(|index| geometry.vertices[index as usize].evaluation.uv),
                    ),
                    payload,
                }),
            control,
            &geometry.source_face_id,
        )?;
        Ok(Self { geometry, index })
    }
    pub(super) fn contains(
        &self,
        uv: [f64; 2],
        control: Option<&dyn GeometryEvaluationControl>,
    ) -> Result<bool, ExactFaceRefinementError> {
        checkpoint(control, &self.geometry.source_face_id)?;
        let mut inside = false;
        let mut failure = None;
        self.index.visit_bounds_with_checkpoint(
            crate::refinement::spatial::point(uv),
            &mut |entry| {
                let vertices = self.geometry.triangles[entry.payload]
                    .triangle
                    .vertex_indices
                    .map(|index| self.geometry.vertices[index as usize].evaluation.uv);
                let signs = [
                    orient2d([vertices[0], vertices[1], uv]),
                    orient2d([vertices[1], vertices[2], uv]),
                    orient2d([vertices[2], vertices[0], uv]),
                ];
                if signs.iter().any(Result::is_err) {
                    failure = Some(invalid(
                        self.geometry,
                        "refinement candidate has invalid predicate input",
                    ));
                    return;
                }
                inside |= signs
                    .iter()
                    .all(|sign| *sign != Ok(PredicateSign::Negative))
                    || signs
                        .iter()
                        .all(|sign| *sign != Ok(PredicateSign::Positive));
            },
            &mut || checkpoint(control, &self.geometry.source_face_id),
        )?;
        if let Some(failure) = failure {
            return Err(failure);
        }
        Ok(inside)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::GeometryEvaluationError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Count(AtomicUsize);
    impl GeometryEvaluationControl for Count {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }
    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Indexed membership matches the direct robust orientation walk on interior, boundary
    //   and exterior points over 1,024 disconnected regions with bounded local work. This is finite
    //   domain-search evidence, not independent CAD or PDE qualification.
    #[test]
    fn domain_index_preserves_direct_membership_and_local_search_work() {
        let mut geometry = crate::refinement::tests::geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
        let source_vertices = geometry.vertices.clone();
        let source_triangle = geometry.triangles[0].clone();
        geometry.vertices.clear();
        geometry.triangles.clear();
        for index in 0..1024 {
            let offset = [(index % 32) as f64 * 10.0, (index / 32) as f64 * 10.0];
            for vertex in &source_vertices {
                let mut current = vertex.clone();
                current.evaluation.uv[0] += offset[0];
                current.evaluation.uv[1] += offset[1];
                geometry.vertices.push(current);
            }
            let mut current = source_triangle.clone();
            current.triangle.vertex_indices = current
                .triangle
                .vertex_indices
                .map(|vertex| vertex + (index * source_vertices.len()) as u32);
            geometry.triangles.push(current);
        }
        let control = Count(AtomicUsize::new(0));
        let queries = DomainQueries::prepare(&geometry, Some(&control)).unwrap();
        for index in 0..1024 {
            for delta in [[0.1, 0.1], [0.0, 0.0], [4.0, 4.0]] {
                let point = [
                    (index % 32) as f64 * 10.0 + delta[0],
                    (index / 32) as f64 * 10.0 + delta[1],
                ];
                assert_eq!(
                    queries.contains(point, Some(&control)).unwrap(),
                    lies_in_trimmed_domain(&geometry, point, None).unwrap()
                );
            }
        }
        assert!(control.0.load(Ordering::Relaxed) < 500 * 1024);
    }
}
