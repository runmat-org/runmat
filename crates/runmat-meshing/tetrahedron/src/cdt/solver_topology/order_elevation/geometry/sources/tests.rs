use super::*;
use crate::cdt::{DelaunayVolumeMeshOptions, DelaunayVolumeQualityOptions};
use runmat_geometry_core::{GeometryEvaluationError, GeometryEvaluationErrorKind};
use runmat_meshing_core::{ElementOrder, NeverCancelled};

struct Denied {
    allocation: bool,
}
impl GeometryEvaluationControl for Denied {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        if self.allocation {
            Ok(())
        } else {
            Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::SearchWorkBudgetExceeded,
                "deny prepared source work",
            ))
        }
    }
    fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::AllocationBudgetExceeded,
            "deny prepared source storage",
        ))
    }
}

struct Fixture {
    exact_topology: runmat_geometry_core::ExactBRepTopology,
    exact_surface: runmat_meshing_surface::ExactSurfaceMesh,
    request: runmat_meshing_core::MeshingRequest,
    volume: crate::cdt::DelaunayVolumeMesh,
}
impl Fixture {
    fn new() -> Self {
        let (exact_topology, exact_surface) = crate::cdt::constraints::tests::tetrahedron();
        let mut request =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4)
                .resolved_request;
        request.metric.global_metric =
            runmat_meshing_core::MetricTensor3::isotropic_length_m(2.0).unwrap();
        let options = DelaunayVolumeMeshOptions {
            quality: DelaunayVolumeQualityOptions {
                maximum_metric_edge_length: 2.0,
                maximum_radius_edge_ratio: 10.0,
                minimum_metric_scaled_jacobian: 0.01,
                ..Default::default()
            },
            ..Default::default()
        };
        let volume = crate::cdt::construct_delaunay_volume_mesh(
            &exact_topology,
            &exact_surface,
            &request.metric,
            options,
            &NeverCancelled,
        )
        .unwrap();
        Self {
            exact_topology,
            exact_surface,
            request,
            volume,
        }
    }
    fn input(&self) -> ProjectionInput<'_> {
        ProjectionInput {
            exact_topology: &self.exact_topology,
            exact_surface: &self.exact_surface,
            volume_mesh: crate::cdt::solver_topology::source::VolumeView {
                topology: &self.volume.topology,
                provenance: &self.volume.provenance,
            },
            request: &self.request,
            exact_evaluation: None,
        }
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Elevation source preparation rejects denied storage and search work before
//   constructing borrowed lookup tables. Its conservative component storage envelope is not
//   measured RSS or a complete worker-memory proof.
#[test]
fn prepared_sources_reject_unadmitted_storage_and_search_before_construction() {
    let fixture = Fixture::new();
    for allocation in [true, false] {
        let failure = ParameterSources::prepare(&fixture.input(), &Denied { allocation })
            .err()
            .unwrap();
        assert_eq!(
            failure.kind,
            if allocation {
                DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded
            } else {
                DelaunaySolverTopologyErrorKind::SearchWorkBudgetExceeded
            }
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Prepared source images reject a present coedge identity belonging to another face or
//   edge. All admitted periodic and singular images remain retained; this component ownership check
//   does not replace original exact-geometry validation.
#[test]
fn source_chart_images_reject_coedges_from_another_face_or_edge() {
    let mut fixture = Fixture::new();
    let node_id = fixture.exact_surface.nodes[0].node_id;
    let triangle = fixture
        .exact_surface
        .triangles
        .iter()
        .find(|triangle| triangle.node_ids.contains(&node_id))
        .unwrap();
    let owner = fixture
        .exact_topology
        .coedges
        .iter()
        .find(|coedge| coedge.face_id == triangle.source_face_id)
        .unwrap();
    let foreign = fixture
        .exact_topology
        .coedges
        .iter()
        .find(|coedge| coedge.face_id != owner.face_id || coedge.edge_id != owner.edge_id)
        .unwrap();
    fixture.exact_surface.nodes[0]
        .uses
        .push(runmat_meshing_surface::ExactFaceMeshNodeUse {
            source_face_id: triangle.source_face_id.clone(),
            chart_id: triangle.chart_id,
            uv: [0.0; 2],
            evaluator_uv: [0.0; 2],
            exact_edge_parameters: vec![runmat_meshing_surface::ExactFaceMeshEdgeParameter {
                source_coedge_id: foreign.id.clone(),
                source_edge_id: owner.edge_id.clone(),
                parameter: 0.0,
            }],
        });
    let failure = ParameterSources::prepare(
        &fixture.input(),
        &crate::cdt::solver_topology::test_evaluator::CONTROL,
    )
    .err()
    .unwrap();
    assert_eq!(failure.kind, DelaunaySolverTopologyErrorKind::InvalidMesh);
    assert!(failure.reason.contains("wrong face or edge owner"));
}

#[test]
fn source_spatial_index_is_lazy_reused_and_released_with_its_original_owner() {
    use crate::cdt::test_fixtures::allocation::AllocationBudget;
    use runmat_geometry_core::GeometryAllocationScope;
    let fixture = Fixture::new();
    let budget = AllocationBudget::new(512 << 20);
    {
        let scope = GeometryAllocationScope::new(&budget);
        let sources = ParameterSources::prepare(&fixture.input(), &scope).unwrap();
        assert!(sources.surface.get().is_none());
        let triangle = &fixture.exact_surface.triangles[0];
        let usage = super::super::SurfaceUse {
            source_face_id: &triangle.source_face_id,
            chart_id: triangle.chart_id,
        };
        let point = std::array::from_fn(|axis| {
            triangle
                .node_ids
                .iter()
                .map(|id| sources.nodes.get(id).unwrap().point_m[axis])
                .sum::<f64>()
                / 3.0
        });
        let mut candidates = Vec::new();
        sources
            .triangle_candidates(&usage, point, &scope, &mut candidates)
            .unwrap();
        assert!(sources.surface.get().is_some());
        assert!(candidates.contains(&0));
        let expected = candidates.clone();
        let live = budget.live_bytes();
        for _ in 0..100 {
            sources
                .triangle_candidates(&usage, point, &scope, &mut candidates)
                .unwrap();
            assert_eq!(candidates, expected);
            assert_eq!(budget.live_bytes(), live);
        }
        drop(candidates);
        drop(sources);
    }
    assert_eq!(budget.live_bytes(), 0);
}
