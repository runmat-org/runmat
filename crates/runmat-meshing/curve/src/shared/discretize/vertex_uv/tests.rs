use super::*;
use runmat_geometry_core::{
    GeometryEvaluationError, ParameterRange, PcurveDerivatives, PcurveEvaluatorId,
    SurfaceCurvature, SurfaceDerivatives, SurfaceEvaluatorId, SurfaceProjection,
};

struct Pcurves {
    representative: PcurveEvaluatorId,
    offset: f64,
}
impl ExactPcurveEvaluator for Pcurves {
    fn parameter_range(
        &self,
        _: &PcurveEvaluatorId,
    ) -> Result<ParameterRange, GeometryEvaluationError> {
        Ok(ParameterRange {
            start: 0.0,
            end: 1.0,
        })
    }
    fn point(
        &self,
        id: &PcurveEvaluatorId,
        _: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[runmat_geometry_core::GeometryParameterCoordinate; 2], GeometryEvaluationError>
    {
        control.checkpoint()?;
        let values = [
            1.0 + if id == &self.representative {
                0.0
            } else {
                self.offset
            },
            2.0,
        ];
        let axes = [
            runmat_geometry_core::GeometryParameterAxis::SurfaceU,
            runmat_geometry_core::GeometryParameterAxis::SurfaceV,
        ];
        let coordinate = |index| {
            runmat_geometry_core::GeometryParameterCoordinate::try_new(
                values[index],
                runmat_geometry_core::GeometryParameterFrame::native(
                    "fixture:surface",
                    axes[index],
                ),
            )
            .map_err(|error| {
                GeometryEvaluationError::new(
                    runmat_geometry_core::GeometryEvaluationErrorKind::InvalidResult,
                    error.to_string(),
                )
            })
        };
        Ok([coordinate(0)?, coordinate(1)?])
    }
    fn derivatives(
        &self,
        _: &PcurveEvaluatorId,
        _: f64,
        _: &dyn GeometryEvaluationControl,
    ) -> Result<PcurveDerivatives, GeometryEvaluationError> {
        panic!("endpoint admission needs only points")
    }
}
/// A plane whose parametrization spans `meters_per_uv` of model space per UV unit.
struct Plane {
    meters_per_uv: f64,
}
impl ExactSurfaceEvaluator for Plane {
    fn parameter_bounds(
        &self,
        _: &SurfaceEvaluatorId,
    ) -> Result<[ParameterRange; 2], GeometryEvaluationError> {
        panic!("vertex admission needs only points")
    }
    fn periodicity(
        &self,
        _: &SurfaceEvaluatorId,
    ) -> Result<[Option<f64>; 2], GeometryEvaluationError> {
        Ok([None, None])
    }
    fn point(
        &self,
        _: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<runmat_geometry_core::physical::Position, GeometryEvaluationError> {
        control.checkpoint()?;
        Ok(runmat_geometry_core::physical::Position::new([
            uv[0] * self.meters_per_uv,
            uv[1] * self.meters_per_uv,
            0.0,
        ]))
    }
    fn derivatives(
        &self,
        _: &SurfaceEvaluatorId,
        _: [f64; 2],
        _: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceDerivatives, GeometryEvaluationError> {
        panic!("vertex admission needs only points")
    }
    fn unit_normal(
        &self,
        _: &SurfaceEvaluatorId,
        _: [f64; 2],
        _: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        panic!("vertex admission needs only points")
    }
    fn principal_curvature(
        &self,
        _: &SurfaceEvaluatorId,
        _: [f64; 2],
        _: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceCurvature, GeometryEvaluationError> {
        panic!("vertex admission needs only points")
    }
    fn closest_point(
        &self,
        _: &SurfaceEvaluatorId,
        _: runmat_geometry_core::physical::Position,
        _: runmat_geometry_core::physical::Length,
        _: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceProjection, GeometryEvaluationError> {
        panic!("vertex admission needs only points")
    }
}
const UNIT_PLANE: Plane = Plane { meters_per_uv: 1.0 };

struct Control;
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
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
fn topology() -> ExactBRepTopology {
    let mut topology = runmat_geometry_fixtures::exact_circle().1;
    let mut edge = topology.edges[0].clone();
    edge.id.source_topology_id.push_str(":other");
    let mut coedge = topology.coedges[0].clone();
    coedge.id.source_topology_id.push_str(":other");
    coedge.edge_id = edge.id.clone();
    coedge.pcurve_evaluator_id = PcurveEvaluatorId::new("pcurve:other").unwrap();
    topology.edges.push(edge);
    topology.coedges.push(coedge);
    topology
}

#[test]
fn incident_pcurves_share_one_coordinate_independently_of_partition_or_traversal() {
    let mut topology = topology();
    let edge = topology.edges[1].clone();
    let coedge = topology.coedges[1].clone();
    let vertex = edge.start_vertex_id.as_ref().unwrap();
    let pcurves = Pcurves {
        representative: topology.coedges[0].pcurve_evaluator_id.clone(),
        offset: 7.0e-15,
    };
    let original = pcurves
        .point(&coedge.pcurve_evaluator_id, 0.0, &Control)
        .unwrap()
        .each_ref()
        .map(|coordinate| coordinate.with_native_coordinate(|value, _frame| value));
    assert_ne!(original, [1.0, 2.0]);
    for _ in 0..2 {
        let uv = VertexUvContext {
            topology: &topology,
            pcurves: &pcurves,
            surfaces: &UNIT_PLANE,
            control: &Control,
            maximum_error: 1.0e-10,
            geometry_error_m: 1.0e-12,
        }
        .canonical_coordinate(&edge, &coedge, vertex, original)
        .unwrap();
        assert_eq!(uv, [1.0, 2.0]);
        topology.coedges.reverse();
    }
}

#[test]
fn endpoint_authority_preserves_distinct_vertices_and_rejects_excess_error() {
    let mut topology = topology();
    let pcurves = Pcurves {
        representative: topology.coedges[0].pcurve_evaluator_id.clone(),
        offset: 0.01,
    };
    let mut edge = topology.edges[1].clone();
    let coedge = &topology.coedges[1];
    let original = pcurves
        .point(&coedge.pcurve_evaluator_id, 0.0, &Control)
        .unwrap()
        .each_ref()
        .map(|coordinate| coordinate.with_native_coordinate(|value, _frame| value));
    let context = VertexUvContext {
        topology: &topology,
        pcurves: &pcurves,
        surfaces: &UNIT_PLANE,
        control: &Control,
        maximum_error: 1.0e-10,
        geometry_error_m: 1.0e-12,
    };
    let error = context
        .canonical_coordinate(
            &edge,
            coedge,
            edge.start_vertex_id.as_ref().unwrap(),
            original,
        )
        .unwrap_err();
    assert_eq!(error.kind, SharedCurveErrorKind::GeometricMismatch);
    let mut distinct = edge.start_vertex_id.clone().unwrap();
    distinct.source_topology_id.push_str(":distinct");
    edge.start_vertex_id = Some(distinct.clone());
    edge.end_vertex_id = Some(distinct.clone());
    topology.edges[1] = edge.clone();
    let uv = VertexUvContext {
        topology: &topology,
        pcurves: &pcurves,
        surfaces: &UNIT_PLANE,
        control: &Control,
        maximum_error: 1.0e-10,
        geometry_error_m: 1.0e-12,
    }
    .canonical_coordinate(&edge, &topology.coedges[1], &distinct, original)
    .unwrap();
    assert_eq!(uv, original);
}

/// CAD vertices close in model space but far apart in UV (a finely parametrized surface)
/// share the representative coordinate; separation beyond the vertex tolerance is rejected.
#[test]
fn incident_pcurves_agree_within_the_source_vertex_tolerance_in_model_space() {
    let mut topology = topology();
    let pcurves = Pcurves {
        representative: topology.coedges[0].pcurve_evaluator_id.clone(),
        offset: 0.01,
    };
    let edge = topology.edges[1].clone();
    let coedge = topology.coedges[1].clone();
    let vertex = edge.start_vertex_id.clone().unwrap();
    let original = pcurves
        .point(&coedge.pcurve_evaluator_id, 0.0, &Control)
        .unwrap()
        .each_ref()
        .map(|coordinate| coordinate.with_native_coordinate(|value, _frame| value));
    // 0.01 UV on a surface spanning 1 um per UV unit is 10 nm in model space.
    let fine = Plane {
        meters_per_uv: 1.0e-6,
    };
    let set_tolerance = |topology: &mut ExactBRepTopology, tolerance_m: f64| {
        let source = topology
            .vertices
            .iter_mut()
            .find(|candidate| candidate.id == vertex)
            .unwrap();
        source.tolerance_m = runmat_types::physical::Length::new(tolerance_m);
    };
    set_tolerance(&mut topology, 1.0e-7);
    let uv = context(&topology, &pcurves, &fine)
        .canonical_coordinate(&edge, &coedge, &vertex, original)
        .unwrap();
    assert_eq!(uv, [1.0, 2.0]);

    set_tolerance(&mut topology, 1.0e-9);
    let error = context(&topology, &pcurves, &fine)
        .canonical_coordinate(&edge, &coedge, &vertex, original)
        .unwrap_err();
    assert_eq!(error.kind, SharedCurveErrorKind::GeometricMismatch);
    assert!(error.reason.contains("admitted 1e-9 m"), "{}", error.reason);
}

fn context<'a>(
    topology: &'a ExactBRepTopology,
    pcurves: &'a Pcurves,
    surfaces: &'a Plane,
) -> VertexUvContext<'a> {
    VertexUvContext {
        topology,
        pcurves,
        surfaces,
        control: &Control,
        maximum_error: 1.0e-10,
        geometry_error_m: 1.0e-12,
    }
}
