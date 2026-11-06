use super::*;
use runmat_geometry_core::{
    GeometryEvaluationError, ParameterRange, PcurveDerivatives, PcurveEvaluatorId,
    SurfaceCurvature, SurfaceDerivatives, SurfaceEvaluatorId, SurfaceProjection,
};

/// Two pcurves on a plane that maps UV (1, 2) onto the vertex and scales UV by
/// `meters_per_uv`. The non-representative pcurve ends `offset` away in U.
struct Pcurves {
    representative: PcurveEvaluatorId,
    offset: f64,
    meters_per_uv: f64,
}

const VERTEX_UV: [f64; 2] = [1.0, 2.0];
const VERTEX_POINT_M: [f64; 3] = [0.25, -0.5, 0.75];

impl ExactSurfaceEvaluator for Pcurves {
    fn parameter_bounds(
        &self,
        _: &SurfaceEvaluatorId,
    ) -> Result<[ParameterRange; 2], GeometryEvaluationError> {
        panic!("endpoint admission needs only points")
    }
    fn periodicity(
        &self,
        _: &SurfaceEvaluatorId,
    ) -> Result<[Option<f64>; 2], GeometryEvaluationError> {
        panic!("endpoint admission needs only points")
    }
    fn point(
        &self,
        _: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<runmat_types::physical::Position, GeometryEvaluationError> {
        control.checkpoint()?;
        Ok(runmat_types::physical::Position::new([
            VERTEX_POINT_M[0] + self.meters_per_uv * (uv[0] - VERTEX_UV[0]),
            VERTEX_POINT_M[1] + self.meters_per_uv * (uv[1] - VERTEX_UV[1]),
            VERTEX_POINT_M[2],
        ]))
    }
    fn derivatives(
        &self,
        _: &SurfaceEvaluatorId,
        _: [f64; 2],
        _: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceDerivatives, GeometryEvaluationError> {
        panic!("endpoint admission needs only points")
    }
    fn unit_normal(
        &self,
        _: &SurfaceEvaluatorId,
        _: [f64; 2],
        _: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        panic!("endpoint admission needs only points")
    }
    fn principal_curvature(
        &self,
        _: &SurfaceEvaluatorId,
        _: [f64; 2],
        _: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceCurvature, GeometryEvaluationError> {
        panic!("endpoint admission needs only points")
    }
    fn closest_point(
        &self,
        _: &SurfaceEvaluatorId,
        _: runmat_types::physical::Position,
        _: runmat_types::physical::Length,
        _: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceProjection, GeometryEvaluationError> {
        panic!("endpoint admission needs only points")
    }
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
            VERTEX_UV[0]
                + if id == &self.representative {
                    0.0
                } else {
                    self.offset
                },
            VERTEX_UV[1],
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
fn topology(vertex_tolerance_m: f64) -> ExactBRepTopology {
    let mut topology = runmat_geometry_fixtures::exact_circle().1;
    for vertex in &mut topology.vertices {
        vertex.point_m = runmat_types::physical::Position::new(VERTEX_POINT_M);
        vertex.tolerance_m = runmat_types::physical::Length::new(vertex_tolerance_m);
    }
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

fn context<'a>(topology: &'a ExactBRepTopology, pcurves: &'a Pcurves) -> VertexUvContext<'a> {
    VertexUvContext {
        topology,
        pcurves,
        control: &Control,
        geometry_absolute_error_m: 1.0e-12,
    }
}

fn endpoint(pcurves: &Pcurves, coedge: &ExactCoedge) -> [f64; 2] {
    ExactPcurveEvaluator::point(pcurves, &coedge.pcurve_evaluator_id, 0.0, &Control)
        .unwrap()
        .each_ref()
        .map(|coordinate| coordinate.with_native_coordinate(|value, _frame| value))
}

#[test]
fn incident_pcurves_share_one_coordinate_independently_of_partition_or_traversal() {
    let mut topology = topology(1.0e-7);
    let edge = topology.edges[1].clone();
    let coedge = topology.coedges[1].clone();
    let vertex = edge.start_vertex_id.as_ref().unwrap();
    let pcurves = Pcurves {
        representative: topology.coedges[0].pcurve_evaluator_id.clone(),
        offset: 7.0e-15,
        meters_per_uv: 1.0,
    };
    let original = endpoint(&pcurves, &coedge);
    assert_ne!(original, VERTEX_UV);
    for _ in 0..2 {
        let uv = context(&topology, &pcurves)
            .canonical_coordinate(&edge, &coedge, vertex, original)
            .unwrap();
        assert_eq!(uv, VERTEX_UV);
        topology.coedges.reverse();
    }
}

/// OCCT boolean results on B-spline faces: incident pcurves meet the vertex within its 1e-7 m
/// tolerance, which on a face of 1 mm per unit UV is a UV disagreement of 5e-5, far above any
/// fixed UV threshold. The endpoints join at the representative coordinate.
#[test]
fn pcurves_within_the_vertex_tolerance_join_whatever_their_uv_disagreement() {
    let topology = topology(1.0e-7);
    let edge = topology.edges[1].clone();
    let coedge = &topology.coedges[1];
    let vertex = edge.start_vertex_id.as_ref().unwrap();
    let pcurves = Pcurves {
        representative: topology.coedges[0].pcurve_evaluator_id.clone(),
        offset: 5.0e-5,
        meters_per_uv: 1.0e-3,
    };
    let original = endpoint(&pcurves, coedge);
    let uv = context(&topology, &pcurves)
        .canonical_coordinate(&edge, coedge, vertex, original)
        .unwrap();
    assert_eq!(uv, VERTEX_UV);
}

/// The admitted UV disagreement is the vertex tolerance mapped through the face's parametric
/// scale: the same UV offset is admitted on a face of 1 um per unit UV and rejected on a face of
/// 1 m per unit UV.
#[test]
fn the_admitted_uv_disagreement_follows_the_face_parametric_scale() {
    let topology = topology(1.0e-7);
    let edge = topology.edges[1].clone();
    let coedge = &topology.coedges[1];
    let vertex = edge.start_vertex_id.as_ref().unwrap();
    let fine = Pcurves {
        representative: topology.coedges[0].pcurve_evaluator_id.clone(),
        offset: 0.01,
        meters_per_uv: 1.0e-6,
    };
    let original = endpoint(&fine, coedge);
    assert_eq!(
        context(&topology, &fine)
            .canonical_coordinate(&edge, coedge, vertex, original)
            .unwrap(),
        VERTEX_UV
    );
    let coarse = Pcurves {
        meters_per_uv: 1.0,
        ..fine
    };
    let error = context(&topology, &coarse)
        .canonical_coordinate(&edge, coedge, vertex, original)
        .unwrap_err();
    assert_eq!(error.kind, SharedCurveErrorKind::GeometricMismatch);
    assert_eq!(error.field, "vertex pcurve coordinate");
}

#[test]
fn endpoint_authority_preserves_distinct_vertices() {
    let mut topology = topology(1.0e-7);
    let pcurves = Pcurves {
        representative: topology.coedges[0].pcurve_evaluator_id.clone(),
        offset: 1.0e-5,
        meters_per_uv: 1.0e-3,
    };
    let mut edge = topology.edges[1].clone();
    let original = endpoint(&pcurves, &topology.coedges[1]);
    let mut distinct = topology.vertices[0].clone();
    distinct.id.source_topology_id.push_str(":distinct");
    edge.start_vertex_id = Some(distinct.id.clone());
    edge.end_vertex_id = Some(distinct.id.clone());
    topology.edges[1] = edge.clone();
    topology.vertices.push(distinct.clone());
    topology
        .vertices
        .sort_by(|left, right| left.id.cmp(&right.id));
    let uv = context(&topology, &pcurves)
        .canonical_coordinate(&edge, &topology.coedges[1], &distinct.id, original)
        .unwrap();
    assert_eq!(uv, original);
}
