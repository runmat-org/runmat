use super::*;
use crate::{
    import::GeometryImportContext, import_exact_cad, ExactCadImportOptions, GeometryFormat,
};
use runmat_geometry_core::{
    GeometryEvaluationError, SurfaceEvaluatorId, TrimClassifierId, TrimDomainLocation, UnitSystem,
};
use sha2::Digest as _;

const BOX: &[u8] = include_bytes!("../../../tests/fixtures/box.brep");

struct Unlimited;

impl GeometryEvaluationControl for Unlimited {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

struct Cancelled;

impl GeometryEvaluationControl for Cancelled {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::Cancelled,
            "test cancellation",
        ))
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        unreachable!()
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        unreachable!()
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        unreachable!()
    }
}

#[test]
fn imported_curve_queries_are_exact_scaled_and_digest_bound() {
    let imported = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    let raw_digest: [u8; 32] = sha2::Sha256::digest(&imported.representation).into();
    assert_eq!(imported.representation_digest(), raw_digest);
    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
    evaluator
        .validate_incidence_consistency(&imported.topology, 1.0e-9, &Unlimited)
        .unwrap();
    assert_eq!(
        evaluator
            .validate_incidence_consistency(&imported.topology, 1.0e-9, &Cancelled)
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::Cancelled
    );
    let mut inconsistent_topology = imported.topology.clone();
    inconsistent_topology.vertices[0].point_m[0] += 1.0;
    assert_eq!(
        evaluator
            .validate_incidence_consistency(&inconsistent_topology, 1.0e-9, &Unlimited)
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::InconsistentGeometry
    );
    let id = &imported.topology.edges[0].curve_evaluator_id;
    let range = evaluator.parameter_range(id).unwrap();
    let start = evaluator.point(id, range.start, &Unlimited).unwrap();
    let end = evaluator.point(id, range.end, &Unlimited).unwrap();
    let expected_length = norm([end[0] - start[0], end[1] - start[1], end[2] - start[2]]);
    let length = evaluator
        .arc_length_m(id, range, 1.0e-12, &Unlimited)
        .unwrap();
    assert!((length - expected_length).abs() < 1.0e-12);

    let parameter = (range.start + range.end) * 0.5;
    let point = evaluator.point(id, parameter, &Unlimited).unwrap();
    let tangent = evaluator.unit_tangent(id, parameter, &Unlimited).unwrap();
    assert!((norm(tangent) - 1.0).abs() < 1.0e-12);
    assert_eq!(
        evaluator
            .curvature_1_per_m(id, parameter, &Unlimited)
            .unwrap(),
        0.0
    );
    let projection = evaluator
        .inverse_project(id, point, 1.0e-12, &Unlimited)
        .unwrap();
    assert!((projection.parameter - parameter).abs() < 1.0e-12);
    assert!(projection.distance_m < 1.0e-12);
    assert_eq!(
        evaluator
            .point(id, range.end + 1.0, &Unlimited)
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    );
    assert_eq!(
        evaluator.point(id, parameter, &Cancelled).unwrap_err().kind,
        GeometryEvaluationErrorKind::Cancelled
    );
    assert_eq!(
        evaluator
            .parameter_range(&CurveEvaluatorId::new("curve:unknown").unwrap())
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::UnknownEvaluator
    );

    let pcurve_id = &imported.topology.coedges[0].pcurve_evaluator_id;
    let pcurve_range =
        runmat_geometry_core::ExactPcurveEvaluator::parameter_range(&evaluator, pcurve_id).unwrap();
    let pcurve_parameter = (pcurve_range.start + pcurve_range.end) * 0.5;
    let pcurve = runmat_geometry_core::ExactPcurveEvaluator::derivatives(
        &evaluator,
        pcurve_id,
        pcurve_parameter,
        &Unlimited,
    )
    .unwrap();
    assert!(pcurve
        .point_uv
        .into_iter()
        .chain(pcurve.first_uv)
        .chain(pcurve.second_uv)
        .all(f64::is_finite));
    for coedge in &imported.topology.coedges {
        let range = runmat_geometry_core::ExactPcurveEvaluator::parameter_range(
            &evaluator,
            &coedge.pcurve_evaluator_id,
        )
        .unwrap();
        runmat_geometry_core::ExactPcurveEvaluator::derivatives(
            &evaluator,
            &coedge.pcurve_evaluator_id,
            (range.start + range.end) * 0.5,
            &Unlimited,
        )
        .unwrap();
    }
    assert_eq!(
        runmat_geometry_core::ExactPcurveEvaluator::point(
            &evaluator,
            pcurve_id,
            pcurve_range.end + 1.0,
            &Unlimited,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    );
    assert_eq!(
        runmat_geometry_core::ExactPcurveEvaluator::point(
            &evaluator,
            pcurve_id,
            pcurve_parameter,
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::Cancelled
    );

    let coedge = &imported.topology.coedges[0];
    let face = imported
        .topology
        .faces
        .iter()
        .find(|face| face.id == coedge.face_id)
        .unwrap();
    assert_eq!(
        runmat_geometry_core::ExactTrimClassifier::classify(
            &evaluator,
            &face.trim_classifier_id,
            pcurve.point_uv,
            1.0e-9,
            &Unlimited,
        )
        .unwrap(),
        TrimDomainLocation::OnBoundary
    );
    assert_eq!(
        runmat_geometry_core::ExactTrimClassifier::classify(
            &evaluator,
            &face.trim_classifier_id,
            [pcurve.point_uv[0] + 1.0e6, pcurve.point_uv[1] + 1.0e6],
            1.0e-9,
            &Unlimited,
        )
        .unwrap(),
        TrimDomainLocation::Outside
    );
    assert_eq!(
        runmat_geometry_core::ExactTrimClassifier::classify(
            &evaluator,
            &face.trim_classifier_id,
            pcurve.point_uv,
            1.0e-9,
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::Cancelled
    );
    assert_eq!(
        runmat_geometry_core::ExactTrimClassifier::classify(
            &evaluator,
            &TrimClassifierId::new("trim:unknown").unwrap(),
            pcurve.point_uv,
            1.0e-9,
            &Unlimited,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::UnknownEvaluator
    );

    let surface_id = &face.surface_evaluator_id;
    let surface_bounds =
        runmat_geometry_core::ExactSurfaceEvaluator::parameter_bounds(&evaluator, surface_id)
            .unwrap();
    assert_eq!(
        runmat_geometry_core::ExactSurfaceEvaluator::periodicity(&evaluator, surface_id).unwrap(),
        [None, None]
    );
    let surface_uv = [
        (surface_bounds[0].start + surface_bounds[0].end) * 0.5,
        (surface_bounds[1].start + surface_bounds[1].end) * 0.5,
    ];
    let surface_derivatives = runmat_geometry_core::ExactSurfaceEvaluator::derivatives(
        &evaluator, surface_id, surface_uv, &Unlimited,
    )
    .unwrap();
    let normal = runmat_geometry_core::ExactSurfaceEvaluator::unit_normal(
        &evaluator, surface_id, surface_uv, &Unlimited,
    )
    .unwrap();
    assert!((norm(normal) - 1.0).abs() < 1.0e-12);
    let curvature = runmat_geometry_core::ExactSurfaceEvaluator::principal_curvature(
        &evaluator, surface_id, surface_uv, &Unlimited,
    )
    .unwrap();
    assert!(curvature.minimum_1_per_m.abs() < 1.0e-12);
    assert!(curvature.maximum_1_per_m.abs() < 1.0e-12);
    let displaced =
        std::array::from_fn(|axis| surface_derivatives.point_m[axis] + normal[axis] * 0.25);
    let surface_projection = runmat_geometry_core::ExactSurfaceEvaluator::closest_point(
        &evaluator, surface_id, displaced, 1.0e-12, &Unlimited,
    )
    .unwrap();
    assert!((surface_projection.distance_m - 0.25).abs() < 1.0e-12);
    assert!(
        norm(std::array::from_fn(|axis| {
            surface_projection.point_m[axis] - surface_derivatives.point_m[axis]
        })) < 1.0e-12
    );
    let u_boundary_uv = [surface_bounds[0].end, surface_uv[1]];
    let u_boundary = runmat_geometry_core::ExactSurfaceEvaluator::point(
        &evaluator,
        surface_id,
        u_boundary_uv,
        &Unlimited,
    )
    .unwrap();
    let u_direction = normalized(surface_derivatives.du_m).unwrap();
    let beyond_u = std::array::from_fn(|axis| u_boundary[axis] + u_direction[axis] * 0.25);
    let boundary_projection = runmat_geometry_core::ExactSurfaceEvaluator::closest_point(
        &evaluator, surface_id, beyond_u, 1.0e-12, &Unlimited,
    )
    .unwrap();
    assert!((boundary_projection.uv[0] - surface_bounds[0].end).abs() < 1.0e-12);
    assert!((boundary_projection.distance_m - 0.25).abs() < 1.0e-12);

    let edge = imported
        .topology
        .edges
        .iter()
        .find(|edge| edge.id == coedge.edge_id)
        .unwrap();
    let boundary_3d = runmat_geometry_core::ExactSurfaceEvaluator::point(
        &evaluator,
        surface_id,
        pcurve.point_uv,
        &Unlimited,
    )
    .unwrap();
    let edge_3d = evaluator
        .point(&edge.curve_evaluator_id, pcurve_parameter, &Unlimited)
        .unwrap();
    assert!(norm(std::array::from_fn(|axis| boundary_3d[axis] - edge_3d[axis])) < 1.0e-12);
    assert_eq!(
        runmat_geometry_core::ExactSurfaceEvaluator::point(
            &evaluator,
            surface_id,
            [surface_bounds[0].end + 1.0, surface_uv[1]],
            &Unlimited,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    );
    assert_eq!(
        runmat_geometry_core::ExactSurfaceEvaluator::point(
            &evaluator, surface_id, surface_uv, &Cancelled,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::Cancelled
    );
    assert_eq!(
        runmat_geometry_core::ExactSurfaceEvaluator::parameter_bounds(
            &evaluator,
            &SurfaceEvaluatorId::new("surface:unknown").unwrap(),
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::UnknownEvaluator
    );

    let millimeter_options = ExactCadImportOptions {
        source_units: UnitSystem::Millimeter,
        ..ExactCadImportOptions::default()
    };
    let millimeter_import = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &millimeter_options,
        &GeometryImportContext::new(),
    )
    .unwrap();
    let millimeter_evaluator = OcctExactEvaluator::new(&millimeter_import).unwrap();
    let millimeter_length = millimeter_evaluator
        .arc_length_m(
            &millimeter_import.topology.edges[0].curve_evaluator_id,
            range,
            1.0e-15,
            &Unlimited,
        )
        .unwrap();
    assert!((millimeter_length - length * 0.001).abs() < 1.0e-15);
    let millimeter_surface_id = &millimeter_import
        .topology
        .faces
        .iter()
        .find(|candidate| candidate.id == face.id)
        .unwrap()
        .surface_evaluator_id;
    let millimeter_surface_point = runmat_geometry_core::ExactSurfaceEvaluator::point(
        &millimeter_evaluator,
        millimeter_surface_id,
        surface_uv,
        &Unlimited,
    )
    .unwrap();
    assert!(
        norm(std::array::from_fn(|axis| {
            millimeter_surface_point[axis] - surface_derivatives.point_m[axis] * 0.001
        })) < 1.0e-15
    );
    let millimeter_displaced =
        std::array::from_fn(|axis| millimeter_surface_point[axis] + normal[axis] * 0.00025);
    let millimeter_projection = runmat_geometry_core::ExactSurfaceEvaluator::closest_point(
        &millimeter_evaluator,
        millimeter_surface_id,
        millimeter_displaced,
        1.0e-15,
        &Unlimited,
    )
    .unwrap();
    assert!((millimeter_projection.distance_m - 0.00025).abs() < 1.0e-15);

    let mut corrupt = imported.clone();
    corrupt.representation[0] ^= 1;
    assert_eq!(
        OcctExactEvaluator::new(&corrupt).err().unwrap().kind,
        GeometryEvaluationErrorKind::InconsistentGeometry
    );
}
