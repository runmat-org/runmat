use super::*;
use crate::{
    import::GeometryImportContext, import_exact_cad, ExactCadImportOptions, GeometryFormat,
};
use runmat_geometry_core::{
    GeometryEvaluationError, TrimClassifierId, TrimDomainLocation, UnitSystem,
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

    let mut corrupt = imported.clone();
    corrupt.representation[0] ^= 1;
    assert_eq!(
        OcctExactEvaluator::new(&corrupt).err().unwrap().kind,
        GeometryEvaluationErrorKind::InconsistentGeometry
    );
}
