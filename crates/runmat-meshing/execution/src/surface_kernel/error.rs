use crate::geometry_failure::category as geometry_category;
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    MeshingDiagnosticEntry, MeshingDiagnosticValue, MeshingFailure, MeshingFailureCategory,
    MeshingOperation, MeshingStageKind, MESHING_FAILURE_SCHEMA_VERSION,
};
use runmat_meshing_surface::{
    ExactFaceAcceptanceErrorKind, ExactFaceChartErrorKind, ExactFaceDelaunayErrorKind,
    ExactFaceGeometryErrorKind, ExactFaceJoinErrorKind, ExactFaceMetricErrorKind,
    ExactFacePartitionError, ExactFacePartitionErrorKind, ExactFaceRefinementErrorKind,
    ExactSurfaceBoundaryErrorKind, ExactSurfaceConvergenceError, ExactSurfaceConvergenceErrorKind,
    ExactSurfaceMeshErrorKind,
};

pub(crate) fn map_convergence_error(error: ExactSurfaceConvergenceError) -> Box<MeshingFailure> {
    let category = match error.kind {
        ExactSurfaceConvergenceErrorKind::InvalidResultSet => {
            MeshingFailureCategory::InvalidGeometry
        }
        ExactSurfaceConvergenceErrorKind::Curve(kind) => {
            crate::curve_kernel::shared_curve_failure_category(kind)
        }
        ExactSurfaceConvergenceErrorKind::Surface(ExactSurfaceMeshErrorKind::Cancelled) => {
            MeshingFailureCategory::Cancelled
        }
        ExactSurfaceConvergenceErrorKind::Surface(ExactSurfaceMeshErrorKind::ResourceLimit) => {
            MeshingFailureCategory::ElementBudgetExceeded
        }
        ExactSurfaceConvergenceErrorKind::Surface(_) => MeshingFailureCategory::InvalidGeometry,
    };
    surface_failure(
        category,
        None,
        "regenerate the complete surface pass from one current curve and face result set",
        &error.to_string(),
    )
}

pub(crate) fn map_surface_error(error: ExactFacePartitionError) -> Box<MeshingFailure> {
    let category = match error.kind {
        ExactFacePartitionErrorKind::Reuse => MeshingFailureCategory::InternalInvariantViolation,
        ExactFacePartitionErrorKind::Coordinates(kind)
        | ExactFacePartitionErrorKind::Chart(ExactFaceChartErrorKind::GeometryEvaluation(kind)) => {
            geometry_category(kind)
        }
        ExactFacePartitionErrorKind::Chart(ExactFaceChartErrorKind::Delaunay(kind))
        | ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::Delaunay(kind)) => {
            delaunay_category(kind)
        }
        ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::Metric(kind))
        | ExactFacePartitionErrorKind::Acceptance(ExactFaceAcceptanceErrorKind::Metric(kind))
        | ExactFacePartitionErrorKind::FaceJoin(ExactFaceJoinErrorKind::Acceptance(
            ExactFaceAcceptanceErrorKind::Metric(kind),
        )) => metric_category(kind),
        ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::Geometry(kind))
        | ExactFacePartitionErrorKind::Acceptance(ExactFaceAcceptanceErrorKind::Geometry(kind))
        | ExactFacePartitionErrorKind::FaceJoin(ExactFaceJoinErrorKind::Acceptance(
            ExactFaceAcceptanceErrorKind::Geometry(kind),
        )) => face_geometry_category(kind),
        ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::ResourceLimit)
        | ExactFacePartitionErrorKind::Acceptance(ExactFaceAcceptanceErrorKind::ResourceLimit)
        | ExactFacePartitionErrorKind::FaceJoin(ExactFaceJoinErrorKind::ResourceLimit)
        | ExactFacePartitionErrorKind::FaceJoin(ExactFaceJoinErrorKind::Acceptance(
            ExactFaceAcceptanceErrorKind::ResourceLimit,
        ))
        | ExactFacePartitionErrorKind::Batch(ExactSurfaceMeshErrorKind::ResourceLimit)
        | ExactFacePartitionErrorKind::Boundary(ExactSurfaceBoundaryErrorKind::ResourceLimit) => {
            MeshingFailureCategory::ElementBudgetExceeded
        }
        ExactFacePartitionErrorKind::Acceptance(
            ExactFaceAcceptanceErrorKind::UnsatisfiedQuality,
        )
        | ExactFacePartitionErrorKind::FaceJoin(ExactFaceJoinErrorKind::Acceptance(
            ExactFaceAcceptanceErrorKind::UnsatisfiedQuality,
        )) => MeshingFailureCategory::QualityTargetUnreachable,
        ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::InvalidQuality) => {
            MeshingFailureCategory::SizingConflict
        }
        ExactFacePartitionErrorKind::Refinement(
            ExactFaceRefinementErrorKind::UnsatisfiedQuality,
        ) => MeshingFailureCategory::QualityTargetUnreachable,
        ExactFacePartitionErrorKind::Boundary(ExactSurfaceBoundaryErrorKind::Cancelled)
        | ExactFacePartitionErrorKind::FaceJoin(ExactFaceJoinErrorKind::Cancelled)
        | ExactFacePartitionErrorKind::Batch(ExactSurfaceMeshErrorKind::Cancelled) => {
            MeshingFailureCategory::Cancelled
        }
        ExactFacePartitionErrorKind::InvalidPartition
        | ExactFacePartitionErrorKind::Boundary(_)
        | ExactFacePartitionErrorKind::Chart(_)
        | ExactFacePartitionErrorKind::Refinement(_)
        | ExactFacePartitionErrorKind::Acceptance(_)
        | ExactFacePartitionErrorKind::FaceJoin(_)
        | ExactFacePartitionErrorKind::Batch(_) => MeshingFailureCategory::InvalidGeometry,
    };
    surface_failure(
        category,
        None,
        "repair the exact face or relax its resolved surface constraints",
        &error.to_string(),
    )
}

fn metric_category(kind: ExactFaceMetricErrorKind) -> MeshingFailureCategory {
    match kind {
        ExactFaceMetricErrorKind::GeometryEvaluation(kind) => geometry_category(kind),
        ExactFaceMetricErrorKind::InvalidRequest => MeshingFailureCategory::SizingConflict,
        ExactFaceMetricErrorKind::UnknownFace | ExactFaceMetricErrorKind::InvalidEvaluation => {
            MeshingFailureCategory::InvalidGeometry
        }
    }
}

fn face_geometry_category(kind: ExactFaceGeometryErrorKind) -> MeshingFailureCategory {
    match kind {
        ExactFaceGeometryErrorKind::Metric(kind) => metric_category(kind),
        ExactFaceGeometryErrorKind::InvalidInput
        | ExactFaceGeometryErrorKind::InvalidEvaluation => MeshingFailureCategory::InvalidGeometry,
    }
}

fn delaunay_category(kind: ExactFaceDelaunayErrorKind) -> MeshingFailureCategory {
    match kind {
        ExactFaceDelaunayErrorKind::GeometryEvaluation(kind) => geometry_category(kind),
        ExactFaceDelaunayErrorKind::Cancelled => MeshingFailureCategory::Cancelled,
        ExactFaceDelaunayErrorKind::ElementLimit => MeshingFailureCategory::ElementBudgetExceeded,
        ExactFaceDelaunayErrorKind::SearchWorkLimit => {
            MeshingFailureCategory::SearchWorkBudgetExceeded
        }
        ExactFaceDelaunayErrorKind::IterationLimit => {
            MeshingFailureCategory::IterationBudgetExceeded
        }
        ExactFaceDelaunayErrorKind::UnsatisfiedConstraint => {
            MeshingFailureCategory::UnsatisfiableConstraints
        }
        ExactFaceDelaunayErrorKind::InvalidPslg
        | ExactFaceDelaunayErrorKind::InvalidOptions
        | ExactFaceDelaunayErrorKind::InvalidTopology => MeshingFailureCategory::InvalidGeometry,
    }
}

pub(crate) fn invalid_input(detail: &str) -> Box<MeshingFailure> {
    surface_failure(
        MeshingFailureCategory::InvalidGeometry,
        None,
        "regenerate the surface partition from one admitted exact geometry and shared curve",
        detail,
    )
}

pub(crate) fn surface_failure(
    category: MeshingFailureCategory,
    entity_id: Option<PersistentEntityId>,
    remediation: &str,
    detail: &str,
) -> Box<MeshingFailure> {
    let detail = crate::diagnostic::bounded_diagnostic_text(detail, "surface stage failure");
    Box::new(MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category,
        stage: MeshingStageKind::SurfaceMesh,
        operation: MeshingOperation::TriangulateSurface,
        entity_ids: entity_id.into_iter().collect(),
        witnesses: Vec::new(),
        request_values: Vec::new(),
        achieved_values: vec![MeshingDiagnosticEntry {
            name: "surface_failure".into(),
            value: MeshingDiagnosticValue::Text(detail),
            unit: None,
        }],
        accounting_error: None,
        remediation: remediation.into(),
    })
}

#[cfg(test)]
mod planarity_control_tests {
    use super::*;
    // witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Joined acceptance preserves cancellation, time, memory, search, iteration and
    //   numerical-result categories through both nested metric paths, retaining the original diagnostic
    //   detail. This category transport fixture does not establish successful convergence, budget
    //   sufficiency or whole-study execution.
    #[test]
    fn joined_acceptance_preserves_nested_geometry_control_causes() {
        use runmat_geometry_core::GeometryEvaluationErrorKind as Kind;
        for cause in [
            Kind::Cancelled,
            Kind::TimeBudgetExceeded,
            Kind::AllocationBudgetExceeded,
            Kind::SearchWorkBudgetExceeded,
            Kind::IterationBudgetExceeded,
            Kind::InvalidResult,
        ] {
            for acceptance in [
                ExactFaceAcceptanceErrorKind::Metric(ExactFaceMetricErrorKind::GeometryEvaluation(
                    cause,
                )),
                ExactFaceAcceptanceErrorKind::Geometry(ExactFaceGeometryErrorKind::Metric(
                    ExactFaceMetricErrorKind::GeometryEvaluation(cause),
                )),
            ] {
                let failure = map_surface_error(ExactFacePartitionError {
                    kind: ExactFacePartitionErrorKind::FaceJoin(
                        ExactFaceJoinErrorKind::Acceptance(acceptance),
                    ),
                    reason: "original joined acceptance cause".into(),
                });
                assert_eq!(failure.category, geometry_category(cause));
                assert!(failure.achieved_values.iter().any(|entry|matches!(&entry.value,MeshingDiagnosticValue::Text(text) if text.contains("original joined acceptance cause"))));
            }
        }
    }
    // witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Surface-stage failure construction preserves memory denial and cancellation
    //   categories, operation and original diagnostic detail from controlled planarity validation.
    #[test]
    fn surface_planarity_failures_preserve_resource_category_and_detail() {
        use runmat_geometry_core::GeometryEvaluationErrorKind;
        for (kind, expected) in [
            (
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                MeshingFailureCategory::MemoryBudgetExceeded,
            ),
            (
                GeometryEvaluationErrorKind::Cancelled,
                MeshingFailureCategory::Cancelled,
            ),
        ] {
            let failure = map_surface_error(ExactFacePartitionError {
                kind: ExactFacePartitionErrorKind::Refinement(
                    ExactFaceRefinementErrorKind::Delaunay(
                        ExactFaceDelaunayErrorKind::GeometryEvaluation(kind),
                    ),
                ),
                reason: "original planarity resource failure".into(),
            });
            assert_eq!(failure.category, expected);
            assert_eq!(failure.stage, MeshingStageKind::SurfaceMesh);
            assert_eq!(failure.operation, MeshingOperation::TriangulateSurface);
            assert!(failure.achieved_values.iter().any(|entry|matches!(&entry.value,MeshingDiagnosticValue::Text(text) if text.contains("original planarity resource failure"))));
        }
    }
}
