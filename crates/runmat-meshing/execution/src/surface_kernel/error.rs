use runmat_geometry_core::{GeometryEvaluationErrorKind, PersistentEntityId};
use runmat_meshing_core::{
    MeshingDiagnosticEntry, MeshingDiagnosticValue, MeshingFailure, MeshingFailureCategory,
    MeshingOperation, MeshingStageKind, MESHING_FAILURE_SCHEMA_VERSION,
};
use runmat_meshing_surface::{
    ExactFaceAcceptanceErrorKind, ExactFaceChartErrorKind, ExactFaceDelaunayErrorKind,
    ExactFaceGeometryErrorKind, ExactFaceJoinErrorKind, ExactFaceMetricErrorKind,
    ExactFacePartitionError, ExactFacePartitionErrorKind, ExactFaceRefinementErrorKind,
    ExactSurfaceBoundaryErrorKind, ExactSurfaceMeshErrorKind,
};

pub(super) fn map_surface_error(error: ExactFacePartitionError) -> Box<MeshingFailure> {
    let category = match error.kind {
        ExactFacePartitionErrorKind::Chart(ExactFaceChartErrorKind::GeometryEvaluation(kind)) => {
            geometry_category(kind)
        }
        ExactFacePartitionErrorKind::Chart(ExactFaceChartErrorKind::Delaunay(kind))
        | ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::Delaunay(kind)) => {
            delaunay_category(kind)
        }
        ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::Metric(kind))
        | ExactFacePartitionErrorKind::Acceptance(ExactFaceAcceptanceErrorKind::Metric(kind)) => {
            metric_category(kind)
        }
        ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::Geometry(kind))
        | ExactFacePartitionErrorKind::Acceptance(ExactFaceAcceptanceErrorKind::Geometry(kind)) => {
            face_geometry_category(kind)
        }
        ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::ResourceLimit)
        | ExactFacePartitionErrorKind::Acceptance(ExactFaceAcceptanceErrorKind::ResourceLimit)
        | ExactFacePartitionErrorKind::FaceJoin(ExactFaceJoinErrorKind::ResourceLimit)
        | ExactFacePartitionErrorKind::Batch(ExactSurfaceMeshErrorKind::ResourceLimit)
        | ExactFacePartitionErrorKind::Boundary(ExactSurfaceBoundaryErrorKind::ResourceLimit) => {
            MeshingFailureCategory::ElementBudgetExceeded
        }
        ExactFacePartitionErrorKind::Acceptance(
            ExactFaceAcceptanceErrorKind::UnsatisfiedQuality,
        ) => MeshingFailureCategory::QualityTargetUnreachable,
        ExactFacePartitionErrorKind::Refinement(ExactFaceRefinementErrorKind::InvalidQuality) => {
            MeshingFailureCategory::SizingConflict
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

fn geometry_category(kind: GeometryEvaluationErrorKind) -> MeshingFailureCategory {
    match kind {
        GeometryEvaluationErrorKind::Cancelled => MeshingFailureCategory::Cancelled,
        GeometryEvaluationErrorKind::TimeBudgetExceeded => {
            MeshingFailureCategory::TimeBudgetExceeded
        }
        GeometryEvaluationErrorKind::AllocationBudgetExceeded => {
            MeshingFailureCategory::MemoryBudgetExceeded
        }
        GeometryEvaluationErrorKind::SearchWorkBudgetExceeded
        | GeometryEvaluationErrorKind::BudgetExceeded => {
            MeshingFailureCategory::SearchWorkBudgetExceeded
        }
        GeometryEvaluationErrorKind::IterationBudgetExceeded => {
            MeshingFailureCategory::IterationBudgetExceeded
        }
        GeometryEvaluationErrorKind::UnknownEvaluator
        | GeometryEvaluationErrorKind::ParameterOutsideDomain
        | GeometryEvaluationErrorKind::InconsistentGeometry => {
            MeshingFailureCategory::InvalidGeometry
        }
        GeometryEvaluationErrorKind::ProjectionDidNotConverge
        | GeometryEvaluationErrorKind::ClassificationDidNotConverge
        | GeometryEvaluationErrorKind::KernelUnavailable
        | GeometryEvaluationErrorKind::KernelFailure
        | GeometryEvaluationErrorKind::InvalidResult => MeshingFailureCategory::NumericalFailure,
    }
}

pub(super) fn invalid_input(detail: &str) -> Box<MeshingFailure> {
    surface_failure(
        MeshingFailureCategory::InvalidGeometry,
        None,
        "regenerate the surface partition from one admitted exact geometry and shared curve",
        detail,
    )
}

pub(super) fn surface_failure(
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
        remediation: remediation.into(),
    })
}
