use runmat_meshing_core::{
    MeshingDiagnosticEntry, MeshingDiagnosticValue, MeshingFailure, MeshingFailureCategory,
    MeshingOperation, MeshingStageKind, MESHING_FAILURE_SCHEMA_VERSION,
};
use runmat_meshing_tetrahedron::cdt::{
    DelaunayVolumeMeshCodecError, DelaunayVolumeMeshCodecErrorKind, DelaunayVolumeMeshError,
    DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshStage,
};

pub(super) fn map_volume_error(error: DelaunayVolumeMeshError) -> Box<MeshingFailure> {
    let category = match error.kind {
        DelaunayVolumeMeshErrorKind::InvalidOptions => {
            MeshingFailureCategory::InternalInvariantViolation
        }
        DelaunayVolumeMeshErrorKind::InvalidGeometry
        | DelaunayVolumeMeshErrorKind::InvalidTopology => MeshingFailureCategory::InvalidGeometry,
        DelaunayVolumeMeshErrorKind::UnsatisfiableConstraint => {
            MeshingFailureCategory::UnsatisfiableConstraints
        }
        DelaunayVolumeMeshErrorKind::InvalidMetric => MeshingFailureCategory::SizingConflict,
        DelaunayVolumeMeshErrorKind::InvalidQuality => {
            MeshingFailureCategory::QualityTargetUnreachable
        }
        DelaunayVolumeMeshErrorKind::Cancelled => MeshingFailureCategory::Cancelled,
        DelaunayVolumeMeshErrorKind::ResourceLimit => resource_category(error.stage),
    };
    volume_failure(
        category,
        "repair the exact boundary, increase its resource budget, or relax volume quality targets",
        &error.to_string(),
    )
}

pub(super) fn map_codec_error(error: DelaunayVolumeMeshCodecError) -> Box<MeshingFailure> {
    let category = match error.kind {
        DelaunayVolumeMeshCodecErrorKind::InvalidEncoding => {
            MeshingFailureCategory::ArtifactBudgetExceeded
        }
        DelaunayVolumeMeshCodecErrorKind::InvalidMesh => {
            MeshingFailureCategory::InternalInvariantViolation
        }
    };
    volume_failure(
        category,
        "regenerate the general CDT artifact from its admitted exact prerequisites",
        &error.to_string(),
    )
}

fn resource_category(stage: DelaunayVolumeMeshStage) -> MeshingFailureCategory {
    match stage {
        DelaunayVolumeMeshStage::PointSet | DelaunayVolumeMeshStage::Refinement => {
            MeshingFailureCategory::NodeBudgetExceeded
        }
        DelaunayVolumeMeshStage::SegmentRecovery | DelaunayVolumeMeshStage::FacetRecovery => {
            MeshingFailureCategory::SearchWorkBudgetExceeded
        }
        DelaunayVolumeMeshStage::Constraints
        | DelaunayVolumeMeshStage::Carving
        | DelaunayVolumeMeshStage::Provenance
        | DelaunayVolumeMeshStage::Quality => MeshingFailureCategory::ElementBudgetExceeded,
    }
}

pub(super) fn invalid_input(detail: &str) -> Box<MeshingFailure> {
    volume_failure(
        MeshingFailureCategory::InvalidGeometry,
        "regenerate exact geometry and the final exact surface from one current revision",
        detail,
    )
}

pub(super) fn volume_failure(
    category: MeshingFailureCategory,
    remediation: &str,
    detail: &str,
) -> Box<MeshingFailure> {
    let detail = crate::diagnostic::bounded_diagnostic_text(detail, "volume stage failure");
    Box::new(MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category,
        stage: MeshingStageKind::Tetrahedralization,
        operation: MeshingOperation::Tetrahedralize,
        entity_ids: Vec::new(),
        witnesses: Vec::new(),
        request_values: Vec::new(),
        achieved_values: vec![MeshingDiagnosticEntry {
            name: "volume_failure".into(),
            value: MeshingDiagnosticValue::Text(detail),
            unit: None,
        }],
        remediation: remediation.into(),
    })
}
