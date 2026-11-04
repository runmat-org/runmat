use super::*;
use crate::codec_control::CodecControl;
use runmat_meshing_tetrahedron::cdt::{
    DelaunayVolumeMeshCodecError, DelaunayVolumeMeshCodecErrorKind,
};

pub(super) fn map_record_error(
    original: Box<MeshingFailure>,
    control: &CodecControl<'_>,
) -> Box<MeshingFailure> {
    match control.take_failure() {
        Some(error) => failure(crate::geometry_failure::category(error.kind), &error.reason),
        None => original,
    }
}
pub(super) fn map_volume_codec_error(
    error: DelaunayVolumeMeshCodecError,
    control: &CodecControl<'_>,
) -> Box<MeshingFailure> {
    if let Some(original) = control.take_failure() {
        return failure(
            crate::geometry_failure::category(original.kind),
            &original.reason,
        );
    }
    let category = match error.kind {
        DelaunayVolumeMeshCodecErrorKind::Cancelled => MeshingFailureCategory::Cancelled,
        DelaunayVolumeMeshCodecErrorKind::ResourceLimit => {
            MeshingFailureCategory::ArtifactBudgetExceeded
        }
        DelaunayVolumeMeshCodecErrorKind::InvalidMesh
        | DelaunayVolumeMeshCodecErrorKind::InvalidEncoding => {
            MeshingFailureCategory::InvalidGeometry
        }
    };
    failure(category, &error.to_string())
}

pub(super) fn map_solver_error(error: DelaunaySolverTopologyError) -> Box<MeshingFailure> {
    if let DelaunaySolverTopologyErrorKind::Accounting(cause) = error.kind {
        let mut failure = failure(MeshingFailureCategory::NumericalAccounting, &error.reason);
        failure.accounting_error = Some(cause);
        failure.remediation =
            "renew the numerical work grant or raise the operation spending cap".into();
        return failure;
    }
    let category = match error.kind {
        DelaunaySolverTopologyErrorKind::Accounting(_) => unreachable!("handled above"),
        DelaunaySolverTopologyErrorKind::InvalidOptions => {
            MeshingFailureCategory::InternalInvariantViolation
        }
        DelaunaySolverTopologyErrorKind::InvalidGeometry => MeshingFailureCategory::InvalidGeometry,
        DelaunaySolverTopologyErrorKind::InvalidMesh => {
            MeshingFailureCategory::QualityTargetUnreachable
        }
        DelaunaySolverTopologyErrorKind::ResourceLimit => resource_category(&error.reason),
        DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded => {
            MeshingFailureCategory::MemoryBudgetExceeded
        }
        DelaunaySolverTopologyErrorKind::TimeBudgetExceeded => {
            MeshingFailureCategory::TimeBudgetExceeded
        }
        DelaunaySolverTopologyErrorKind::SearchWorkBudgetExceeded => {
            MeshingFailureCategory::SearchWorkBudgetExceeded
        }
        DelaunaySolverTopologyErrorKind::IterationBudgetExceeded => {
            MeshingFailureCategory::IterationBudgetExceeded
        }
        DelaunaySolverTopologyErrorKind::Cancelled => MeshingFailureCategory::Cancelled,
    };
    failure(category, &error.to_string())
}

pub(super) fn resource_category(reason: &str) -> MeshingFailureCategory {
    if reason.contains("IterationBudgetExceeded") || reason.contains("round limit") {
        MeshingFailureCategory::IterationBudgetExceeded
    } else if reason.contains("TimeBudgetExceeded") {
        MeshingFailureCategory::TimeBudgetExceeded
    } else if reason.contains("AllocationBudgetExceeded") {
        MeshingFailureCategory::MemoryBudgetExceeded
    } else if reason.contains("node inventory") {
        MeshingFailureCategory::NodeBudgetExceeded
    } else if reason.contains("boundary face") || reason.contains("boundary edge") {
        MeshingFailureCategory::ElementBudgetExceeded
    } else {
        MeshingFailureCategory::SearchWorkBudgetExceeded
    }
}

pub(super) fn invalid_inputs() -> Box<MeshingFailure> {
    failure(
        MeshingFailureCategory::InvalidGeometry,
        "solver projection requires one exact geometry, final surface, and final volume",
    )
}

pub(super) fn failure(category: MeshingFailureCategory, detail: &str) -> Box<MeshingFailure> {
    let detail = crate::diagnostic::bounded_diagnostic_text(detail, "solver projection failed");
    Box::new(MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category,
        stage: MeshingStageKind::OrderElevation,
        operation: MeshingStageKind::OrderElevation.operation(),
        entity_ids: Vec::new(),
        witnesses: Vec::new(),
        request_values: Vec::new(),
        achieved_values: vec![MeshingDiagnosticEntry {
            name: "solver_projection_failure".into(),
            value: MeshingDiagnosticValue::Text(detail),
            unit: None,
        }],
        accounting_error: None,
        remediation:
            "regenerate terminal inputs, increase the exhausted budget, or repair exact geometry"
                .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    };
    #[test]
    fn solver_accounting_failure_retains_original_cause() {
        let cause = runmat_simulation_usage::LedgerError::CountOverflow;
        let mapped = map_solver_error(DelaunaySolverTopologyError {
            kind: DelaunaySolverTopologyErrorKind::Accounting(cause),
            reason: "operation count overflow".into(),
        });
        assert_eq!(mapped.category, MeshingFailureCategory::NumericalAccounting);
        assert_eq!(mapped.accounting_error, Some(cause));
        mapped.validate().unwrap();
    }
    struct Control(Option<GeometryEvaluationErrorKind>);
    impl GeometryEvaluationControl for Control {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            match self.0 {
                Some(kind) => Err(GeometryEvaluationError::new(
                    kind,
                    "original control failure",
                )),
                None => Ok(()),
            }
        }
        fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            self.checkpoint()
        }
        fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            self.checkpoint()
        }
        fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            self.checkpoint()
        }
    }
    // witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Projection retains originating deadline, allocation and cancellation failure
    //   categories through its volume decode adapter, distinguishes malformed artifacts and preserves
    //   the order-elevation stage. This finite control fixture does not establish full native admission
    //   or study acceptance.
    #[test]
    fn volume_decode_preserves_original_budget_and_failure_categories() {
        for (kind, expected) in [
            (
                GeometryEvaluationErrorKind::TimeBudgetExceeded,
                MeshingFailureCategory::TimeBudgetExceeded,
            ),
            (
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                MeshingFailureCategory::MemoryBudgetExceeded,
            ),
            (
                GeometryEvaluationErrorKind::Cancelled,
                MeshingFailureCategory::Cancelled,
            ),
        ] {
            let source = Control(Some(kind));
            let codec = CodecControl::new(&source);
            assert!(MeshingCancellationSignal::is_cancelled(&codec));
            let mapped = map_volume_codec_error(
                DelaunayVolumeMeshCodecError {
                    kind: DelaunayVolumeMeshCodecErrorKind::Cancelled,
                    reason: "codec interrupted".into(),
                },
                &codec,
            );
            assert_eq!(mapped.category, expected);
            assert_eq!(mapped.stage, MeshingStageKind::OrderElevation);
        }
        let source = Control(None);
        let codec = CodecControl::new(&source);
        for (kind, expected) in [
            (
                DelaunayVolumeMeshCodecErrorKind::ResourceLimit,
                MeshingFailureCategory::ArtifactBudgetExceeded,
            ),
            (
                DelaunayVolumeMeshCodecErrorKind::InvalidMesh,
                MeshingFailureCategory::InvalidGeometry,
            ),
        ] {
            assert_eq!(
                map_volume_codec_error(
                    DelaunayVolumeMeshCodecError {
                        kind,
                        reason: "original artifact failure".into()
                    },
                    &codec
                )
                .category,
                expected
            );
        }
    }
}
