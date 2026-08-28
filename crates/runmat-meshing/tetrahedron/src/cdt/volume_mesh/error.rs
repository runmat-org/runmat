use super::{error, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshStage};
use crate::cdt::{
    DelaunayCarvingError, DelaunayCarvingErrorKind, DelaunayConstraintError,
    DelaunayConstraintErrorKind, DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind,
    DelaunayPointSetError, DelaunayPointSetErrorKind, DelaunaySegmentRecoveryError,
    DelaunaySegmentRecoveryErrorKind, DelaunayVolumeProvenanceError,
    DelaunayVolumeProvenanceErrorKind, DelaunayVolumeQualityError, DelaunayVolumeQualityErrorKind,
    DelaunayVolumeRefinementStepError, DelaunayVolumeRefinementStepErrorKind,
};

pub(super) fn constraint_error(failure: DelaunayConstraintError) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        DelaunayConstraintErrorKind::InvalidOptions => DelaunayVolumeMeshErrorKind::InvalidOptions,
        DelaunayConstraintErrorKind::InvalidGeometry
        | DelaunayConstraintErrorKind::InvalidBoundary
        | DelaunayConstraintErrorKind::InvalidIdentity
        | DelaunayConstraintErrorKind::IdentityCollision => {
            DelaunayVolumeMeshErrorKind::InvalidGeometry
        }
        DelaunayConstraintErrorKind::ResourceLimit => DelaunayVolumeMeshErrorKind::ResourceLimit,
        DelaunayConstraintErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
    };
    mapped(DelaunayVolumeMeshStage::Constraints, kind, failure)
}

pub(super) fn point_set_error(failure: DelaunayPointSetError) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        DelaunayPointSetErrorKind::InvalidOptions => DelaunayVolumeMeshErrorKind::InvalidOptions,
        DelaunayPointSetErrorKind::InvalidNode
        | DelaunayPointSetErrorKind::InsufficientDimension => {
            DelaunayVolumeMeshErrorKind::InvalidGeometry
        }
        DelaunayPointSetErrorKind::InvalidTopology => DelaunayVolumeMeshErrorKind::InvalidTopology,
        DelaunayPointSetErrorKind::ResourceLimit => DelaunayVolumeMeshErrorKind::ResourceLimit,
        DelaunayPointSetErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
    };
    mapped(DelaunayVolumeMeshStage::PointSet, kind, failure)
}

pub(super) fn segment_error(failure: DelaunaySegmentRecoveryError) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        DelaunaySegmentRecoveryErrorKind::InvalidOptions => {
            DelaunayVolumeMeshErrorKind::InvalidOptions
        }
        DelaunaySegmentRecoveryErrorKind::InvalidConstraints
        | DelaunaySegmentRecoveryErrorKind::IdentityCollision => {
            DelaunayVolumeMeshErrorKind::InvalidGeometry
        }
        DelaunaySegmentRecoveryErrorKind::InvalidTopology => {
            DelaunayVolumeMeshErrorKind::InvalidTopology
        }
        DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint => {
            DelaunayVolumeMeshErrorKind::UnsatisfiableConstraint
        }
        DelaunaySegmentRecoveryErrorKind::ResourceLimit => {
            DelaunayVolumeMeshErrorKind::ResourceLimit
        }
        DelaunaySegmentRecoveryErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
    };
    mapped(DelaunayVolumeMeshStage::SegmentRecovery, kind, failure)
}

pub(super) fn facet_error(failure: DelaunayFacetRecoveryError) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        DelaunayFacetRecoveryErrorKind::InvalidOptions => {
            DelaunayVolumeMeshErrorKind::InvalidOptions
        }
        DelaunayFacetRecoveryErrorKind::InvalidConstraints => {
            DelaunayVolumeMeshErrorKind::InvalidGeometry
        }
        DelaunayFacetRecoveryErrorKind::InvalidTopology => {
            DelaunayVolumeMeshErrorKind::InvalidTopology
        }
        DelaunayFacetRecoveryErrorKind::UnsatisfiableConstraint => {
            DelaunayVolumeMeshErrorKind::UnsatisfiableConstraint
        }
        DelaunayFacetRecoveryErrorKind::ResourceLimit => DelaunayVolumeMeshErrorKind::ResourceLimit,
        DelaunayFacetRecoveryErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
    };
    mapped(DelaunayVolumeMeshStage::FacetRecovery, kind, failure)
}

pub(super) fn carving_error(failure: DelaunayCarvingError) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        DelaunayCarvingErrorKind::InvalidOptions => DelaunayVolumeMeshErrorKind::InvalidOptions,
        DelaunayCarvingErrorKind::InvalidConstraints => {
            DelaunayVolumeMeshErrorKind::InvalidGeometry
        }
        DelaunayCarvingErrorKind::InvalidTopology => DelaunayVolumeMeshErrorKind::InvalidTopology,
        DelaunayCarvingErrorKind::AmbiguousClassification => {
            DelaunayVolumeMeshErrorKind::UnsatisfiableConstraint
        }
        DelaunayCarvingErrorKind::ResourceLimit => DelaunayVolumeMeshErrorKind::ResourceLimit,
        DelaunayCarvingErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
    };
    mapped(DelaunayVolumeMeshStage::Carving, kind, failure)
}

pub(super) fn provenance_error(failure: DelaunayVolumeProvenanceError) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        DelaunayVolumeProvenanceErrorKind::InvalidOptions => {
            DelaunayVolumeMeshErrorKind::InvalidOptions
        }
        DelaunayVolumeProvenanceErrorKind::InvalidTopology => {
            DelaunayVolumeMeshErrorKind::InvalidTopology
        }
        DelaunayVolumeProvenanceErrorKind::InvalidProvenance => {
            DelaunayVolumeMeshErrorKind::InvalidGeometry
        }
        DelaunayVolumeProvenanceErrorKind::ResourceLimit => {
            DelaunayVolumeMeshErrorKind::ResourceLimit
        }
        DelaunayVolumeProvenanceErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
    };
    mapped(DelaunayVolumeMeshStage::Provenance, kind, failure)
}

pub(super) fn quality_error(failure: DelaunayVolumeQualityError) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        DelaunayVolumeQualityErrorKind::InvalidOptions => {
            DelaunayVolumeMeshErrorKind::InvalidOptions
        }
        DelaunayVolumeQualityErrorKind::InvalidTopology => {
            DelaunayVolumeMeshErrorKind::InvalidTopology
        }
        DelaunayVolumeQualityErrorKind::InvalidMetric
        | DelaunayVolumeQualityErrorKind::InvalidMetricContext
        | DelaunayVolumeQualityErrorKind::NumericalFailure => {
            DelaunayVolumeMeshErrorKind::InvalidMetric
        }
        DelaunayVolumeQualityErrorKind::InvalidQuality => {
            DelaunayVolumeMeshErrorKind::InvalidQuality
        }
        DelaunayVolumeQualityErrorKind::ResourceLimit => DelaunayVolumeMeshErrorKind::ResourceLimit,
        DelaunayVolumeQualityErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
    };
    mapped(DelaunayVolumeMeshStage::Quality, kind, failure)
}

pub(super) fn refinement_error(
    failure: DelaunayVolumeRefinementStepError,
) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        DelaunayVolumeRefinementStepErrorKind::InvalidOptions => {
            DelaunayVolumeMeshErrorKind::InvalidOptions
        }
        DelaunayVolumeRefinementStepErrorKind::InvalidInput
        | DelaunayVolumeRefinementStepErrorKind::InvalidTopology
        | DelaunayVolumeRefinementStepErrorKind::InvalidProvenance => {
            DelaunayVolumeMeshErrorKind::InvalidTopology
        }
        DelaunayVolumeRefinementStepErrorKind::InvalidCandidate
        | DelaunayVolumeRefinementStepErrorKind::InvalidQuality => {
            DelaunayVolumeMeshErrorKind::InvalidQuality
        }
        DelaunayVolumeRefinementStepErrorKind::ResourceLimit => {
            DelaunayVolumeMeshErrorKind::ResourceLimit
        }
        DelaunayVolumeRefinementStepErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
    };
    mapped(DelaunayVolumeMeshStage::Refinement, kind, failure)
}

fn mapped(
    stage: DelaunayVolumeMeshStage,
    kind: DelaunayVolumeMeshErrorKind,
    failure: impl std::fmt::Display,
) -> DelaunayVolumeMeshError {
    error(stage, kind, failure.to_string())
}
