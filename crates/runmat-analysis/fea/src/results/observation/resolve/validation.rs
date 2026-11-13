//! Typed observation/reduction compatibility checks, independent of target materialization.
use super::*;

pub(super) fn validate_supported_spec(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), ObservationError> {
    validate_output_numeric_contract(spec)?;
    let native_location =
        spec.source_location == spec.output_location && spec.recovery == RecoveryPolicy::Native;
    let reduced_location = spec.output_location == FieldLocation::Global
        && spec.recovery == RecoveryPolicy::Native
        && spec
            .reduction
            .as_ref()
            .is_some_and(|reduction| reduction.spatial != SpatialReduction::None);
    let solid_point_interpolation = spec.source_location == FieldLocation::Node
        && spec.output_location == FieldLocation::Point
        && spec.recovery == RecoveryPolicy::ShapeFunctionInterpolation
        && matches!(
            spec.target,
            ObservationTarget::Point { .. } | ObservationTarget::Path { .. }
        );
    let materialized_location_transfer = spec.source_location == FieldLocation::Element
        && spec.output_location == FieldLocation::Node
        && spec.recovery == RecoveryPolicy::VolumeWeightedAverage;
    if !native_location
        && !reduced_location
        && !solid_point_interpolation
        && !materialized_location_transfer
    {
        return Err(ObservationError::Unsupported(
            "field recovery and location transfer must be completed before observation evaluation",
        ));
    }
    if !matches!(
        spec.source_location,
        FieldLocation::Node
            | FieldLocation::Element
            | FieldLocation::Cell
            | FieldLocation::Face
            | FieldLocation::IntegrationPoint
            | FieldLocation::Connector
            | FieldLocation::Constraint
            | FieldLocation::Edge
    ) {
        return Err(ObservationError::Unsupported(
            "this evaluator accepts only nodal, element, integration-point, connector, and constraint-group fields",
        ));
    }
    match request {
        ObservationRequest::History(_) if spec.sampling == SamplingPolicy::Final => {
            return Err(ObservationError::Invalid(
                "history requests require a sampled series rather than one final state".into(),
            ));
        }
        ObservationRequest::Field(_)
        | ObservationRequest::Probe(_)
        | ObservationRequest::Reduction(_)
            if spec.sampling != SamplingPolicy::Final =>
        {
            return Err(ObservationError::Unsupported(
                "sampled field, probe, and reduction series require a history request",
            ));
        }
        _ => {}
    }
    if !matches!(
        spec.storage.compression,
        CompressionPolicy::None | CompressionPolicy::Lossless
    ) {
        return Err(ObservationError::Unsupported(
            "error-bounded observation compression is not implemented",
        ));
    }
    match &spec.storage.retention {
        RetentionPolicy::LatestSamples { count } if *count < 1 => {
            return Err(ObservationError::Invalid(
                "retention count must be nonzero".into(),
            ));
        }
        RetentionPolicy::ExplicitSamples { maximum_count } if *maximum_count < 1 => {
            return Err(ObservationError::Invalid(
                "retention count must be nonzero".into(),
            ));
        }
        _ => {}
    }
    if let Some(reduction) = &spec.reduction {
        if matches!(spec.sampling, SamplingPolicy::Event(_))
            && reduction.series != SeriesReduction::None
        {
            return Err(ObservationError::Invalid(
                "event histories may be empty, so they cannot request a series reduction".into(),
            ));
        }
        if reduction.series != SeriesReduction::None
            && !matches!(request, ObservationRequest::History(_))
        {
            return Err(ObservationError::Invalid(
                "series reductions require a history request".into(),
            ));
        }
        if !matches!(
            reduction.spatial,
            SpatialReduction::None
                | SpatialReduction::Integral
                | SpatialReduction::Average
                | SpatialReduction::RootMeanSquare
                | SpatialReduction::EuclideanNorm
                | SpatialReduction::Minimum
                | SpatialReduction::Maximum
                | SpatialReduction::Resultant
                | SpatialReduction::Moment { .. }
                | SpatialReduction::NormalFlux
                | SpatialReduction::MeanDifference
                | SpatialReduction::TransmissionLoss
        ) {
            return Err(ObservationError::Unsupported(
                "unsupported spatial reduction",
            ));
        }
    }
    match request {
        ObservationRequest::Field(_) if spec.reduction.is_some() => Err(ObservationError::Invalid(
            "field requests cannot apply reductions".into(),
        )),
        ObservationRequest::Probe(_) if spec.reduction.is_some() => Err(ObservationError::Invalid(
            "probe requests cannot apply reductions".into(),
        )),
        ObservationRequest::Reduction(_)
            if spec
                .reduction
                .as_ref()
                .is_none_or(|reduction| reduction.spatial == SpatialReduction::None) =>
        {
            Err(ObservationError::Invalid(
                "reduction requests require a spatial reduction".into(),
            ))
        }
        ObservationRequest::History(ObservationSpec {
            reduction: Some(_), ..
        }) if spec.reduction.as_ref().is_none_or(|reduction| {
            reduction.spatial == SpatialReduction::None && reduction.series == SeriesReduction::None
        }) =>
        {
            Err(ObservationError::Invalid(
                "history reduction requests require a spatial or series reduction".into(),
            ))
        }
        _ => Ok(()),
    }
}

pub(super) fn validate_resolved_spatial_reduction(
    spec: &ObservationSpec,
    source: &RecoveredFieldDefinition,
    target: &ResolvedObservationTarget,
) -> Result<(), ObservationError> {
    let Some(operation) = spec.reduction.as_ref().map(|reduction| reduction.spatial) else {
        return Ok(());
    };
    if matches!(
        operation,
        SpatialReduction::Integral
            | SpatialReduction::Average
            | SpatialReduction::RootMeanSquare
            | SpatialReduction::EuclideanNorm
            | SpatialReduction::Resultant
            | SpatialReduction::Moment { .. }
            | SpatialReduction::NormalFlux
            | SpatialReduction::MeanDifference
            | SpatialReduction::TransmissionLoss
    ) && spec.output_location != FieldLocation::Global
    {
        return Err(ObservationError::Invalid(
            "integrated spatial reductions require global output location".into(),
        ));
    }
    if operation.is_paired() {
        if source.value_shape() != &ObservationValueShape::Scalar
            || source.numeric_representation() != RecoveredFieldNumericRepresentation::Real
            || spec.component != ObservationComponent::All
            || !matches!(target, ResolvedObservationTarget::BoundaryPair { .. })
        {
            return Err(ObservationError::Unsupported(
                "a paired reduction reads a real scalar field over two boundary face regions",
            ));
        }
        if operation == SpatialReduction::TransmissionLoss
            && spec.quantity != ObservationQuantity::AcousticPortIntensity
        {
            return Err(ObservationError::Unsupported(
                "transmission loss reads the acoustic port intensity",
            ));
        }
        return Ok(());
    }
    if operation == SpatialReduction::NormalFlux {
        if source.value_shape() != &ObservationValueShape::Vector3
            || source.numeric_representation() != RecoveredFieldNumericRepresentation::Real
            || spec.component != ObservationComponent::All
            || !matches!(target, ResolvedObservationTarget::Boundary { .. })
        {
            return Err(ObservationError::Unsupported(
                "a normal flux reduces a real vector field through boundary faces",
            ));
        }
        return Ok(());
    }
    if !matches!(
        operation,
        SpatialReduction::Resultant | SpatialReduction::Moment { .. }
    ) {
        return Ok(());
    }
    if spec.component != ObservationComponent::All {
        return Err(ObservationError::Invalid(
            "resultant and moment reductions require all source components".into(),
        ));
    }
    let representation_is_valid = matches!(
        (source.numeric_representation(), spec.complex_representation),
        (
            RecoveredFieldNumericRepresentation::Real,
            ComplexRepresentation::Real
        ) | (
            RecoveredFieldNumericRepresentation::ComplexInterleaved,
            ComplexRepresentation::Real
                | ComplexRepresentation::Imaginary
                | ComplexRepresentation::Complex
        )
    );
    if !representation_is_valid {
        return Err(ObservationError::Unsupported(
            "resultants require real values, one Cartesian phasor component, or the complete complex phasor",
        ));
    }
    let source_shape = source.value_shape();
    match (source_shape, target) {
        (ObservationValueShape::Vector3, ResolvedObservationTarget::Nodes { .. })
            if matches!(
                spec.quantity,
                ObservationQuantity::ReactionForce | ObservationQuantity::ReactionMoment
            ) => {}
        (ObservationValueShape::Vector3, ResolvedObservationTarget::ConstraintGroups { .. })
            if matches!(
                spec.quantity,
                ObservationQuantity::ReactionForce | ObservationQuantity::ReactionMoment
            ) => {}
        (ObservationValueShape::Vector3, ResolvedObservationTarget::Volume { .. })
        | (ObservationValueShape::Vector3, ResolvedObservationTarget::Boundary { .. }) => {}
        (ObservationValueShape::SymmetricTensor3, ResolvedObservationTarget::Boundary { .. })
            if spec.quantity == ObservationQuantity::Stress
                && matches!(
                    spec.sign_convention,
                    SignConvention::Native | SignConvention::TensionPositive
                ) => {}
        (ObservationValueShape::Scalar, ResolvedObservationTarget::Boundary { .. })
            if matches!(
                spec.quantity,
                ObservationQuantity::FluidPressure
                    | ObservationQuantity::AcousticPressure
                    | ObservationQuantity::ContactPressure
            ) && matches!(
                spec.sign_convention,
                SignConvention::OutwardPositive | SignConvention::InwardPositive
            ) => {}
        _ => {
            return Err(ObservationError::Unsupported(
                "resultants require a vector volume/surface field, boundary stress, or explicitly oriented boundary pressure",
            ))
        }
    }
    if matches!(operation, SpatialReduction::Moment { .. })
        && spec.quantity == ObservationQuantity::ReactionMoment
    {
        return Err(ObservationError::Invalid(
            "a moment reduction cannot take a second moment of reaction moments".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_numeric_representation(
    spec: &ObservationSpec,
    source: &RecoveredFieldDefinition,
) -> Result<(), ObservationError> {
    match source.numeric_representation() {
        RecoveredFieldNumericRepresentation::Real
            if spec.complex_representation == ComplexRepresentation::Real =>
        {
            Ok(())
        }
        RecoveredFieldNumericRepresentation::ComplexInterleaved => Ok(()),
        RecoveredFieldNumericRepresentation::Real => Err(ObservationError::Invalid(
            "a real recovered field cannot provide imaginary, phase, or complex values".into(),
        )),
    }
}
