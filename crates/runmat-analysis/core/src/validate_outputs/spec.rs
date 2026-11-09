use crate::{
    AnalysisModel, AnalysisProcedure, AnalysisValidationError, ComplexRepresentation,
    CoordinateFrame, CoupledPhysicsFamily, ExcitationTarget, FieldLocation, ObservationComponent,
    ObservationQuantity, ObservationRequest, ObservationSpec, ObservationTarget,
    ObservationValueShape, PhysicsFamily, ProcedureStep, RecoveryPolicy, SignConvention,
    SignalPrecision, SpatialReduction,
};

use super::{invalid_observation, observation_error};

pub(super) fn validate_spec(
    model: &AnalysisModel,
    step: &ProcedureStep,
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    validate_quantity_family(step, request, spec.quantity)?;
    validate_shape(request, spec)?;
    validate_target(model, request, &spec.target)?;
    crate::validate_authored::validate_coordinate_frame(&spec.frame)
        .map_err(|_| invalid_observation(request, "coordinate frame is invalid"))?;
    validate_component(request, spec)?;
    validate_sign(request, spec)?;
    validate_complex(step, request, spec)?;
    validate_target_location(request, spec)?;
    validate_pairing(request, spec)?;
    validate_recovery(request, spec)?;
    super::sampling::validate_sampling(step, request, &spec.sampling)?;
    super::sampling::validate_reduction(step, request, spec)?;
    super::sampling::validate_storage(request, spec)
}

pub(super) fn validate_authored_spec(
    step: &ProcedureStep,
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    validate_quantity_family(step, request, spec.quantity)?;
    validate_shape(request, spec)?;
    crate::validate_authored::validate_coordinate_frame(&spec.frame)
        .map_err(|_| invalid_observation(request, "coordinate frame is invalid"))?;
    validate_component(request, spec)?;
    validate_sign(request, spec)?;
    validate_complex(step, request, spec)?;
    validate_target_location(request, spec)?;
    validate_pairing(request, spec)?;
    validate_recovery(request, spec)?;
    super::sampling::validate_sampling(step, request, &spec.sampling)?;
    super::sampling::validate_reduction(step, request, spec)?;
    super::sampling::validate_storage(request, spec)
}

/// A paired reduction (pressure drop, transmission loss) reads exactly a two-face-region target,
/// and a two-region target is read only by a paired reduction.
fn validate_pairing(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    let paired_target = matches!(spec.target, ObservationTarget::RegionPair { .. });
    let paired_reduction = spec.spatial_reduction().is_paired();
    if paired_target != paired_reduction {
        return observation_error(
            request,
            "a two-region target requires a paired reduction, and a paired reduction requires two face regions",
        );
    }
    if paired_reduction
        && (!matches!(
            request,
            ObservationRequest::Reduction(_) | ObservationRequest::History(_)
        ) || spec.value_shape != ObservationValueShape::Scalar
            || spec.component != ObservationComponent::All
            || spec.complex_representation != ComplexRepresentation::Real
            || spec.output_location != FieldLocation::Global)
    {
        return observation_error(
            request,
            "a paired reduction reads one real scalar field into one global value",
        );
    }
    if spec.spatial_reduction() == SpatialReduction::TransmissionLoss
        && spec.quantity != ObservationQuantity::AcousticPortIntensity
    {
        return observation_error(
            request,
            "transmission loss reads the acoustic port intensity",
        );
    }
    Ok(())
}

fn validate_target_location(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    match &spec.target {
        ObservationTarget::Path { .. }
            if spec.source_location == FieldLocation::Node
                && spec.output_location == FieldLocation::Point
                && spec.recovery == RecoveryPolicy::ShapeFunctionInterpolation =>
        {
            Ok(())
        }
        ObservationTarget::Path { .. } => observation_error(
            request,
            "path observations require nodal source values, point output, and shape-function interpolation",
        ),
        ObservationTarget::Point { .. }
            if spec.output_location == FieldLocation::Point
                && (spec.source_location != FieldLocation::Node
                    || spec.recovery != RecoveryPolicy::ShapeFunctionInterpolation) =>
        {
            observation_error(
                request,
                "interpolated point observations require nodal source values and shape-function interpolation",
            )
        }
        ObservationTarget::Point { .. }
            if spec.recovery == RecoveryPolicy::ShapeFunctionInterpolation
                && spec.output_location != FieldLocation::Point =>
        {
            observation_error(
                request,
                "shape-function point interpolation requires point output location",
            )
        }
        ObservationTarget::Point { .. } => Ok(()),
        _ if spec.output_location == FieldLocation::Point => observation_error(
            request,
            "point output location requires a point or path target",
        ),
        _ => Ok(()),
    }
}

fn validate_quantity_family(
    step: &ProcedureStep,
    request: &ObservationRequest,
    quantity: ObservationQuantity,
) -> Result<(), AnalysisValidationError> {
    let supported = quantity_supported(step, quantity);
    if supported {
        Ok(())
    } else {
        observation_error(
            request,
            "quantity is not produced by the step's physics family",
        )
    }
}

pub fn quantity_supported(step: &ProcedureStep, quantity: ObservationQuantity) -> bool {
    (match step.family {
        PhysicsFamily::Structural => is_structural(quantity),
        PhysicsFamily::Thermal => is_thermal(quantity),
        PhysicsFamily::Electromagnetic => is_electromagnetic(quantity),
        PhysicsFamily::Acoustic => is_acoustic(quantity),
        PhysicsFamily::Fluid => is_fluid(quantity),
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical) => {
            is_structural(quantity) || is_thermal(quantity)
        }
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal) => {
            is_electromagnetic(quantity) || is_thermal(quantity)
        }
        PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer) => {
            is_fluid(quantity) || is_thermal(quantity)
        }
        PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction) => {
            is_fluid(quantity) || is_structural(quantity)
        }
    }) || is_universal(quantity)
}

fn is_structural(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::Displacement
            | ObservationQuantity::Rotation
            | ObservationQuantity::Velocity
            | ObservationQuantity::Acceleration
            | ObservationQuantity::Strain
            | ObservationQuantity::Stress
            | ObservationQuantity::VonMisesStress
            | ObservationQuantity::ReactionForce
            | ObservationQuantity::ReactionMoment
            | ObservationQuantity::StrainEnergy
            | ObservationQuantity::BeamAxialForce
            | ObservationQuantity::BeamShearForce
            | ObservationQuantity::BeamTorsionMoment
            | ObservationQuantity::BeamBendingMoment
            | ObservationQuantity::BeamBendingStress
            | ObservationQuantity::BeamTorsionStress
            | ObservationQuantity::ShellMembraneForce
            | ObservationQuantity::ShellBendingMoment
            | ObservationQuantity::ShellTransverseShear
            | ObservationQuantity::ShellVonMises
            | ObservationQuantity::ContactGap
            | ObservationQuantity::ContactPressure
            | ObservationQuantity::ModalFrequency
            | ObservationQuantity::ModalParticipation
            | ObservationQuantity::Eigenvalue
    )
}

fn is_thermal(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::Temperature
            | ObservationQuantity::TemperatureGradient
            | ObservationQuantity::HeatFlux
            | ObservationQuantity::HeatRate
            | ObservationQuantity::ThermalEnergyStorageRateDensity
    )
}

fn is_electromagnetic(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::ElectricPotential
            | ObservationQuantity::ElectricField
            | ObservationQuantity::ElectricDisplacement
            | ObservationQuantity::ElectricEnergyDensity
            | ObservationQuantity::JoulePowerDensity
            | ObservationQuantity::ElectricCurrent
            | ObservationQuantity::ElectricCurrentDensity
            | ObservationQuantity::ElectricCharge
            | ObservationQuantity::ElectricEdgeCirculation
            | ObservationQuantity::MagneticEdgeCirculation
            | ObservationQuantity::MagneticVectorPotential
            | ObservationQuantity::MagneticFluxDensity
            | ObservationQuantity::MagneticFieldIntensity
            | ObservationQuantity::HeatRate
    )
}

fn is_acoustic(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::AcousticPressure
            | ObservationQuantity::AcousticParticleVelocity
            | ObservationQuantity::AcousticIntensity
            | ObservationQuantity::AcousticPortIntensity
            | ObservationQuantity::SoundPressureLevel
            | ObservationQuantity::TransmissionLoss
    )
}

fn is_fluid(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::FluidPressure
            | ObservationQuantity::FluidVelocity
            | ObservationQuantity::Vorticity
            | ObservationQuantity::MassFlowRate
            | ObservationQuantity::VolumeFlowRate
            | ObservationQuantity::WallShearStress
            | ObservationQuantity::HeatRate
    )
}

fn is_universal(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::Residual
            | ObservationQuantity::Balance
            | ObservationQuantity::ConstraintViolation
    )
}

fn validate_shape(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    if let ObservationValueShape::Tensor { extents } = &spec.value_shape {
        if extents.is_empty()
            || extents.contains(&0)
            || extents
                .iter()
                .try_fold(1usize, |total, extent| total.checked_mul(*extent))
                .is_none()
        {
            return observation_error(
                request,
                "tensor extents must be nonempty, nonzero, and representable",
            );
        }
    }
    let valid = spec
        .value_shape
        .is_storage_equivalent_to(&spec.quantity.canonical_value_shape());
    if valid {
        Ok(())
    } else {
        observation_error(request, "value shape does not match the requested quantity")
    }
}

fn validate_target(
    model: &AnalysisModel,
    request: &ObservationRequest,
    target: &ObservationTarget,
) -> Result<(), AnalysisValidationError> {
    let valid = match target {
        ObservationTarget::Spatial(ExcitationTarget::Region(region)) => {
            !region.region_id.trim().is_empty()
        }
        ObservationTarget::Spatial(ExcitationTarget::ReferencePoint(point)) => {
            !point.reference_point_id.trim().is_empty() && point.position.is_finite()
        }
        ObservationTarget::Spatial(ExcitationTarget::Port { port_id }) => {
            !port_id.trim().is_empty()
        }
        ObservationTarget::Spatial(ExcitationTarget::Interface { interface_id }) => model
            .interfaces
            .iter()
            .any(|interface| interface.interface_id == *interface_id),
        ObservationTarget::RegionPair { first, second } => [first, second].iter().all(|region| {
            !region.region_id.trim().is_empty()
                && region.entity_dimension == crate::EntityDimension::Face
        }),
        ObservationTarget::Spatial(ExcitationTarget::WholeModel)
        | ObservationTarget::WholeModel => true,
        ObservationTarget::Point { position } => position.is_finite(),
        ObservationTarget::Path { positions } => {
            positions.len() >= 2 && positions.iter().all(|position| position.is_finite())
        }
        ObservationTarget::Connector { connector_id } => model
            .connectors
            .iter()
            .any(|connector| connector.connector_id == *connector_id),
        ObservationTarget::Constraint { constraint_id } => model
            .constraints
            .iter()
            .any(|constraint| constraint.constraint_id() == constraint_id),
    };
    if valid {
        Ok(())
    } else {
        observation_error(
            request,
            "target is invalid or does not resolve in the analysis model",
        )
    }
}

fn validate_component(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    use ObservationComponent as C;
    let boundary_component = matches!(spec.component, C::Normal | C::Tangent1 | C::Tangent2);
    if boundary_component && !matches!(&spec.frame, CoordinateFrame::BoundaryNormalTangent) {
        return observation_error(
            request,
            "normal and tangent components require the boundary-normal/tangent frame",
        );
    }
    let valid = match spec.quantity.canonical_value_shape() {
        ObservationValueShape::Scalar => matches!(spec.component, C::All),
        ObservationValueShape::Vector3 => matches!(
            spec.component,
            C::All | C::X | C::Y | C::Z | C::Normal | C::Tangent1 | C::Tangent2 | C::Magnitude
        ),
        ObservationValueShape::SymmetricTensor3 => matches!(
            spec.component,
            C::All
                | C::Normal
                | C::Tangent1
                | C::Tangent2
                | C::Trace
                | C::Hydrostatic
                | C::Deviatoric
                | C::VonMises
                | C::MaximumPrincipal
                | C::MinimumPrincipal
        ),
        ObservationValueShape::Tensor { .. } => matches!(spec.component, C::All),
    };
    if valid {
        Ok(())
    } else {
        observation_error(
            request,
            "component is incompatible with the quantity's value shape",
        )
    }
}

fn validate_sign(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    let valid = match spec.sign_convention {
        SignConvention::Native => true,
        SignConvention::TensionPositive | SignConvention::CompressionPositive => matches!(
            spec.quantity,
            ObservationQuantity::Strain
                | ObservationQuantity::Stress
                | ObservationQuantity::ContactPressure
        ),
        SignConvention::OutwardPositive | SignConvention::InwardPositive => matches!(
            spec.quantity,
            ObservationQuantity::HeatFlux
                | ObservationQuantity::HeatRate
                | ObservationQuantity::ElectricCurrent
                | ObservationQuantity::ElectricCurrentDensity
                | ObservationQuantity::MassFlowRate
                | ObservationQuantity::AcousticParticleVelocity
                | ObservationQuantity::FluidVelocity
        ),
    };
    if valid {
        Ok(())
    } else {
        observation_error(
            request,
            "sign convention is not meaningful for this quantity",
        )
    }
}

fn validate_complex(
    step: &ProcedureStep,
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    let complex_procedure = matches!(
        step.procedure,
        AnalysisProcedure::HarmonicResponse
            | AnalysisProcedure::PrestressedHarmonicResponse
            | AnalysisProcedure::RandomResponse
    );
    if !matches!(spec.complex_representation, ComplexRepresentation::Real) && !complex_procedure {
        return observation_error(
            request,
            "complex representation requires a complex procedure",
        );
    }
    let complex_storage = matches!(
        spec.storage.precision,
        SignalPrecision::Complex64 | SignalPrecision::Complex128
    );
    if matches!(spec.complex_representation, ComplexRepresentation::Complex) != complex_storage {
        return observation_error(
            request,
            "storage precision must be complex exactly when the representation is complex",
        );
    }
    Ok(())
}

fn validate_recovery(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    if spec.output_location == FieldLocation::Global
        && spec.recovery == RecoveryPolicy::Native
        && spec
            .reduction
            .as_ref()
            .is_some_and(|reduction| reduction.spatial != SpatialReduction::None)
    {
        return Ok(());
    }
    if spec.source_location == spec.output_location {
        return if matches!(spec.recovery, RecoveryPolicy::Native) {
            Ok(())
        } else {
            observation_error(
                request,
                "recovery must be native when source and output locations match",
            )
        };
    }
    let valid = match spec.recovery {
        RecoveryPolicy::Native => false,
        RecoveryPolicy::Extrapolate | RecoveryPolicy::VolumeWeightedAverage => matches!(
            spec.source_location,
            FieldLocation::IntegrationPoint | FieldLocation::Element | FieldLocation::Cell
        ),
        RecoveryPolicy::AreaWeightedAverage => matches!(
            spec.source_location,
            FieldLocation::Face | FieldLocation::Interface
        ),
        RecoveryPolicy::ConservativeProjection => true,
        RecoveryPolicy::ShapeFunctionInterpolation => matches!(
            spec.source_location,
            FieldLocation::Node | FieldLocation::Element | FieldLocation::IntegrationPoint
        ),
    };
    if valid {
        Ok(())
    } else {
        observation_error(
            request,
            "recovery policy is incompatible with the location transfer",
        )
    }
}

pub(super) fn is_scalar_result(spec: &ObservationSpec) -> bool {
    spec.result_value_shape() == ObservationValueShape::Scalar
}

#[cfg(test)]
mod tests {
    use crate::{ObservationQuantity, ProcedureStep};

    use super::quantity_supported;

    #[test]
    fn quantity_support_follows_physics_family() {
        let structural = ProcedureStep::structural_static("structural");
        assert!(quantity_supported(&structural, ObservationQuantity::Stress));
        for quantity in [
            ObservationQuantity::VonMisesStress,
            ObservationQuantity::BeamAxialForce,
            ObservationQuantity::BeamShearForce,
            ObservationQuantity::BeamTorsionMoment,
            ObservationQuantity::BeamBendingMoment,
            ObservationQuantity::BeamBendingStress,
            ObservationQuantity::BeamTorsionStress,
            ObservationQuantity::ShellMembraneForce,
            ObservationQuantity::ShellBendingMoment,
            ObservationQuantity::ShellTransverseShear,
            ObservationQuantity::ShellVonMises,
        ] {
            assert!(quantity_supported(&structural, quantity));
        }
        assert!(!quantity_supported(
            &structural,
            ObservationQuantity::Temperature
        ));
        assert!(quantity_supported(
            &structural,
            ObservationQuantity::Residual
        ));

        let thermal = ProcedureStep::thermal_transient("thermal");
        assert!(quantity_supported(&thermal, ObservationQuantity::HeatFlux));
        assert!(quantity_supported(
            &thermal,
            ObservationQuantity::ThermalEnergyStorageRateDensity
        ));
        assert!(!quantity_supported(
            &structural,
            ObservationQuantity::ThermalEnergyStorageRateDensity
        ));
        for family in [
            crate::PhysicsFamily::Electromagnetic,
            crate::PhysicsFamily::Acoustic,
            crate::PhysicsFamily::Fluid,
            crate::PhysicsFamily::Coupled(crate::CoupledPhysicsFamily::FluidStructureInteraction),
        ] {
            let mut nonthermal = structural.clone();
            nonthermal.family = family;
            assert!(!quantity_supported(
                &nonthermal,
                ObservationQuantity::ThermalEnergyStorageRateDensity
            ));
        }
        assert!(!quantity_supported(
            &thermal,
            ObservationQuantity::Displacement
        ));
    }
}
