use std::sync::Once;

use runmat_analysis_core::{
    ComplexRepresentation, DiagnosticKind, ExtremaTiePolicy, FieldLocation, ObservationComponent,
    ObservationPurpose, ObservationQuantity, PartialAccessPolicy, Position, RecoveryPolicy,
    SeriesReduction, SeriesWeighting, SignConvention, SpatialReduction,
};

use super::super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, register_immutable_class, to_typed_value, VALUE_FIELD,
};

pub(super) const QUANTITY_CLASS: &str = "fea.ObservationQuantity";
pub(super) const FIELD_LOCATION_CLASS: &str = "fea.FieldLocation";
pub(super) const RECOVERY_CLASS: &str = "fea.RecoveryPolicy";
pub(super) const COMPONENT_CLASS: &str = "fea.ObservationComponent";
pub(super) const SIGN_CLASS: &str = "fea.SignConvention";
pub(super) const COMPLEX_CLASS: &str = "fea.ComplexRepresentation";
pub(super) const SPATIAL_REDUCTION_CLASS: &str = "fea.SpatialReduction";
pub(super) const SERIES_REDUCTION_CLASS: &str = "fea.SeriesReduction";
pub(super) const SERIES_WEIGHTING_CLASS: &str = "fea.SeriesWeighting";
pub(super) const EXTREMA_TIE_CLASS: &str = "fea.ExtremaTiePolicy";
pub(super) const PARTIAL_ACCESS_CLASS: &str = "fea.PartialAccessPolicy";
pub(super) const PURPOSE_CLASS: &str = "fea.ObservationPurpose";
pub(super) const DIAGNOSTIC_KIND_CLASS: &str = "fea.DiagnosticKind";

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}
const CHOICE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA observation-policy constructors have no integer inputs.",
};

macro_rules! choice {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $class:expr, $value:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!("value = ", $builtin, "()"),
            inputs: &NO_INPUTS,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        #[runtime_builtin(
                            name = $builtin,
                            category = "fea",
                            summary = "Select a typed FEA observation policy.",
                            keywords = "fea,observation,output,policy,typed",
                            descriptor($descriptor),
                            integer_audit(CHOICE_INTEGER_AUDIT),
                            builtin_path = "crate::builtins::fea::study_policies::observation"
                        )]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value($builtin, $class, $value)
        }
    };
}

macro_rules! quantity_choices {
    ($(($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $value:ident)),+ $(,)?) => {
        $(choice!($function, $descriptor, $signatures, $builtin, QUANTITY_CLASS, ObservationQuantity::$value);)+
    };
}

quantity_choices!(
    (
        quantity_displacement,
        Q_DISPLACEMENT_DESCRIPTOR,
        Q_DISPLACEMENT_SIGNATURES,
        "fea.ObservationQuantity.displacement",
        Displacement
    ),
    (
        quantity_rotation,
        Q_ROTATION_DESCRIPTOR,
        Q_ROTATION_SIGNATURES,
        "fea.ObservationQuantity.rotation",
        Rotation
    ),
    (
        quantity_velocity,
        Q_VELOCITY_DESCRIPTOR,
        Q_VELOCITY_SIGNATURES,
        "fea.ObservationQuantity.velocity",
        Velocity
    ),
    (
        quantity_acceleration,
        Q_ACCELERATION_DESCRIPTOR,
        Q_ACCELERATION_SIGNATURES,
        "fea.ObservationQuantity.acceleration",
        Acceleration
    ),
    (
        quantity_strain,
        Q_STRAIN_DESCRIPTOR,
        Q_STRAIN_SIGNATURES,
        "fea.ObservationQuantity.strain",
        Strain
    ),
    (
        quantity_stress,
        Q_STRESS_DESCRIPTOR,
        Q_STRESS_SIGNATURES,
        "fea.ObservationQuantity.stress",
        Stress
    ),
    (
        quantity_reaction_force,
        Q_REACTION_FORCE_DESCRIPTOR,
        Q_REACTION_FORCE_SIGNATURES,
        "fea.ObservationQuantity.reactionForce",
        ReactionForce
    ),
    (
        quantity_reaction_moment,
        Q_REACTION_MOMENT_DESCRIPTOR,
        Q_REACTION_MOMENT_SIGNATURES,
        "fea.ObservationQuantity.reactionMoment",
        ReactionMoment
    ),
    (
        quantity_strain_energy,
        Q_STRAIN_ENERGY_DESCRIPTOR,
        Q_STRAIN_ENERGY_SIGNATURES,
        "fea.ObservationQuantity.strainEnergy",
        StrainEnergy
    ),
    (
        quantity_temperature,
        Q_TEMPERATURE_DESCRIPTOR,
        Q_TEMPERATURE_SIGNATURES,
        "fea.ObservationQuantity.temperature",
        Temperature
    ),
    (
        quantity_temperature_gradient,
        Q_TEMPERATURE_GRADIENT_DESCRIPTOR,
        Q_TEMPERATURE_GRADIENT_SIGNATURES,
        "fea.ObservationQuantity.temperatureGradient",
        TemperatureGradient
    ),
    (
        quantity_heat_flux,
        Q_HEAT_FLUX_DESCRIPTOR,
        Q_HEAT_FLUX_SIGNATURES,
        "fea.ObservationQuantity.heatFlux",
        HeatFlux
    ),
    (
        quantity_heat_rate,
        Q_HEAT_RATE_DESCRIPTOR,
        Q_HEAT_RATE_SIGNATURES,
        "fea.ObservationQuantity.heatRate",
        HeatRate
    ),
    (
        quantity_electric_potential,
        Q_ELECTRIC_POTENTIAL_DESCRIPTOR,
        Q_ELECTRIC_POTENTIAL_SIGNATURES,
        "fea.ObservationQuantity.electricPotential",
        ElectricPotential
    ),
    (
        quantity_electric_field,
        Q_ELECTRIC_FIELD_DESCRIPTOR,
        Q_ELECTRIC_FIELD_SIGNATURES,
        "fea.ObservationQuantity.electricField",
        ElectricField
    ),
    (
        quantity_electric_current,
        Q_ELECTRIC_CURRENT_DESCRIPTOR,
        Q_ELECTRIC_CURRENT_SIGNATURES,
        "fea.ObservationQuantity.electricCurrent",
        ElectricCurrent
    ),
    (
        quantity_electric_current_density,
        Q_ELECTRIC_CURRENT_DENSITY_DESCRIPTOR,
        Q_ELECTRIC_CURRENT_DENSITY_SIGNATURES,
        "fea.ObservationQuantity.electricCurrentDensity",
        ElectricCurrentDensity
    ),
    (
        quantity_electric_charge,
        Q_ELECTRIC_CHARGE_DESCRIPTOR,
        Q_ELECTRIC_CHARGE_SIGNATURES,
        "fea.ObservationQuantity.electricCharge",
        ElectricCharge
    ),
    (
        quantity_magnetic_vector_potential,
        Q_MAGNETIC_VECTOR_POTENTIAL_DESCRIPTOR,
        Q_MAGNETIC_VECTOR_POTENTIAL_SIGNATURES,
        "fea.ObservationQuantity.magneticVectorPotential",
        MagneticVectorPotential
    ),
    (
        quantity_magnetic_flux_density,
        Q_MAGNETIC_FLUX_DENSITY_DESCRIPTOR,
        Q_MAGNETIC_FLUX_DENSITY_SIGNATURES,
        "fea.ObservationQuantity.magneticFluxDensity",
        MagneticFluxDensity
    ),
    (
        quantity_magnetic_field_intensity,
        Q_MAGNETIC_FIELD_INTENSITY_DESCRIPTOR,
        Q_MAGNETIC_FIELD_INTENSITY_SIGNATURES,
        "fea.ObservationQuantity.magneticFieldIntensity",
        MagneticFieldIntensity
    ),
    (
        quantity_acoustic_pressure,
        Q_ACOUSTIC_PRESSURE_DESCRIPTOR,
        Q_ACOUSTIC_PRESSURE_SIGNATURES,
        "fea.ObservationQuantity.acousticPressure",
        AcousticPressure
    ),
    (
        quantity_acoustic_particle_velocity,
        Q_ACOUSTIC_PARTICLE_VELOCITY_DESCRIPTOR,
        Q_ACOUSTIC_PARTICLE_VELOCITY_SIGNATURES,
        "fea.ObservationQuantity.acousticParticleVelocity",
        AcousticParticleVelocity
    ),
    (
        quantity_sound_pressure_level,
        Q_SOUND_PRESSURE_LEVEL_DESCRIPTOR,
        Q_SOUND_PRESSURE_LEVEL_SIGNATURES,
        "fea.ObservationQuantity.soundPressureLevel",
        SoundPressureLevel
    ),
    (
        quantity_fluid_pressure,
        Q_FLUID_PRESSURE_DESCRIPTOR,
        Q_FLUID_PRESSURE_SIGNATURES,
        "fea.ObservationQuantity.fluidPressure",
        FluidPressure
    ),
    (
        quantity_fluid_velocity,
        Q_FLUID_VELOCITY_DESCRIPTOR,
        Q_FLUID_VELOCITY_SIGNATURES,
        "fea.ObservationQuantity.fluidVelocity",
        FluidVelocity
    ),
    (
        quantity_vorticity,
        Q_VORTICITY_DESCRIPTOR,
        Q_VORTICITY_SIGNATURES,
        "fea.ObservationQuantity.vorticity",
        Vorticity
    ),
    (
        quantity_mass_flow_rate,
        Q_MASS_FLOW_RATE_DESCRIPTOR,
        Q_MASS_FLOW_RATE_SIGNATURES,
        "fea.ObservationQuantity.massFlowRate",
        MassFlowRate
    ),
    (
        quantity_wall_shear_stress,
        Q_WALL_SHEAR_STRESS_DESCRIPTOR,
        Q_WALL_SHEAR_STRESS_SIGNATURES,
        "fea.ObservationQuantity.wallShearStress",
        WallShearStress
    ),
    (
        quantity_modal_frequency,
        Q_MODAL_FREQUENCY_DESCRIPTOR,
        Q_MODAL_FREQUENCY_SIGNATURES,
        "fea.ObservationQuantity.modalFrequency",
        ModalFrequency
    ),
    (
        quantity_modal_participation,
        Q_MODAL_PARTICIPATION_DESCRIPTOR,
        Q_MODAL_PARTICIPATION_SIGNATURES,
        "fea.ObservationQuantity.modalParticipation",
        ModalParticipation
    ),
    (
        quantity_eigenvalue,
        Q_EIGENVALUE_DESCRIPTOR,
        Q_EIGENVALUE_SIGNATURES,
        "fea.ObservationQuantity.eigenvalue",
        Eigenvalue
    ),
    (
        quantity_residual,
        Q_RESIDUAL_DESCRIPTOR,
        Q_RESIDUAL_SIGNATURES,
        "fea.ObservationQuantity.residual",
        Residual
    ),
    (
        quantity_balance,
        Q_BALANCE_DESCRIPTOR,
        Q_BALANCE_SIGNATURES,
        "fea.ObservationQuantity.balance",
        Balance
    ),
    (
        quantity_constraint_violation,
        Q_CONSTRAINT_VIOLATION_DESCRIPTOR,
        Q_CONSTRAINT_VIOLATION_SIGNATURES,
        "fea.ObservationQuantity.constraintViolation",
        ConstraintViolation
    ),
    (
        quantity_contact_gap,
        Q_CONTACT_GAP_DESCRIPTOR,
        Q_CONTACT_GAP_SIGNATURES,
        "fea.ObservationQuantity.contactGap",
        ContactGap
    ),
    (
        quantity_contact_pressure,
        Q_CONTACT_PRESSURE_DESCRIPTOR,
        Q_CONTACT_PRESSURE_SIGNATURES,
        "fea.ObservationQuantity.contactPressure",
        ContactPressure
    ),
);

macro_rules! choices {
    ($class:expr, $(($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $value:expr)),+ $(,)?) => {
        $(choice!($function, $descriptor, $signatures, $builtin, $class, $value);)+
    };
}

choices!(
    FIELD_LOCATION_CLASS,
    (
        location_node,
        LOCATION_NODE_DESCRIPTOR,
        LOCATION_NODE_SIGNATURES,
        "fea.FieldLocation.node",
        FieldLocation::Node
    ),
    (
        location_point,
        LOCATION_POINT_DESCRIPTOR,
        LOCATION_POINT_SIGNATURES,
        "fea.FieldLocation.point",
        FieldLocation::Point
    ),
    (
        location_cell,
        LOCATION_CELL_DESCRIPTOR,
        LOCATION_CELL_SIGNATURES,
        "fea.FieldLocation.cell",
        FieldLocation::Cell
    ),
    (
        location_element,
        LOCATION_ELEMENT_DESCRIPTOR,
        LOCATION_ELEMENT_SIGNATURES,
        "fea.FieldLocation.element",
        FieldLocation::Element
    ),
    (
        location_integration_point,
        LOCATION_INTEGRATION_POINT_DESCRIPTOR,
        LOCATION_INTEGRATION_POINT_SIGNATURES,
        "fea.FieldLocation.integrationPoint",
        FieldLocation::IntegrationPoint
    ),
    (
        location_face,
        LOCATION_FACE_DESCRIPTOR,
        LOCATION_FACE_SIGNATURES,
        "fea.FieldLocation.face",
        FieldLocation::Face
    ),
    (
        location_edge,
        LOCATION_EDGE_DESCRIPTOR,
        LOCATION_EDGE_SIGNATURES,
        "fea.FieldLocation.edge",
        FieldLocation::Edge
    ),
    (
        location_interface,
        LOCATION_INTERFACE_DESCRIPTOR,
        LOCATION_INTERFACE_SIGNATURES,
        "fea.FieldLocation.interface",
        FieldLocation::Interface
    ),
    (
        location_connector,
        LOCATION_CONNECTOR_DESCRIPTOR,
        LOCATION_CONNECTOR_SIGNATURES,
        "fea.FieldLocation.connector",
        FieldLocation::Connector
    ),
    (
        location_constraint,
        LOCATION_CONSTRAINT_DESCRIPTOR,
        LOCATION_CONSTRAINT_SIGNATURES,
        "fea.FieldLocation.constraint",
        FieldLocation::Constraint
    ),
    (
        location_reference_point,
        LOCATION_REFERENCE_POINT_DESCRIPTOR,
        LOCATION_REFERENCE_POINT_SIGNATURES,
        "fea.FieldLocation.referencePoint",
        FieldLocation::ReferencePoint
    ),
    (
        location_mode,
        LOCATION_MODE_DESCRIPTOR,
        LOCATION_MODE_SIGNATURES,
        "fea.FieldLocation.mode",
        FieldLocation::Mode
    ),
    (
        location_global,
        LOCATION_GLOBAL_DESCRIPTOR,
        LOCATION_GLOBAL_SIGNATURES,
        "fea.FieldLocation.globalLocation",
        FieldLocation::Global
    )
);
choices!(
    RECOVERY_CLASS,
    (
        recovery_native,
        RECOVERY_NATIVE_DESCRIPTOR,
        RECOVERY_NATIVE_SIGNATURES,
        "fea.RecoveryPolicy.native",
        RecoveryPolicy::Native
    ),
    (
        recovery_extrapolate,
        RECOVERY_EXTRAPOLATE_DESCRIPTOR,
        RECOVERY_EXTRAPOLATE_SIGNATURES,
        "fea.RecoveryPolicy.extrapolate",
        RecoveryPolicy::Extrapolate
    ),
    (
        recovery_volume_average,
        RECOVERY_VOLUME_AVERAGE_DESCRIPTOR,
        RECOVERY_VOLUME_AVERAGE_SIGNATURES,
        "fea.RecoveryPolicy.volumeWeightedAverage",
        RecoveryPolicy::VolumeWeightedAverage
    ),
    (
        recovery_area_average,
        RECOVERY_AREA_AVERAGE_DESCRIPTOR,
        RECOVERY_AREA_AVERAGE_SIGNATURES,
        "fea.RecoveryPolicy.areaWeightedAverage",
        RecoveryPolicy::AreaWeightedAverage
    ),
    (
        recovery_conservative,
        RECOVERY_CONSERVATIVE_DESCRIPTOR,
        RECOVERY_CONSERVATIVE_SIGNATURES,
        "fea.RecoveryPolicy.conservativeProjection",
        RecoveryPolicy::ConservativeProjection
    ),
    (
        recovery_shape_function,
        RECOVERY_SHAPE_FUNCTION_DESCRIPTOR,
        RECOVERY_SHAPE_FUNCTION_SIGNATURES,
        "fea.RecoveryPolicy.shapeFunctionInterpolation",
        RecoveryPolicy::ShapeFunctionInterpolation
    )
);
choices!(
    COMPONENT_CLASS,
    (
        component_all,
        COMPONENT_ALL_DESCRIPTOR,
        COMPONENT_ALL_SIGNATURES,
        "fea.ObservationComponent.all",
        ObservationComponent::All
    ),
    (
        component_x,
        COMPONENT_X_DESCRIPTOR,
        COMPONENT_X_SIGNATURES,
        "fea.ObservationComponent.x",
        ObservationComponent::X
    ),
    (
        component_y,
        COMPONENT_Y_DESCRIPTOR,
        COMPONENT_Y_SIGNATURES,
        "fea.ObservationComponent.y",
        ObservationComponent::Y
    ),
    (
        component_z,
        COMPONENT_Z_DESCRIPTOR,
        COMPONENT_Z_SIGNATURES,
        "fea.ObservationComponent.z",
        ObservationComponent::Z
    ),
    (
        component_normal,
        COMPONENT_NORMAL_DESCRIPTOR,
        COMPONENT_NORMAL_SIGNATURES,
        "fea.ObservationComponent.normal",
        ObservationComponent::Normal
    ),
    (
        component_tangent1,
        COMPONENT_TANGENT1_DESCRIPTOR,
        COMPONENT_TANGENT1_SIGNATURES,
        "fea.ObservationComponent.tangent1",
        ObservationComponent::Tangent1
    ),
    (
        component_tangent2,
        COMPONENT_TANGENT2_DESCRIPTOR,
        COMPONENT_TANGENT2_SIGNATURES,
        "fea.ObservationComponent.tangent2",
        ObservationComponent::Tangent2
    ),
    (
        component_trace,
        COMPONENT_TRACE_DESCRIPTOR,
        COMPONENT_TRACE_SIGNATURES,
        "fea.ObservationComponent.trace",
        ObservationComponent::Trace
    ),
    (
        component_hydrostatic,
        COMPONENT_HYDROSTATIC_DESCRIPTOR,
        COMPONENT_HYDROSTATIC_SIGNATURES,
        "fea.ObservationComponent.hydrostatic",
        ObservationComponent::Hydrostatic
    ),
    (
        component_deviatoric,
        COMPONENT_DEVIATORIC_DESCRIPTOR,
        COMPONENT_DEVIATORIC_SIGNATURES,
        "fea.ObservationComponent.deviatoric",
        ObservationComponent::Deviatoric
    ),
    (
        component_von_mises,
        COMPONENT_VON_MISES_DESCRIPTOR,
        COMPONENT_VON_MISES_SIGNATURES,
        "fea.ObservationComponent.vonMises",
        ObservationComponent::VonMises
    ),
    (
        component_max_principal,
        COMPONENT_MAX_PRINCIPAL_DESCRIPTOR,
        COMPONENT_MAX_PRINCIPAL_SIGNATURES,
        "fea.ObservationComponent.maximumPrincipal",
        ObservationComponent::MaximumPrincipal
    ),
    (
        component_min_principal,
        COMPONENT_MIN_PRINCIPAL_DESCRIPTOR,
        COMPONENT_MIN_PRINCIPAL_SIGNATURES,
        "fea.ObservationComponent.minimumPrincipal",
        ObservationComponent::MinimumPrincipal
    ),
    (
        component_magnitude,
        COMPONENT_MAGNITUDE_DESCRIPTOR,
        COMPONENT_MAGNITUDE_SIGNATURES,
        "fea.ObservationComponent.magnitude",
        ObservationComponent::Magnitude
    )
);
choices!(
    SIGN_CLASS,
    (
        sign_native,
        SIGN_NATIVE_DESCRIPTOR,
        SIGN_NATIVE_SIGNATURES,
        "fea.SignConvention.native",
        SignConvention::Native
    ),
    (
        sign_tension,
        SIGN_TENSION_DESCRIPTOR,
        SIGN_TENSION_SIGNATURES,
        "fea.SignConvention.tensionPositive",
        SignConvention::TensionPositive
    ),
    (
        sign_compression,
        SIGN_COMPRESSION_DESCRIPTOR,
        SIGN_COMPRESSION_SIGNATURES,
        "fea.SignConvention.compressionPositive",
        SignConvention::CompressionPositive
    ),
    (
        sign_outward,
        SIGN_OUTWARD_DESCRIPTOR,
        SIGN_OUTWARD_SIGNATURES,
        "fea.SignConvention.outwardPositive",
        SignConvention::OutwardPositive
    ),
    (
        sign_inward,
        SIGN_INWARD_DESCRIPTOR,
        SIGN_INWARD_SIGNATURES,
        "fea.SignConvention.inwardPositive",
        SignConvention::InwardPositive
    )
);
choices!(
    COMPLEX_CLASS,
    (
        complex_real,
        COMPLEX_REAL_DESCRIPTOR,
        COMPLEX_REAL_SIGNATURES,
        "fea.ComplexRepresentation.real",
        ComplexRepresentation::Real
    ),
    (
        complex_imaginary,
        COMPLEX_IMAGINARY_DESCRIPTOR,
        COMPLEX_IMAGINARY_SIGNATURES,
        "fea.ComplexRepresentation.imaginary",
        ComplexRepresentation::Imaginary
    ),
    (
        complex_magnitude,
        COMPLEX_MAGNITUDE_DESCRIPTOR,
        COMPLEX_MAGNITUDE_SIGNATURES,
        "fea.ComplexRepresentation.magnitude",
        ComplexRepresentation::Magnitude
    ),
    (
        complex_phase,
        COMPLEX_PHASE_DESCRIPTOR,
        COMPLEX_PHASE_SIGNATURES,
        "fea.ComplexRepresentation.phaseRadians",
        ComplexRepresentation::PhaseRadians
    ),
    (
        complex_complex,
        COMPLEX_COMPLEX_DESCRIPTOR,
        COMPLEX_COMPLEX_SIGNATURES,
        "fea.ComplexRepresentation.complexValue",
        ComplexRepresentation::Complex
    )
);
choices!(
    SPATIAL_REDUCTION_CLASS,
    (
        spatial_none,
        SPATIAL_NONE_DESCRIPTOR,
        SPATIAL_NONE_SIGNATURES,
        "fea.SpatialReduction.none",
        SpatialReduction::None
    ),
    (
        spatial_integral,
        SPATIAL_INTEGRAL_DESCRIPTOR,
        SPATIAL_INTEGRAL_SIGNATURES,
        "fea.SpatialReduction.integral",
        SpatialReduction::Integral
    ),
    (
        spatial_average,
        SPATIAL_AVERAGE_DESCRIPTOR,
        SPATIAL_AVERAGE_SIGNATURES,
        "fea.SpatialReduction.average",
        SpatialReduction::Average
    ),
    (
        spatial_rms,
        SPATIAL_RMS_DESCRIPTOR,
        SPATIAL_RMS_SIGNATURES,
        "fea.SpatialReduction.rootMeanSquare",
        SpatialReduction::RootMeanSquare
    ),
    (
        spatial_norm,
        SPATIAL_NORM_DESCRIPTOR,
        SPATIAL_NORM_SIGNATURES,
        "fea.SpatialReduction.euclideanNorm",
        SpatialReduction::EuclideanNorm
    ),
    (
        spatial_minimum,
        SPATIAL_MINIMUM_DESCRIPTOR,
        SPATIAL_MINIMUM_SIGNATURES,
        "fea.SpatialReduction.minimum",
        SpatialReduction::Minimum
    ),
    (
        spatial_maximum,
        SPATIAL_MAXIMUM_DESCRIPTOR,
        SPATIAL_MAXIMUM_SIGNATURES,
        "fea.SpatialReduction.maximum",
        SpatialReduction::Maximum
    ),
    (
        spatial_resultant,
        SPATIAL_RESULTANT_DESCRIPTOR,
        SPATIAL_RESULTANT_SIGNATURES,
        "fea.SpatialReduction.resultant",
        SpatialReduction::Resultant
    )
);

const SPATIAL_MOMENT_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "about",
    BuiltinParamType::Any,
    "Typed fea.Position defining the moment reference point.",
)];
const SPATIAL_MOMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.SpatialReduction.moment(about)",
    inputs: &SPATIAL_MOMENT_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const SPATIAL_MOMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPATIAL_MOMENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.SpatialReduction.moment",
    category = "fea",
    summary = "Integrate the moment of a force-like field about an explicit point.",
    keywords = "fea,observation,reduction,moment,reference point,typed",
    descriptor(SPATIAL_MOMENT_DESCRIPTOR),
    integer_audit(CHOICE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::study_policies::observation"
)]
pub async fn spatial_moment(about: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.SpatialReduction.moment";
    let about = Position::new(super::super::values::position_meters(NAME, &about)?);
    to_typed_value(
        NAME,
        SPATIAL_REDUCTION_CLASS,
        SpatialReduction::Moment { about },
    )
}
choices!(
    SERIES_REDUCTION_CLASS,
    (
        series_none,
        SERIES_NONE_DESCRIPTOR,
        SERIES_NONE_SIGNATURES,
        "fea.SeriesReduction.none",
        SeriesReduction::None
    ),
    (
        series_minimum,
        SERIES_MINIMUM_DESCRIPTOR,
        SERIES_MINIMUM_SIGNATURES,
        "fea.SeriesReduction.minimum",
        SeriesReduction::Minimum
    ),
    (
        series_maximum,
        SERIES_MAXIMUM_DESCRIPTOR,
        SERIES_MAXIMUM_SIGNATURES,
        "fea.SeriesReduction.maximum",
        SeriesReduction::Maximum
    ),
    (
        series_absolute_maximum,
        SERIES_ABSOLUTE_MAXIMUM_DESCRIPTOR,
        SERIES_ABSOLUTE_MAXIMUM_SIGNATURES,
        "fea.SeriesReduction.absoluteMaximum",
        SeriesReduction::AbsoluteMaximum
    ),
    (
        series_envelope,
        SERIES_ENVELOPE_DESCRIPTOR,
        SERIES_ENVELOPE_SIGNATURES,
        "fea.SeriesReduction.envelope",
        SeriesReduction::Envelope
    )
);
choices!(
    SERIES_WEIGHTING_CLASS,
    (
        series_weighting_uniform,
        SERIES_WEIGHTING_UNIFORM_DESCRIPTOR,
        SERIES_WEIGHTING_UNIFORM_SIGNATURES,
        "fea.SeriesWeighting.uniformSamples",
        SeriesWeighting::UniformSamples
    ),
    (
        series_weighting_trapezoidal,
        SERIES_WEIGHTING_TRAPEZOIDAL_DESCRIPTOR,
        SERIES_WEIGHTING_TRAPEZOIDAL_SIGNATURES,
        "fea.SeriesWeighting.trapezoidalIndependentDomain",
        SeriesWeighting::TrapezoidalIndependentDomain
    )
);

const SERIES_STATISTIC_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "weighting",
    BuiltinParamType::Any,
    "Typed weighting over the independent-domain samples.",
)];

macro_rules! series_statistic {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $variant:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!("value = ", $builtin, "(weighting)"),
            inputs: &SERIES_STATISTIC_INPUTS,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        #[runtime_builtin(
                            name = $builtin,
                            category = "fea",
                            summary = "Select a typed, explicitly weighted series statistic.",
                            keywords = "fea,observation,series,reduction,weighting,typed",
                            descriptor($descriptor),
                            integer_audit(CHOICE_INTEGER_AUDIT),
                            builtin_path = "crate::builtins::fea::study_policies::observation"
                        )]
        pub async fn $function(weighting: Value) -> BuiltinResult<Value> {
            to_typed_value(
                $builtin,
                SERIES_REDUCTION_CLASS,
                SeriesReduction::$variant {
                    weighting: series_weighting($builtin, &weighting)?,
                },
            )
        }
    };
}

series_statistic!(
    series_rms,
    SERIES_RMS_DESCRIPTOR,
    SERIES_RMS_SIGNATURES,
    "fea.SeriesReduction.rootMeanSquare",
    RootMeanSquare
);
series_statistic!(
    series_mean,
    SERIES_MEAN_DESCRIPTOR,
    SERIES_MEAN_SIGNATURES,
    "fea.SeriesReduction.mean",
    Mean
);
series_statistic!(
    series_variance,
    SERIES_VARIANCE_DESCRIPTOR,
    SERIES_VARIANCE_SIGNATURES,
    "fea.SeriesReduction.variance",
    Variance
);
choices!(
    EXTREMA_TIE_CLASS,
    (
        tie_first,
        TIE_FIRST_DESCRIPTOR,
        TIE_FIRST_SIGNATURES,
        "fea.ExtremaTiePolicy.firstCanonical",
        ExtremaTiePolicy::FirstCanonical
    ),
    (
        tie_all,
        TIE_ALL_DESCRIPTOR,
        TIE_ALL_SIGNATURES,
        "fea.ExtremaTiePolicy.allLocations",
        ExtremaTiePolicy::AllLocations
    )
);
choices!(
    PARTIAL_ACCESS_CLASS,
    (
        partial_chunks,
        PARTIAL_CHUNKS_DESCRIPTOR,
        PARTIAL_CHUNKS_SIGNATURES,
        "fea.PartialAccessPolicy.manifestAndChunks",
        PartialAccessPolicy::ManifestAndChunks
    ),
    (
        partial_whole,
        PARTIAL_WHOLE_DESCRIPTOR,
        PARTIAL_WHOLE_SIGNATURES,
        "fea.PartialAccessPolicy.wholeArtifact",
        PartialAccessPolicy::WholeArtifact
    )
);
choices!(
    PURPOSE_CLASS,
    (
        purpose_publication,
        PURPOSE_PUBLICATION_DESCRIPTOR,
        PURPOSE_PUBLICATION_SIGNATURES,
        "fea.ObservationPurpose.publication",
        ObservationPurpose::Publication
    ),
    (
        purpose_solver,
        PURPOSE_SOLVER_DESCRIPTOR,
        PURPOSE_SOLVER_SIGNATURES,
        "fea.ObservationPurpose.solverControl",
        ObservationPurpose::SolverControl
    ),
    (
        purpose_visualization,
        PURPOSE_VISUALIZATION_DESCRIPTOR,
        PURPOSE_VISUALIZATION_SIGNATURES,
        "fea.ObservationPurpose.visualization",
        ObservationPurpose::Visualization
    ),
    (
        purpose_downstream,
        PURPOSE_DOWNSTREAM_DESCRIPTOR,
        PURPOSE_DOWNSTREAM_SIGNATURES,
        "fea.ObservationPurpose.downstreamState",
        ObservationPurpose::DownstreamState
    )
);
choices!(
    DIAGNOSTIC_KIND_CLASS,
    (
        diagnostic_convergence,
        DIAGNOSTIC_CONVERGENCE_DESCRIPTOR,
        DIAGNOSTIC_CONVERGENCE_SIGNATURES,
        "fea.DiagnosticKind.convergence",
        DiagnosticKind::Convergence
    ),
    (
        diagnostic_balance,
        DIAGNOSTIC_BALANCE_DESCRIPTOR,
        DIAGNOSTIC_BALANCE_SIGNATURES,
        "fea.DiagnosticKind.balance",
        DiagnosticKind::Balance
    ),
    (
        diagnostic_constraint,
        DIAGNOSTIC_CONSTRAINT_DESCRIPTOR,
        DIAGNOSTIC_CONSTRAINT_SIGNATURES,
        "fea.DiagnosticKind.constraint",
        DiagnosticKind::Constraint
    ),
    (
        diagnostic_contact,
        DIAGNOSTIC_CONTACT_DESCRIPTOR,
        DIAGNOSTIC_CONTACT_SIGNATURES,
        "fea.DiagnosticKind.contact",
        DiagnosticKind::Contact
    ),
    (
        diagnostic_adaptation,
        DIAGNOSTIC_ADAPTATION_DESCRIPTOR,
        DIAGNOSTIC_ADAPTATION_SIGNATURES,
        "fea.DiagnosticKind.adaptation",
        DiagnosticKind::Adaptation
    ),
    (
        diagnostic_resource,
        DIAGNOSTIC_RESOURCE_DESCRIPTOR,
        DIAGNOSTIC_RESOURCE_SIGNATURES,
        "fea.DiagnosticKind.resource",
        DiagnosticKind::Resource
    ),
    (
        diagnostic_equivalence,
        DIAGNOSTIC_EQUIVALENCE_DESCRIPTOR,
        DIAGNOSTIC_EQUIVALENCE_SIGNATURES,
        "fea.DiagnosticKind.executionEquivalence",
        DiagnosticKind::ExecutionEquivalence
    )
);

pub(crate) fn quantity(builtin: &'static str, value: &Value) -> BuiltinResult<ObservationQuantity> {
    from_typed_value(builtin, QUANTITY_CLASS, value)
}
pub(crate) fn field_location(builtin: &'static str, value: &Value) -> BuiltinResult<FieldLocation> {
    from_typed_value(builtin, FIELD_LOCATION_CLASS, value)
}
pub(crate) fn recovery(builtin: &'static str, value: &Value) -> BuiltinResult<RecoveryPolicy> {
    from_typed_value(builtin, RECOVERY_CLASS, value)
}
pub(crate) fn component(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<ObservationComponent> {
    from_typed_value(builtin, COMPONENT_CLASS, value)
}
pub(crate) fn sign(builtin: &'static str, value: &Value) -> BuiltinResult<SignConvention> {
    from_typed_value(builtin, SIGN_CLASS, value)
}
pub(crate) fn complex_representation(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<ComplexRepresentation> {
    from_typed_value(builtin, COMPLEX_CLASS, value)
}
pub(crate) fn spatial_reduction(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<SpatialReduction> {
    from_typed_value(builtin, SPATIAL_REDUCTION_CLASS, value)
}
pub(crate) fn series_reduction(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<SeriesReduction> {
    from_typed_value(builtin, SERIES_REDUCTION_CLASS, value)
}
pub(crate) fn series_weighting(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<SeriesWeighting> {
    from_typed_value(builtin, SERIES_WEIGHTING_CLASS, value)
}
pub(crate) fn extrema_ties(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<ExtremaTiePolicy> {
    from_typed_value(builtin, EXTREMA_TIE_CLASS, value)
}
pub(crate) fn partial_access(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<PartialAccessPolicy> {
    from_typed_value(builtin, PARTIAL_ACCESS_CLASS, value)
}
pub(crate) fn purpose(builtin: &'static str, value: &Value) -> BuiltinResult<ObservationPurpose> {
    from_typed_value(builtin, PURPOSE_CLASS, value)
}
pub(crate) fn diagnostic_kind(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<DiagnosticKind> {
    from_typed_value(builtin, DIAGNOSTIC_KIND_CLASS, value)
}

pub(super) fn ensure_classes_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        register_immutable_class(
            QUANTITY_CLASS,
            "runmat.fea.observation-quantity",
            VALUE_FIELD,
            &QUANTITY_METHODS,
        );
        register_immutable_class(
            FIELD_LOCATION_CLASS,
            "runmat.fea.field-location",
            VALUE_FIELD,
            &FIELD_LOCATION_METHODS,
        );
        register_immutable_class(
            RECOVERY_CLASS,
            "runmat.fea.recovery-policy",
            VALUE_FIELD,
            &RECOVERY_METHODS,
        );
        register_immutable_class(
            COMPONENT_CLASS,
            "runmat.fea.observation-component",
            VALUE_FIELD,
            &COMPONENT_METHODS,
        );
        register_immutable_class(
            SIGN_CLASS,
            "runmat.fea.sign-convention",
            VALUE_FIELD,
            &SIGN_METHODS,
        );
        register_immutable_class(
            COMPLEX_CLASS,
            "runmat.fea.complex-representation",
            VALUE_FIELD,
            &COMPLEX_METHODS,
        );
        register_immutable_class(
            SPATIAL_REDUCTION_CLASS,
            "runmat.fea.spatial-reduction",
            VALUE_FIELD,
            &SPATIAL_METHODS,
        );
        register_immutable_class(
            SERIES_REDUCTION_CLASS,
            "runmat.fea.series-reduction",
            VALUE_FIELD,
            &SERIES_METHODS,
        );
        register_immutable_class(
            SERIES_WEIGHTING_CLASS,
            "runmat.fea.series-weighting",
            VALUE_FIELD,
            &SERIES_WEIGHTING_METHODS,
        );
        register_immutable_class(
            EXTREMA_TIE_CLASS,
            "runmat.fea.extrema-tie-policy",
            VALUE_FIELD,
            &[
                ("firstCanonical", "fea.ExtremaTiePolicy.firstCanonical"),
                ("allLocations", "fea.ExtremaTiePolicy.allLocations"),
            ],
        );
        register_immutable_class(
            PARTIAL_ACCESS_CLASS,
            "runmat.fea.partial-access-policy",
            VALUE_FIELD,
            &[
                (
                    "manifestAndChunks",
                    "fea.PartialAccessPolicy.manifestAndChunks",
                ),
                ("wholeArtifact", "fea.PartialAccessPolicy.wholeArtifact"),
            ],
        );
        register_immutable_class(
            PURPOSE_CLASS,
            "runmat.fea.observation-purpose",
            VALUE_FIELD,
            &[
                ("publication", "fea.ObservationPurpose.publication"),
                ("solverControl", "fea.ObservationPurpose.solverControl"),
                ("visualization", "fea.ObservationPurpose.visualization"),
                ("downstreamState", "fea.ObservationPurpose.downstreamState"),
            ],
        );
        register_immutable_class(
            DIAGNOSTIC_KIND_CLASS,
            "runmat.fea.diagnostic-kind",
            VALUE_FIELD,
            &[
                ("convergence", "fea.DiagnosticKind.convergence"),
                ("balance", "fea.DiagnosticKind.balance"),
                ("constraint", "fea.DiagnosticKind.constraint"),
                ("contact", "fea.DiagnosticKind.contact"),
                ("adaptation", "fea.DiagnosticKind.adaptation"),
                ("resource", "fea.DiagnosticKind.resource"),
                (
                    "executionEquivalence",
                    "fea.DiagnosticKind.executionEquivalence",
                ),
            ],
        );
    });
}

const QUANTITY_METHODS: [(&str, &str); 37] = [
    ("displacement", "fea.ObservationQuantity.displacement"),
    ("rotation", "fea.ObservationQuantity.rotation"),
    ("velocity", "fea.ObservationQuantity.velocity"),
    ("acceleration", "fea.ObservationQuantity.acceleration"),
    ("strain", "fea.ObservationQuantity.strain"),
    ("stress", "fea.ObservationQuantity.stress"),
    ("reactionForce", "fea.ObservationQuantity.reactionForce"),
    ("reactionMoment", "fea.ObservationQuantity.reactionMoment"),
    ("strainEnergy", "fea.ObservationQuantity.strainEnergy"),
    ("temperature", "fea.ObservationQuantity.temperature"),
    (
        "temperatureGradient",
        "fea.ObservationQuantity.temperatureGradient",
    ),
    ("heatFlux", "fea.ObservationQuantity.heatFlux"),
    ("heatRate", "fea.ObservationQuantity.heatRate"),
    (
        "electricPotential",
        "fea.ObservationQuantity.electricPotential",
    ),
    ("electricField", "fea.ObservationQuantity.electricField"),
    ("electricCurrent", "fea.ObservationQuantity.electricCurrent"),
    (
        "electricCurrentDensity",
        "fea.ObservationQuantity.electricCurrentDensity",
    ),
    ("electricCharge", "fea.ObservationQuantity.electricCharge"),
    (
        "magneticVectorPotential",
        "fea.ObservationQuantity.magneticVectorPotential",
    ),
    (
        "magneticFluxDensity",
        "fea.ObservationQuantity.magneticFluxDensity",
    ),
    (
        "magneticFieldIntensity",
        "fea.ObservationQuantity.magneticFieldIntensity",
    ),
    (
        "acousticPressure",
        "fea.ObservationQuantity.acousticPressure",
    ),
    (
        "acousticParticleVelocity",
        "fea.ObservationQuantity.acousticParticleVelocity",
    ),
    (
        "soundPressureLevel",
        "fea.ObservationQuantity.soundPressureLevel",
    ),
    ("fluidPressure", "fea.ObservationQuantity.fluidPressure"),
    ("fluidVelocity", "fea.ObservationQuantity.fluidVelocity"),
    ("vorticity", "fea.ObservationQuantity.vorticity"),
    ("massFlowRate", "fea.ObservationQuantity.massFlowRate"),
    ("wallShearStress", "fea.ObservationQuantity.wallShearStress"),
    ("modalFrequency", "fea.ObservationQuantity.modalFrequency"),
    (
        "modalParticipation",
        "fea.ObservationQuantity.modalParticipation",
    ),
    ("eigenvalue", "fea.ObservationQuantity.eigenvalue"),
    ("residual", "fea.ObservationQuantity.residual"),
    ("balance", "fea.ObservationQuantity.balance"),
    (
        "constraintViolation",
        "fea.ObservationQuantity.constraintViolation",
    ),
    ("contactGap", "fea.ObservationQuantity.contactGap"),
    ("contactPressure", "fea.ObservationQuantity.contactPressure"),
];
const FIELD_LOCATION_METHODS: [(&str, &str); 13] = [
    ("node", "fea.FieldLocation.node"),
    ("point", "fea.FieldLocation.point"),
    ("cell", "fea.FieldLocation.cell"),
    ("element", "fea.FieldLocation.element"),
    ("integrationPoint", "fea.FieldLocation.integrationPoint"),
    ("face", "fea.FieldLocation.face"),
    ("edge", "fea.FieldLocation.edge"),
    ("interface", "fea.FieldLocation.interface"),
    ("connector", "fea.FieldLocation.connector"),
    ("constraint", "fea.FieldLocation.constraint"),
    ("referencePoint", "fea.FieldLocation.referencePoint"),
    ("mode", "fea.FieldLocation.mode"),
    ("globalLocation", "fea.FieldLocation.globalLocation"),
];
const RECOVERY_METHODS: [(&str, &str); 6] = [
    ("native", "fea.RecoveryPolicy.native"),
    ("extrapolate", "fea.RecoveryPolicy.extrapolate"),
    (
        "volumeWeightedAverage",
        "fea.RecoveryPolicy.volumeWeightedAverage",
    ),
    (
        "areaWeightedAverage",
        "fea.RecoveryPolicy.areaWeightedAverage",
    ),
    (
        "conservativeProjection",
        "fea.RecoveryPolicy.conservativeProjection",
    ),
    (
        "shapeFunctionInterpolation",
        "fea.RecoveryPolicy.shapeFunctionInterpolation",
    ),
];
const COMPONENT_METHODS: [(&str, &str); 14] = [
    ("all", "fea.ObservationComponent.all"),
    ("x", "fea.ObservationComponent.x"),
    ("y", "fea.ObservationComponent.y"),
    ("z", "fea.ObservationComponent.z"),
    ("normal", "fea.ObservationComponent.normal"),
    ("tangent1", "fea.ObservationComponent.tangent1"),
    ("tangent2", "fea.ObservationComponent.tangent2"),
    ("trace", "fea.ObservationComponent.trace"),
    ("hydrostatic", "fea.ObservationComponent.hydrostatic"),
    ("deviatoric", "fea.ObservationComponent.deviatoric"),
    ("vonMises", "fea.ObservationComponent.vonMises"),
    (
        "maximumPrincipal",
        "fea.ObservationComponent.maximumPrincipal",
    ),
    (
        "minimumPrincipal",
        "fea.ObservationComponent.minimumPrincipal",
    ),
    ("magnitude", "fea.ObservationComponent.magnitude"),
];
const SIGN_METHODS: [(&str, &str); 5] = [
    ("native", "fea.SignConvention.native"),
    ("tensionPositive", "fea.SignConvention.tensionPositive"),
    (
        "compressionPositive",
        "fea.SignConvention.compressionPositive",
    ),
    ("outwardPositive", "fea.SignConvention.outwardPositive"),
    ("inwardPositive", "fea.SignConvention.inwardPositive"),
];
const COMPLEX_METHODS: [(&str, &str); 5] = [
    ("real", "fea.ComplexRepresentation.real"),
    ("imaginary", "fea.ComplexRepresentation.imaginary"),
    ("magnitude", "fea.ComplexRepresentation.magnitude"),
    ("phaseRadians", "fea.ComplexRepresentation.phaseRadians"),
    ("complexValue", "fea.ComplexRepresentation.complexValue"),
];
const SPATIAL_METHODS: [(&str, &str); 9] = [
    ("none", "fea.SpatialReduction.none"),
    ("integral", "fea.SpatialReduction.integral"),
    ("average", "fea.SpatialReduction.average"),
    ("rootMeanSquare", "fea.SpatialReduction.rootMeanSquare"),
    ("euclideanNorm", "fea.SpatialReduction.euclideanNorm"),
    ("minimum", "fea.SpatialReduction.minimum"),
    ("maximum", "fea.SpatialReduction.maximum"),
    ("resultant", "fea.SpatialReduction.resultant"),
    ("moment", "fea.SpatialReduction.moment"),
];
const SERIES_METHODS: [(&str, &str); 8] = [
    ("none", "fea.SeriesReduction.none"),
    ("minimum", "fea.SeriesReduction.minimum"),
    ("maximum", "fea.SeriesReduction.maximum"),
    ("absoluteMaximum", "fea.SeriesReduction.absoluteMaximum"),
    ("rootMeanSquare", "fea.SeriesReduction.rootMeanSquare"),
    ("mean", "fea.SeriesReduction.mean"),
    ("variance", "fea.SeriesReduction.variance"),
    ("envelope", "fea.SeriesReduction.envelope"),
];
const SERIES_WEIGHTING_METHODS: [(&str, &str); 2] = [
    ("uniformSamples", "fea.SeriesWeighting.uniformSamples"),
    (
        "trapezoidalIndependentDomain",
        "fea.SeriesWeighting.trapezoidalIndependentDomain",
    ),
];

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use super::*;

    #[test]
    fn point_field_location_constructor_preserves_the_typed_location() {
        let value = block_on(location_point()).unwrap();
        assert_eq!(
            field_location("test", &value).unwrap(),
            FieldLocation::Point
        );
    }

    #[test]
    fn constraint_field_location_constructor_preserves_the_typed_location() {
        let value = block_on(location_constraint()).unwrap();
        assert_eq!(
            field_location("test", &value).unwrap(),
            FieldLocation::Constraint
        );
    }

    #[test]
    fn series_statistics_preserve_explicit_typed_weighting() {
        let uniform = block_on(series_weighting_uniform()).unwrap();
        let trapezoidal = block_on(series_weighting_trapezoidal()).unwrap();
        assert_eq!(
            series_reduction("test", &block_on(series_mean(uniform)).unwrap()).unwrap(),
            SeriesReduction::Mean {
                weighting: SeriesWeighting::UniformSamples,
            }
        );
        assert_eq!(
            series_reduction("test", &block_on(series_rms(trapezoidal)).unwrap()).unwrap(),
            SeriesReduction::RootMeanSquare {
                weighting: SeriesWeighting::TrapezoidalIndependentDomain,
            }
        );
    }

    #[test]
    fn moment_reduction_preserves_its_typed_reference_point() {
        let about = Position::new([1.0, -2.0, 3.5]);
        let position = block_on(crate::builtins::fea::values::position_meters_builtin(
            Value::Tensor(Tensor::new_2d(about.meters.to_vec(), 1, 3).unwrap()),
        ))
        .unwrap();
        let reduction = block_on(spatial_moment(position)).unwrap();
        assert_eq!(
            spatial_reduction("test", &reduction).unwrap(),
            SpatialReduction::Moment { about }
        );
    }
}
