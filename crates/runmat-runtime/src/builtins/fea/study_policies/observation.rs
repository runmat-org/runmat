use std::sync::Once;

use runmat_analysis_core::{
    ComplexRepresentation, DiagnosticKind, ExtremaTiePolicy, FieldLocation, ObservationComponent,
    ObservationPurpose, ObservationQuantity, PartialAccessPolicy, Position, RecoveryPolicy,
    SeriesReduction, SeriesWeighting, SignConvention, SpatialReduction,
};

use super::super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, register_immutable_runtime_classes, register_immutable_schemas,
    to_typed_value, ImmutableStudyClass, VALUE_FIELD,
};

use runmat_builtins::fea_contract::policy_types::classes;

pub(super) const QUANTITY_CLASS: &str = classes::OBSERVATION_QUANTITY.display_name();
pub(super) const FIELD_LOCATION_CLASS: &str = classes::FIELD_LOCATION.display_name();
pub(super) const RECOVERY_CLASS: &str = classes::RECOVERY_POLICY.display_name();
pub(super) const COMPONENT_CLASS: &str = classes::OBSERVATION_COMPONENT.display_name();
pub(super) const SIGN_CLASS: &str = classes::SIGN_CONVENTION.display_name();
pub(super) const COMPLEX_CLASS: &str = classes::COMPLEX_REPRESENTATION.display_name();
pub(super) const SPATIAL_REDUCTION_CLASS: &str = classes::SPATIAL_REDUCTION.display_name();
pub(super) const SERIES_REDUCTION_CLASS: &str = classes::SERIES_REDUCTION.display_name();
pub(super) const SERIES_WEIGHTING_CLASS: &str = classes::SERIES_WEIGHTING.display_name();
pub(super) const EXTREMA_TIE_CLASS: &str = classes::EXTREMA_TIE_POLICY.display_name();
pub(super) const PARTIAL_ACCESS_CLASS: &str = classes::PARTIAL_ACCESS_POLICY.display_name();
pub(super) const PURPOSE_CLASS: &str = classes::OBSERVATION_PURPOSE.display_name();
pub(super) const DIAGNOSTIC_KIND_CLASS: &str = classes::DIAGNOSTIC_KIND.display_name();

macro_rules! choice {
    ($function:ident, $builtin:literal, $class:expr, $value:expr) => {
        #[runtime_builtin(name = $builtin, binding_variant = "default", builtin_path = "crate::builtins::fea::study_policies::observation")]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value($builtin, $class, $value)
        }
    };
}

macro_rules! quantity_choices {
    ($(($function:ident, $builtin:literal, $value:ident)),+ $(,)?) => {
        $(choice!($function, $builtin, QUANTITY_CLASS, ObservationQuantity::$value);)+
    };
}

quantity_choices!(
    (
        quantity_displacement,
        "fea.ObservationQuantity.displacement",
        Displacement
    ),
    (
        quantity_rotation,
        "fea.ObservationQuantity.rotation",
        Rotation
    ),
    (
        quantity_velocity,
        "fea.ObservationQuantity.velocity",
        Velocity
    ),
    (
        quantity_acceleration,
        "fea.ObservationQuantity.acceleration",
        Acceleration
    ),
    (quantity_strain, "fea.ObservationQuantity.strain", Strain),
    (quantity_stress, "fea.ObservationQuantity.stress", Stress),
    (
        quantity_von_mises_stress,
        "fea.ObservationQuantity.vonMisesStress",
        VonMisesStress
    ),
    (
        quantity_reaction_force,
        "fea.ObservationQuantity.reactionForce",
        ReactionForce
    ),
    (
        quantity_reaction_moment,
        "fea.ObservationQuantity.reactionMoment",
        ReactionMoment
    ),
    (
        quantity_strain_energy,
        "fea.ObservationQuantity.strainEnergy",
        StrainEnergy
    ),
    (
        quantity_beam_axial_force,
        "fea.ObservationQuantity.beamAxialForce",
        BeamAxialForce
    ),
    (
        quantity_beam_shear_force,
        "fea.ObservationQuantity.beamShearForce",
        BeamShearForce
    ),
    (
        quantity_beam_torsion_moment,
        "fea.ObservationQuantity.beamTorsionMoment",
        BeamTorsionMoment
    ),
    (
        quantity_beam_bending_moment,
        "fea.ObservationQuantity.beamBendingMoment",
        BeamBendingMoment
    ),
    (
        quantity_beam_bending_stress,
        "fea.ObservationQuantity.beamBendingStress",
        BeamBendingStress
    ),
    (
        quantity_beam_torsion_stress,
        "fea.ObservationQuantity.beamTorsionStress",
        BeamTorsionStress
    ),
    (
        quantity_shell_membrane_force,
        "fea.ObservationQuantity.shellMembraneForce",
        ShellMembraneForce
    ),
    (
        quantity_shell_bending_moment,
        "fea.ObservationQuantity.shellBendingMoment",
        ShellBendingMoment
    ),
    (
        quantity_shell_transverse_shear,
        "fea.ObservationQuantity.shellTransverseShear",
        ShellTransverseShear
    ),
    (
        quantity_shell_von_mises,
        "fea.ObservationQuantity.shellVonMises",
        ShellVonMises
    ),
    (
        quantity_temperature,
        "fea.ObservationQuantity.temperature",
        Temperature
    ),
    (
        quantity_temperature_gradient,
        "fea.ObservationQuantity.temperatureGradient",
        TemperatureGradient
    ),
    (
        quantity_heat_flux,
        "fea.ObservationQuantity.heatFlux",
        HeatFlux
    ),
    (
        quantity_heat_rate,
        "fea.ObservationQuantity.heatRate",
        HeatRate
    ),
    (
        quantity_electric_potential,
        "fea.ObservationQuantity.electricPotential",
        ElectricPotential
    ),
    (
        quantity_electric_field,
        "fea.ObservationQuantity.electricField",
        ElectricField
    ),
    (
        quantity_electric_current,
        "fea.ObservationQuantity.electricCurrent",
        ElectricCurrent
    ),
    (
        quantity_electric_current_density,
        "fea.ObservationQuantity.electricCurrentDensity",
        ElectricCurrentDensity
    ),
    (
        quantity_electric_charge,
        "fea.ObservationQuantity.electricCharge",
        ElectricCharge
    ),
    (
        quantity_magnetic_vector_potential,
        "fea.ObservationQuantity.magneticVectorPotential",
        MagneticVectorPotential
    ),
    (
        quantity_magnetic_flux_density,
        "fea.ObservationQuantity.magneticFluxDensity",
        MagneticFluxDensity
    ),
    (
        quantity_magnetic_field_intensity,
        "fea.ObservationQuantity.magneticFieldIntensity",
        MagneticFieldIntensity
    ),
    (
        quantity_acoustic_pressure,
        "fea.ObservationQuantity.acousticPressure",
        AcousticPressure
    ),
    (
        quantity_acoustic_particle_velocity,
        "fea.ObservationQuantity.acousticParticleVelocity",
        AcousticParticleVelocity
    ),
    (
        quantity_sound_pressure_level,
        "fea.ObservationQuantity.soundPressureLevel",
        SoundPressureLevel
    ),
    (
        quantity_fluid_pressure,
        "fea.ObservationQuantity.fluidPressure",
        FluidPressure
    ),
    (
        quantity_fluid_velocity,
        "fea.ObservationQuantity.fluidVelocity",
        FluidVelocity
    ),
    (
        quantity_vorticity,
        "fea.ObservationQuantity.vorticity",
        Vorticity
    ),
    (
        quantity_mass_flow_rate,
        "fea.ObservationQuantity.massFlowRate",
        MassFlowRate
    ),
    (
        quantity_wall_shear_stress,
        "fea.ObservationQuantity.wallShearStress",
        WallShearStress
    ),
    (
        quantity_modal_frequency,
        "fea.ObservationQuantity.modalFrequency",
        ModalFrequency
    ),
    (
        quantity_modal_participation,
        "fea.ObservationQuantity.modalParticipation",
        ModalParticipation
    ),
    (
        quantity_eigenvalue,
        "fea.ObservationQuantity.eigenvalue",
        Eigenvalue
    ),
    (
        quantity_residual,
        "fea.ObservationQuantity.residual",
        Residual
    ),
    (quantity_balance, "fea.ObservationQuantity.balance", Balance),
    (
        quantity_constraint_violation,
        "fea.ObservationQuantity.constraintViolation",
        ConstraintViolation
    ),
    (
        quantity_contact_gap,
        "fea.ObservationQuantity.contactGap",
        ContactGap
    ),
    (
        quantity_contact_pressure,
        "fea.ObservationQuantity.contactPressure",
        ContactPressure
    ),
);

macro_rules! choices {
    ($class:expr, $(($function:ident, $builtin:literal, $value:expr)),+ $(,)?) => {
        $(choice!($function, $builtin, $class, $value);)+
    };
}

choices!(
    FIELD_LOCATION_CLASS,
    (location_node, "fea.FieldLocation.node", FieldLocation::Node),
    (
        location_point,
        "fea.FieldLocation.point",
        FieldLocation::Point
    ),
    (location_cell, "fea.FieldLocation.cell", FieldLocation::Cell),
    (
        location_element,
        "fea.FieldLocation.element",
        FieldLocation::Element
    ),
    (
        location_integration_point,
        "fea.FieldLocation.integrationPoint",
        FieldLocation::IntegrationPoint
    ),
    (location_face, "fea.FieldLocation.face", FieldLocation::Face),
    (location_edge, "fea.FieldLocation.edge", FieldLocation::Edge),
    (
        location_interface,
        "fea.FieldLocation.interface",
        FieldLocation::Interface
    ),
    (
        location_connector,
        "fea.FieldLocation.connector",
        FieldLocation::Connector
    ),
    (
        location_constraint,
        "fea.FieldLocation.constraint",
        FieldLocation::Constraint
    ),
    (
        location_reference_point,
        "fea.FieldLocation.referencePoint",
        FieldLocation::ReferencePoint
    ),
    (location_mode, "fea.FieldLocation.mode", FieldLocation::Mode),
    (
        location_global,
        "fea.FieldLocation.globalLocation",
        FieldLocation::Global
    )
);
choices!(
    RECOVERY_CLASS,
    (
        recovery_native,
        "fea.RecoveryPolicy.native",
        RecoveryPolicy::Native
    ),
    (
        recovery_extrapolate,
        "fea.RecoveryPolicy.extrapolate",
        RecoveryPolicy::Extrapolate
    ),
    (
        recovery_volume_average,
        "fea.RecoveryPolicy.volumeWeightedAverage",
        RecoveryPolicy::VolumeWeightedAverage
    ),
    (
        recovery_area_average,
        "fea.RecoveryPolicy.areaWeightedAverage",
        RecoveryPolicy::AreaWeightedAverage
    ),
    (
        recovery_conservative,
        "fea.RecoveryPolicy.conservativeProjection",
        RecoveryPolicy::ConservativeProjection
    ),
    (
        recovery_shape_function,
        "fea.RecoveryPolicy.shapeFunctionInterpolation",
        RecoveryPolicy::ShapeFunctionInterpolation
    )
);
choices!(
    COMPONENT_CLASS,
    (
        component_all,
        "fea.ObservationComponent.all",
        ObservationComponent::All
    ),
    (
        component_x,
        "fea.ObservationComponent.x",
        ObservationComponent::X
    ),
    (
        component_y,
        "fea.ObservationComponent.y",
        ObservationComponent::Y
    ),
    (
        component_z,
        "fea.ObservationComponent.z",
        ObservationComponent::Z
    ),
    (
        component_normal,
        "fea.ObservationComponent.normal",
        ObservationComponent::Normal
    ),
    (
        component_tangent1,
        "fea.ObservationComponent.tangent1",
        ObservationComponent::Tangent1
    ),
    (
        component_tangent2,
        "fea.ObservationComponent.tangent2",
        ObservationComponent::Tangent2
    ),
    (
        component_trace,
        "fea.ObservationComponent.trace",
        ObservationComponent::Trace
    ),
    (
        component_hydrostatic,
        "fea.ObservationComponent.hydrostatic",
        ObservationComponent::Hydrostatic
    ),
    (
        component_deviatoric,
        "fea.ObservationComponent.deviatoric",
        ObservationComponent::Deviatoric
    ),
    (
        component_von_mises,
        "fea.ObservationComponent.vonMises",
        ObservationComponent::VonMises
    ),
    (
        component_max_principal,
        "fea.ObservationComponent.maximumPrincipal",
        ObservationComponent::MaximumPrincipal
    ),
    (
        component_min_principal,
        "fea.ObservationComponent.minimumPrincipal",
        ObservationComponent::MinimumPrincipal
    ),
    (
        component_magnitude,
        "fea.ObservationComponent.magnitude",
        ObservationComponent::Magnitude
    )
);
choices!(
    SIGN_CLASS,
    (
        sign_native,
        "fea.SignConvention.native",
        SignConvention::Native
    ),
    (
        sign_tension,
        "fea.SignConvention.tensionPositive",
        SignConvention::TensionPositive
    ),
    (
        sign_compression,
        "fea.SignConvention.compressionPositive",
        SignConvention::CompressionPositive
    ),
    (
        sign_outward,
        "fea.SignConvention.outwardPositive",
        SignConvention::OutwardPositive
    ),
    (
        sign_inward,
        "fea.SignConvention.inwardPositive",
        SignConvention::InwardPositive
    )
);
choices!(
    COMPLEX_CLASS,
    (
        complex_real,
        "fea.ComplexRepresentation.real",
        ComplexRepresentation::Real
    ),
    (
        complex_imaginary,
        "fea.ComplexRepresentation.imaginary",
        ComplexRepresentation::Imaginary
    ),
    (
        complex_magnitude,
        "fea.ComplexRepresentation.magnitude",
        ComplexRepresentation::Magnitude
    ),
    (
        complex_phase,
        "fea.ComplexRepresentation.phaseRadians",
        ComplexRepresentation::PhaseRadians
    ),
    (
        complex_complex,
        "fea.ComplexRepresentation.complexValue",
        ComplexRepresentation::Complex
    )
);
choices!(
    SPATIAL_REDUCTION_CLASS,
    (
        spatial_none,
        "fea.SpatialReduction.none",
        SpatialReduction::None
    ),
    (
        spatial_integral,
        "fea.SpatialReduction.integral",
        SpatialReduction::Integral
    ),
    (
        spatial_average,
        "fea.SpatialReduction.average",
        SpatialReduction::Average
    ),
    (
        spatial_rms,
        "fea.SpatialReduction.rootMeanSquare",
        SpatialReduction::RootMeanSquare
    ),
    (
        spatial_norm,
        "fea.SpatialReduction.euclideanNorm",
        SpatialReduction::EuclideanNorm
    ),
    (
        spatial_minimum,
        "fea.SpatialReduction.minimum",
        SpatialReduction::Minimum
    ),
    (
        spatial_maximum,
        "fea.SpatialReduction.maximum",
        SpatialReduction::Maximum
    ),
    (
        spatial_resultant,
        "fea.SpatialReduction.resultant",
        SpatialReduction::Resultant
    )
);

#[runtime_builtin(
    name = "fea.SpatialReduction.moment",
    binding_variant = "default",
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
        "fea.SeriesReduction.none",
        SeriesReduction::None
    ),
    (
        series_minimum,
        "fea.SeriesReduction.minimum",
        SeriesReduction::Minimum
    ),
    (
        series_maximum,
        "fea.SeriesReduction.maximum",
        SeriesReduction::Maximum
    ),
    (
        series_absolute_maximum,
        "fea.SeriesReduction.absoluteMaximum",
        SeriesReduction::AbsoluteMaximum
    ),
    (
        series_envelope,
        "fea.SeriesReduction.envelope",
        SeriesReduction::Envelope
    )
);
choices!(
    SERIES_WEIGHTING_CLASS,
    (
        series_weighting_uniform,
        "fea.SeriesWeighting.uniformSamples",
        SeriesWeighting::UniformSamples
    ),
    (
        series_weighting_trapezoidal,
        "fea.SeriesWeighting.trapezoidalIndependentDomain",
        SeriesWeighting::TrapezoidalIndependentDomain
    )
);

macro_rules! series_statistic {
    ($function:ident, $builtin:literal, $variant:ident) => {
        #[runtime_builtin(name = $builtin, binding_variant = "default", builtin_path = "crate::builtins::fea::study_policies::observation")]
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
    "fea.SeriesReduction.rootMeanSquare",
    RootMeanSquare
);
series_statistic!(series_mean, "fea.SeriesReduction.mean", Mean);
series_statistic!(series_variance, "fea.SeriesReduction.variance", Variance);
choices!(
    EXTREMA_TIE_CLASS,
    (
        tie_first,
        "fea.ExtremaTiePolicy.firstCanonical",
        ExtremaTiePolicy::FirstCanonical
    ),
    (
        tie_all,
        "fea.ExtremaTiePolicy.allLocations",
        ExtremaTiePolicy::AllLocations
    )
);
choices!(
    PARTIAL_ACCESS_CLASS,
    (
        partial_chunks,
        "fea.PartialAccessPolicy.manifestAndChunks",
        PartialAccessPolicy::ManifestAndChunks
    ),
    (
        partial_whole,
        "fea.PartialAccessPolicy.wholeArtifact",
        PartialAccessPolicy::WholeArtifact
    )
);
choices!(
    PURPOSE_CLASS,
    (
        purpose_publication,
        "fea.ObservationPurpose.publication",
        ObservationPurpose::Publication
    ),
    (
        purpose_solver,
        "fea.ObservationPurpose.solverControl",
        ObservationPurpose::SolverControl
    ),
    (
        purpose_visualization,
        "fea.ObservationPurpose.visualization",
        ObservationPurpose::Visualization
    ),
    (
        purpose_downstream,
        "fea.ObservationPurpose.downstreamState",
        ObservationPurpose::DownstreamState
    )
);
choices!(
    DIAGNOSTIC_KIND_CLASS,
    (
        diagnostic_convergence,
        "fea.DiagnosticKind.convergence",
        DiagnosticKind::Convergence
    ),
    (
        diagnostic_balance,
        "fea.DiagnosticKind.balance",
        DiagnosticKind::Balance
    ),
    (
        diagnostic_constraint,
        "fea.DiagnosticKind.constraint",
        DiagnosticKind::Constraint
    ),
    (
        diagnostic_contact,
        "fea.DiagnosticKind.contact",
        DiagnosticKind::Contact
    ),
    (
        diagnostic_adaptation,
        "fea.DiagnosticKind.adaptation",
        DiagnosticKind::Adaptation
    ),
    (
        diagnostic_resource,
        "fea.DiagnosticKind.resource",
        DiagnosticKind::Resource
    ),
    (
        diagnostic_equivalence,
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

const OBSERVATION_POLICY_CLASSES: [ImmutableStudyClass; 13] = [
    ImmutableStudyClass {
        class_name: QUANTITY_CLASS,
        type_identity: "runmat.fea.observation-quantity",
        fields: VALUE_FIELD,
        methods: &QUANTITY_METHODS,
    },
    ImmutableStudyClass {
        class_name: FIELD_LOCATION_CLASS,
        type_identity: "runmat.fea.field-location",
        fields: VALUE_FIELD,
        methods: &FIELD_LOCATION_METHODS,
    },
    ImmutableStudyClass {
        class_name: RECOVERY_CLASS,
        type_identity: "runmat.fea.recovery-policy",
        fields: VALUE_FIELD,
        methods: &RECOVERY_METHODS,
    },
    ImmutableStudyClass {
        class_name: COMPONENT_CLASS,
        type_identity: "runmat.fea.observation-component",
        fields: VALUE_FIELD,
        methods: &COMPONENT_METHODS,
    },
    ImmutableStudyClass {
        class_name: SIGN_CLASS,
        type_identity: "runmat.fea.sign-convention",
        fields: VALUE_FIELD,
        methods: &SIGN_METHODS,
    },
    ImmutableStudyClass {
        class_name: COMPLEX_CLASS,
        type_identity: "runmat.fea.complex-representation",
        fields: VALUE_FIELD,
        methods: &COMPLEX_METHODS,
    },
    ImmutableStudyClass {
        class_name: SPATIAL_REDUCTION_CLASS,
        type_identity: "runmat.fea.spatial-reduction",
        fields: VALUE_FIELD,
        methods: &SPATIAL_METHODS,
    },
    ImmutableStudyClass {
        class_name: SERIES_REDUCTION_CLASS,
        type_identity: "runmat.fea.series-reduction",
        fields: VALUE_FIELD,
        methods: &SERIES_METHODS,
    },
    ImmutableStudyClass {
        class_name: SERIES_WEIGHTING_CLASS,
        type_identity: "runmat.fea.series-weighting",
        fields: VALUE_FIELD,
        methods: &SERIES_WEIGHTING_METHODS,
    },
    ImmutableStudyClass {
        class_name: EXTREMA_TIE_CLASS,
        type_identity: "runmat.fea.extrema-tie-policy",
        fields: VALUE_FIELD,
        methods: &[
            ("firstCanonical", "fea.ExtremaTiePolicy.firstCanonical"),
            ("allLocations", "fea.ExtremaTiePolicy.allLocations"),
        ],
    },
    ImmutableStudyClass {
        class_name: PARTIAL_ACCESS_CLASS,
        type_identity: "runmat.fea.partial-access-policy",
        fields: VALUE_FIELD,
        methods: &[
            (
                "manifestAndChunks",
                "fea.PartialAccessPolicy.manifestAndChunks",
            ),
            ("wholeArtifact", "fea.PartialAccessPolicy.wholeArtifact"),
        ],
    },
    ImmutableStudyClass {
        class_name: PURPOSE_CLASS,
        type_identity: "runmat.fea.observation-purpose",
        fields: VALUE_FIELD,
        methods: &[
            ("publication", "fea.ObservationPurpose.publication"),
            ("solverControl", "fea.ObservationPurpose.solverControl"),
            ("visualization", "fea.ObservationPurpose.visualization"),
            ("downstreamState", "fea.ObservationPurpose.downstreamState"),
        ],
    },
    ImmutableStudyClass {
        class_name: DIAGNOSTIC_KIND_CLASS,
        type_identity: "runmat.fea.diagnostic-kind",
        fields: VALUE_FIELD,
        methods: &[
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
    },
];

pub(super) fn ensure_classes_registered() {
    static SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(classes::OBSERVATION_QUANTITY);
    SCHEMAS.call_once(|| {
        static FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        register_immutable_schemas(&FIELDS, &OBSERVATION_POLICY_CLASSES);
    });
    RUNTIME_CLASSES.ensure(|| register_immutable_runtime_classes(&OBSERVATION_POLICY_CLASSES));
}

const QUANTITY_METHODS: [(&str, &str); 48] = [
    ("displacement", "fea.ObservationQuantity.displacement"),
    ("rotation", "fea.ObservationQuantity.rotation"),
    ("velocity", "fea.ObservationQuantity.velocity"),
    ("acceleration", "fea.ObservationQuantity.acceleration"),
    ("strain", "fea.ObservationQuantity.strain"),
    ("stress", "fea.ObservationQuantity.stress"),
    ("vonMisesStress", "fea.ObservationQuantity.vonMisesStress"),
    ("reactionForce", "fea.ObservationQuantity.reactionForce"),
    ("reactionMoment", "fea.ObservationQuantity.reactionMoment"),
    ("strainEnergy", "fea.ObservationQuantity.strainEnergy"),
    ("beamAxialForce", "fea.ObservationQuantity.beamAxialForce"),
    ("beamShearForce", "fea.ObservationQuantity.beamShearForce"),
    (
        "beamTorsionMoment",
        "fea.ObservationQuantity.beamTorsionMoment",
    ),
    (
        "beamBendingMoment",
        "fea.ObservationQuantity.beamBendingMoment",
    ),
    (
        "beamBendingStress",
        "fea.ObservationQuantity.beamBendingStress",
    ),
    (
        "beamTorsionStress",
        "fea.ObservationQuantity.beamTorsionStress",
    ),
    (
        "shellMembraneForce",
        "fea.ObservationQuantity.shellMembraneForce",
    ),
    (
        "shellBendingMoment",
        "fea.ObservationQuantity.shellBendingMoment",
    ),
    (
        "shellTransverseShear",
        "fea.ObservationQuantity.shellTransverseShear",
    ),
    ("shellVonMises", "fea.ObservationQuantity.shellVonMises"),
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
    fn structural_recovery_quantity_constructors_preserve_every_typed_value() {
        let values = [
            (
                block_on(quantity_von_mises_stress()).unwrap(),
                ObservationQuantity::VonMisesStress,
            ),
            (
                block_on(quantity_beam_axial_force()).unwrap(),
                ObservationQuantity::BeamAxialForce,
            ),
            (
                block_on(quantity_beam_shear_force()).unwrap(),
                ObservationQuantity::BeamShearForce,
            ),
            (
                block_on(quantity_beam_torsion_moment()).unwrap(),
                ObservationQuantity::BeamTorsionMoment,
            ),
            (
                block_on(quantity_beam_bending_moment()).unwrap(),
                ObservationQuantity::BeamBendingMoment,
            ),
            (
                block_on(quantity_beam_bending_stress()).unwrap(),
                ObservationQuantity::BeamBendingStress,
            ),
            (
                block_on(quantity_beam_torsion_stress()).unwrap(),
                ObservationQuantity::BeamTorsionStress,
            ),
            (
                block_on(quantity_shell_membrane_force()).unwrap(),
                ObservationQuantity::ShellMembraneForce,
            ),
            (
                block_on(quantity_shell_bending_moment()).unwrap(),
                ObservationQuantity::ShellBendingMoment,
            ),
            (
                block_on(quantity_shell_transverse_shear()).unwrap(),
                ObservationQuantity::ShellTransverseShear,
            ),
            (
                block_on(quantity_shell_von_mises()).unwrap(),
                ObservationQuantity::ShellVonMises,
            ),
        ];
        for (value, expected) in values {
            assert_eq!(quantity("test", &value).unwrap(), expected);
        }
    }

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
            Value::Tensor(Tensor::new_2d(about.canonical_value().to_vec(), 1, 3).unwrap()),
        ))
        .unwrap();
        let reduction = block_on(spatial_moment(position)).unwrap();
        assert_eq!(
            spatial_reduction("test", &reduction).unwrap(),
            SpatialReduction::Moment { about }
        );
    }
}
