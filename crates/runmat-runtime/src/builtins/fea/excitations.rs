use std::collections::HashMap;
use std::sync::Once;

use super::*;
use crate::builtins::fea::constraints::require_geometry_binding;
use crate::builtins::geometry::region::region_ref_from_value;
use runmat_analysis_core::{
    AnalysisExcitation, ExcitationCaseActivation, ExcitationCaseId, ExcitationDistribution,
    ExcitationId, ExcitationKinematics, ProjectionRequirement,
};

pub(super) const STRUCTURAL_EXCITATION_CLASS: &str = "fea.StructuralExcitation";
pub(super) const THERMAL_EXCITATION_CLASS: &str = "fea.ThermalExcitation";
pub(super) const ELECTRICAL_EXCITATION_CLASS: &str = "fea.ElectricalExcitation";
pub(super) const MAGNETIC_EXCITATION_CLASS: &str = "fea.MagneticExcitation";
pub(super) const ACOUSTIC_EXCITATION_CLASS: &str = "fea.AcousticExcitation";
pub(super) const FLUID_EXCITATION_CLASS: &str = "fea.FluidExcitation";
pub(super) const EXCITATION_CASE_CLASS: &str = "fea.ExcitationCase";

const EXCITATION_FIELDS: &[&str] = &["geometry_id", "geometry_revision", "action"];
const CASE_FIELDS: &[&str] = &["case_id", "action_ids", "activation"];
const TYPED_EXCITATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Typed excitation constructors accept identifiers, geometry.Region values, and typed signal objects; physical numeric conversion belongs to the signal constructor.",
    };

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GeometryBoundExcitation {
    pub(super) geometry_id: String,
    pub(super) geometry_revision: u32,
    pub(super) action: ExcitationAction,
}

impl GeometryBoundExcitation {
    pub(super) fn require_geometry(
        &self,
        builtin: &'static str,
        geometry: &GeometryAsset,
    ) -> BuiltinResult<()> {
        require_geometry_binding(
            builtin,
            "excitation",
            &self.geometry_id,
            self.geometry_revision,
            geometry,
        )
    }

    fn require_class(&self, builtin: &'static str, class_name: &'static str) -> BuiltinResult<()> {
        let expected = action_class(&self.action);
        if class_name != expected {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("excitation action belongs to `{expected}`, not `{class_name}`"),
            ));
        }
        Ok(())
    }
}

const ACTION_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Stable excitation identifier.",
    ),
    required(
        "caseId",
        BuiltinParamType::StringScalar,
        "Identifier of the containing excitation case.",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "Typed geometry.Region target.",
    ),
    required(
        "signal",
        BuiltinParamType::Any,
        "Typed physical signal with the quantity required by this action.",
    ),
];
const CASE_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Stable excitation-case identifier.",
    ),
    required(
        "actions",
        BuiltinParamType::Any,
        "Typed excitation object or cell array of typed excitations.",
    ),
];
const STEP_CASE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Stable excitation-case identifier.",
    ),
    required(
        "actions",
        BuiltinParamType::Any,
        "Typed excitation object or cell array of typed excitations.",
    ),
    required(
        "stepIds",
        BuiltinParamType::Any,
        "Nonempty string or cell array of procedure-step identifiers.",
    ),
];
const GROUP_CASE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Stable excitation-case identifier.",
    ),
    required(
        "actions",
        BuiltinParamType::Any,
        "Typed excitation object or cell array of typed excitations.",
    ),
    required(
        "groupId",
        BuiltinParamType::StringScalar,
        "Mutually-exclusive combination-group identifier.",
    ),
];

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

fn action_to_object(
    builtin: &'static str,
    class_name: &'static str,
    value: GeometryBoundExcitation,
) -> BuiltinResult<Value> {
    let revision = value.geometry_revision;
    let mut object = serializable_to_object_value(builtin, &ERROR_INTERNAL, class_name, &value)?;
    object.properties.insert(
        "geometry_revision".into(),
        Value::Int(IntValue::U32(revision)),
    );
    Ok(Value::Object(object))
}

#[derive(Clone, Copy)]
enum RegionExpectation {
    Boundary,
    Volume,
}

fn create_action(
    builtin: &'static str,
    class_name: &'static str,
    id: String,
    case_id: String,
    region: &Value,
    signal: &Value,
    expected_quantity: runmat_analysis_core::PhysicalQuantityKind,
    region_expectation: RegionExpectation,
    distribution: ExcitationDistribution,
    preserve_moment: bool,
    wrap: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) -> BuiltinResult<Value> {
    if id.trim().is_empty() || case_id.trim().is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "excitation and case identifiers must not be empty",
        ));
    }
    let region = region_ref_from_value(builtin, region)?;
    let signal = super::signals::physical_signal_from_value(builtin, signal)?;
    if signal.quantity() != expected_quantity {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "{builtin} requires a {expected_quantity:?} signal, got {:?}",
                signal.quantity()
            ),
        ));
    }
    let frame = signal.frame().cloned().unwrap_or(CoordinateFrame::Global);
    let target = match region_expectation {
        RegionExpectation::Boundary => RegionTarget::boundary(region.region_id),
        RegionExpectation::Volume => RegionTarget::volume(region.region_id),
    };
    let action = wrap(AnalysisExcitation {
        excitation_id: ExcitationId(id),
        case_id: ExcitationCaseId(case_id),
        target: ExcitationTarget::Region(target),
        frame,
        direction: None,
        distribution,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: signal.signal_id().clone(),
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: distribution == ExcitationDistribution::PreserveResultant,
            preserve_moment,
            relative_tolerance: 1.0e-10,
        },
    });
    action_to_object(
        builtin,
        class_name,
        GeometryBoundExcitation {
            geometry_id: region.geometry_id,
            geometry_revision: region.geometry_revision,
            action,
        },
    )
}

macro_rules! action_builtin {
    (
        $function:ident, $descriptor:ident, $signatures:ident,
        $builtin:literal, $method:literal, $summary:literal, $class:expr,
        $quantity:path, $region:expr, $distribution:path, $preserve_moment:expr, $variant:path
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!("action = ", $builtin, "(id, caseId, region, signal)"),
            inputs: &ACTION_INPUTS,
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
                                            summary = $summary,
                                            keywords = "fea,excitation,signal,region,typed",
                                            descriptor($descriptor),
                                            integer_audit(TYPED_EXCITATION_INTEGER_AUDIT),
                                            builtin_path = "crate::builtins::fea::excitations"
                                        )]
        pub async fn $function(
            id: String,
            case_id: String,
            region: Value,
            signal: Value,
        ) -> BuiltinResult<Value> {
            create_action(
                $builtin,
                $class,
                id,
                case_id,
                &region,
                &signal,
                $quantity,
                $region,
                $distribution,
                $preserve_moment,
                $variant,
            )
        }
    };
}

use runmat_analysis_core::PhysicalQuantityKind as Quantity;

action_builtin!(
    structural_force,
    STRUCTURAL_FORCE_DESCRIPTOR,
    STRUCTURAL_FORCE_SIGNATURES,
    "fea.StructuralExcitation.force",
    "force",
    "Bind a force signal to a structural boundary region.",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Force,
    RegionExpectation::Boundary,
    ExcitationDistribution::PreserveResultant,
    false,
    ExcitationAction::Force
);
action_builtin!(
    structural_moment,
    STRUCTURAL_MOMENT_DESCRIPTOR,
    STRUCTURAL_MOMENT_SIGNATURES,
    "fea.StructuralExcitation.moment",
    "moment",
    "Bind a moment signal to a structural boundary region.",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Moment,
    RegionExpectation::Boundary,
    ExcitationDistribution::PreserveResultant,
    true,
    ExcitationAction::Moment
);
action_builtin!(
    structural_traction,
    STRUCTURAL_TRACTION_DESCRIPTOR,
    STRUCTURAL_TRACTION_SIGNATURES,
    "fea.StructuralExcitation.traction",
    "traction",
    "Bind a traction signal to a structural boundary region.",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Traction,
    RegionExpectation::Boundary,
    ExcitationDistribution::IntensityPerArea,
    false,
    ExcitationAction::Traction
);
action_builtin!(
    structural_pressure,
    STRUCTURAL_PRESSURE_DESCRIPTOR,
    STRUCTURAL_PRESSURE_SIGNATURES,
    "fea.StructuralExcitation.pressure",
    "pressure",
    "Bind a pressure signal to a structural boundary region.",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Pressure,
    RegionExpectation::Boundary,
    ExcitationDistribution::IntensityPerArea,
    false,
    ExcitationAction::Pressure
);
action_builtin!(
    structural_body_force,
    STRUCTURAL_BODY_FORCE_DESCRIPTOR,
    STRUCTURAL_BODY_FORCE_SIGNATURES,
    "fea.StructuralExcitation.bodyForce",
    "bodyForce",
    "Bind a body-force-density signal to a structural volume region.",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::BodyForceDensity,
    RegionExpectation::Volume,
    ExcitationDistribution::IntensityPerVolume,
    false,
    ExcitationAction::BodyForce
);
action_builtin!(
    structural_base_displacement,
    STRUCTURAL_BASE_DISPLACEMENT_DESCRIPTOR,
    STRUCTURAL_BASE_DISPLACEMENT_SIGNATURES,
    "fea.StructuralExcitation.baseDisplacement",
    "baseDisplacement",
    "Bind a base-displacement signal to a structural boundary region.",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Displacement,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::BaseDisplacement
);
action_builtin!(
    structural_base_velocity,
    STRUCTURAL_BASE_VELOCITY_DESCRIPTOR,
    STRUCTURAL_BASE_VELOCITY_SIGNATURES,
    "fea.StructuralExcitation.baseVelocity",
    "baseVelocity",
    "Bind a base-velocity signal to a structural boundary region.",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Velocity,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::BaseVelocity
);
action_builtin!(
    structural_base_acceleration,
    STRUCTURAL_BASE_ACCELERATION_DESCRIPTOR,
    STRUCTURAL_BASE_ACCELERATION_SIGNATURES,
    "fea.StructuralExcitation.baseAcceleration",
    "baseAcceleration",
    "Bind a base-acceleration signal to a structural boundary region.",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Acceleration,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::BaseAcceleration
);
action_builtin!(
    thermal_heat_generation,
    THERMAL_HEAT_GENERATION_DESCRIPTOR,
    THERMAL_HEAT_GENERATION_SIGNATURES,
    "fea.ThermalExcitation.heatGeneration",
    "heatGeneration",
    "Bind a volumetric heat-generation signal to a thermal volume region.",
    THERMAL_EXCITATION_CLASS,
    Quantity::HeatGenerationDensity,
    RegionExpectation::Volume,
    ExcitationDistribution::IntensityPerVolume,
    false,
    ExcitationAction::HeatGeneration
);
action_builtin!(
    thermal_heat_flux,
    THERMAL_HEAT_FLUX_DESCRIPTOR,
    THERMAL_HEAT_FLUX_SIGNATURES,
    "fea.ThermalExcitation.heatFlux",
    "heatFlux",
    "Bind a heat-flux signal to a thermal boundary region.",
    THERMAL_EXCITATION_CLASS,
    Quantity::HeatFlux,
    RegionExpectation::Boundary,
    ExcitationDistribution::IntensityPerArea,
    false,
    ExcitationAction::HeatFlux
);
action_builtin!(
    thermal_temperature_field,
    THERMAL_TEMPERATURE_FIELD_DESCRIPTOR,
    THERMAL_TEMPERATURE_FIELD_SIGNATURES,
    "fea.ThermalExcitation.temperatureField",
    "temperatureField",
    "Bind an absolute-temperature signal to a volume region for coupling.",
    THERMAL_EXCITATION_CLASS,
    Quantity::Temperature,
    RegionExpectation::Volume,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::TemperatureField
);
action_builtin!(
    thermal_ambient_temperature,
    THERMAL_AMBIENT_TEMPERATURE_DESCRIPTOR,
    THERMAL_AMBIENT_TEMPERATURE_SIGNATURES,
    "fea.ThermalExcitation.ambientTemperature",
    "ambientTemperature",
    "Bind an ambient-temperature signal to a thermal boundary region.",
    THERMAL_EXCITATION_CLASS,
    Quantity::Temperature,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::AmbientTemperature
);
action_builtin!(
    electrical_terminal_voltage,
    ELECTRICAL_TERMINAL_VOLTAGE_DESCRIPTOR,
    ELECTRICAL_TERMINAL_VOLTAGE_SIGNATURES,
    "fea.ElectricalExcitation.terminalVoltage",
    "terminalVoltage",
    "Bind an electric-potential signal to a terminal region.",
    ELECTRICAL_EXCITATION_CLASS,
    Quantity::ElectricPotential,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::TerminalVoltage
);
action_builtin!(
    electrical_current,
    ELECTRICAL_CURRENT_DESCRIPTOR,
    ELECTRICAL_CURRENT_SIGNATURES,
    "fea.ElectricalExcitation.current",
    "current",
    "Bind an electric-current signal to a conducting volume region.",
    ELECTRICAL_EXCITATION_CLASS,
    Quantity::ElectricCurrent,
    RegionExpectation::Volume,
    ExcitationDistribution::PreserveResultant,
    false,
    ExcitationAction::ElectricCurrent
);
action_builtin!(
    electrical_charge,
    ELECTRICAL_CHARGE_DESCRIPTOR,
    ELECTRICAL_CHARGE_SIGNATURES,
    "fea.ElectricalExcitation.charge",
    "charge",
    "Bind an electric-charge signal to a volume region.",
    ELECTRICAL_EXCITATION_CLASS,
    Quantity::ElectricCharge,
    RegionExpectation::Volume,
    ExcitationDistribution::PreserveResultant,
    false,
    ExcitationAction::ElectricCharge
);
action_builtin!(
    electrical_current_density,
    ELECTRICAL_CURRENT_DENSITY_DESCRIPTOR,
    ELECTRICAL_CURRENT_DENSITY_SIGNATURES,
    "fea.ElectricalExcitation.currentDensity",
    "currentDensity",
    "Bind a current-density signal to a conducting volume region.",
    ELECTRICAL_EXCITATION_CLASS,
    Quantity::ElectricCurrentDensity,
    RegionExpectation::Volume,
    ExcitationDistribution::IntensityPerVolume,
    false,
    ExcitationAction::CurrentDensity
);
action_builtin!(
    magnetic_vector_potential,
    MAGNETIC_VECTOR_POTENTIAL_DESCRIPTOR,
    MAGNETIC_VECTOR_POTENTIAL_SIGNATURES,
    "fea.MagneticExcitation.vectorPotential",
    "vectorPotential",
    "Bind a magnetic-vector-potential signal to a magnetic volume region.",
    MAGNETIC_EXCITATION_CLASS,
    Quantity::MagneticVectorPotential,
    RegionExpectation::Volume,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::MagneticVectorPotential
);
action_builtin!(
    acoustic_pressure,
    ACOUSTIC_PRESSURE_DESCRIPTOR,
    ACOUSTIC_PRESSURE_SIGNATURES,
    "fea.AcousticExcitation.pressure",
    "pressure",
    "Bind an acoustic-pressure signal to an acoustic boundary region.",
    ACOUSTIC_EXCITATION_CLASS,
    Quantity::Pressure,
    RegionExpectation::Boundary,
    ExcitationDistribution::IntensityPerArea,
    false,
    ExcitationAction::AcousticPressure
);
action_builtin!(
    acoustic_normal_velocity,
    ACOUSTIC_NORMAL_VELOCITY_DESCRIPTOR,
    ACOUSTIC_NORMAL_VELOCITY_SIGNATURES,
    "fea.AcousticExcitation.normalVelocity",
    "normalVelocity",
    "Bind a velocity signal to an acoustic boundary normal.",
    ACOUSTIC_EXCITATION_CLASS,
    Quantity::Velocity,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::NormalVelocity
);
action_builtin!(
    fluid_inlet_velocity,
    FLUID_INLET_VELOCITY_DESCRIPTOR,
    FLUID_INLET_VELOCITY_SIGNATURES,
    "fea.FluidExcitation.inletVelocity",
    "inletVelocity",
    "Bind a velocity signal to a fluid inlet region.",
    FLUID_EXCITATION_CLASS,
    Quantity::Velocity,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::InletVelocity
);
action_builtin!(
    fluid_mass_flow,
    FLUID_MASS_FLOW_DESCRIPTOR,
    FLUID_MASS_FLOW_SIGNATURES,
    "fea.FluidExcitation.massFlow",
    "massFlow",
    "Bind a mass-flow signal to a fluid boundary region.",
    FLUID_EXCITATION_CLASS,
    Quantity::MassFlowRate,
    RegionExpectation::Boundary,
    ExcitationDistribution::PreserveResultant,
    false,
    ExcitationAction::MassFlow
);
action_builtin!(
    fluid_wall_motion,
    FLUID_WALL_MOTION_DESCRIPTOR,
    FLUID_WALL_MOTION_SIGNATURES,
    "fea.FluidExcitation.wallMotion",
    "wallMotion",
    "Bind a wall-velocity signal to a fluid boundary region.",
    FLUID_EXCITATION_CLASS,
    Quantity::Velocity,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::WallMotion
);

const ALWAYS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "case = fea.ExcitationCase.always(id, actions)",
    inputs: &CASE_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const ALWAYS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ALWAYS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const DURING_STEPS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "case = fea.ExcitationCase.duringSteps(id, actions, stepIds)",
    inputs: &STEP_CASE_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const DURING_STEPS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURING_STEPS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const EXCLUSIVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "case = fea.ExcitationCase.mutuallyExclusive(id, actions, groupId)",
    inputs: &GROUP_CASE_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const EXCLUSIVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EXCLUSIVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.ExcitationCase.always",
    category = "fea",
    summary = "Create an always-active excitation case.",
    keywords = "fea,excitation,case,activation,typed",
    descriptor(crate::builtins::fea::excitations::ALWAYS_DESCRIPTOR),
    integer_audit(TYPED_EXCITATION_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::excitations"
)]
pub async fn excitation_case_always(id: String, actions: Value) -> BuiltinResult<Value> {
    create_case(id, &actions, ExcitationCaseActivation::Always)
}

#[runtime_builtin(
    name = "fea.ExcitationCase.duringSteps",
    category = "fea",
    summary = "Create an excitation case active during named procedure steps.",
    keywords = "fea,excitation,case,steps,activation,typed",
    descriptor(crate::builtins::fea::excitations::DURING_STEPS_DESCRIPTOR),
    integer_audit(TYPED_EXCITATION_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::excitations"
)]
pub async fn excitation_case_during_steps(
    id: String,
    actions: Value,
    step_ids: Value,
) -> BuiltinResult<Value> {
    let step_ids = string_vec_from_value("fea.ExcitationCase.duringSteps", &step_ids)?;
    create_case(id, &actions, ExcitationCaseActivation::Steps(step_ids))
}

#[runtime_builtin(
    name = "fea.ExcitationCase.mutuallyExclusive",
    category = "fea",
    summary = "Create an excitation case in a mutually-exclusive group.",
    keywords = "fea,excitation,case,group,activation,typed",
    descriptor(crate::builtins::fea::excitations::EXCLUSIVE_DESCRIPTOR),
    integer_audit(TYPED_EXCITATION_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::excitations"
)]
pub async fn excitation_case_mutually_exclusive(
    id: String,
    actions: Value,
    group_id: String,
) -> BuiltinResult<Value> {
    if group_id.trim().is_empty() {
        return Err(builtin_error(
            "fea.ExcitationCase.mutuallyExclusive",
            &ERROR_INPUT,
            "group id must not be empty",
        ));
    }
    create_case(
        id,
        &actions,
        ExcitationCaseActivation::MutuallyExclusiveGroup { group_id },
    )
}

fn create_case(
    id: String,
    actions: &Value,
    activation: ExcitationCaseActivation,
) -> BuiltinResult<Value> {
    let builtin = "fea.ExcitationCase";
    if id.trim().is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "excitation-case id must not be empty",
        ));
    }
    let actions = bound_excitation_vec_from_value(builtin, actions)?;
    if actions.is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "excitation cases require at least one action",
        ));
    }
    let mut action_ids = Vec::with_capacity(actions.len());
    for action in actions {
        let excitation = action.action.excitation();
        if excitation.case_id.0 != id {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!(
                    "excitation `{}` names case `{}`, not `{id}`",
                    excitation.excitation_id.0, excitation.case_id.0
                ),
            ));
        }
        action_ids.push(excitation.excitation_id.clone());
    }
    serializable_to_object(
        builtin,
        &ERROR_INTERNAL,
        EXCITATION_CASE_CLASS,
        &ExcitationCase {
            case_id: ExcitationCaseId(id),
            action_ids,
            activation,
        },
    )
}

pub(super) fn bound_excitation_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<GeometryBoundExcitation>> {
    let values = match value {
        Value::Cell(cell) => cell.data.iter().collect::<Vec<_>>(),
        Value::Object(_) => vec![value],
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("expected a typed FEA excitation object or cell array; got {other:?}"),
            ))
        }
    };
    values
        .into_iter()
        .map(|value| {
            let Value::Object(object) = value else {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!(
                        "excitation cells may contain only typed FEA excitation objects; got {value:?}"
                    ),
                ));
            };
            let class_name = excitation_class_name(&object.class_name).ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported FEA excitation class `{}`", object.class_name),
                )
            })?;
            let payload: GeometryBoundExcitation = object_payload(builtin, value, class_name)?;
            payload.require_class(builtin, class_name)?;
            Ok(payload)
        })
        .collect()
}

pub(super) fn excitation_case_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<ExcitationCase>> {
    object_vec_from_value(builtin, value, EXCITATION_CASE_CLASS)
}

fn excitation_class_name(class_name: &str) -> Option<&'static str> {
    Some(match class_name {
        STRUCTURAL_EXCITATION_CLASS => STRUCTURAL_EXCITATION_CLASS,
        THERMAL_EXCITATION_CLASS => THERMAL_EXCITATION_CLASS,
        ELECTRICAL_EXCITATION_CLASS => ELECTRICAL_EXCITATION_CLASS,
        MAGNETIC_EXCITATION_CLASS => MAGNETIC_EXCITATION_CLASS,
        ACOUSTIC_EXCITATION_CLASS => ACOUSTIC_EXCITATION_CLASS,
        FLUID_EXCITATION_CLASS => FLUID_EXCITATION_CLASS,
        _ => return None,
    })
}

fn action_class(action: &ExcitationAction) -> &'static str {
    match action {
        ExcitationAction::Force(_)
        | ExcitationAction::Moment(_)
        | ExcitationAction::Traction(_)
        | ExcitationAction::Pressure(_)
        | ExcitationAction::BodyForce(_)
        | ExcitationAction::BaseDisplacement(_)
        | ExcitationAction::BaseVelocity(_)
        | ExcitationAction::BaseAcceleration(_) => STRUCTURAL_EXCITATION_CLASS,
        ExcitationAction::HeatGeneration(_)
        | ExcitationAction::HeatFlux(_)
        | ExcitationAction::TemperatureField(_)
        | ExcitationAction::AmbientTemperature(_) => THERMAL_EXCITATION_CLASS,
        ExcitationAction::TerminalVoltage(_)
        | ExcitationAction::ElectricCurrent(_)
        | ExcitationAction::ElectricCharge(_)
        | ExcitationAction::CurrentDensity(_) => ELECTRICAL_EXCITATION_CLASS,
        ExcitationAction::MagneticVectorPotential(_) => MAGNETIC_EXCITATION_CLASS,
        ExcitationAction::AcousticPressure(_) | ExcitationAction::NormalVelocity(_) => {
            ACOUSTIC_EXCITATION_CLASS
        }
        ExcitationAction::InletVelocity(_)
        | ExcitationAction::MassFlow(_)
        | ExcitationAction::WallMotion(_) => FLUID_EXCITATION_CLASS,
    }
}

pub(super) fn ensure_excitation_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(STRUCTURAL_EXCITATION_CLASS);
    PORTABLE_SCHEMAS.call_once(|| {
        for (class_name, type_identity) in excitation_schemas() {
            crate::execution::value_codec::register_immutable_value_class(
                crate::execution::value_codec::ImmutableValueClassSchema {
                    class_name,
                    type_identity,
                    schema_version: 1,
                    allowed_fields: EXCITATION_FIELDS,
                    required_fields: EXCITATION_FIELDS,
                    validate: validate_excitation_object,
                },
            )
            .unwrap_or_else(|error| {
                panic!("failed to register immutable FEA excitation class {class_name}: {error}")
            });
        }
        crate::execution::value_codec::register_immutable_value_class(
            crate::execution::value_codec::ImmutableValueClassSchema {
                class_name: EXCITATION_CASE_CLASS,
                type_identity: "runmat.fea.excitation-case",
                schema_version: 1,
                allowed_fields: CASE_FIELDS,
                required_fields: CASE_FIELDS,
                validate: validate_case_object,
            },
        )
        .unwrap_or_else(|error| panic!("failed to register {EXCITATION_CASE_CLASS}: {error}"));
    });
    RUNTIME_CLASSES.ensure(|| {
        for (class_name, _) in excitation_schemas() {
            crate::class_registry::register_class(crate::class_registry::RuntimeClass {
                name: class_name.into(),
                parent: None,
                properties: immutable_properties(EXCITATION_FIELDS),
                methods: excitation_methods(class_name),
            });
        }
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: EXCITATION_CASE_CLASS.into(),
            parent: None,
            properties: immutable_properties(CASE_FIELDS),
            methods: HashMap::from([
                (
                    "always".into(),
                    static_method("always", "fea.ExcitationCase.always"),
                ),
                (
                    "duringSteps".into(),
                    static_method("duringSteps", "fea.ExcitationCase.duringSteps"),
                ),
                (
                    "mutuallyExclusive".into(),
                    static_method("mutuallyExclusive", "fea.ExcitationCase.mutuallyExclusive"),
                ),
            ]),
        });
    });
}

fn excitation_schemas() -> [(&'static str, &'static str); 6] {
    [
        (
            STRUCTURAL_EXCITATION_CLASS,
            "runmat.fea.excitation.structural",
        ),
        (THERMAL_EXCITATION_CLASS, "runmat.fea.excitation.thermal"),
        (
            ELECTRICAL_EXCITATION_CLASS,
            "runmat.fea.excitation.electrical",
        ),
        (MAGNETIC_EXCITATION_CLASS, "runmat.fea.excitation.magnetic"),
        (ACOUSTIC_EXCITATION_CLASS, "runmat.fea.excitation.acoustic"),
        (FLUID_EXCITATION_CLASS, "runmat.fea.excitation.fluid"),
    ]
}

fn excitation_methods(class_name: &str) -> HashMap<String, crate::class_registry::RuntimeMethod> {
    let methods: &[(&str, &str)] = match class_name {
        STRUCTURAL_EXCITATION_CLASS => &[
            ("force", "fea.StructuralExcitation.force"),
            ("moment", "fea.StructuralExcitation.moment"),
            ("traction", "fea.StructuralExcitation.traction"),
            ("pressure", "fea.StructuralExcitation.pressure"),
            ("bodyForce", "fea.StructuralExcitation.bodyForce"),
            (
                "baseDisplacement",
                "fea.StructuralExcitation.baseDisplacement",
            ),
            ("baseVelocity", "fea.StructuralExcitation.baseVelocity"),
            (
                "baseAcceleration",
                "fea.StructuralExcitation.baseAcceleration",
            ),
        ],
        THERMAL_EXCITATION_CLASS => &[
            ("heatGeneration", "fea.ThermalExcitation.heatGeneration"),
            ("heatFlux", "fea.ThermalExcitation.heatFlux"),
            ("temperatureField", "fea.ThermalExcitation.temperatureField"),
            (
                "ambientTemperature",
                "fea.ThermalExcitation.ambientTemperature",
            ),
        ],
        ELECTRICAL_EXCITATION_CLASS => &[
            (
                "terminalVoltage",
                "fea.ElectricalExcitation.terminalVoltage",
            ),
            ("current", "fea.ElectricalExcitation.current"),
            ("charge", "fea.ElectricalExcitation.charge"),
            ("currentDensity", "fea.ElectricalExcitation.currentDensity"),
        ],
        MAGNETIC_EXCITATION_CLASS => {
            &[("vectorPotential", "fea.MagneticExcitation.vectorPotential")]
        }
        ACOUSTIC_EXCITATION_CLASS => &[
            ("pressure", "fea.AcousticExcitation.pressure"),
            ("normalVelocity", "fea.AcousticExcitation.normalVelocity"),
        ],
        FLUID_EXCITATION_CLASS => &[
            ("inletVelocity", "fea.FluidExcitation.inletVelocity"),
            ("massFlow", "fea.FluidExcitation.massFlow"),
            ("wallMotion", "fea.FluidExcitation.wallMotion"),
        ],
        _ => unreachable!("excitation schemas contain only known classes"),
    };
    methods
        .iter()
        .map(|(name, builtin)| ((*name).into(), static_method(name, builtin)))
        .collect()
}

fn immutable_properties(
    fields: &[&str],
) -> HashMap<String, crate::class_registry::RuntimeProperty> {
    fields
        .iter()
        .map(|field| {
            (
                (*field).into(),
                crate::class_registry::RuntimeProperty {
                    name: (*field).into(),
                    is_static: false,
                    is_constant: true,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            )
        })
        .collect()
}

fn static_method(name: &str, builtin: &str) -> crate::class_registry::RuntimeMethod {
    crate::class_registry::RuntimeMethod {
        name: name.into(),
        is_static: true,
        is_abstract: false,
        is_sealed: false,
        access: MemberAccess::Public,
        function_name: builtin.into(),
        implicit_class_argument: None,
    }
}

fn validate_excitation_object(object: &ObjectInstance) -> Result<(), String> {
    if !matches!(object.properties.get("geometry_id"), Some(Value::String(value)) if !value.is_empty())
    {
        return Err(format!(
            "{}.geometry_id must be a non-empty string",
            object.class_name
        ));
    }
    if !matches!(
        object.properties.get("geometry_revision"),
        Some(Value::Int(IntValue::U32(_)))
    ) {
        return Err(format!(
            "{}.geometry_revision must be uint32",
            object.class_name
        ));
    }
    if !matches!(object.properties.get("action"), Some(Value::Struct(_))) {
        return Err(format!(
            "{}.action must be a typed structure",
            object.class_name
        ));
    }
    Ok(())
}

fn validate_case_object(object: &ObjectInstance) -> Result<(), String> {
    if !matches!(object.properties.get("case_id"), Some(Value::String(value)) if !value.is_empty())
        || !matches!(
            object.properties.get("action_ids"),
            Some(Value::StringArray(_))
        )
        || !matches!(
            object.properties.get("activation"),
            Some(Value::String(_) | Value::Struct(_))
        )
    {
        return Err("fea.ExcitationCase properties are malformed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::signals::force_constant;
    use crate::builtins::geometry::region::{
        ensure_region_class_registered, GEOMETRY_REGION_CLASS,
    };
    use futures::executor::block_on;
    use runmat_value::CellArray;

    fn region() -> Value {
        ensure_region_class_registered();
        let mut object = ObjectInstance::new(GEOMETRY_REGION_CLASS.into());
        object
            .properties
            .insert("geometry_id".into(), Value::String("geom".into()));
        object
            .properties
            .insert("geometry_revision".into(), Value::Int(IntValue::U32(2)));
        object
            .properties
            .insert("region_id".into(), Value::String("load_face".into()));
        object
            .properties
            .insert("name".into(), Value::String("Load Face".into()));
        Value::Object(object)
    }

    #[test]
    fn force_action_and_case_preserve_separate_typed_roles() {
        let signal = block_on(force_constant(
            "tip.signal".into(),
            Value::Tensor(Tensor::new_2d(vec![0.0, -1250.0, 0.0], 1, 3).unwrap()),
        ))
        .unwrap();
        let action = block_on(structural_force(
            "tip".into(),
            "service".into(),
            region(),
            signal,
        ))
        .unwrap();
        let case = block_on(excitation_case_always("service".into(), action.clone())).unwrap();
        for value in [&action, &case] {
            assert_eq!(
                crate::execution::value_codec::decode_inline_value(
                    &crate::execution::value_codec::encode_inline_value(value).unwrap()
                )
                .unwrap(),
                value.clone()
            );
        }
        let actions = bound_excitation_vec_from_value("test", &action).unwrap();
        assert_eq!(actions[0].action.excitation().signal_id.0, "tip.signal");
        let cases = excitation_case_vec_from_value("test", &case).unwrap();
        assert_eq!(cases[0].action_ids, vec![ExcitationId("tip".into())]);
    }

    #[test]
    fn action_rejects_wrong_signal_quantity_and_case_rejects_mismatch() {
        let pressure = block_on(super::super::signals::pressure_constant(
            "pressure.signal".into(),
            Value::Num(2.0),
        ))
        .unwrap();
        assert!(block_on(structural_force(
            "tip".into(),
            "service".into(),
            region(),
            pressure,
        ))
        .is_err());

        let force = block_on(force_constant(
            "tip.signal".into(),
            Value::Tensor(Tensor::new_2d(vec![0.0, -1.0, 0.0], 1, 3).unwrap()),
        ))
        .unwrap();
        let action = block_on(structural_force(
            "tip".into(),
            "service".into(),
            region(),
            force,
        ))
        .unwrap();
        assert!(block_on(excitation_case_always("other".into(), action)).is_err());

        let malformed = Value::Cell(
            CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("test cell should be valid"),
        );
        assert!(bound_excitation_vec_from_value("test", &malformed).is_err());
    }

    #[test]
    fn model_composition_accepts_the_separate_signal_action_and_case_objects() {
        let temp = tempfile::TempDir::new().unwrap();
        let geometry =
            super::super::test_support::load_stl_fixture(&temp.path().join("loaded.stl"), 1.0);
        let asset =
            crate::builtins::geometry::geometry_asset_from_value_with_builtin(&geometry, "test")
                .unwrap();
        let region_id = asset.regions[0].region_id.clone();
        let region = block_on(crate::builtins::geometry::region::geometry_region_builtin(
            geometry.clone(),
            region_id,
        ))
        .unwrap();
        let signal = block_on(force_constant(
            "tip.signal".into(),
            Value::Tensor(Tensor::new_2d(vec![0.0, -1250.0, 0.0], 1, 3).unwrap()),
        ))
        .unwrap();
        let action = block_on(structural_force(
            "tip".into(),
            "service".into(),
            region,
            signal.clone(),
        ))
        .unwrap();
        let case = block_on(excitation_case_always("service".into(), action.clone())).unwrap();
        let model = super::super::modeling::create_model_object_from_args(vec![
            Value::String("bracket".into()),
            geometry,
            Value::String("Template".into()),
            Value::String("linear_static_structural".into()),
            Value::String("Signals".into()),
            signal,
            Value::String("Excitations".into()),
            action,
            Value::String("ExcitationCases".into()),
            case,
        ])
        .unwrap();
        let model: AnalysisModel = object_payload("test", &model, FEA_MODEL_CLASS).unwrap();
        assert_eq!(model.signals.len(), 1);
        assert_eq!(model.excitations.len(), 1);
        assert_eq!(model.excitation_cases.len(), 1);
        runmat_analysis_core::validate_model_against_geometry(
            &model,
            asset.units,
            &CoordinateFrame::Global,
        )
        .unwrap();
    }
}
