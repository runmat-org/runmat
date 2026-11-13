use std::collections::HashMap;
use std::sync::Once;

use super::*;
use crate::builtins::fea::constraints::require_geometry_binding;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::geometry::region::region_ref_from_value;
use crate::class_registry::declaration::{immutable_properties, static_methods};
use runmat_analysis_core::{
    AnalysisExcitation, ExcitationCaseActivation, ExcitationCaseId, ExcitationDistribution,
    ExcitationId, ExcitationKinematics, ProjectionRequirement,
};

pub(super) const STRUCTURAL_EXCITATION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_EXCITATION_CLASS
        .display_name();
pub(super) const THERMAL_EXCITATION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_THERMAL_EXCITATION_CLASS
        .display_name();
pub(super) const ELECTRICAL_EXCITATION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_ELECTRICAL_EXCITATION_CLASS
        .display_name();
pub(super) const MAGNETIC_EXCITATION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_MAGNETIC_EXCITATION_CLASS
        .display_name();
pub(super) const ACOUSTIC_EXCITATION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_ACOUSTIC_EXCITATION_CLASS
        .display_name();
pub(super) const FLUID_EXCITATION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_FLUID_EXCITATION_CLASS.display_name();
pub(super) const EXCITATION_CASE_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_EXCITATION_CASE_CLASS.display_name();

const EXCITATION_FIELDS: &[&str] = &["geometry_id", "geometry_revision", "action"];
const CASE_FIELDS: &[&str] = &["case_id", "action_ids", "activation"];
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

fn action_to_object(
    builtin: &'static str,
    class_name: &'static str,
    value: GeometryBoundExcitation,
) -> BuiltinResult<Value> {
    let revision = value.geometry_revision;
    let mut object = serializable_to_object_value(builtin, &ERROR_INTERNAL, class_name, &value)?;
    crate::builtins::fea::object_api::replace(
        builtin,
        &mut object,
        "geometry_revision",
        Value::Int(IntValue::U32(revision)),
    )?;
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
    region: &Value,
    signal: &Value,
    expected_quantity: runmat_analysis_core::PhysicalQuantityKind,
    region_expectation: RegionExpectation,
    distribution: ExcitationDistribution,
    preserve_moment: bool,
    wrap: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) -> BuiltinResult<Value> {
    if id.trim().is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "excitation identifier must not be empty",
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
        $function:ident, $descriptor:ident, $builtin:literal, $class:expr,
        $quantity:path, $region:expr, $distribution:path, $preserve_moment:expr, $variant:path
    ) => {
        #[runtime_builtin(
                                            name = $builtin,
                                            binding_variant = "default",
                                            builtin_path = "crate::builtins::fea::excitations"
                                        )]
        pub async fn $function(id: String, region: Value, signal: Value) -> BuiltinResult<Value> {
            crate::builtins::fea::constraints::fixed_outputs(
                &runmat_builtins::$descriptor,
                $builtin,
            )?;
            create_action(
                $builtin,
                $class,
                id,
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
    FEA_STRUCTURAL_EXCITATION_FORCE_DESCRIPTOR,
    "fea.StructuralExcitation.force",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Force,
    RegionExpectation::Boundary,
    ExcitationDistribution::PreserveResultant,
    false,
    ExcitationAction::Force
);
action_builtin!(
    structural_moment,
    FEA_STRUCTURAL_EXCITATION_MOMENT_DESCRIPTOR,
    "fea.StructuralExcitation.moment",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Moment,
    RegionExpectation::Boundary,
    ExcitationDistribution::PreserveResultant,
    true,
    ExcitationAction::Moment
);
action_builtin!(
    structural_traction,
    FEA_STRUCTURAL_EXCITATION_TRACTION_DESCRIPTOR,
    "fea.StructuralExcitation.traction",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Traction,
    RegionExpectation::Boundary,
    ExcitationDistribution::IntensityPerArea,
    false,
    ExcitationAction::Traction
);
action_builtin!(
    structural_pressure,
    FEA_STRUCTURAL_EXCITATION_PRESSURE_DESCRIPTOR,
    "fea.StructuralExcitation.pressure",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::Pressure,
    RegionExpectation::Boundary,
    ExcitationDistribution::IntensityPerArea,
    false,
    ExcitationAction::Pressure
);
action_builtin!(
    structural_body_force,
    FEA_STRUCTURAL_EXCITATION_BODY_FORCE_DESCRIPTOR,
    "fea.StructuralExcitation.bodyForce",
    STRUCTURAL_EXCITATION_CLASS,
    Quantity::BodyForceDensity,
    RegionExpectation::Volume,
    ExcitationDistribution::IntensityPerVolume,
    false,
    ExcitationAction::BodyForce
);
action_builtin!(
    thermal_heat_generation,
    FEA_THERMAL_EXCITATION_HEAT_GENERATION_DESCRIPTOR,
    "fea.ThermalExcitation.heatGeneration",
    THERMAL_EXCITATION_CLASS,
    Quantity::HeatGenerationDensity,
    RegionExpectation::Volume,
    ExcitationDistribution::IntensityPerVolume,
    false,
    ExcitationAction::HeatGeneration
);
action_builtin!(
    thermal_heat_flux,
    FEA_THERMAL_EXCITATION_HEAT_FLUX_DESCRIPTOR,
    "fea.ThermalExcitation.heatFlux",
    THERMAL_EXCITATION_CLASS,
    Quantity::HeatFlux,
    RegionExpectation::Boundary,
    ExcitationDistribution::IntensityPerArea,
    false,
    ExcitationAction::HeatFlux
);
action_builtin!(
    thermal_temperature_field,
    FEA_THERMAL_EXCITATION_TEMPERATURE_FIELD_DESCRIPTOR,
    "fea.ThermalExcitation.temperatureField",
    THERMAL_EXCITATION_CLASS,
    Quantity::Temperature,
    RegionExpectation::Volume,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::TemperatureField
);
action_builtin!(
    thermal_ambient_temperature,
    FEA_THERMAL_EXCITATION_AMBIENT_TEMPERATURE_DESCRIPTOR,
    "fea.ThermalExcitation.ambientTemperature",
    THERMAL_EXCITATION_CLASS,
    Quantity::Temperature,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::AmbientTemperature
);
action_builtin!(
    electrical_terminal_voltage,
    FEA_ELECTRICAL_EXCITATION_TERMINAL_VOLTAGE_DESCRIPTOR,
    "fea.ElectricalExcitation.terminalVoltage",
    ELECTRICAL_EXCITATION_CLASS,
    Quantity::ElectricPotential,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::TerminalVoltage
);
action_builtin!(
    electrical_current,
    FEA_ELECTRICAL_EXCITATION_CURRENT_DESCRIPTOR,
    "fea.ElectricalExcitation.current",
    ELECTRICAL_EXCITATION_CLASS,
    Quantity::ElectricCurrent,
    RegionExpectation::Volume,
    ExcitationDistribution::PreserveResultant,
    false,
    ExcitationAction::ElectricCurrent
);
action_builtin!(
    electrical_charge,
    FEA_ELECTRICAL_EXCITATION_CHARGE_DESCRIPTOR,
    "fea.ElectricalExcitation.charge",
    ELECTRICAL_EXCITATION_CLASS,
    Quantity::ElectricCharge,
    RegionExpectation::Volume,
    ExcitationDistribution::PreserveResultant,
    false,
    ExcitationAction::ElectricCharge
);
action_builtin!(
    electrical_current_density,
    FEA_ELECTRICAL_EXCITATION_CURRENT_DENSITY_DESCRIPTOR,
    "fea.ElectricalExcitation.currentDensity",
    ELECTRICAL_EXCITATION_CLASS,
    Quantity::ElectricCurrentDensity,
    RegionExpectation::Volume,
    ExcitationDistribution::IntensityPerVolume,
    false,
    ExcitationAction::CurrentDensity
);
action_builtin!(
    magnetic_vector_potential,
    FEA_MAGNETIC_EXCITATION_VECTOR_POTENTIAL_DESCRIPTOR,
    "fea.MagneticExcitation.vectorPotential",
    MAGNETIC_EXCITATION_CLASS,
    Quantity::MagneticVectorPotential,
    RegionExpectation::Volume,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::MagneticVectorPotential
);
action_builtin!(
    acoustic_pressure,
    FEA_ACOUSTIC_EXCITATION_PRESSURE_DESCRIPTOR,
    "fea.AcousticExcitation.pressure",
    ACOUSTIC_EXCITATION_CLASS,
    Quantity::Pressure,
    RegionExpectation::Boundary,
    ExcitationDistribution::IntensityPerArea,
    false,
    ExcitationAction::AcousticPressure
);
action_builtin!(
    acoustic_normal_velocity,
    FEA_ACOUSTIC_EXCITATION_NORMAL_VELOCITY_DESCRIPTOR,
    "fea.AcousticExcitation.normalVelocity",
    ACOUSTIC_EXCITATION_CLASS,
    Quantity::Velocity,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::NormalVelocity
);
action_builtin!(
    acoustic_volume_velocity,
    FEA_ACOUSTIC_EXCITATION_VOLUME_VELOCITY_DESCRIPTOR,
    "fea.AcousticExcitation.volumeVelocity",
    ACOUSTIC_EXCITATION_CLASS,
    Quantity::VolumeFlowRate,
    RegionExpectation::Boundary,
    ExcitationDistribution::PreserveResultant,
    false,
    ExcitationAction::VolumeVelocity
);
action_builtin!(
    fluid_inlet_velocity,
    FEA_FLUID_EXCITATION_INLET_VELOCITY_DESCRIPTOR,
    "fea.FluidExcitation.inletVelocity",
    FLUID_EXCITATION_CLASS,
    Quantity::Velocity,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::InletVelocity
);
action_builtin!(
    fluid_mass_flow,
    FEA_FLUID_EXCITATION_MASS_FLOW_DESCRIPTOR,
    "fea.FluidExcitation.massFlow",
    FLUID_EXCITATION_CLASS,
    Quantity::MassFlowRate,
    RegionExpectation::Boundary,
    ExcitationDistribution::PreserveResultant,
    false,
    ExcitationAction::MassFlow
);
action_builtin!(
    fluid_wall_motion,
    FEA_FLUID_EXCITATION_WALL_MOTION_DESCRIPTOR,
    "fea.FluidExcitation.wallMotion",
    FLUID_EXCITATION_CLASS,
    Quantity::Velocity,
    RegionExpectation::Boundary,
    ExcitationDistribution::UniformKinematic,
    false,
    ExcitationAction::WallMotion
);

#[runtime_builtin(
    name = "fea.ExcitationCase.always",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::excitations"
)]
pub async fn excitation_case_always(id: String, actions: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_EXCITATION_CASE_ALWAYS_DESCRIPTOR,
        "fea.ExcitationCase.always",
    )?;
    create_case(
        "fea.ExcitationCase.always",
        id,
        &actions,
        ExcitationCaseActivation::Always,
    )
}

#[runtime_builtin(
    name = "fea.ExcitationCase.duringSteps",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::excitations"
)]
pub async fn excitation_case_during_steps(
    id: String,
    actions: Value,
    step_ids: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_EXCITATION_CASE_DURING_STEPS_DESCRIPTOR,
        "fea.ExcitationCase.duringSteps",
    )?;
    let step_ids = string_vec_from_value("fea.ExcitationCase.duringSteps", &step_ids)?;
    create_case(
        "fea.ExcitationCase.duringSteps",
        id,
        &actions,
        ExcitationCaseActivation::Steps(step_ids),
    )
}

#[runtime_builtin(
    name = "fea.ExcitationCase.mutuallyExclusive",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::excitations"
)]
pub async fn excitation_case_mutually_exclusive(
    id: String,
    actions: Value,
    group_id: String,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_EXCITATION_CASE_MUTUALLY_EXCLUSIVE_DESCRIPTOR,
        "fea.ExcitationCase.mutuallyExclusive",
    )?;
    if group_id.trim().is_empty() {
        return Err(builtin_error(
            "fea.ExcitationCase.mutuallyExclusive",
            &ERROR_INPUT,
            "group id must not be empty",
        ));
    }
    create_case(
        "fea.ExcitationCase.mutuallyExclusive",
        id,
        &actions,
        ExcitationCaseActivation::MutuallyExclusiveGroup { group_id },
    )
}

fn create_case(
    builtin: &'static str,
    id: String,
    actions: &Value,
    activation: ExcitationCaseActivation,
) -> BuiltinResult<Value> {
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
        action_ids.push(action.action.excitation().excitation_id.clone());
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
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
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
            let class_name = excitation_class_name(object.class_name().display_name()).ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported FEA excitation class `{}`", object.class_name()),
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
        | ExcitationAction::BodyForce(_) => STRUCTURAL_EXCITATION_CLASS,
        ExcitationAction::HeatGeneration(_)
        | ExcitationAction::HeatFlux(_)
        | ExcitationAction::TemperatureField(_)
        | ExcitationAction::AmbientTemperature(_) => THERMAL_EXCITATION_CLASS,
        ExcitationAction::TerminalVoltage(_)
        | ExcitationAction::ElectricCurrent(_)
        | ExcitationAction::ElectricCharge(_)
        | ExcitationAction::ElectricChargeDensity(_)
        | ExcitationAction::CurrentDensity(_)
        | ExcitationAction::MagneticVolumeCurrentDensity(_) => ELECTRICAL_EXCITATION_CLASS,
        ExcitationAction::MagneticVectorPotential(_) => MAGNETIC_EXCITATION_CLASS,
        ExcitationAction::AcousticPressure(_)
        | ExcitationAction::AcousticIncidentPressure(_)
        | ExcitationAction::NormalVelocity(_)
        | ExcitationAction::VolumeVelocity(_) => ACOUSTIC_EXCITATION_CLASS,
        ExcitationAction::InletVelocity(_)
        | ExcitationAction::MassFlow(_)
        | ExcitationAction::WallMotion(_) => FLUID_EXCITATION_CLASS,
    }
}

pub(super) fn ensure_excitation_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            STRUCTURAL_EXCITATION_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let mut schemas = Vec::new();
        for (class_name, type_identity) in excitation_schemas() {
            schemas.push(crate::execution::value_codec::ImmutableValueClassSchema {
                class_name,
                type_identity,
                schema_version: 1,
                allowed_fields: EXCITATION_FIELDS,
                required_fields: EXCITATION_FIELDS,
                validate: validate_excitation_object,
            });
        }
        schemas.push(crate::execution::value_codec::ImmutableValueClassSchema {
            class_name: EXCITATION_CASE_CLASS,
            type_identity: "runmat.fea.excitation-case",
            schema_version: 1,
            allowed_fields: CASE_FIELDS,
            required_fields: CASE_FIELDS,
            validate: validate_case_object,
        });

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASSES.ensure(|| {
        for (class_name, _) in excitation_schemas() {
            register_value_class(
                class_name,
                immutable_properties(EXCITATION_FIELDS),
                excitation_methods(class_name),
            );
        }
        register_value_class(
            EXCITATION_CASE_CLASS,
            immutable_properties(CASE_FIELDS),
            static_methods([
                ("always", "fea.ExcitationCase.always"),
                ("duringSteps", "fea.ExcitationCase.duringSteps"),
                ("mutuallyExclusive", "fea.ExcitationCase.mutuallyExclusive"),
            ]),
        );
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

fn excitation_methods(class_name: &str) -> HashMap<MethodName, RuntimeMethod> {
    let methods: &[(&str, &str)] = match class_name {
        STRUCTURAL_EXCITATION_CLASS => &[
            ("force", "fea.StructuralExcitation.force"),
            ("moment", "fea.StructuralExcitation.moment"),
            ("traction", "fea.StructuralExcitation.traction"),
            ("pressure", "fea.StructuralExcitation.pressure"),
            ("bodyForce", "fea.StructuralExcitation.bodyForce"),
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
            ("volumeVelocity", "fea.AcousticExcitation.volumeVelocity"),
        ],
        FLUID_EXCITATION_CLASS => &[
            ("inletVelocity", "fea.FluidExcitation.inletVelocity"),
            ("massFlow", "fea.FluidExcitation.massFlow"),
            ("wallMotion", "fea.FluidExcitation.wallMotion"),
        ],
        _ => unreachable!("excitation schemas contain only known classes"),
    };
    static_methods(methods.iter().copied())
}

fn validate_excitation_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    if !matches!(object.properties().get("geometry_id"), Some(Value::String(runmat_value::StringElement::Text(value))) if !value.is_empty())
    {
        return Err(format!(
            "{}.geometry_id must be a non-empty string",
            object.class_name()
        ));
    }
    if !matches!(
        object.properties().get("geometry_revision"),
        Some(Value::Int(IntValue::U32(_)))
    ) {
        return Err(format!(
            "{}.geometry_revision must be uint32",
            object.class_name()
        ));
    }
    if !matches!(object.properties().get("action"), Some(Value::Struct(_))) {
        return Err(format!(
            "{}.action must be a typed structure",
            object.class_name()
        ));
    }
    Ok(())
}

fn validate_case_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    if !matches!(object.properties().get("case_id"), Some(Value::String(runmat_value::StringElement::Text(value))) if !value.is_empty())
        || !matches!(
            object.properties().get("action_ids"),
            Some(Value::StringArray(_))
        )
        || !matches!(
            object.properties().get("activation"),
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
        let mut object = ObjectInstance::new(GEOMETRY_REGION_CLASS);
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            "geometry_id",
            Value::String("geom".into()),
        )
        .expect("ordinary fixture property edit");
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            "geometry_revision",
            Value::Int(IntValue::U32(2)),
        )
        .expect("ordinary fixture property edit");
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            "region_id",
            Value::String("load_face".into()),
        )
        .expect("ordinary fixture property edit");
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            "name",
            Value::String("Load Face".into()),
        )
        .expect("ordinary fixture property edit");
        Value::Object(object)
    }

    #[test]
    fn volume_flow_action_retains_total_quantity_policy_and_portable_class() {
        let signal = block_on(super::super::signals::volume_flow_constant(
            "flow".into(),
            Value::Num(-0.25),
        ))
        .unwrap();
        let value = block_on(acoustic_volume_velocity("speaker".into(), region(), signal)).unwrap();
        let decoded = crate::execution::value_codec::decode_inline_value(
            &crate::execution::value_codec::encode_inline_value(&value).unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, value);
        let actions = bound_excitation_vec_from_value("test", &decoded).unwrap();
        let ExcitationAction::VolumeVelocity(action) = &actions[0].action else {
            panic!("volume flow must remain a typed total-flow action");
        };
        assert_eq!(action.signal_id.0, "flow");
        assert_eq!(
            action.distribution,
            ExcitationDistribution::PreserveResultant
        );
        assert!(action.projection.preserve_resultant && !action.projection.preserve_moment);
        assert!(action.direction.is_none());
        let wrong = block_on(super::super::signals::velocity_constant(
            "speed".into(),
            Value::Tensor(Tensor::new_2d(vec![1.0, 0.0, 0.0], 1, 3).unwrap()),
        ))
        .unwrap();
        assert!(block_on(acoustic_volume_velocity("bad".into(), region(), wrong)).is_err());
    }

    #[test]
    fn force_action_and_case_preserve_separate_typed_roles() {
        let signal = block_on(force_constant(
            "tip.signal".into(),
            Value::Tensor(Tensor::new_2d(vec![0.0, -1250.0, 0.0], 1, 3).unwrap()),
        ))
        .unwrap();
        let action = block_on(structural_force("tip".into(), region(), signal)).unwrap();
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
    fn action_rejects_wrong_signal_quantity_and_cases_own_membership() {
        let pressure = block_on(super::super::signals::pressure_constant(
            "pressure.signal".into(),
            Value::Num(2.0),
        ))
        .unwrap();
        assert!(block_on(structural_force("tip".into(), region(), pressure,)).is_err());

        let force = block_on(force_constant(
            "tip.signal".into(),
            Value::Tensor(Tensor::new_2d(vec![0.0, -1.0, 0.0], 1, 3).unwrap()),
        ))
        .unwrap();
        let action = block_on(structural_force("tip".into(), region(), force)).unwrap();
        assert!(block_on(excitation_case_always("other".into(), action)).is_ok());

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
        let action = block_on(structural_force("tip".into(), region, signal.clone())).unwrap();
        let case = block_on(excitation_case_always("service".into(), action.clone())).unwrap();
        let model = super::super::modeling::create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("bracket".into()),
                geometry,
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Signals".into()),
                signal,
                Value::String("Excitations".into()),
                action,
                Value::String("ExcitationCases".into()),
                case,
            ]),
        )
        .unwrap();
        let model: AnalysisModel = object_payload("test", &model, FEA_MODEL_CLASS).unwrap();
        assert_eq!(model.signals.len(), 1);
        assert_eq!(model.excitations.len(), 1);
        assert_eq!(model.excitation_cases.len(), 1);
    }
}
