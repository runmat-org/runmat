use std::sync::Once;

use super::*;
use crate::builtins::fea::constraints::require_geometry_binding;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::geometry::region::region_ref_from_value;
use crate::class_registry::declaration::{immutable_properties, static_methods};
use runmat_analysis_core::{
    InitialCondition, InitialConditionTarget, InitialConditionValue, InitialValue, PhysicalSignal,
    SignalId, SignalShape,
};

pub(super) const TARGET_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_INITIAL_CONDITION_TARGET_CLASS
        .display_name();
pub(super) const STRUCTURAL_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_INITIAL_CONDITION_CLASS
        .display_name();
pub(super) const THERMAL_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_THERMAL_INITIAL_CONDITION_CLASS
        .display_name();
pub(super) const ELECTRICAL_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_ELECTRICAL_INITIAL_CONDITION_CLASS
        .display_name();
pub(super) const MAGNETIC_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_MAGNETIC_INITIAL_CONDITION_CLASS
        .display_name();
pub(super) const ACOUSTIC_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_ACOUSTIC_INITIAL_CONDITION_CLASS
        .display_name();
pub(super) const FLUID_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_FLUID_INITIAL_CONDITION_CLASS
        .display_name();

const TARGET_FIELDS: &[&str] = &["whole_model"];
const CONDITION_FIELDS: &[&str] = &["geometry_id", "geometry_revision", "condition"];
const CONDITION_REQUIRED_FIELDS: &[&str] = &["condition"];
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WholeModelTarget {
    whole_model: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GeometryBoundInitialCondition {
    pub(super) geometry_id: Option<String>,
    pub(super) geometry_revision: Option<u32>,
    pub(super) condition: InitialCondition,
}

impl GeometryBoundInitialCondition {
    pub(super) fn require_geometry(
        &self,
        builtin: &'static str,
        geometry: &GeometryAsset,
    ) -> BuiltinResult<()> {
        match (&self.geometry_id, self.geometry_revision) {
            (Some(id), Some(revision)) => {
                require_geometry_binding(builtin, "initial condition", id, revision, geometry)
            }
            (None, None) => Ok(()),
            _ => Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "initial-condition geometry binding is incomplete",
            )),
        }
    }
}

#[runtime_builtin(
    name = "fea.InitialConditionTarget.wholeModel",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::initial_conditions"
)]
pub async fn whole_model_target() -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_INITIAL_CONDITION_TARGET_WHOLE_MODEL_DESCRIPTOR,
        "fea.InitialConditionTarget.wholeModel",
    )?;
    serializable_to_object(
        "fea.InitialConditionTarget.wholeModel",
        &ERROR_INTERNAL,
        TARGET_CLASS,
        &WholeModelTarget { whole_model: true },
    )
}

fn target_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<(Option<(String, u32)>, InitialConditionTarget)> {
    if matches!(value, Value::Object(object) if object.class_name().is(runmat_builtins::fea_contract::model_types::classes::FEA_INITIAL_CONDITION_TARGET_CLASS))
    {
        let target: WholeModelTarget = object_payload(builtin, value, TARGET_CLASS)?;
        if !target.whole_model {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "whole-model target is malformed",
            ));
        }
        return Ok((None, InitialConditionTarget::WholeModel));
    }
    let region = region_ref_from_value(builtin, value)?;
    Ok((
        Some((region.geometry_id, region.geometry_revision)),
        InitialConditionTarget::Region(RegionTarget::volume(region.region_id)),
    ))
}

fn condition_to_object(
    builtin: &'static str,
    class_name: &'static str,
    binding: Option<(String, u32)>,
    condition: InitialCondition,
) -> BuiltinResult<Value> {
    let (geometry_id, geometry_revision) = binding
        .map(|(id, revision)| (Some(id), Some(revision)))
        .unwrap_or((None, None));
    let revision = geometry_revision;
    let mut object = serializable_to_object_value(
        builtin,
        &ERROR_INTERNAL,
        class_name,
        &GeometryBoundInitialCondition {
            geometry_id,
            geometry_revision,
            condition,
        },
    )?;
    if let Some(revision) = revision {
        crate::builtins::fea::object_api::replace(
            builtin,
            &mut object,
            "geometry_revision",
            Value::Int(IntValue::U32(revision)),
        )?;
    }
    Ok(Value::Object(object))
}

macro_rules! initial_condition_builtin {
    (
        $function:ident, $descriptor:ident, $builtin:literal, $class:expr,
        $signal_variant:path, $condition_variant:path, $shape:pat
    ) => {
        #[runtime_builtin(
                                    name = $builtin,
                                    binding_variant = "default",
                                    builtin_path = "crate::builtins::fea::initial_conditions"
                                )]
        pub async fn $function(
            id: String,
            target: Value,
            signal: Value,
            abscissa: Value,
            sample_approach: Value,
            before_step_id: String,
        ) -> BuiltinResult<Value> {
            crate::builtins::fea::constraints::fixed_outputs(
                &runmat_builtins::$descriptor,
                $builtin,
            )?;
            if id.trim().is_empty() || before_step_id.trim().is_empty() {
                return Err(builtin_error(
                    $builtin,
                    &ERROR_INPUT,
                    "initial-condition and following-step identifiers must not be empty",
                ));
            }
            let abscissa = crate::builtins::fea::values::finite_numeric_scalar(
                $builtin, &abscissa, "abscissa",
            )?;
            let approach =
                super::signal_policies::sample_approach_from_value($builtin, &sample_approach)?;
            let signal = super::signals::physical_signal_from_value($builtin, &signal)?;
            let $signal_variant(signal) = signal else {
                return Err(builtin_error(
                    $builtin,
                    &ERROR_INPUT,
                    "signal has the wrong physical quantity for this initial condition",
                ));
            };
            if !matches!(&signal.shape, $shape) {
                return Err(builtin_error(
                    $builtin,
                    &ERROR_INPUT,
                    "signal has the wrong shape for this initial condition",
                ));
            }
            let (binding, target) = target_from_value($builtin, &target)?;
            let condition = $condition_variant(InitialConditionValue {
                initial_condition_id: id,
                target,
                frame: signal.frame.unwrap_or(CoordinateFrame::Global),
                value: InitialValue::SignalSample {
                    signal_id: SignalId(signal.signal_id.0),
                    abscissa,
                    approach,
                },
                applies_before_step_id: before_step_id,
            });
            condition_to_object($builtin, $class, binding, condition)
        }
    };
}

initial_condition_builtin!(
    structural_displacement,
    FEA_STRUCTURAL_INITIAL_CONDITION_DISPLACEMENT_DESCRIPTOR,
    "fea.StructuralInitialCondition.displacement",
    STRUCTURAL_CLASS,
    PhysicalSignal::Displacement,
    InitialCondition::Displacement,
    SignalShape::Vector3
);
initial_condition_builtin!(
    structural_velocity,
    FEA_STRUCTURAL_INITIAL_CONDITION_VELOCITY_DESCRIPTOR,
    "fea.StructuralInitialCondition.velocity",
    STRUCTURAL_CLASS,
    PhysicalSignal::Velocity,
    InitialCondition::Velocity,
    SignalShape::Vector3
);
initial_condition_builtin!(
    structural_angular_velocity,
    FEA_STRUCTURAL_INITIAL_CONDITION_ANGULAR_VELOCITY_DESCRIPTOR,
    "fea.StructuralInitialCondition.angularVelocity",
    STRUCTURAL_CLASS,
    PhysicalSignal::AngularVelocity,
    InitialCondition::AngularVelocity,
    SignalShape::Vector3
);
initial_condition_builtin!(
    thermal_temperature,
    FEA_THERMAL_INITIAL_CONDITION_TEMPERATURE_DESCRIPTOR,
    "fea.ThermalInitialCondition.temperature",
    THERMAL_CLASS,
    PhysicalSignal::Temperature,
    InitialCondition::Temperature,
    SignalShape::Scalar
);
initial_condition_builtin!(
    electrical_potential,
    FEA_ELECTRICAL_INITIAL_CONDITION_POTENTIAL_DESCRIPTOR,
    "fea.ElectricalInitialCondition.potential",
    ELECTRICAL_CLASS,
    PhysicalSignal::ElectricPotential,
    InitialCondition::ElectricPotential,
    SignalShape::Scalar
);
initial_condition_builtin!(
    magnetic_vector_potential,
    FEA_MAGNETIC_INITIAL_CONDITION_VECTOR_POTENTIAL_DESCRIPTOR,
    "fea.MagneticInitialCondition.vectorPotential",
    MAGNETIC_CLASS,
    PhysicalSignal::MagneticVectorPotential,
    InitialCondition::MagneticVectorPotential,
    SignalShape::Vector3
);
initial_condition_builtin!(
    acoustic_pressure,
    FEA_ACOUSTIC_INITIAL_CONDITION_PRESSURE_DESCRIPTOR,
    "fea.AcousticInitialCondition.pressure",
    ACOUSTIC_CLASS,
    PhysicalSignal::Pressure,
    InitialCondition::AcousticPressure,
    SignalShape::Scalar
);
initial_condition_builtin!(
    fluid_pressure,
    FEA_FLUID_INITIAL_CONDITION_PRESSURE_DESCRIPTOR,
    "fea.FluidInitialCondition.pressure",
    FLUID_CLASS,
    PhysicalSignal::Pressure,
    InitialCondition::FluidPressure,
    SignalShape::Scalar
);
initial_condition_builtin!(
    fluid_velocity,
    FEA_FLUID_INITIAL_CONDITION_VELOCITY_DESCRIPTOR,
    "fea.FluidInitialCondition.velocity",
    FLUID_CLASS,
    PhysicalSignal::Velocity,
    InitialCondition::FluidVelocity,
    SignalShape::Vector3
);

pub(super) fn initial_condition_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<GeometryBoundInitialCondition>> {
    let values = match value {
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
        Value::Object(_) => vec![value],
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!(
                    "expected a typed FEA initial-condition object or cell array; got {other:?}"
                ),
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
                    "initial-condition cells may contain only typed objects",
                ));
            };
            let class_name =
                condition_class(object.class_name().display_name()).ok_or_else(|| {
                    builtin_error(
                        builtin,
                        &ERROR_INPUT,
                        format!(
                            "unsupported FEA initial-condition class `{}`",
                            object.class_name()
                        ),
                    )
                })?;
            object_payload(builtin, value, class_name)
        })
        .collect()
}

fn condition_class(class_name: &str) -> Option<&'static str> {
    Some(match class_name {
        STRUCTURAL_CLASS => STRUCTURAL_CLASS,
        THERMAL_CLASS => THERMAL_CLASS,
        ELECTRICAL_CLASS => ELECTRICAL_CLASS,
        MAGNETIC_CLASS => MAGNETIC_CLASS,
        ACOUSTIC_CLASS => ACOUSTIC_CLASS,
        FLUID_CLASS => FLUID_CLASS,
        _ => return None,
    })
}

pub(super) fn ensure_initial_condition_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            TARGET_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let mut schemas = vec![crate::execution::value_codec::ImmutableValueClassSchema {
            class_name: TARGET_CLASS,
            type_identity: "runmat.fea.initial-condition-target",
            schema_version: 1,
            allowed_fields: TARGET_FIELDS,
            required_fields: TARGET_FIELDS,
            validate: validate_target_object,
        }];
        for (class_name, type_identity, _) in condition_schemas() {
            schemas.push(crate::execution::value_codec::ImmutableValueClassSchema {
                class_name,
                type_identity,
                schema_version: 1,
                allowed_fields: CONDITION_FIELDS,
                required_fields: CONDITION_REQUIRED_FIELDS,
                validate: validate_condition_object,
            });
        }

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASSES.ensure(|| {
        register_value_class(
            TARGET_CLASS,
            immutable_properties(TARGET_FIELDS),
            static_methods([("wholeModel", "fea.InitialConditionTarget.wholeModel")]),
        );
        for (class_name, _, methods) in condition_schemas() {
            register_value_class(
                class_name,
                immutable_properties(CONDITION_FIELDS),
                static_methods(methods.iter().copied()),
            );
        }
    });
}

type ConditionSchema = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
);

fn condition_schemas() -> [ConditionSchema; 6] {
    [
        (
            STRUCTURAL_CLASS,
            "runmat.fea.initial-condition.structural",
            &[
                (
                    "displacement",
                    "fea.StructuralInitialCondition.displacement",
                ),
                ("velocity", "fea.StructuralInitialCondition.velocity"),
                (
                    "angularVelocity",
                    "fea.StructuralInitialCondition.angularVelocity",
                ),
            ],
        ),
        (
            THERMAL_CLASS,
            "runmat.fea.initial-condition.thermal",
            &[("temperature", "fea.ThermalInitialCondition.temperature")],
        ),
        (
            ELECTRICAL_CLASS,
            "runmat.fea.initial-condition.electrical",
            &[("potential", "fea.ElectricalInitialCondition.potential")],
        ),
        (
            MAGNETIC_CLASS,
            "runmat.fea.initial-condition.magnetic",
            &[(
                "vectorPotential",
                "fea.MagneticInitialCondition.vectorPotential",
            )],
        ),
        (
            ACOUSTIC_CLASS,
            "runmat.fea.initial-condition.acoustic",
            &[("pressure", "fea.AcousticInitialCondition.pressure")],
        ),
        (
            FLUID_CLASS,
            "runmat.fea.initial-condition.fluid",
            &[
                ("pressure", "fea.FluidInitialCondition.pressure"),
                ("velocity", "fea.FluidInitialCondition.velocity"),
            ],
        ),
    ]
}

fn validate_target_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    if matches!(
        object.properties().get("whole_model"),
        Some(Value::Bool(true))
    ) {
        Ok(())
    } else {
        Err("fea.InitialConditionTarget.whole_model must be true".into())
    }
}

fn validate_condition_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    let binding_valid = matches!(
        (
            object.properties().get("geometry_id"),
            object.properties().get("geometry_revision")
        ),
        (Some(Value::String(_)), Some(Value::Int(IntValue::U32(_)))) | (None, None)
    );
    if binding_valid && matches!(object.properties().get("condition"), Some(Value::Struct(_))) {
        Ok(())
    } else {
        Err(format!(
            "{} has malformed initial-condition data",
            object.class_name()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::geometry::region::{
        ensure_region_class_registered, GEOMETRY_REGION_CLASS,
    };
    use futures::executor::block_on;

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
            Value::Int(IntValue::U32(4)),
        )
        .expect("ordinary fixture property edit");
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            "region_id",
            Value::String("fluid".into()),
        )
        .expect("ordinary fixture property edit");
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            "name",
            Value::String("Fluid".into()),
        )
        .expect("ordinary fixture property edit");
        Value::Object(object)
    }

    fn velocity_signal() -> Value {
        block_on(super::super::signals::velocity_constant(
            "initial.velocity".into(),
            Value::Tensor(Tensor::new_2d(vec![1.0, 2.0, 3.0], 1, 3).unwrap()),
        ))
        .unwrap()
    }

    #[test]
    fn whole_model_initial_state_is_typed_and_portable() {
        let condition = block_on(structural_velocity(
            "initial".into(),
            block_on(whole_model_target()).unwrap(),
            velocity_signal(),
            Value::Num(0.0),
            block_on(super::super::signal_policies::sample_approach_at()).unwrap(),
            "transient".into(),
        ))
        .unwrap();
        let decoded = initial_condition_vec_from_value("test", &condition).unwrap();
        assert!(decoded[0].geometry_id.is_none());
        assert!(matches!(
            decoded[0].condition,
            InitialCondition::Velocity(InitialConditionValue {
                target: InitialConditionTarget::WholeModel,
                ..
            })
        ));
        let encoded = crate::execution::value_codec::encode_inline_value(&condition).unwrap();
        assert_eq!(
            crate::execution::value_codec::decode_inline_value(&encoded).unwrap(),
            condition
        );
    }

    #[test]
    fn region_initial_state_retains_geometry_and_rejects_wrong_quantity() {
        let condition = block_on(fluid_velocity(
            "initial".into(),
            region(),
            velocity_signal(),
            Value::Num(0.0),
            block_on(super::super::signal_policies::sample_approach_at()).unwrap(),
            "transient".into(),
        ))
        .unwrap();
        let decoded = initial_condition_vec_from_value("test", &condition).unwrap();
        assert_eq!(decoded[0].geometry_id.as_deref(), Some("geom"));
        assert_eq!(decoded[0].geometry_revision, Some(4));

        let pressure = block_on(super::super::signals::pressure_constant(
            "pressure".into(),
            Value::Num(2.0),
        ))
        .unwrap();
        assert!(block_on(fluid_velocity(
            "bad".into(),
            region(),
            pressure,
            Value::Num(0.0),
            block_on(super::super::signal_policies::sample_approach_at()).unwrap(),
            "transient".into(),
        ))
        .is_err());
    }
}
