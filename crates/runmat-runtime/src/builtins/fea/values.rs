use std::collections::HashMap;
use std::sync::Once;

use super::*;
use crate::analysis::value_admission::{
    admit_borrowed_real64, AdmittedReal64Array, AnalysisValueAdmissionPolicy,
};
use crate::builtins::fea::object_projection::register_value_class;
use crate::class_registry::declaration::{immutable_properties, static_methods};
use runmat_builtins::fea_contract::identities;

pub(super) const DISPLACEMENT_CLASS: &str = identities::FEA_DISPLACEMENT_CLASS.display_name();
pub(super) const ROTATION_CLASS: &str = identities::FEA_ROTATION_CLASS.display_name();
pub(super) const TEMPERATURE_CLASS: &str = identities::FEA_TEMPERATURE_CLASS.display_name();
pub(super) const TEMPERATURE_DIFFERENCE_CLASS: &str =
    identities::FEA_TEMPERATURE_DIFFERENCE_CLASS.display_name();
pub(super) const PRESSURE_CLASS: &str = identities::FEA_PRESSURE_CLASS.display_name();
pub(super) const VELOCITY_CLASS: &str = identities::FEA_VELOCITY_CLASS.display_name();
pub(super) const HEAT_FLUX_CLASS: &str = identities::FEA_HEAT_FLUX_CLASS.display_name();
pub(super) const HEAT_TRANSFER_COEFFICIENT_CLASS: &str =
    identities::FEA_HEAT_TRANSFER_COEFFICIENT_CLASS.display_name();
pub(super) const ACOUSTIC_IMPEDANCE_CLASS: &str =
    identities::FEA_SPECIFIC_ACOUSTIC_IMPEDANCE_CLASS.display_name();
pub(super) const ELECTRIC_POTENTIAL_CLASS: &str =
    identities::FEA_ELECTRIC_POTENTIAL_CLASS.display_name();
pub(super) const ELECTRIC_CURRENT_CLASS: &str =
    identities::FEA_ELECTRIC_CURRENT_CLASS.display_name();
pub(super) const MAGNETIC_VECTOR_POTENTIAL_CLASS: &str =
    identities::FEA_MAGNETIC_VECTOR_POTENTIAL_CLASS.display_name();
pub(super) const LENGTH_CLASS: &str = identities::FEA_LENGTH_CLASS.display_name();
pub(super) const POSITION_CLASS: &str = identities::FEA_POSITION_CLASS.display_name();
pub(super) const TRANSLATION_CLASS: &str = identities::FEA_TRANSLATION_CLASS.display_name();
pub(super) const COMPLEX_PHASE_CLASS: &str = identities::FEA_COMPLEX_PHASE_CLASS.display_name();
pub(super) const THERMAL_CONTACT_RESISTANCE_CLASS: &str =
    identities::FEA_THERMAL_CONTACT_RESISTANCE_CLASS.display_name();
pub(super) const FRICTION_COEFFICIENT_CLASS: &str =
    identities::FEA_FRICTION_COEFFICIENT_CLASS.display_name();
pub(super) const FREQUENCY_CLASS: &str = identities::FEA_FREQUENCY_CLASS.display_name();
pub(super) const TIME_CLASS: &str = identities::FEA_TIME_CLASS.display_name();
pub(super) const MASS_DENSITY_CLASS: &str = identities::FEA_MASS_DENSITY_CLASS.display_name();
pub(super) const DYNAMIC_VISCOSITY_CLASS: &str =
    identities::FEA_DYNAMIC_VISCOSITY_CLASS.display_name();
pub(super) const TURBULENCE_INTENSITY_CLASS: &str =
    identities::FEA_TURBULENCE_INTENSITY_CLASS.display_name();
pub(super) const RAYLEIGH_MASS_COEFFICIENT_CLASS: &str =
    identities::FEA_RAYLEIGH_MASS_COEFFICIENT_CLASS.display_name();
pub(super) const RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS: &str =
    identities::FEA_RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS.display_name();

const METERS: &[&str] = &["meters"];
const RADIANS: &[&str] = &["radians"];
const KELVIN: &[&str] = &["kelvin"];
const PASCALS: &[&str] = &["pascals"];
const METERS_PER_SECOND: &[&str] = &["meters_per_second"];
const WATTS_PER_SQUARE_METER: &[&str] = &["watts_per_square_meter"];
const WATTS_PER_SQUARE_METER_KELVIN: &[&str] = &["watts_per_square_meter_kelvin"];
const PASCAL_SECONDS_PER_METER: &[&str] = &["pascal_seconds_per_meter"];
const VOLTS: &[&str] = &["volts"];
const AMPERES: &[&str] = &["amperes"];
const WEBERS_PER_METER: &[&str] = &["webers_per_meter"];
const SQUARE_METER_KELVIN_PER_WATT: &[&str] = &["square_meter_kelvin_per_watt"];
const RATIO: &[&str] = &["ratio"];
const HERTZ: &[&str] = &["hertz"];
const KILOGRAMS_PER_CUBIC_METER: &[&str] = &["kilograms_per_cubic_meter"];
const PASCAL_SECONDS: &[&str] = &["pascal_seconds"];
const PER_SECOND: &[&str] = &["per_second"];
const SECONDS: &[&str] = &["seconds"];

macro_rules! scalar_quantity_builtin {
    (
        $function:ident, $builtin_name:literal,
        $class:expr, $field:literal, $scale:expr, $offset:expr
    ) => {
        #[runtime_builtin(
                                                    name = $builtin_name,
                                                    binding_variant = "default",
                                                    builtin_path = "crate::builtins::fea::values"
                                                )]
        pub async fn $function(value: Value) -> BuiltinResult<Value> {
            let value = finite_numeric_scalar($builtin_name, &value, "value")?;
            quantity_to_value($builtin_name, $class, $field, value * $scale + $offset)
        }
    };
}

scalar_quantity_builtin!(
    displacement_meters_builtin,
    "fea.Displacement.meters",
    DISPLACEMENT_CLASS,
    "meters",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    thermal_contact_resistance_builtin,
    "fea.ThermalContactResistance.squareMeterKelvinPerWatt",
    THERMAL_CONTACT_RESISTANCE_CLASS,
    "square_meter_kelvin_per_watt",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    friction_coefficient_builtin,
    "fea.FrictionCoefficient.ratio",
    FRICTION_COEFFICIENT_CLASS,
    "ratio",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    time_seconds_builtin,
    "fea.Time.seconds",
    TIME_CLASS,
    "seconds",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    frequency_builtin,
    "fea.Frequency.hertz",
    FREQUENCY_CLASS,
    "hertz",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    mass_density_builtin,
    "fea.MassDensity.kilogramsPerCubicMeter",
    MASS_DENSITY_CLASS,
    "kilograms_per_cubic_meter",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    dynamic_viscosity_builtin,
    "fea.DynamicViscosity.pascalSeconds",
    DYNAMIC_VISCOSITY_CLASS,
    "pascal_seconds",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    turbulence_intensity_builtin,
    "fea.TurbulenceIntensity.ratio",
    TURBULENCE_INTENSITY_CLASS,
    "ratio",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    rayleigh_mass_coefficient_builtin,
    "fea.RayleighMassCoefficient.perSecond",
    RAYLEIGH_MASS_COEFFICIENT_CLASS,
    "per_second",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    rayleigh_stiffness_coefficient_builtin,
    "fea.RayleighStiffnessCoefficient.seconds",
    RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS,
    "seconds",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    complex_phase_radians_builtin,
    "fea.ComplexPhase.radians",
    COMPLEX_PHASE_CLASS,
    "radians",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    complex_phase_degrees_builtin,
    "fea.ComplexPhase.degrees",
    COMPLEX_PHASE_CLASS,
    "radians",
    std::f64::consts::PI / 180.0,
    0.0
);
scalar_quantity_builtin!(
    displacement_millimeters_builtin,
    "fea.Displacement.millimeters",
    DISPLACEMENT_CLASS,
    "meters",
    1.0e-3,
    0.0
);
scalar_quantity_builtin!(
    displacement_micrometers_builtin,
    "fea.Displacement.micrometers",
    DISPLACEMENT_CLASS,
    "meters",
    1.0e-6,
    0.0
);
scalar_quantity_builtin!(
    rotation_radians_builtin,
    "fea.Rotation.radians",
    ROTATION_CLASS,
    "radians",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    rotation_degrees_builtin,
    "fea.Rotation.degrees",
    ROTATION_CLASS,
    "radians",
    std::f64::consts::PI / 180.0,
    0.0
);
scalar_quantity_builtin!(
    temperature_kelvin_builtin,
    "fea.Temperature.kelvin",
    TEMPERATURE_CLASS,
    "kelvin",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    temperature_celsius_builtin,
    "fea.Temperature.celsius",
    TEMPERATURE_CLASS,
    "kelvin",
    1.0,
    273.15
);
scalar_quantity_builtin!(
    temperature_difference_kelvin_builtin,
    "fea.TemperatureDifference.kelvin",
    TEMPERATURE_DIFFERENCE_CLASS,
    "kelvin",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    temperature_difference_celsius_builtin,
    "fea.TemperatureDifference.celsius",
    TEMPERATURE_DIFFERENCE_CLASS,
    "kelvin",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    pressure_pascals_builtin,
    "fea.Pressure.pascals",
    PRESSURE_CLASS,
    "pascals",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    velocity_meters_per_second_builtin,
    "fea.Velocity.metersPerSecond",
    VELOCITY_CLASS,
    "meters_per_second",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    heat_flux_watts_per_square_meter_builtin,
    "fea.HeatFlux.wattsPerSquareMeter",
    HEAT_FLUX_CLASS,
    "watts_per_square_meter",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    heat_transfer_coefficient_builtin,
    "fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin",
    HEAT_TRANSFER_COEFFICIENT_CLASS,
    "watts_per_square_meter_kelvin",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    specific_acoustic_impedance_builtin,
    "fea.SpecificAcousticImpedance.pascalSecondsPerMeter",
    ACOUSTIC_IMPEDANCE_CLASS,
    "pascal_seconds_per_meter",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    electric_potential_volts_builtin,
    "fea.ElectricPotential.volts",
    ELECTRIC_POTENTIAL_CLASS,
    "volts",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    electric_current_amperes_builtin,
    "fea.ElectricCurrent.amperes",
    ELECTRIC_CURRENT_CLASS,
    "amperes",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    magnetic_vector_potential_builtin,
    "fea.MagneticVectorPotential.webersPerMeter",
    MAGNETIC_VECTOR_POTENTIAL_CLASS,
    "webers_per_meter",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    length_meters_builtin,
    "fea.Length.meters",
    LENGTH_CLASS,
    "meters",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    length_millimeters_builtin,
    "fea.Length.millimeters",
    LENGTH_CLASS,
    "meters",
    1.0e-3,
    0.0
);
scalar_quantity_builtin!(
    length_micrometers_builtin,
    "fea.Length.micrometers",
    LENGTH_CLASS,
    "meters",
    1.0e-6,
    0.0
);

macro_rules! vector_quantity_builtin {
    (
        $function:ident, $builtin_name:literal, $class:expr, $scale:expr
    ) => {
        #[runtime_builtin(
                                                    name = $builtin_name,
                                                    binding_variant = "default",
                                                    builtin_path = "crate::builtins::fea::values"
                                                )]
        pub async fn $function(value: Value) -> BuiltinResult<Value> {
            let components = finite_numeric_vector3($builtin_name, &value, "components")?;
            let components = components.map(|value| value * $scale);
            vector_quantity_to_value($builtin_name, $class, components)
        }
    };
}

vector_quantity_builtin!(
    position_meters_builtin,
    "fea.Position.meters",
    POSITION_CLASS,
    1.0
);
vector_quantity_builtin!(
    position_millimeters_builtin,
    "fea.Position.millimeters",
    POSITION_CLASS,
    1.0e-3
);
vector_quantity_builtin!(
    translation_meters_builtin,
    "fea.Translation.meters",
    TRANSLATION_CLASS,
    1.0
);
vector_quantity_builtin!(
    translation_millimeters_builtin,
    "fea.Translation.millimeters",
    TRANSLATION_CLASS,
    1.0e-3
);

pub(super) fn ensure_quantity_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            DISPLACEMENT_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let mut schemas = Vec::new();
        for (class_name, type_identity, fields) in quantity_schemas() {
            schemas.push(crate::execution::value_codec::ImmutableValueClassSchema {
                class_name,
                type_identity,
                schema_version: 1,
                allowed_fields: fields,
                required_fields: fields,
                validate: validate_quantity_object,
            });
        }

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASSES.ensure(|| {
        for (class_name, _, fields) in quantity_schemas() {
            register_value_class(
                class_name,
                immutable_properties(fields),
                quantity_methods(class_name),
            );
        }
    });
}

fn quantity_schemas() -> [(&'static str, &'static str, &'static [&'static str]); 25] {
    [
        (DISPLACEMENT_CLASS, "runmat.fea.displacement", METERS),
        (ROTATION_CLASS, "runmat.fea.rotation", RADIANS),
        (TEMPERATURE_CLASS, "runmat.fea.temperature", KELVIN),
        (
            TEMPERATURE_DIFFERENCE_CLASS,
            "runmat.fea.temperature-difference",
            KELVIN,
        ),
        (PRESSURE_CLASS, "runmat.fea.pressure", PASCALS),
        (VELOCITY_CLASS, "runmat.fea.velocity", METERS_PER_SECOND),
        (
            HEAT_FLUX_CLASS,
            "runmat.fea.heat-flux",
            WATTS_PER_SQUARE_METER,
        ),
        (
            HEAT_TRANSFER_COEFFICIENT_CLASS,
            "runmat.fea.heat-transfer-coefficient",
            WATTS_PER_SQUARE_METER_KELVIN,
        ),
        (
            ACOUSTIC_IMPEDANCE_CLASS,
            "runmat.fea.specific-acoustic-impedance",
            PASCAL_SECONDS_PER_METER,
        ),
        (
            ELECTRIC_POTENTIAL_CLASS,
            "runmat.fea.electric-potential",
            VOLTS,
        ),
        (
            ELECTRIC_CURRENT_CLASS,
            "runmat.fea.electric-current",
            AMPERES,
        ),
        (
            MAGNETIC_VECTOR_POTENTIAL_CLASS,
            "runmat.fea.magnetic-vector-potential",
            WEBERS_PER_METER,
        ),
        (LENGTH_CLASS, "runmat.fea.length", METERS),
        (POSITION_CLASS, "runmat.fea.position", METERS),
        (TRANSLATION_CLASS, "runmat.fea.translation", METERS),
        (COMPLEX_PHASE_CLASS, "runmat.fea.complex-phase", RADIANS),
        (
            THERMAL_CONTACT_RESISTANCE_CLASS,
            "runmat.fea.thermal-contact-resistance",
            SQUARE_METER_KELVIN_PER_WATT,
        ),
        (
            FRICTION_COEFFICIENT_CLASS,
            "runmat.fea.friction-coefficient",
            RATIO,
        ),
        (FREQUENCY_CLASS, "runmat.fea.frequency", HERTZ),
        (TIME_CLASS, "runmat.fea.time", SECONDS),
        (
            MASS_DENSITY_CLASS,
            "runmat.fea.mass-density",
            KILOGRAMS_PER_CUBIC_METER,
        ),
        (
            DYNAMIC_VISCOSITY_CLASS,
            "runmat.fea.dynamic-viscosity",
            PASCAL_SECONDS,
        ),
        (
            TURBULENCE_INTENSITY_CLASS,
            "runmat.fea.turbulence-intensity",
            RATIO,
        ),
        (
            RAYLEIGH_MASS_COEFFICIENT_CLASS,
            "runmat.fea.rayleigh-mass-coefficient",
            PER_SECOND,
        ),
        (
            RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS,
            "runmat.fea.rayleigh-stiffness-coefficient",
            SECONDS,
        ),
    ]
}

fn quantity_methods(class_name: &str) -> HashMap<MethodName, RuntimeMethod> {
    let names: &[(&str, &str)] = match class_name {
        DISPLACEMENT_CLASS => &[
            ("meters", "fea.Displacement.meters"),
            ("millimeters", "fea.Displacement.millimeters"),
            ("micrometers", "fea.Displacement.micrometers"),
        ],
        ROTATION_CLASS => &[
            ("radians", "fea.Rotation.radians"),
            ("degrees", "fea.Rotation.degrees"),
        ],
        TEMPERATURE_CLASS => &[
            ("kelvin", "fea.Temperature.kelvin"),
            ("celsius", "fea.Temperature.celsius"),
        ],
        TEMPERATURE_DIFFERENCE_CLASS => &[
            ("kelvin", "fea.TemperatureDifference.kelvin"),
            ("celsius", "fea.TemperatureDifference.celsius"),
        ],
        PRESSURE_CLASS => &[("pascals", "fea.Pressure.pascals")],
        VELOCITY_CLASS => &[("metersPerSecond", "fea.Velocity.metersPerSecond")],
        HEAT_FLUX_CLASS => &[("wattsPerSquareMeter", "fea.HeatFlux.wattsPerSquareMeter")],
        HEAT_TRANSFER_COEFFICIENT_CLASS => &[(
            "wattsPerSquareMeterKelvin",
            "fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin",
        )],
        ACOUSTIC_IMPEDANCE_CLASS => &[(
            "pascalSecondsPerMeter",
            "fea.SpecificAcousticImpedance.pascalSecondsPerMeter",
        )],
        ELECTRIC_POTENTIAL_CLASS => &[("volts", "fea.ElectricPotential.volts")],
        ELECTRIC_CURRENT_CLASS => &[("amperes", "fea.ElectricCurrent.amperes")],
        MAGNETIC_VECTOR_POTENTIAL_CLASS => &[(
            "webersPerMeter",
            "fea.MagneticVectorPotential.webersPerMeter",
        )],
        LENGTH_CLASS => &[
            ("meters", "fea.Length.meters"),
            ("millimeters", "fea.Length.millimeters"),
            ("micrometers", "fea.Length.micrometers"),
        ],
        POSITION_CLASS => &[
            ("meters", "fea.Position.meters"),
            ("millimeters", "fea.Position.millimeters"),
        ],
        TRANSLATION_CLASS => &[
            ("meters", "fea.Translation.meters"),
            ("millimeters", "fea.Translation.millimeters"),
        ],
        COMPLEX_PHASE_CLASS => &[
            ("radians", "fea.ComplexPhase.radians"),
            ("degrees", "fea.ComplexPhase.degrees"),
        ],
        THERMAL_CONTACT_RESISTANCE_CLASS => &[(
            "squareMeterKelvinPerWatt",
            "fea.ThermalContactResistance.squareMeterKelvinPerWatt",
        )],
        FRICTION_COEFFICIENT_CLASS => &[("ratio", "fea.FrictionCoefficient.ratio")],
        FREQUENCY_CLASS => &[("hertz", "fea.Frequency.hertz")],
        TIME_CLASS => &[("seconds", "fea.Time.seconds")],
        MASS_DENSITY_CLASS => &[(
            "kilogramsPerCubicMeter",
            "fea.MassDensity.kilogramsPerCubicMeter",
        )],
        DYNAMIC_VISCOSITY_CLASS => &[("pascalSeconds", "fea.DynamicViscosity.pascalSeconds")],
        TURBULENCE_INTENSITY_CLASS => &[("ratio", "fea.TurbulenceIntensity.ratio")],
        RAYLEIGH_MASS_COEFFICIENT_CLASS => {
            &[("perSecond", "fea.RayleighMassCoefficient.perSecond")]
        }
        RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS => {
            &[("seconds", "fea.RayleighStiffnessCoefficient.seconds")]
        }
        _ => unreachable!("quantity schemas contain only known classes"),
    };
    static_methods(names.iter().copied())
}

fn validate_quantity_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    let fields = quantity_schemas()
        .into_iter()
        .find(|(class_name, _, _)| *class_name == object.class_name().display_name())
        .map(|(_, _, fields)| fields)
        .ok_or_else(|| format!("unknown FEA quantity class `{}`", object.class_name()))?;
    let Some(value) = object.properties().get(fields[0]) else {
        return Err(format!(
            "{} is missing `{}`",
            object.class_name(),
            fields[0]
        ));
    };
    if object.class_name().display_name() == POSITION_CLASS
        || object.class_name().display_name() == TRANSLATION_CLASS
    {
        finite_numeric_vector3("FEA vector quantity", value, "components")
            .map(|_| ())
            .map_err(|error| error.message().to_string())
    } else {
        let value = finite_numeric_scalar("FEA quantity", value, fields[0])
            .map_err(|error| error.message().to_string())?;
        validate_quantity_range(object.class_name().display_name(), value)
    }
}

fn validate_quantity_range(class_name: &str, value: f64) -> Result<(), String> {
    match class_name {
        TEMPERATURE_CLASS if value < 0.0 => {
            Err("fea.Temperature cannot be below zero kelvin".into())
        }
        LENGTH_CLASS
        | HEAT_TRANSFER_COEFFICIENT_CLASS
        | THERMAL_CONTACT_RESISTANCE_CLASS
        | FRICTION_COEFFICIENT_CLASS
        | RAYLEIGH_MASS_COEFFICIENT_CLASS
        | RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS
            if value < 0.0 =>
        {
            Err(format!("{class_name} cannot be negative"))
        }
        ACOUSTIC_IMPEDANCE_CLASS if value <= 0.0 => {
            Err("fea.SpecificAcousticImpedance must be positive".into())
        }
        FREQUENCY_CLASS | MASS_DENSITY_CLASS | DYNAMIC_VISCOSITY_CLASS if value <= 0.0 => {
            Err(format!("{class_name} must be positive"))
        }
        TURBULENCE_INTENSITY_CLASS if !(0.0..=1.0).contains(&value) => {
            Err("fea.TurbulenceIntensity must be in [0, 1]".into())
        }
        _ => Ok(()),
    }
}

pub(super) fn quantity_to_value(
    builtin: &'static str,
    class_name: &'static str,
    field: &'static str,
    value: f64,
) -> BuiltinResult<Value> {
    if !value.is_finite() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "unit conversion produced a non-finite canonical value",
        ));
    }
    validate_quantity_range(class_name, value)
        .map_err(|message| builtin_error(builtin, &ERROR_INPUT, message))?;
    ensure_quantity_classes_registered();
    let mut object = crate::builtins::fea::object_api::ordinary_builder(builtin, class_name)?;
    object.insert(field.into(), Value::Num(value));
    Ok(Value::Object(object.finish()))
}

fn vector_quantity_to_value(
    builtin: &'static str,
    class_name: &'static str,
    coordinates: [f64; 3],
) -> BuiltinResult<Value> {
    ensure_quantity_classes_registered();
    let mut object = crate::builtins::fea::object_api::ordinary_builder(builtin, class_name)?;
    let tensor = Tensor::new_2d(coordinates.to_vec(), 1, 3).map_err(|error| {
        builtin_error(
            builtin,
            &ERROR_INTERNAL,
            format!("failed to construct dimensioned vector: {error}"),
        )
    })?;
    object.insert("meters".into(), Value::Tensor(tensor));
    Ok(Value::Object(object.finish()))
}

pub(super) fn quantity_scalar(
    builtin: &'static str,
    value: &Value,
    class_name: &'static str,
    field: &'static str,
) -> BuiltinResult<f64> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {class_name}"),
        ));
    };
    if object.class_name().display_name() != class_name {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {class_name}, got {}", object.class_name()),
        ));
    }
    let value = object.properties().get(field).ok_or_else(|| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{class_name} is missing `{field}`"),
        )
    })?;
    finite_numeric_scalar(builtin, value, field)
}

pub(super) fn position_meters(builtin: &'static str, value: &Value) -> BuiltinResult<[f64; 3]> {
    vector_quantity(builtin, value, POSITION_CLASS, "position")
}

pub(super) fn translation_meters(builtin: &'static str, value: &Value) -> BuiltinResult<[f64; 3]> {
    vector_quantity(builtin, value, TRANSLATION_CLASS, "translation")
}

fn vector_quantity(
    builtin: &'static str,
    value: &Value,
    class_name: &'static str,
    label: &str,
) -> BuiltinResult<[f64; 3]> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {class_name}"),
        ));
    };
    if object.class_name().display_name() != class_name {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {class_name}, got {}", object.class_name()),
        ));
    }
    let coordinates = object.properties().get("meters").ok_or_else(|| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{class_name} is missing `meters`"),
        )
    })?;
    finite_numeric_vector3(builtin, coordinates, label)
}

pub(super) fn finite_numeric_scalar(
    builtin: &'static str,
    value: &Value,
    label: &str,
) -> BuiltinResult<f64> {
    if !matches!(value, Value::Num(_) | Value::Int(_))
        && !matches!(value, Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor))
    {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must be a real numeric scalar"),
        ));
    }
    let admitted = admit_numeric_real64(
        builtin,
        value,
        format!("{label} must be a real numeric scalar"),
    )?;
    let converted = admitted.values()[0];
    if !converted.is_finite() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must be finite"),
        ));
    }
    Ok(converted)
}

pub(super) fn finite_numeric_vector3(
    builtin: &'static str,
    value: &Value,
    label: &str,
) -> BuiltinResult<[f64; 3]> {
    let Value::Tensor(tensor) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must be a 1-by-3 or 3-by-1 real numeric vector"),
        ));
    };
    if tensor.shape() != [1, 3] && tensor.shape() != [3, 1] {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must be a 1-by-3 or 3-by-1 real numeric vector"),
        ));
    }
    let admitted = admit_numeric_real64(
        builtin,
        value,
        format!("{label} must be a 1-by-3 or 3-by-1 real numeric vector"),
    )?;
    let mut result = [0.0; 3];
    for (index, target) in result.iter_mut().enumerate() {
        *target = admitted.values()[index];
        if !target.is_finite() {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("{label} must contain only finite values"),
            ));
        }
    }
    Ok(result)
}

pub(super) fn finite_numeric_vector(
    builtin: &'static str,
    value: &Value,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    let Value::Tensor(tensor) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must be a real numeric row or column vector"),
        ));
    };
    if tensor.shape().len() != 2
        || (tensor.shape()[0] != 1 && tensor.shape()[1] != 1)
        || tensor.shape()[0].saturating_mul(tensor.shape()[1]) == 0
    {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must be a nonempty real numeric row or column vector"),
        ));
    }
    let admitted = admit_numeric_real64(
        builtin,
        value,
        format!("{label} must be a real numeric row or column vector"),
    )?;
    if admitted.values().iter().any(|value| !value.is_finite()) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must contain only finite values"),
        ));
    }
    let (values, _, _admission_receipt) = admitted.into_owned().into_parts();
    Ok(values)
}

pub(super) fn finite_numeric_vector3_samples(
    builtin: &'static str,
    value: &Value,
    sample_count: usize,
    label: &str,
) -> BuiltinResult<Vec<[f64; 3]>> {
    let Value::Tensor(tensor) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must be an N-by-3 or 3-by-N real numeric matrix"),
        ));
    };
    let [rows, columns] = tensor.shape() else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must be an N-by-3 or 3-by-N real numeric matrix"),
        ));
    };
    let sample_rows = *rows == sample_count && *columns == 3;
    let sample_columns = *rows == 3 && *columns == sample_count;
    if sample_count == 0 || (!sample_rows && !sample_columns) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must contain exactly {sample_count} three-component samples"),
        ));
    }
    let admitted = admit_numeric_real64(
        builtin,
        value,
        format!("{label} must be an N-by-3 or 3-by-N real numeric matrix"),
    )?;
    (0..sample_count)
        .map(|sample| {
            let mut components = [0.0; 3];
            for (component, target) in components.iter_mut().enumerate() {
                let index = if sample_rows {
                    sample + component * sample_count
                } else {
                    component + sample * 3
                };
                *target = admitted.values()[index];
                if !target.is_finite() {
                    return Err(builtin_error(
                        builtin,
                        &ERROR_INPUT,
                        format!("{label} must contain only finite values"),
                    ));
                }
            }
            Ok(components)
        })
        .collect()
}

fn admit_numeric_real64<'a>(
    builtin: &'static str,
    value: &'a Value,
    invalid_message: String,
) -> BuiltinResult<AdmittedReal64Array<'a>> {
    admit_borrowed_real64(
        value,
        AnalysisValueAdmissionPolicy::explicit_host_conversion(),
    )
    .map_err(|_| builtin_error(builtin, &ERROR_INPUT, invalid_message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::value_codec::{decode_inline_value, encode_inline_value};
    use futures::executor::block_on;

    #[test]
    fn scalar_quantities_convert_to_canonical_si_and_remain_portable() {
        let millimeters = block_on(displacement_millimeters_builtin(Value::Int(IntValue::U16(
            25,
        ))))
        .expect("displacement");
        assert_eq!(
            quantity_scalar("test", &millimeters, DISPLACEMENT_CLASS, "meters").unwrap(),
            0.025
        );
        assert_eq!(
            decode_inline_value(&encode_inline_value(&millimeters).expect("encode"))
                .expect("decode"),
            millimeters
        );

        let celsius = block_on(temperature_celsius_builtin(Value::Num(20.0))).expect("temperature");
        assert_eq!(
            quantity_scalar("test", &celsius, TEMPERATURE_CLASS, "kelvin").unwrap(),
            293.15
        );
        let difference = block_on(temperature_difference_celsius_builtin(Value::Num(-20.0)))
            .expect("temperature difference");
        assert_eq!(
            quantity_scalar("test", &difference, TEMPERATURE_DIFFERENCE_CLASS, "kelvin").unwrap(),
            -20.0
        );
    }

    #[test]
    fn quantity_classes_reject_wrong_dimensions_and_invalid_values() {
        let pressure = block_on(pressure_pascals_builtin(Value::Num(5.0))).expect("pressure");
        assert!(quantity_scalar("test", &pressure, DISPLACEMENT_CLASS, "meters").is_err());
        assert!(block_on(rotation_radians_builtin(Value::Num(f64::NAN))).is_err());
        assert!(block_on(temperature_kelvin_builtin(Value::Num(-0.01))).is_err());
        assert!(block_on(length_meters_builtin(Value::Num(-1.0))).is_err());
        assert!(block_on(heat_transfer_coefficient_builtin(Value::Num(-1.0))).is_err());
        assert!(block_on(specific_acoustic_impedance_builtin(Value::Num(0.0))).is_err());
        assert!(block_on(position_meters_builtin(Value::Tensor(
            Tensor::new_2d(vec![1.0, 2.0], 1, 2).unwrap()
        )))
        .is_err());
    }

    #[test]
    fn positions_preserve_vector_shape_and_unit_conversion() {
        let value = block_on(position_millimeters_builtin(Value::Tensor(
            Tensor::new_2d(vec![1000.0, 2000.0, 3000.0], 1, 3).unwrap(),
        )))
        .expect("position");
        let Value::Object(object) = value else {
            panic!("expected position object")
        };
        let Some(Value::Tensor(coordinates)) = object.properties().get("meters") else {
            panic!("expected coordinate tensor")
        };
        assert_eq!(coordinates.shape(), vec![1, 3]);
        assert_eq!(coordinates.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }
}
