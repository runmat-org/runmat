use std::collections::HashMap;
use std::sync::Once;

use super::*;

pub(super) const DISPLACEMENT_CLASS: &str = "fea.Displacement";
pub(super) const ROTATION_CLASS: &str = "fea.Rotation";
pub(super) const TEMPERATURE_CLASS: &str = "fea.Temperature";
pub(super) const TEMPERATURE_DIFFERENCE_CLASS: &str = "fea.TemperatureDifference";
pub(super) const PRESSURE_CLASS: &str = "fea.Pressure";
pub(super) const VELOCITY_CLASS: &str = "fea.Velocity";
pub(super) const HEAT_FLUX_CLASS: &str = "fea.HeatFlux";
pub(super) const HEAT_TRANSFER_COEFFICIENT_CLASS: &str = "fea.HeatTransferCoefficient";
pub(super) const ACOUSTIC_IMPEDANCE_CLASS: &str = "fea.SpecificAcousticImpedance";
pub(super) const ELECTRIC_POTENTIAL_CLASS: &str = "fea.ElectricPotential";
pub(super) const ELECTRIC_CURRENT_CLASS: &str = "fea.ElectricCurrent";
pub(super) const MAGNETIC_VECTOR_POTENTIAL_CLASS: &str = "fea.MagneticVectorPotential";
pub(super) const LENGTH_CLASS: &str = "fea.Length";
pub(super) const POSITION_CLASS: &str = "fea.Position";
pub(super) const TRANSLATION_CLASS: &str = "fea.Translation";
pub(super) const COMPLEX_PHASE_CLASS: &str = "fea.ComplexPhase";
pub(super) const THERMAL_CONTACT_RESISTANCE_CLASS: &str = "fea.ThermalContactResistance";
pub(super) const FRICTION_COEFFICIENT_CLASS: &str = "fea.FrictionCoefficient";

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

const QUANTITY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real scalar in the unit named by the constructor.",
}];
const QUANTITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "quantity",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Immutable dimensioned FEA value in canonical SI storage.",
}];
const VECTOR_QUANTITY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "components",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real three-component vector in the unit named by the constructor.",
}];
const VECTOR_QUANTITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "quantity",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Immutable three-component dimensioned FEA value in canonical meter storage.",
}];

const QUANTITY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The exact scalar is validated before one explicit conversion to canonical binary64 SI storage; sufficiently wide integers can round.",
    }];
pub(super) const QUANTITY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "dimensioned scalar value",
        inputs: &QUANTITY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Unit conversion and canonical storage are binary64 operations; resident values are rejected without provider execution.",
    }];
const VECTOR_QUANTITY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "components",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A three-component real integer vector is shape-checked before one explicit conversion to canonical binary64 meter storage; sufficiently wide integers can round.",
    }];
pub(super) const VECTOR_QUANTITY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "three-component dimensioned vector",
        inputs: &VECTOR_QUANTITY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Unit conversion and canonical vector storage are binary64 operations; resident values are rejected without provider execution.",
    }];

macro_rules! scalar_quantity_builtin {
    (
        $function:ident, $descriptor:ident, $signatures:ident,
        $builtin_name:literal, $label:literal, $summary:literal,
        $class:expr, $field:literal, $scale:expr, $offset:expr
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &QUANTITY_INPUTS,
            outputs: &QUANTITY_OUTPUTS,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        #[runtime_builtin(
                                            name = $builtin_name,
                                            category = "fea",
                                            summary = $summary,
                                            keywords = "fea,quantity,units,typed",
                                            descriptor($descriptor),
                                            integer_capabilities(QUANTITY_INTEGER_CAPABILITIES),
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
    DISPLACEMENT_METERS_DESCRIPTOR,
    DISPLACEMENT_METERS_SIGNATURES,
    "fea.Displacement.meters",
    "value = fea.Displacement.meters(meters)",
    "Create a displacement value from meters.",
    DISPLACEMENT_CLASS,
    "meters",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    thermal_contact_resistance_builtin,
    THERMAL_CONTACT_RESISTANCE_DESCRIPTOR,
    THERMAL_CONTACT_RESISTANCE_SIGNATURES,
    "fea.ThermalContactResistance.squareMeterKelvinPerWatt",
    "value = fea.ThermalContactResistance.squareMeterKelvinPerWatt(value)",
    "Create a thermal contact resistance from square meter-kelvin per watt.",
    THERMAL_CONTACT_RESISTANCE_CLASS,
    "square_meter_kelvin_per_watt",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    friction_coefficient_builtin,
    FRICTION_COEFFICIENT_DESCRIPTOR,
    FRICTION_COEFFICIENT_SIGNATURES,
    "fea.FrictionCoefficient.ratio",
    "value = fea.FrictionCoefficient.ratio(value)",
    "Create a non-negative friction coefficient.",
    FRICTION_COEFFICIENT_CLASS,
    "ratio",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    complex_phase_radians_builtin,
    COMPLEX_PHASE_RADIANS_DESCRIPTOR,
    COMPLEX_PHASE_RADIANS_SIGNATURES,
    "fea.ComplexPhase.radians",
    "phase = fea.ComplexPhase.radians(radians)",
    "Create a complex phase shift from radians.",
    COMPLEX_PHASE_CLASS,
    "radians",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    complex_phase_degrees_builtin,
    COMPLEX_PHASE_DEGREES_DESCRIPTOR,
    COMPLEX_PHASE_DEGREES_SIGNATURES,
    "fea.ComplexPhase.degrees",
    "phase = fea.ComplexPhase.degrees(degrees)",
    "Create a complex phase shift from degrees.",
    COMPLEX_PHASE_CLASS,
    "radians",
    std::f64::consts::PI / 180.0,
    0.0
);
scalar_quantity_builtin!(
    displacement_millimeters_builtin,
    DISPLACEMENT_MILLIMETERS_DESCRIPTOR,
    DISPLACEMENT_MILLIMETERS_SIGNATURES,
    "fea.Displacement.millimeters",
    "value = fea.Displacement.millimeters(millimeters)",
    "Create a displacement value from millimeters.",
    DISPLACEMENT_CLASS,
    "meters",
    1.0e-3,
    0.0
);
scalar_quantity_builtin!(
    displacement_micrometers_builtin,
    DISPLACEMENT_MICROMETERS_DESCRIPTOR,
    DISPLACEMENT_MICROMETERS_SIGNATURES,
    "fea.Displacement.micrometers",
    "value = fea.Displacement.micrometers(micrometers)",
    "Create a displacement value from micrometers.",
    DISPLACEMENT_CLASS,
    "meters",
    1.0e-6,
    0.0
);
scalar_quantity_builtin!(
    rotation_radians_builtin,
    ROTATION_RADIANS_DESCRIPTOR,
    ROTATION_RADIANS_SIGNATURES,
    "fea.Rotation.radians",
    "value = fea.Rotation.radians(radians)",
    "Create a rotation value from radians.",
    ROTATION_CLASS,
    "radians",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    rotation_degrees_builtin,
    ROTATION_DEGREES_DESCRIPTOR,
    ROTATION_DEGREES_SIGNATURES,
    "fea.Rotation.degrees",
    "value = fea.Rotation.degrees(degrees)",
    "Create a rotation value from degrees.",
    ROTATION_CLASS,
    "radians",
    std::f64::consts::PI / 180.0,
    0.0
);
scalar_quantity_builtin!(
    temperature_kelvin_builtin,
    TEMPERATURE_KELVIN_DESCRIPTOR,
    TEMPERATURE_KELVIN_SIGNATURES,
    "fea.Temperature.kelvin",
    "value = fea.Temperature.kelvin(kelvin)",
    "Create an absolute temperature value from kelvin.",
    TEMPERATURE_CLASS,
    "kelvin",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    temperature_celsius_builtin,
    TEMPERATURE_CELSIUS_DESCRIPTOR,
    TEMPERATURE_CELSIUS_SIGNATURES,
    "fea.Temperature.celsius",
    "value = fea.Temperature.celsius(celsius)",
    "Create an absolute temperature value from degrees Celsius.",
    TEMPERATURE_CLASS,
    "kelvin",
    1.0,
    273.15
);
scalar_quantity_builtin!(
    temperature_difference_kelvin_builtin,
    TEMPERATURE_DIFFERENCE_KELVIN_DESCRIPTOR,
    TEMPERATURE_DIFFERENCE_KELVIN_SIGNATURES,
    "fea.TemperatureDifference.kelvin",
    "difference = fea.TemperatureDifference.kelvin(value)",
    "Create a temperature difference from kelvin.",
    TEMPERATURE_DIFFERENCE_CLASS,
    "kelvin",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    temperature_difference_celsius_builtin,
    TEMPERATURE_DIFFERENCE_CELSIUS_DESCRIPTOR,
    TEMPERATURE_DIFFERENCE_CELSIUS_SIGNATURES,
    "fea.TemperatureDifference.celsius",
    "difference = fea.TemperatureDifference.celsius(value)",
    "Create a temperature difference from degrees Celsius.",
    TEMPERATURE_DIFFERENCE_CLASS,
    "kelvin",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    pressure_pascals_builtin,
    PRESSURE_PASCALS_DESCRIPTOR,
    PRESSURE_PASCALS_SIGNATURES,
    "fea.Pressure.pascals",
    "value = fea.Pressure.pascals(pascals)",
    "Create a pressure value from pascals.",
    PRESSURE_CLASS,
    "pascals",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    velocity_meters_per_second_builtin,
    VELOCITY_METERS_PER_SECOND_DESCRIPTOR,
    VELOCITY_METERS_PER_SECOND_SIGNATURES,
    "fea.Velocity.metersPerSecond",
    "value = fea.Velocity.metersPerSecond(value)",
    "Create a velocity value from meters per second.",
    VELOCITY_CLASS,
    "meters_per_second",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    heat_flux_watts_per_square_meter_builtin,
    HEAT_FLUX_DESCRIPTOR,
    HEAT_FLUX_SIGNATURES,
    "fea.HeatFlux.wattsPerSquareMeter",
    "value = fea.HeatFlux.wattsPerSquareMeter(value)",
    "Create a heat-flux value from watts per square meter.",
    HEAT_FLUX_CLASS,
    "watts_per_square_meter",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    heat_transfer_coefficient_builtin,
    HEAT_TRANSFER_COEFFICIENT_DESCRIPTOR,
    HEAT_TRANSFER_COEFFICIENT_SIGNATURES,
    "fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin",
    "value = fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin(value)",
    "Create a heat-transfer coefficient from watts per square meter-kelvin.",
    HEAT_TRANSFER_COEFFICIENT_CLASS,
    "watts_per_square_meter_kelvin",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    specific_acoustic_impedance_builtin,
    ACOUSTIC_IMPEDANCE_DESCRIPTOR,
    ACOUSTIC_IMPEDANCE_SIGNATURES,
    "fea.SpecificAcousticImpedance.pascalSecondsPerMeter",
    "value = fea.SpecificAcousticImpedance.pascalSecondsPerMeter(value)",
    "Create a specific acoustic impedance from pascal-seconds per meter.",
    ACOUSTIC_IMPEDANCE_CLASS,
    "pascal_seconds_per_meter",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    electric_potential_volts_builtin,
    ELECTRIC_POTENTIAL_DESCRIPTOR,
    ELECTRIC_POTENTIAL_SIGNATURES,
    "fea.ElectricPotential.volts",
    "value = fea.ElectricPotential.volts(volts)",
    "Create an electric-potential value from volts.",
    ELECTRIC_POTENTIAL_CLASS,
    "volts",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    electric_current_amperes_builtin,
    ELECTRIC_CURRENT_DESCRIPTOR,
    ELECTRIC_CURRENT_SIGNATURES,
    "fea.ElectricCurrent.amperes",
    "value = fea.ElectricCurrent.amperes(amperes)",
    "Create an electric-current value from amperes.",
    ELECTRIC_CURRENT_CLASS,
    "amperes",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    magnetic_vector_potential_builtin,
    MAGNETIC_VECTOR_POTENTIAL_DESCRIPTOR,
    MAGNETIC_VECTOR_POTENTIAL_SIGNATURES,
    "fea.MagneticVectorPotential.webersPerMeter",
    "value = fea.MagneticVectorPotential.webersPerMeter(value)",
    "Create a magnetic-vector-potential value from webers per meter.",
    MAGNETIC_VECTOR_POTENTIAL_CLASS,
    "webers_per_meter",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    length_meters_builtin,
    LENGTH_METERS_DESCRIPTOR,
    LENGTH_METERS_SIGNATURES,
    "fea.Length.meters",
    "value = fea.Length.meters(meters)",
    "Create a length value from meters.",
    LENGTH_CLASS,
    "meters",
    1.0,
    0.0
);
scalar_quantity_builtin!(
    length_millimeters_builtin,
    LENGTH_MILLIMETERS_DESCRIPTOR,
    LENGTH_MILLIMETERS_SIGNATURES,
    "fea.Length.millimeters",
    "value = fea.Length.millimeters(millimeters)",
    "Create a length value from millimeters.",
    LENGTH_CLASS,
    "meters",
    1.0e-3,
    0.0
);
scalar_quantity_builtin!(
    length_micrometers_builtin,
    LENGTH_MICROMETERS_DESCRIPTOR,
    LENGTH_MICROMETERS_SIGNATURES,
    "fea.Length.micrometers",
    "value = fea.Length.micrometers(micrometers)",
    "Create a length value from micrometers.",
    LENGTH_CLASS,
    "meters",
    1.0e-6,
    0.0
);

macro_rules! vector_quantity_builtin {
    (
        $function:ident, $descriptor:ident, $signatures:ident,
        $builtin_name:literal, $label:literal, $summary:literal, $class:expr, $scale:expr
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &VECTOR_QUANTITY_INPUTS,
            outputs: &VECTOR_QUANTITY_OUTPUTS,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        #[runtime_builtin(
                                                    name = $builtin_name,
                                                    category = "fea",
                                                    summary = $summary,
                                            keywords = "fea,quantity,vector,components,units,typed",
                                            descriptor($descriptor),
                                            integer_capabilities(VECTOR_QUANTITY_INTEGER_CAPABILITIES),
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
    POSITION_METERS_DESCRIPTOR,
    POSITION_METERS_SIGNATURES,
    "fea.Position.meters",
    "position = fea.Position.meters(coordinates)",
    "Create a position from meter coordinates.",
    POSITION_CLASS,
    1.0
);
vector_quantity_builtin!(
    position_millimeters_builtin,
    POSITION_MILLIMETERS_DESCRIPTOR,
    POSITION_MILLIMETERS_SIGNATURES,
    "fea.Position.millimeters",
    "position = fea.Position.millimeters(coordinates)",
    "Create a position from millimeter coordinates.",
    POSITION_CLASS,
    1.0e-3
);
vector_quantity_builtin!(
    translation_meters_builtin,
    TRANSLATION_METERS_DESCRIPTOR,
    TRANSLATION_METERS_SIGNATURES,
    "fea.Translation.meters",
    "translation = fea.Translation.meters(components)",
    "Create a translation vector from meter components.",
    TRANSLATION_CLASS,
    1.0
);
vector_quantity_builtin!(
    translation_millimeters_builtin,
    TRANSLATION_MILLIMETERS_DESCRIPTOR,
    TRANSLATION_MILLIMETERS_SIGNATURES,
    "fea.Translation.millimeters",
    "translation = fea.Translation.millimeters(components)",
    "Create a translation vector from millimeter components.",
    TRANSLATION_CLASS,
    1.0e-3
);

pub(super) fn ensure_quantity_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(DISPLACEMENT_CLASS);
    PORTABLE_SCHEMAS.call_once(|| {
        for (class_name, type_identity, fields) in quantity_schemas() {
            crate::execution::value_codec::register_immutable_value_class(
                crate::execution::value_codec::ImmutableValueClassSchema {
                    class_name,
                    type_identity,
                    schema_version: 1,
                    allowed_fields: fields,
                    required_fields: fields,
                    validate: validate_quantity_object,
                },
            )
            .unwrap_or_else(|error| {
                panic!("failed to register immutable FEA quantity class {class_name}: {error}")
            });
        }
    });
    RUNTIME_CLASSES.ensure(|| {
        for (class_name, _, fields) in quantity_schemas() {
            crate::class_registry::register_class(crate::class_registry::RuntimeClass {
                name: class_name.into(),
                parent: None,
                properties: immutable_properties(fields),
                methods: quantity_methods(class_name),
            });
        }
    });
}

fn quantity_schemas() -> [(&'static str, &'static str, &'static [&'static str]); 18] {
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
    ]
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

fn quantity_methods(class_name: &str) -> HashMap<String, crate::class_registry::RuntimeMethod> {
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
        _ => unreachable!("quantity schemas contain only known classes"),
    };
    names
        .iter()
        .map(|(name, function_name)| {
            (
                (*name).into(),
                crate::class_registry::RuntimeMethod {
                    name: (*name).into(),
                    is_static: true,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: (*function_name).into(),
                    implicit_class_argument: None,
                },
            )
        })
        .collect()
}

fn validate_quantity_object(object: &ObjectInstance) -> Result<(), String> {
    let fields = quantity_schemas()
        .into_iter()
        .find(|(class_name, _, _)| *class_name == object.class_name)
        .map(|(_, _, fields)| fields)
        .ok_or_else(|| format!("unknown FEA quantity class `{}`", object.class_name))?;
    let Some(value) = object.properties.get(fields[0]) else {
        return Err(format!("{} is missing `{}`", object.class_name, fields[0]));
    };
    if object.class_name == POSITION_CLASS || object.class_name == TRANSLATION_CLASS {
        finite_numeric_vector3("FEA vector quantity", value, "components")
            .map(|_| ())
            .map_err(|error| error.message().to_string())
    } else {
        let value = finite_numeric_scalar("FEA quantity", value, fields[0])
            .map_err(|error| error.message().to_string())?;
        validate_quantity_range(&object.class_name, value)
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
            if value < 0.0 =>
        {
            Err(format!("{class_name} cannot be negative"))
        }
        ACOUSTIC_IMPEDANCE_CLASS if value <= 0.0 => {
            Err("fea.SpecificAcousticImpedance must be positive".into())
        }
        _ => Ok(()),
    }
}

fn quantity_to_value(
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
    let mut object = ObjectInstance::new(class_name.into());
    object.properties.insert(field.into(), Value::Num(value));
    Ok(Value::Object(object))
}

fn vector_quantity_to_value(
    builtin: &'static str,
    class_name: &'static str,
    coordinates: [f64; 3],
) -> BuiltinResult<Value> {
    ensure_quantity_classes_registered();
    let mut object = ObjectInstance::new(class_name.into());
    let tensor = Tensor::new_2d(coordinates.to_vec(), 1, 3).map_err(|error| {
        builtin_error(
            builtin,
            &ERROR_INTERNAL,
            format!("failed to construct dimensioned vector: {error}"),
        )
    })?;
    object
        .properties
        .insert("meters".into(), Value::Tensor(tensor));
    Ok(Value::Object(object))
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
    if object.class_name != class_name {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {class_name}, got {}", object.class_name),
        ));
    }
    let value = object.properties.get(field).ok_or_else(|| {
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
    if object.class_name != class_name {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {class_name}, got {}", object.class_name),
        ));
    }
    let coordinates = object.properties.get("meters").ok_or_else(|| {
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
    let converted = match value {
        Value::Num(value) => *value,
        Value::Int(value) => integer_to_f64(value),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => numeric_to_f64(
            tensor
                .numeric_value_at(0)
                .expect("validated scalar tensor storage"),
        ),
        _ => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("{label} must be a real numeric scalar"),
            ))
        }
    };
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
    if tensor.shape != [1, 3] && tensor.shape != [3, 1] {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{label} must be a 1-by-3 or 3-by-1 real numeric vector"),
        ));
    }
    let mut result = [0.0; 3];
    for (index, target) in result.iter_mut().enumerate() {
        *target = numeric_to_f64(
            tensor
                .numeric_value_at(index)
                .expect("validated three-element tensor storage"),
        );
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

fn integer_to_f64(value: &IntValue) -> f64 {
    match value {
        IntValue::I8(value) => f64::from(*value),
        IntValue::I16(value) => f64::from(*value),
        IntValue::I32(value) => f64::from(*value),
        IntValue::I64(value) => *value as f64,
        IntValue::U8(value) => f64::from(*value),
        IntValue::U16(value) => f64::from(*value),
        IntValue::U32(value) => f64::from(*value),
        IntValue::U64(value) => *value as f64,
    }
}

fn numeric_to_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        NumericScalar::I8(value) => f64::from(value),
        NumericScalar::I16(value) => f64::from(value),
        NumericScalar::I32(value) => f64::from(value),
        NumericScalar::I64(value) => value as f64,
        NumericScalar::U8(value) => f64::from(value),
        NumericScalar::U16(value) => f64::from(value),
        NumericScalar::U32(value) => f64::from(value),
        NumericScalar::U64(value) => value as f64,
    }
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
        let Some(Value::Tensor(coordinates)) = object.properties.get("meters") else {
            panic!("expected coordinate tensor")
        };
        assert_eq!(coordinates.shape, vec![1, 3]);
        assert_eq!(coordinates.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }
}
