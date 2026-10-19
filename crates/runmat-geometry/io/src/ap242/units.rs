use super::{
    p21::{P21Value, PhysicalFile},
    Ap242InspectionError,
};
use runmat_geometry_core::{Ap242Measure, Ap242PhysicalDimension};

pub(super) fn measure(
    file: &PhysicalFile,
    entity_id: u64,
) -> Result<Ap242Measure, Ap242InspectionError> {
    let entity = file
        .entities
        .get(&entity_id)
        .ok_or_else(|| inconsistent(format!("missing measure #{entity_id}")))?;
    let component = entity
        .component("MEASURE_WITH_UNIT")
        .or_else(|| entity.component("MEASURE_REPRESENTATION_ITEM"))
        .ok_or_else(|| unsupported(entity_id, "not a measure"))?;
    let offset = usize::from(component.name == "MEASURE_REPRESENTATION_ITEM");
    let (kind, source_value) = match component.arguments.get(offset) {
        Some(P21Value::Typed(kind, values)) => match values.as_slice() {
            [P21Value::Number(value)] if value.is_finite() => (kind.as_str(), *value),
            _ => return Err(unsupported(entity_id, "value is not one finite scalar")),
        },
        _ => return Err(unsupported(entity_id, "missing typed value")),
    };
    let dimension =
        dimension(kind).ok_or_else(|| unsupported(entity_id, "unsupported measure type"))?;
    let unit_id = component
        .arguments
        .get(offset + 1)
        .and_then(P21Value::as_reference)
        .ok_or_else(|| unsupported(entity_id, "missing unit"))?;
    let unit = resolve_unit(file, unit_id, &mut Vec::new())?;
    if unit.dimension != dimension {
        return Err(inconsistent(format!(
            "measure #{entity_id} type/unit mismatch"
        )));
    }
    Ok(Ap242Measure {
        dimension,
        value_si: source_value * unit.scale,
        source_value,
        source_unit: unit.label,
    })
}

#[derive(Clone)]
struct Unit {
    dimension: Ap242PhysicalDimension,
    scale: f64,
    label: String,
}

fn resolve_unit(
    file: &PhysicalFile,
    id: u64,
    stack: &mut Vec<u64>,
) -> Result<Unit, Ap242InspectionError> {
    if stack.contains(&id) {
        return Err(inconsistent("unit dependency cycle"));
    }
    stack.push(id);
    let entity = file
        .entities
        .get(&id)
        .ok_or_else(|| inconsistent(format!("missing unit #{id}")))?;
    let result = if let Some(value) = entity.component("SI_UNIT") {
        si_unit(id, &value.arguments)
    } else if let Some(value) = entity.component("DERIVED_UNIT") {
        derived_unit(file, id, &value.arguments, stack)
    } else {
        Err(unsupported(id, "unit is not SI or derived SI"))
    };
    stack.pop();
    result
}

fn si_unit(id: u64, args: &[P21Value]) -> Result<Unit, Ap242InspectionError> {
    let (factor, prefix) = match args.first() {
        Some(P21Value::Null) | None => (1.0, ""),
        Some(P21Value::Enumeration(value)) => {
            metric_prefix(value).ok_or_else(|| unsupported(id, "unsupported SI prefix"))?
        }
        _ => return Err(unsupported(id, "invalid SI prefix")),
    };
    let name = match args.get(1) {
        Some(P21Value::Enumeration(value)) => value.as_str(),
        _ => return Err(unsupported(id, "invalid SI unit")),
    };
    let (dimension, base, symbol) = match name {
        "METRE" => (Ap242PhysicalDimension::Length, 1.0, "m"),
        "GRAM" => (Ap242PhysicalDimension::Mass, 1e-3, "g"),
        "KELVIN" => (Ap242PhysicalDimension::Temperature, 1.0, "K"),
        "SECOND" => (Ap242PhysicalDimension::Time, 1.0, "s"),
        "RADIAN" => (Ap242PhysicalDimension::PlaneAngle, 1.0, "rad"),
        "PASCAL" => (Ap242PhysicalDimension::Pressure, 1.0, "Pa"),
        "NEWTON" => (Ap242PhysicalDimension::Force, 1.0, "N"),
        "JOULE" => (Ap242PhysicalDimension::Energy, 1.0, "J"),
        "WATT" => (Ap242PhysicalDimension::Power, 1.0, "W"),
        "VOLT" => (Ap242PhysicalDimension::ElectricPotential, 1.0, "V"),
        "AMPERE" => (Ap242PhysicalDimension::ElectricCurrent, 1.0, "A"),
        "TESLA" => (Ap242PhysicalDimension::MagneticFluxDensity, 1.0, "T"),
        _ => return Err(unsupported(id, "unsupported SI unit")),
    };
    Ok(Unit {
        dimension,
        scale: factor * base,
        label: format!("{prefix}{symbol}"),
    })
}

fn derived_unit(
    file: &PhysicalFile,
    id: u64,
    args: &[P21Value],
    stack: &mut Vec<u64>,
) -> Result<Unit, Ap242InspectionError> {
    let elements = args
        .first()
        .and_then(P21Value::as_list)
        .ok_or_else(|| unsupported(id, "missing derived-unit elements"))?;
    let mut powers = [0_i32; 5];
    let mut scale = 1.0;
    let mut labels = Vec::new();
    for value in elements {
        let element_id = value
            .as_reference()
            .ok_or_else(|| unsupported(id, "invalid unit element"))?;
        let element = file
            .entities
            .get(&element_id)
            .and_then(|value| value.component("DERIVED_UNIT_ELEMENT"))
            .ok_or_else(|| unsupported(element_id, "invalid unit element"))?;
        let unit_id = element
            .arguments
            .first()
            .and_then(P21Value::as_reference)
            .ok_or_else(|| unsupported(element_id, "missing element unit"))?;
        let exponent = match element.arguments.get(1) {
            Some(P21Value::Number(value)) if value.fract() == 0.0 => *value as i32,
            _ => return Err(unsupported(element_id, "noninteger exponent")),
        };
        let unit = resolve_unit(file, unit_id, stack)?;
        let slot = base_slot(unit.dimension)
            .ok_or_else(|| unsupported(element_id, "unsupported nested dimension"))?;
        powers[slot] += exponent;
        scale *= unit.scale.powi(exponent);
        labels.push(format!("{}^{exponent}", unit.label));
    }
    let dimension = match powers {
        [0, 0, 0, 0, 0] => Ap242PhysicalDimension::Dimensionless,
        [2, 0, 0, 0, 0] => Ap242PhysicalDimension::Area,
        [3, 0, 0, 0, 0] => Ap242PhysicalDimension::Volume,
        [-3, 1, 0, 0, 0] => Ap242PhysicalDimension::MassDensity,
        [-1, 1, -2, 0, 0] => Ap242PhysicalDimension::Pressure,
        [1, 1, -2, 0, 0] => Ap242PhysicalDimension::Force,
        [2, 1, -2, 0, 0] => Ap242PhysicalDimension::Energy,
        [2, 1, -3, 0, 0] => Ap242PhysicalDimension::Power,
        [1, 1, -3, -1, 0] => Ap242PhysicalDimension::ThermalConductivity,
        [2, 0, -2, -1, 0] => Ap242PhysicalDimension::SpecificHeat,
        [0, 0, 0, -1, 0] => Ap242PhysicalDimension::InverseTemperature,
        [-3, -1, 3, 0, 2] => Ap242PhysicalDimension::ElectricalConductivity,
        [-2, 1, -1, 0, 0] => Ap242PhysicalDimension::AcousticImpedance,
        [-1, 1, -1, 0, 0] => Ap242PhysicalDimension::DynamicViscosity,
        _ => return Err(unsupported(id, "unsupported derived dimension")),
    };
    Ok(Unit {
        dimension,
        scale,
        label: labels.join("*"),
    })
}

fn base_slot(value: Ap242PhysicalDimension) -> Option<usize> {
    match value {
        Ap242PhysicalDimension::Length => Some(0),
        Ap242PhysicalDimension::Mass => Some(1),
        Ap242PhysicalDimension::Time => Some(2),
        Ap242PhysicalDimension::Temperature => Some(3),
        Ap242PhysicalDimension::ElectricCurrent => Some(4),
        _ => None,
    }
}
fn dimension(value: &str) -> Option<Ap242PhysicalDimension> {
    Some(match value {
        "RATIO_MEASURE" | "POSITIVE_RATIO_MEASURE" | "COUNT_MEASURE" => {
            Ap242PhysicalDimension::Dimensionless
        }
        "LENGTH_MEASURE" => Ap242PhysicalDimension::Length,
        "PLANE_ANGLE_MEASURE" => Ap242PhysicalDimension::PlaneAngle,
        "AREA_MEASURE" => Ap242PhysicalDimension::Area,
        "VOLUME_MEASURE" => Ap242PhysicalDimension::Volume,
        "MASS_MEASURE" => Ap242PhysicalDimension::Mass,
        "MASS_DENSITY_MEASURE" => Ap242PhysicalDimension::MassDensity,
        "THERMODYNAMIC_TEMPERATURE_MEASURE" => Ap242PhysicalDimension::Temperature,
        "PRESSURE_MEASURE" => Ap242PhysicalDimension::Pressure,
        "FORCE_MEASURE" => Ap242PhysicalDimension::Force,
        "ENERGY_MEASURE" => Ap242PhysicalDimension::Energy,
        "POWER_MEASURE" => Ap242PhysicalDimension::Power,
        "THERMAL_CONDUCTIVITY_MEASURE" => Ap242PhysicalDimension::ThermalConductivity,
        "SPECIFIC_HEAT_CAPACITY_MEASURE" => Ap242PhysicalDimension::SpecificHeat,
        "THERMAL_EXPANSION_COEFFICIENT_MEASURE" => Ap242PhysicalDimension::InverseTemperature,
        "ELECTRIC_CONDUCTIVITY_MEASURE" => Ap242PhysicalDimension::ElectricalConductivity,
        "ACOUSTIC_IMPEDANCE_MEASURE" => Ap242PhysicalDimension::AcousticImpedance,
        "DYNAMIC_VISCOSITY_MEASURE" => Ap242PhysicalDimension::DynamicViscosity,
        "ELECTRIC_POTENTIAL_MEASURE" => Ap242PhysicalDimension::ElectricPotential,
        "ELECTRIC_CURRENT_MEASURE" => Ap242PhysicalDimension::ElectricCurrent,
        "MAGNETIC_FLUX_DENSITY_MEASURE" => Ap242PhysicalDimension::MagneticFluxDensity,
        _ => return None,
    })
}
fn metric_prefix(value: &str) -> Option<(f64, &'static str)> {
    Some(match value {
        "MILLI" => (1e-3, "m"),
        "CENTI" => (1e-2, "c"),
        "DECI" => (1e-1, "d"),
        "DECA" => (1e1, "da"),
        "HECTO" => (1e2, "h"),
        "KILO" => (1e3, "k"),
        "MEGA" => (1e6, "M"),
        "GIGA" => (1e9, "G"),
        "MICRO" => (1e-6, "u"),
        "NANO" => (1e-9, "n"),
        _ => return None,
    })
}
fn unsupported(id: u64, detail: &str) -> Ap242InspectionError {
    Ap242InspectionError::Unsupported(format!("entity #{id}: {detail}"))
}
fn inconsistent(detail: impl Into<String>) -> Ap242InspectionError {
    Ap242InspectionError::Inconsistent(detail.into())
}
