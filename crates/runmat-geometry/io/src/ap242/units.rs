use super::{
    p21::{P21Value, PhysicalFile},
    Ap242InspectionError,
};
use runmat_geometry_core::physical::{UnitAttribution, UnitDefinition};
use runmat_geometry_core::{Ap242Measure, QuantityKind};

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
        .or_else(|| entity.component("LENGTH_MEASURE_WITH_UNIT"))
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
    if unit.definition.dimension() != dimension.dimension() {
        return Err(inconsistent(format!(
            "measure #{entity_id} type/unit mismatch"
        )));
    }
    Ap242Measure::try_new(
        dimension,
        source_value,
        unit.definition,
        unit.label,
        UnitAttribution::FileDeclaration {
            context: format!("STEP measure #{entity_id}"),
        },
    )
    .map_err(|error| inconsistent(error.to_string()))
}

#[derive(Clone)]
struct Unit {
    definition: UnitDefinition,
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
        "METRE" => (QuantityKind::Length, 1.0, "m"),
        "GRAM" => (QuantityKind::Mass, 1e-3, "g"),
        "KELVIN" => (QuantityKind::Temperature, 1.0, "K"),
        "SECOND" => (QuantityKind::Time, 1.0, "s"),
        "RADIAN" => (QuantityKind::Angle, 1.0, "rad"),
        "PASCAL" => (QuantityKind::Pressure, 1.0, "Pa"),
        "NEWTON" => (QuantityKind::Force, 1.0, "N"),
        "JOULE" => (QuantityKind::Energy, 1.0, "J"),
        "WATT" => (QuantityKind::Power, 1.0, "W"),
        "VOLT" => (QuantityKind::ElectricPotential, 1.0, "V"),
        "AMPERE" => (QuantityKind::ElectricCurrent, 1.0, "A"),
        "TESLA" => (QuantityKind::MagneticFluxDensity, 1.0, "T"),
        _ => return Err(unsupported(id, "unsupported SI unit")),
    };
    Ok(Unit {
        definition: UnitDefinition::try_for_kind(dimension, factor * base, 0.0)
            .map_err(|e| inconsistent(e.to_string()))?,
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
    let mut dimension = runmat_geometry_core::physical::QuantityKind::Dimensionless.dimension();
    let mut scale = 1.0;
    let mut angle_power = 0_i8;
    let mut labels = Vec::new();
    for value in elements {
        let element_id = value
            .as_reference()
            .ok_or_else(|| unsupported(id, "invalid unit element"))?;
        let element = file
            .entities
            .get(&element_id)
            .and_then(|v| v.component("DERIVED_UNIT_ELEMENT"))
            .ok_or_else(|| unsupported(element_id, "invalid unit element"))?;
        let unit_id = element
            .arguments
            .first()
            .and_then(P21Value::as_reference)
            .ok_or_else(|| unsupported(element_id, "missing element unit"))?;
        let exponent = match element.arguments.get(1) {
            Some(P21Value::Number(value))
                if value.is_finite()
                    && value.fract() == 0.0
                    && *value >= i8::MIN as f64
                    && *value <= i8::MAX as f64 =>
            {
                *value as i8
            }
            _ => return Err(unsupported(element_id, "invalid or out-of-range exponent")),
        };
        let unit = resolve_unit(file, unit_id, stack)?;
        if unit.definition.offset() != 0.0 {
            return Err(unsupported(element_id, "affine units cannot be composed"));
        }
        dimension = dimension
            .multiply(
                unit.definition
                    .dimension()
                    .power(exponent)
                    .ok_or_else(|| unsupported(element_id, "unit exponent overflow"))?,
            )
            .ok_or_else(|| unsupported(element_id, "unit dimension overflow"))?;
        angle_power = angle_power
            .checked_add(
                unit.definition
                    .angle_power()
                    .checked_mul(exponent)
                    .ok_or_else(|| unsupported(element_id, "angle exponent overflow"))?,
            )
            .ok_or_else(|| unsupported(element_id, "angle exponent overflow"))?;
        scale *= unit.definition.scale().powi(i32::from(exponent));
        labels.push(format!("{}^{exponent}", unit.label));
    }
    Ok(Unit {
        definition: UnitDefinition::try_with_angle(dimension, scale, 0.0, angle_power)
            .map_err(|e| inconsistent(e.to_string()))?,
        label: labels.join("*"),
    })
}

fn dimension(value: &str) -> Option<QuantityKind> {
    Some(match value {
        "RATIO_MEASURE" | "POSITIVE_RATIO_MEASURE" | "COUNT_MEASURE" => QuantityKind::Dimensionless,
        "LENGTH_MEASURE" => QuantityKind::Length,
        "PLANE_ANGLE_MEASURE" => QuantityKind::Angle,
        "AREA_MEASURE" => QuantityKind::Area,
        "VOLUME_MEASURE" => QuantityKind::Volume,
        "MASS_MEASURE" => QuantityKind::Mass,
        "MASS_DENSITY_MEASURE" => QuantityKind::MassDensity,
        "THERMODYNAMIC_TEMPERATURE_MEASURE" => QuantityKind::Temperature,
        "PRESSURE_MEASURE" => QuantityKind::Pressure,
        "FORCE_MEASURE" => QuantityKind::Force,
        "ENERGY_MEASURE" => QuantityKind::Energy,
        "POWER_MEASURE" => QuantityKind::Power,
        "THERMAL_CONDUCTIVITY_MEASURE" => QuantityKind::ThermalConductivity,
        "SPECIFIC_HEAT_CAPACITY_MEASURE" => QuantityKind::SpecificHeat,
        "THERMAL_EXPANSION_COEFFICIENT_MEASURE" => QuantityKind::InverseTemperature,
        "ELECTRIC_CONDUCTIVITY_MEASURE" => QuantityKind::ElectricalConductivity,
        "ACOUSTIC_IMPEDANCE_MEASURE" => QuantityKind::SpecificAcousticImpedance,
        "DYNAMIC_VISCOSITY_MEASURE" => QuantityKind::DynamicViscosity,
        "ELECTRIC_POTENTIAL_MEASURE" => QuantityKind::ElectricPotential,
        "ELECTRIC_CURRENT_MEASURE" => QuantityKind::ElectricCurrent,
        "MAGNETIC_FLUX_DENSITY_MEASURE" => QuantityKind::MagneticFluxDensity,
        _ => return None,
    })
}
fn metric_prefix(value: &str) -> Option<(f64, &'static str)> {
    Some(match value {
        "EXA" => (1e18, "E"),
        "PETA" => (1e15, "P"),
        "TERA" => (1e12, "T"),
        "PICO" => (1e-12, "p"),
        "FEMTO" => (1e-15, "f"),
        "ATTO" => (1e-18, "a"),
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

/// The single length unit a STEP file's global unit contexts declare, when it
/// is one RunMat geometry can import in (m, mm or inch). `None` when the file
/// declares no length unit, several different ones, or one RunMat cannot use.
pub fn declared_length_unit(source: &str) -> Option<runmat_geometry_core::LengthUnit> {
    inspect_length_units(source).uniform_unit()
}

/// Preserve each STEP representation context. A mixed-unit assembly is not an
/// unknown-unit file and a conversion-based unit is resolved from its factor,
/// not guessed from its name.
pub fn inspect_length_units(source: &str) -> runmat_geometry_core::physical::LengthUnitEvidence {
    use runmat_geometry_core::physical::{LengthUnitDeclaration, LengthUnitEvidence};
    let file = match super::p21::parse(source) {
        Ok(file) => file,
        Err(error) => {
            return LengthUnitEvidence::Unsupported {
                contexts: Vec::new(),
                reason: format!("cannot parse STEP unit contexts: {error}"),
            }
        }
    };
    let mut declarations = Vec::new();
    let mut unsupported = Vec::new();
    let mut conflicts = Vec::new();
    for entity in file.entities.values() {
        let Some(context) = entity.component("GLOBAL_UNIT_ASSIGNED_CONTEXT") else {
            continue;
        };
        let Some(references) = context.arguments.first().and_then(P21Value::as_list) else {
            unsupported.push(format!("#{}", entity.id));
            continue;
        };
        let mut local = Vec::new();
        for id in references.iter().filter_map(P21Value::as_reference) {
            if !file
                .entities
                .get(&id)
                .is_some_and(|unit| unit.has("LENGTH_UNIT"))
            {
                continue;
            }
            match length_definition(&file, id, &mut Vec::new()) {
                Ok((unit, symbol)) => local.push(LengthUnitDeclaration {
                    context: format!("STEP representation #{}", entity.id),
                    source_symbol: symbol,
                    unit,
                }),
                Err(_) => unsupported.push(format!("#{} unit #{}", entity.id, id)),
            }
        }
        if local.len() > 1 {
            conflicts.extend(local.iter().cloned());
        }
        if local.is_empty() {
            unsupported.push(format!("#{} has no length declaration", entity.id));
        }
        declarations.extend(local);
    }
    if !conflicts.is_empty() {
        return LengthUnitEvidence::Conflicting {
            contexts: conflicts,
            reason: "a STEP representation context declares multiple length units".into(),
        };
    }
    if !unsupported.is_empty() {
        return LengthUnitEvidence::Unsupported {
            contexts: unsupported,
            reason: "STEP contains unresolved length-unit contexts".into(),
        };
    }
    if declarations.is_empty() {
        return LengthUnitEvidence::Unsupported {
            contexts:Vec::new(),reason: "STEP declares no representation-context length unit; repair the exchange declaration before physical analysis".into(),
        };
    }
    LengthUnitEvidence::Declared {
        contexts: declarations,
    }
}

fn length_definition(
    file: &PhysicalFile,
    id: u64,
    stack: &mut Vec<u64>,
) -> Result<(runmat_geometry_core::LengthUnit, String), Ap242InspectionError> {
    use runmat_geometry_core::physical::QuantityKind;
    use runmat_geometry_core::physical::{LengthUnit, UnitDefinition};
    if stack.len() >= 64 || stack.contains(&id) {
        return Err(inconsistent("length-unit dependency cycle or depth limit"));
    }
    stack.push(id);
    let entity = file
        .entities
        .get(&id)
        .ok_or_else(|| inconsistent("missing length unit"))?;
    let result = if let Some(si) = entity.component("SI_UNIT") {
        let metre =
            matches!(si.arguments.get(1),Some(P21Value::Enumeration(base)) if base=="METRE");
        if !metre {
            return Err(unsupported(id, "length context has a non-length SI unit"));
        }
        let (scale, prefix) = match si.arguments.first() {
            Some(P21Value::Null) | None => (1.0, ""),
            Some(P21Value::Enumeration(prefix)) => {
                metric_prefix(prefix).ok_or_else(|| unsupported(id, "unknown length prefix"))?
            }
            _ => return Err(unsupported(id, "invalid length prefix")),
        };
        let definition = UnitDefinition::try_new(QuantityKind::Length.dimension(), scale, 0.0)
            .map_err(|e| inconsistent(e.to_string()))?;
        Ok((
            LengthUnit::try_from_definition(definition).map_err(|e| inconsistent(e.to_string()))?,
            format!("{prefix}m"),
        ))
    } else if let Some(conversion) = entity.component("CONVERSION_BASED_UNIT") {
        let symbol = conversion
            .arguments
            .first()
            .and_then(P21Value::as_string)
            .ok_or_else(|| unsupported(id, "conversion unit has no name"))?;
        let factor = conversion
            .arguments
            .get(1)
            .and_then(P21Value::as_reference)
            .and_then(|id| file.entities.get(&id));
        // The declared factor governs; a standard unit name defines a unit whose factor
        // entity is absent.
        let Some(factor) = factor else {
            let unit = LengthUnit::from_conversion_name(symbol)
                .ok_or_else(|| unsupported(id, "conversion unit has no factor"))?;
            stack.pop();
            return Ok((unit, symbol.to_string()));
        };
        let measure = factor
            .component("LENGTH_MEASURE_WITH_UNIT")
            .or_else(|| factor.component("MEASURE_WITH_UNIT"))
            .ok_or_else(|| unsupported(id, "conversion factor is not a length measure"))?;
        let value = match measure.arguments.first() {
            Some(P21Value::Typed(kind, v))
                if kind == "LENGTH_MEASURE" || kind == "POSITIVE_LENGTH_MEASURE" =>
            {
                match v.as_slice() {
                    [P21Value::Number(v)] => *v,
                    _ => return Err(unsupported(id, "invalid conversion factor")),
                }
            }
            Some(P21Value::Number(v)) => *v,
            _ => return Err(unsupported(id, "invalid conversion factor")),
        };
        let base = measure
            .arguments
            .get(1)
            .and_then(P21Value::as_reference)
            .ok_or_else(|| unsupported(id, "conversion factor has no base unit"))?;
        let (base, _) = length_definition(file, base, stack)?;
        let definition = UnitDefinition::try_new(
            QuantityKind::Length.dimension(),
            value * base.meters_per_unit(),
            0.0,
        )
        .map_err(|e| inconsistent(e.to_string()))?;
        Ok((
            LengthUnit::try_from_definition(definition).map_err(|e| inconsistent(e.to_string()))?,
            symbol.to_string(),
        ))
    } else {
        Err(unsupported(id, "unrecognized length-unit declaration"))
    };
    stack.pop();
    result
}

#[cfg(test)]
mod declared_length_unit_tests {
    use super::declared_length_unit;
    use runmat_geometry_core::LengthUnit;

    fn step(units: &str) -> String {
        format!(
            "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('AP242'));\nENDSEC;\nDATA;\n\
             #1=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#2,#3)) REPRESENTATION_CONTEXT('',''));\n\
             #2=({units});\n\
             #3=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));\n\
             ENDSEC;\nEND-ISO-10303-21;\n"
        )
    }

    #[test]
    fn reads_the_declared_length_unit() {
        assert_eq!(
            declared_length_unit(&step(
                "LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.)"
            )),
            Some(LengthUnit::Millimeter)
        );
        assert_eq!(
            declared_length_unit(&step("LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.)")),
            Some(LengthUnit::Meter)
        );
        assert_eq!(
            declared_length_unit(&step(
                "LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.CENTI.,.METRE.)"
            )),
            Some(LengthUnit::Centimeter)
        );
        assert_eq!(declared_length_unit("not a step file"), None);
    }

    #[test]
    fn conversion_based_units_use_the_declared_factor() {
        let foot = step(
            "CONVERSION_BASED_UNIT('FOOT',#4) LENGTH_UNIT() NAMED_UNIT(#6));\n\
             #4=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(304.8),#5);\n\
             #5=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));\n\
             #6=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.",
        );
        let meters =
            declared_length_unit(&foot).map(runmat_geometry_core::LengthUnit::meters_per_unit);
        assert!((meters.unwrap() - 0.3048).abs() < 1.0e-15, "{meters:?}");
        // A named unit without a readable factor falls back to its definition.
        let mil = step("CONVERSION_BASED_UNIT('MIL',#9) LENGTH_UNIT() NAMED_UNIT(*)");
        let meters =
            declared_length_unit(&mil).map(runmat_geometry_core::LengthUnit::meters_per_unit);
        assert!((meters.unwrap() - 2.54e-5).abs() < 1.0e-18, "{meters:?}");
        // An unknown conversion-based unit without a factor is unresolvable.
        let cubit = step("CONVERSION_BASED_UNIT('CUBIT',#9) LENGTH_UNIT() NAMED_UNIT(*)");
        assert_eq!(declared_length_unit(&cubit), None);
        // Every SI prefix on the metre resolves.
        for (prefix, meters) in [("MICRO", 1.0e-6), ("KILO", 1.0e3), ("PICO", 1.0e-12)] {
            let unit = step(&format!(
                "LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.{prefix}.,.METRE.)"
            ));
            let resolved =
                declared_length_unit(&unit).map(runmat_geometry_core::LengthUnit::meters_per_unit);
            assert_eq!(resolved, Some(meters), "{prefix}");
        }
    }
}

#[cfg(test)]
mod source_context_tests {
    use super::inspect_length_units;
    use runmat_geometry_core::{physical::LengthUnitEvidence, LengthUnit};
    fn step(data: &str) -> String {
        format!(
            "ISO-10303-21;HEADER;FILE_SCHEMA(('AP242'));ENDSEC;DATA;{data}ENDSEC;END-ISO-10303-21;"
        )
    }
    #[test]
    fn mixed_contexts_remain_distinct_and_resolved() {
        let evidence = inspect_length_units(&step("#1=(GLOBAL_UNIT_ASSIGNED_CONTEXT((#3)) REPRESENTATION_CONTEXT('mm',''));#2=(GLOBAL_UNIT_ASSIGNED_CONTEXT((#4)) REPRESENTATION_CONTEXT('m',''));#3=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));#4=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));"));
        assert!(evidence.physical_scale_resolved());
        assert_eq!(evidence.uniform_unit(), None);
        let LengthUnitEvidence::Declared { contexts } = evidence else {
            panic!("declared contexts expected")
        };
        assert_eq!(contexts.len(), 2);
        assert_eq!(contexts[0].unit, LengthUnit::Millimeter);
        assert_eq!(contexts[1].unit, LengthUnit::Meter);
        assert_ne!(contexts[0].context, contexts[1].context);
    }
    #[test]
    fn competing_units_in_one_context_are_not_a_fallback_opportunity() {
        let evidence = inspect_length_units(&step("#1=(GLOBAL_UNIT_ASSIGNED_CONTEXT((#3,#4)) REPRESENTATION_CONTEXT('',''));#3=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));#4=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));"));
        assert!(matches!(&evidence, LengthUnitEvidence::Conflicting { .. }));
        assert!(!evidence
            .clone()
            .resolve_fallback(Some(LengthUnit::Meter), "test override")
            .physical_scale_resolved());
    }
    #[test]
    fn conversion_factor_controls_scale_instead_of_the_label() {
        let evidence = inspect_length_units(&step("#1=(GLOBAL_UNIT_ASSIGNED_CONTEXT((#2)) REPRESENTATION_CONTEXT('',''));#2=(CONVERSION_BASED_UNIT('not-inch',#3) LENGTH_UNIT() NAMED_UNIT(*));#3=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.0254),#4);#4=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));"));
        assert_eq!(evidence.uniform_unit(), Some(LengthUnit::Inch));
        let LengthUnitEvidence::Declared { contexts } = evidence else {
            panic!("declared context expected")
        };
        assert_eq!(contexts[0].source_symbol, "not-inch");
    }
    #[test]
    fn cyclic_conversion_unit_is_explicitly_unresolved() {
        let evidence = inspect_length_units(&step("#1=(GLOBAL_UNIT_ASSIGNED_CONTEXT((#2)) REPRESENTATION_CONTEXT('',''));#2=(CONVERSION_BASED_UNIT('loop',#3) LENGTH_UNIT() NAMED_UNIT(*));#3=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0),#2);"));
        assert!(matches!(evidence, LengthUnitEvidence::Unsupported { .. }));
    }
}
