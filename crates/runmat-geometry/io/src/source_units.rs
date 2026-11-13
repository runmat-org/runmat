//! Source-unit inspection is shared by exact and preview imports. Output
//! coordinate preferences do not enter this evidence.
use crate::occt::OcctCadFormat;
use runmat_geometry_core::{
    physical::{LengthUnitDeclaration, LengthUnitEvidence, QuantityKind, UnitDefinition},
    LengthUnit,
};

pub fn source_unit_evidence(path: &str, bytes: &[u8]) -> LengthUnitEvidence {
    match crate::detect_geometry_format(path, bytes) {
        crate::GeometryFormat::Step => inspect_source_units(path, bytes, OcctCadFormat::Step),
        crate::GeometryFormat::Iges => inspect_source_units(path, bytes, OcctCadFormat::Iges),
        crate::GeometryFormat::Brep => inspect_source_units(path, bytes, OcctCadFormat::Brep),
        crate::GeometryFormat::Gltf => LengthUnitEvidence::FormatConvention {
            format: "glTF 2.0".into(),
            declaration: LengthUnitDeclaration {
                context: "glTF coordinate convention".into(),
                source_symbol: "m".into(),
                unit: LengthUnit::Meter,
            },
        },
        _ => LengthUnitEvidence::Missing {
            reason: "this source format does not declare physical length units".into(),
        },
    }
}
pub(crate) fn inspect_source_units(
    _path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
) -> LengthUnitEvidence {
    match format {
        OcctCadFormat::Step => crate::ap242::inspect_length_units(&String::from_utf8_lossy(bytes)),
        OcctCadFormat::Iges => iges_units(bytes),
        OcctCadFormat::Brep => LengthUnitEvidence::Missing {
            reason: "BREP does not declare physical units".into(),
        },
    }
}
fn iges_units(bytes: &[u8]) -> LengthUnitEvidence {
    let mut global = Vec::new();
    for line in bytes.split(|&b| b == b'\n') {
        if line.get(72) == Some(&b'G') {
            global.extend_from_slice(&line[..72]);
        }
    }
    let Some(fields) = global_parameters(&global) else {
        return LengthUnitEvidence::Unsupported {
            contexts: vec!["IGES global section".into()],
            reason: "cannot decode IGES global parameters".into(),
        };
    };
    let Some(flag) = fields.get(13).and_then(|s| s.trim().parse::<u32>().ok()) else {
        return LengthUnitEvidence::Unsupported {
            contexts: vec!["IGES global section".into()],
            reason: "IGES has no valid declared length unit flag; repair the source file".into(),
        };
    };
    let name = fields.get(14).map(|s| s.trim()).unwrap_or("");
    let (scale, default_symbol) = match flag {
        1 => (0.0254, "in"),
        2 => (0.001, "mm"),
        4 => (0.3048, "ft"),
        5 => (1609.344, "mile"),
        6 => (1.0, "m"),
        7 => (1000.0, "km"),
        8 => (0.0000254, "mil"),
        9 => (0.000001, "um"),
        10 => (0.01, "cm"),
        11 => (0.0000000254, "microinch"),
        3 => {
            // IGES 5.3 global parameter 15 names that may accompany flag 3.
            let meters = match name.to_ascii_lowercase().as_str() {
                "inch" | "in" => Some(0.0254),
                "mm" | "millimeter" | "millimetre" => Some(0.001),
                "m" | "meter" | "metre" => Some(1.0),
                "cm" => Some(0.01),
                "ft" | "foot" => Some(0.3048),
                "mi" => Some(1609.344),
                "km" => Some(1000.0),
                "mil" => Some(2.54e-5),
                "um" | "micron" => Some(1.0e-6),
                "uin" => Some(2.54e-8),
                _ => None,
            };
            let definition = meters.and_then(LengthUnit::from_meters_per_unit);
            return match definition {
                Some(unit) => declared(unit, name),
                None => LengthUnitEvidence::Unsupported {
                    contexts: vec!["IGES global section".into()],
                    reason: format!("unsupported IGES user-defined unit `{name}`"),
                },
            };
        }
        _ => {
            return LengthUnitEvidence::Unsupported {
                contexts: vec!["IGES global section".into()],
                reason: format!("unsupported IGES unit flag {flag}"),
            }
        }
    };
    let definition = UnitDefinition::try_new(QuantityKind::Length.dimension(), scale, 0.0)
        .expect("catalog scale");
    declared(
        LengthUnit::try_from_definition(definition).expect("length definition"),
        if name.is_empty() {
            default_symbol
        } else {
            name
        },
    )
}
fn declared(unit: LengthUnit, symbol: &str) -> LengthUnitEvidence {
    LengthUnitEvidence::Declared {
        contexts: vec![LengthUnitDeclaration {
            context: "IGES global section".into(),
            source_symbol: symbol.into(),
            unit,
        }],
    }
}
fn global_parameters(bytes: &[u8]) -> Option<Vec<String>> {
    let mut fields = Vec::new();
    let mut at = 0;
    let mut delimiter = b',';
    while at < bytes.len() && fields.len() < 32 {
        while bytes.get(at).is_some_and(|b| b.is_ascii_whitespace()) {
            at += 1;
        }
        if at == bytes.len() {
            break;
        }
        let start = at;
        while bytes.get(at).is_some_and(u8::is_ascii_digit) {
            at += 1;
        }
        let field = if bytes.get(at).is_some_and(|b| matches!(b, b'H' | b'h')) && at > start {
            let size = std::str::from_utf8(&bytes[start..at])
                .ok()?
                .parse::<usize>()
                .ok()?;
            at += 1;
            let end = at.checked_add(size)?;
            let field = String::from_utf8(bytes.get(at..end)?.to_vec()).ok()?;
            at = end;
            field
        } else {
            at = start;
            while bytes.get(at).is_some_and(|b| *b != delimiter && *b != b';') {
                at += 1;
            }
            String::from_utf8(bytes.get(start..at)?.to_vec())
                .ok()?
                .trim()
                .to_string()
        };
        if fields.is_empty() && field.len() == 1 {
            delimiter = field.as_bytes()[0];
        }
        fields.push(field);
        while bytes.get(at).is_some_and(|b| b.is_ascii_whitespace()) {
            at += 1;
        }
        match bytes.get(at) {
            Some(&b) if b == delimiter => at += 1,
            Some(b';') | None => break,
            _ => return None,
        }
    }
    Some(fields)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hollerith_delimiters_are_not_split_as_fields() {
        let f = global_parameters(b"1H,,1H;,4Htest,1Hx,1Hx,1Hx,32,38,6,308,15,1Hx,1.0,2,2HMM;")
            .unwrap();
        assert_eq!(f[13], "2");
        assert_eq!(f[14], "MM");
    }
}

#[cfg(test)]
mod preview_scale_tests {
    use crate::{import_geometry_with_context, GeometryImportContext, GeometryImportOptions};
    use runmat_geometry_core::{physical::LengthUnitEvidence, LengthUnit};
    const STL: &[u8] = b"solid part\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1000 0 0\nvertex 0 1000 0\nendloop\nendfacet\nendsolid part\n";
    #[test]
    fn explicit_source_fallback_converts_faceted_coordinates_to_preview_units() {
        let imported = import_geometry_with_context(
            "part.stl",
            STL,
            GeometryImportOptions {
                source_units: Some(LengthUnit::Millimeter),
                units: LengthUnit::Meter,
                ..Default::default()
            },
            &GeometryImportContext::new(),
        )
        .unwrap();
        assert!(imported.asset.surface_meshes[0]
            .vertices
            .contains(&[1.0, 0.0, 0.0]));
        assert_eq!(
            imported.asset.source.unit_evidence.uniform_unit(),
            Some(LengthUnit::Millimeter)
        );
    }
    #[test]
    fn absent_fallback_stays_unresolved_instead_of_becoming_file_declared_meters() {
        let imported = import_geometry_with_context(
            "part.stl",
            STL,
            GeometryImportOptions::default(),
            &GeometryImportContext::new(),
        )
        .unwrap();
        assert!(matches!(
            imported.asset.source.unit_evidence,
            LengthUnitEvidence::Missing { .. }
        ));
        assert!(imported.asset.surface_meshes[0]
            .vertices
            .contains(&[1000.0, 0.0, 0.0]));
    }
}

/// An OCCT FFI observation, immediately checked into the shared length-unit type.
#[cfg(any(feature = "occt-native", test))]
pub(crate) struct StepContextUnits {
    pub entity_id: u64,
    pub unit: Option<LengthUnit>,
}

#[cfg(any(feature = "occt-native", test))]
pub(crate) fn verify_step_context_units(
    bytes: &[u8],
    actual: &[StepContextUnits],
    require_declared: bool,
) -> Result<(), String> {
    use std::collections::BTreeMap;
    let evidence = crate::ap242::inspect_length_units(&String::from_utf8_lossy(bytes));
    evidence
        .validate()
        .map_err(|error| format!("invalid STEP source-unit evidence: {error}"))?;
    let declarations = match evidence {
        LengthUnitEvidence::Declared { contexts } => contexts,
        LengthUnitEvidence::Missing { .. } | LengthUnitEvidence::Unsupported { .. } if !require_declared => {
            // A read-only preview can retain unresolved scale, never promote an
            // OCCT default into source evidence or physical measurement authority.
            return Ok(());
        }
        _ => return Err("STEP source-unit contexts are missing, conflicting or unsupported; repair the file declaration before physical analysis".into()),
    };
    let mut observed = BTreeMap::new();
    for context in actual {
        if context.entity_id == 0
            || observed
                .insert(
                    format!("STEP representation #{}", context.entity_id),
                    context.unit,
                )
                .is_some()
        {
            return Err("OCCT returned duplicate or unidentified STEP unit contexts".into());
        }
    }
    if observed.len() != declarations.len() {
        return Err("OCCT STEP source-unit contexts differ from the file declarations".into());
    }
    for declaration in declarations {
        let Some(Some(unit)) = observed.get(&declaration.context) else {
            return Err(format!(
                "OCCT did not resolve the declared length unit for {}",
                declaration.context
            ));
        };
        let expected = declaration.unit.meters_per_unit();
        let observed = unit.meters_per_unit();
        // Relative tolerance only accommodates independent floating-point
        // evaluation of conversion-based factors; it has no physical unit.
        if (expected - observed).abs() > expected.abs().max(observed.abs()) * 1.0e-12 {
            return Err(format!(
                "OCCT STEP source-unit factor disagrees with {}",
                declaration.context
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod step_kernel_factor_tests {
    use super::*;
    fn source(contexts: &str) -> Vec<u8> {
        format!("ISO-10303-21;HEADER;FILE_SCHEMA(('AP242'));ENDSEC;DATA;{contexts}ENDSEC;END-ISO-10303-21;").into_bytes()
    }
    fn observation(entity_id: u64, unit: LengthUnit) -> StepContextUnits {
        StepContextUnits {
            entity_id,
            unit: Some(unit),
        }
    }
    const MIXED: &str = "#1=(GLOBAL_UNIT_ASSIGNED_CONTEXT((#3)) REPRESENTATION_CONTEXT('a',''));#2=(GLOBAL_UNIT_ASSIGNED_CONTEXT((#4)) REPRESENTATION_CONTEXT('b',''));#3=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));#4=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));";
    #[test]
    fn mixed_contexts_are_verified_by_identity_and_factor_not_transfer_units() {
        let bytes = source(MIXED);
        let actual = [
            observation(2, LengthUnit::Meter),
            observation(1, LengthUnit::Millimeter),
        ];
        assert!(verify_step_context_units(&bytes, &actual, true).is_ok());
        let swapped = [
            observation(1, LengthUnit::Meter),
            observation(2, LengthUnit::Millimeter),
        ];
        assert!(verify_step_context_units(&bytes, &swapped, true).is_err());
    }
    #[test]
    fn missing_unresolved_and_duplicate_kernel_contexts_are_rejected() {
        let bytes = source(MIXED);
        assert!(
            verify_step_context_units(&bytes, &[observation(1, LengthUnit::Millimeter)], true)
                .is_err()
        );
        assert!(verify_step_context_units(
            &bytes,
            &[
                observation(1, LengthUnit::Millimeter),
                observation(1, LengthUnit::Meter)
            ],
            true
        )
        .is_err());
        assert!(verify_step_context_units(
            &bytes,
            &[
                observation(1, LengthUnit::Millimeter),
                StepContextUnits {
                    entity_id: 2,
                    unit: None
                }
            ],
            true
        )
        .is_err());
    }
    #[test]
    fn unknown_source_scale_can_preview_but_never_acquire_default_physical_authority() {
        let bytes = source("");
        assert!(verify_step_context_units(&bytes, &[], false).is_ok());
        assert!(verify_step_context_units(&bytes, &[], true).is_err());
        assert!(matches!(
            crate::ap242::inspect_length_units(&String::from_utf8_lossy(&bytes)),
            LengthUnitEvidence::Unsupported { .. }
        ));
    }
    #[test]
    fn conflicting_file_declarations_never_accept_the_kernels_chosen_default() {
        let bytes = source("#1=(GLOBAL_UNIT_ASSIGNED_CONTEXT((#3,#4)) REPRESENTATION_CONTEXT('a',''));#3=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));#4=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));");
        assert!(verify_step_context_units(
            &bytes,
            &[observation(1, LengthUnit::Millimeter)],
            false
        )
        .is_err());
    }
    #[test]
    fn conversion_factor_is_authoritative_even_when_the_unit_name_is_misleading() {
        let bytes = source("#1=(GLOBAL_UNIT_ASSIGNED_CONTEXT((#3)) REPRESENTATION_CONTEXT('a',''));#3=(CONVERSION_BASED_UNIT('MILLIMETRE',#5) LENGTH_UNIT() NAMED_UNIT(#6));#4=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));#5=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.0254),#4);#6=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);");
        assert!(
            verify_step_context_units(&bytes, &[observation(1, LengthUnit::Inch)], true).is_ok()
        );
        assert!(
            verify_step_context_units(&bytes, &[observation(1, LengthUnit::Millimeter)], true)
                .is_err()
        );
    }
}
