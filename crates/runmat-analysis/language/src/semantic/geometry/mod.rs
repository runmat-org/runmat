mod command;
mod feature;
mod generated;
mod regeneration;
mod value;

use super::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, CheckedGeometryInspection,
};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey,
    FeaSymbolKind, FeaSymbolOrigin, GeometryInspectionCapability, LogicalSourcePath,
};
use runmat_geometry_core::{
    Ap242ParameterAccess, Ap242PhysicalDimension, CadParameterAccess, CadParameterDefinition,
    CadParameterPhysicalType, CadParameterSelector, DerivedGeometryDefinition,
    GeometryAuthoringCapabilities, GeometryDefinition, GeometryDefinitionRef,
    GeometryDefinitionSource, GeometryDigest, GeometryRegionCardinality, GeometryRegionDefinition,
    GeometryRegionRole, GeometryRegionSelector, GeometrySourceFormat, ImportedGeometryDefinition,
    UnitSystem,
};
use std::collections::BTreeMap;

pub fn lower_geometry_definitions(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    inspections: &BTreeMap<(LogicalSourcePath, String), CheckedGeometryInspection>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition> {
    let mut result = BTreeMap::new();
    for (identity, declaration) in declarations {
        if identity.1.kind != FeaSymbolKind::Geometry {
            continue;
        }
        let initial_errors = diagnostics.len();
        let definition = definition(
            bound,
            declaration,
            inspections.get(&(identity.0.clone(), identity.1.name.clone())),
            diagnostics,
        );
        if diagnostics.len() == initial_errors {
            let Some(definition) = definition else {
                continue;
            };
            match definition.validate() {
                Ok(()) => {
                    result.insert(identity.clone(), definition);
                }
                Err(validation) => error(
                    declaration,
                    declaration.span,
                    validation.to_string(),
                    diagnostics,
                ),
            }
        }
    }
    regeneration::remove_unqualified(&mut result, declarations, diagnostics);
    result
}

fn definition(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    inspection: Option<&CheckedGeometryInspection>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryDefinition> {
    let regions = declaration
        .statements
        .iter()
        .filter_map(|statement| lower_region(declaration, statement, diagnostics))
        .collect::<Vec<_>>();
    let imported = imported_file(&declaration.statements).is_some();
    let construct = named_block(&declaration.statements, "construct");
    let derive = named_block(&declaration.statements, "derive");
    let generated = generated::block(&declaration.statements);
    let source_count = usize::from(imported)
        + usize::from(construct.is_some())
        + usize::from(derive.is_some())
        + usize::from(generated.is_some());
    if source_count != 1 {
        error(
            declaration,
            declaration.span,
            "geometry requires exactly one imported, constructed, derived, or generated source",
            diagnostics,
        );
        return None;
    }
    let source = if imported {
        if declaration.source.is_some() {
            error(
                declaration,
                declaration.span,
                "imported geometry cannot declare a `from` source",
                diagnostics,
            );
        }
        let Some(inspection) = inspection else {
            error(
                declaration,
                declaration.span,
                "imported geometry has no inspection facts",
                diagnostics,
            );
            return None;
        };
        imported_source(declaration, inspection, diagnostics)
    } else if let Some(statements) = construct {
        if declaration.source.is_some() {
            error(
                declaration,
                declaration.span,
                "constructed geometry cannot declare a `from` source",
                diagnostics,
            );
        }
        GeometryDefinitionSource::Constructed(feature::graph(
            declaration,
            statements,
            &regions,
            diagnostics,
        )?)
    } else if let Some(statements) = derive {
        let source = derived_source(bound, declaration, diagnostics)?;
        GeometryDefinitionSource::Derived(DerivedGeometryDefinition {
            source,
            graph: feature::graph(declaration, statements, &regions, diagnostics)?,
        })
    } else {
        if declaration.source.is_some() {
            error(
                declaration,
                declaration.span,
                "generated geometry cannot declare a `from` source",
                diagnostics,
            );
        }
        generated::source(
            bound,
            declaration,
            generated.expect("source count establishes generated block"),
            diagnostics,
        )?
    };
    Some(GeometryDefinition {
        module_path: declaration.module.as_str().to_string(),
        geometry_id: declaration.key.name.clone(),
        parameters: value::parameters(declaration, diagnostics),
        source,
        regions,
        cad_parameters: declaration
            .statements
            .iter()
            .filter_map(|statement| {
                lower_cad_parameter(declaration, statement, inspection, diagnostics)
            })
            .collect(),
    })
}

fn imported_source(
    declaration: &CheckedFeaDeclaration,
    inspection: &CheckedGeometryInspection,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> GeometryDefinitionSource {
    let units = imported_units(declaration, diagnostics).unwrap_or(UnitSystem::Unspecified);
    let format = source_format(inspection.source_path.as_str()).unwrap_or_else(|| {
        error(
            declaration,
            declaration.span,
            "geometry import requires a recognized exact-CAD extension",
            diagnostics,
        );
        GeometrySourceFormat::Step
    });
    GeometryDefinitionSource::Imported(ImportedGeometryDefinition {
        logical_path: inspection.source_path.as_str().to_string(),
        format,
        units,
        content_digest: GeometryDigest::from_bytes(inspection.facts.content_digest.0),
        topology_digest: GeometryDigest::from_bytes(inspection.facts.topology_digest.0),
        capabilities: capabilities(inspection),
        exchange: inspection.facts.exchange.clone(),
    })
}

fn named_block<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a [CheckedFeaStatement]> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Block {
            header, statements, ..
        } = &statement.kind
        else {
            return None;
        };
        (header.trim() == name).then_some(statements.as_slice())
    })
}

fn derived_source(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryDefinitionRef> {
    let Some([namespace, name]) = declaration.source.as_deref() else {
        error(
            declaration,
            declaration.span,
            "derived geometry requires `from geometry.<name>`",
            diagnostics,
        );
        return None;
    };
    if namespace != "geometry" {
        error(
            declaration,
            declaration.span,
            "derived geometry source must use the `geometry` namespace",
            diagnostics,
        );
        return None;
    }
    let key = FeaSymbolKey {
        kind: FeaSymbolKind::Geometry,
        name: name.clone(),
    };
    let Some(symbol) = bound
        .symbols
        .get(&declaration.module)
        .and_then(|symbols| symbols.get(&key))
    else {
        error(
            declaration,
            declaration.span,
            format!("derived geometry references unknown geometry `{name}`"),
            diagnostics,
        );
        return None;
    };
    Some(match &symbol.origin {
        FeaSymbolOrigin::Local => GeometryDefinitionRef {
            module_path: declaration.module.as_str().to_string(),
            geometry_id: name.clone(),
        },
        FeaSymbolOrigin::Import {
            target_module,
            target,
        } => GeometryDefinitionRef {
            module_path: target_module.as_str().to_string(),
            geometry_id: target.name.clone(),
        },
        FeaSymbolOrigin::Function => {
            error(
                declaration,
                declaration.span,
                "derived geometry source resolved to a function",
                diagnostics,
            );
            return None;
        }
    })
}

fn imported_units(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<UnitSystem> {
    let Some(value) = assignment(&declaration.statements, "units") else {
        error(
            declaration,
            declaration.span,
            "imported geometry requires explicit `units`",
            diagnostics,
        );
        return None;
    };
    let CheckedExpressionKind::Enum(unit) = &value.kind else {
        error(
            declaration,
            value.span,
            "geometry units require a unit-system name",
            diagnostics,
        );
        return None;
    };
    match unit.as_str() {
        "m" | "meter" | "meters" => Some(UnitSystem::Meter),
        "mm" | "millimeter" | "millimeters" => Some(UnitSystem::Millimeter),
        "in" | "inch" | "inches" => Some(UnitSystem::Inch),
        _ => {
            error(
                declaration,
                value.span,
                format!("unsupported geometry unit system `{unit}`"),
                diagnostics,
            );
            None
        }
    }
}

fn lower_region(
    declaration: &CheckedFeaDeclaration,
    statement: &CheckedFeaStatement,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryRegionDefinition> {
    let CheckedFeaStatementKind::Block {
        header,
        header_span,
        statements,
    } = &statement.kind
    else {
        return None;
    };
    let rest = header.trim().strip_prefix("region ")?;
    let Some((name, role)) = rest.split_once(':') else {
        error(
            declaration,
            *header_span,
            "region requires `region <name> : <role>`",
            diagnostics,
        );
        return None;
    };
    let Some(role) = region_role(role.trim()) else {
        error(
            declaration,
            *header_span,
            format!("unsupported region role `{}`", role.trim()),
            diagnostics,
        );
        return None;
    };
    let selector = command::region_selector(statements).unwrap_or_else(|message| {
        error(declaration, statement.span, message, diagnostics);
        GeometryRegionSelector::Result
    });
    let cardinality = command::region_cardinality(statements).unwrap_or_else(|message| {
        error(declaration, statement.span, message, diagnostics);
        GeometryRegionCardinality::ExactlyOne
    });
    Some(GeometryRegionDefinition {
        region_id: name.trim().to_string(),
        role,
        selector,
        cardinality,
    })
}

fn lower_cad_parameter(
    declaration: &CheckedFeaDeclaration,
    statement: &CheckedFeaStatement,
    inspection: Option<&CheckedGeometryInspection>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<CadParameterDefinition> {
    let CheckedFeaStatementKind::Block {
        header,
        header_span,
        statements,
    } = &statement.kind
    else {
        return None;
    };
    let rest = header.trim().strip_prefix("cad_parameter ")?;
    let Some((name, physical_type)) = rest.split_once(':') else {
        error(
            declaration,
            *header_span,
            "CAD parameter requires `cad_parameter <name> : <type>`",
            diagnostics,
        );
        return None;
    };
    let Some(physical_type) = parameter_type(physical_type.trim()) else {
        error(
            declaration,
            *header_span,
            format!("unsupported CAD parameter type `{}`", physical_type.trim()),
            diagnostics,
        );
        return None;
    };
    let selector = command::parameter_selector(statements).unwrap_or_else(|message| {
        error(declaration, statement.span, message, diagnostics);
        CadParameterSelector::Name(String::new())
    });
    let access = command::parameter_access(statements).unwrap_or_else(|message| {
        error(declaration, statement.span, message, diagnostics);
        CadParameterAccess::Readable
    });
    let Some(inventory) = inspection.and_then(|value| value.facts.exchange.as_ref()) else {
        error(
            declaration,
            statement.span,
            "CAD parameter declarations require a normalized AP242 exchange inventory",
            diagnostics,
        );
        return None;
    };
    let matches = inventory
        .parameters
        .iter()
        .filter(|parameter| match &selector {
            CadParameterSelector::Name(name) => parameter.name == *name,
            CadParameterSelector::PersistentId(id) => {
                parameter.persistent_id.as_deref() == Some(id.as_str())
            }
        })
        .collect::<Vec<_>>();
    let [source] = matches.as_slice() else {
        let detail = if matches.is_empty() {
            "does not resolve to a source parameter"
        } else {
            "is ambiguous in the source exchange inventory"
        };
        error(
            declaration,
            statement.span,
            format!("CAD parameter `{}` {detail}", name.trim()),
            diagnostics,
        );
        return None;
    };
    if source.value.dimension != physical_dimension(physical_type) {
        error(
            declaration,
            *header_span,
            format!(
                "CAD parameter `{}` is declared as `{}` but the source has dimension `{:?}`",
                name.trim(),
                physical_type_name(physical_type),
                source.value.dimension
            ),
            diagnostics,
        );
        return None;
    }
    if access == CadParameterAccess::Writable && source.access != Ap242ParameterAccess::Regenerable
    {
        error(
            declaration,
            statement.span,
            format!(
                "CAD parameter `{}` is read-only; writable access requires a qualified regenerative source",
                name.trim()
            ),
            diagnostics,
        );
        return None;
    }
    Some(CadParameterDefinition {
        parameter_id: name.trim().to_string(),
        physical_type,
        selector,
        access,
        source: (*source).clone(),
    })
}

fn physical_dimension(value: CadParameterPhysicalType) -> Ap242PhysicalDimension {
    match value {
        CadParameterPhysicalType::Dimensionless => Ap242PhysicalDimension::Dimensionless,
        CadParameterPhysicalType::Length => Ap242PhysicalDimension::Length,
        CadParameterPhysicalType::Angle => Ap242PhysicalDimension::PlaneAngle,
        CadParameterPhysicalType::Area => Ap242PhysicalDimension::Area,
        CadParameterPhysicalType::Volume => Ap242PhysicalDimension::Volume,
        CadParameterPhysicalType::Mass => Ap242PhysicalDimension::Mass,
        CadParameterPhysicalType::Temperature => Ap242PhysicalDimension::Temperature,
    }
}

fn physical_type_name(value: CadParameterPhysicalType) -> &'static str {
    match value {
        CadParameterPhysicalType::Dimensionless => "dimensionless",
        CadParameterPhysicalType::Length => "length",
        CadParameterPhysicalType::Angle => "angle",
        CadParameterPhysicalType::Area => "area",
        CadParameterPhysicalType::Volume => "volume",
        CadParameterPhysicalType::Mass => "mass",
        CadParameterPhysicalType::Temperature => "temperature",
    }
}

fn capabilities(inspection: &CheckedGeometryInspection) -> GeometryAuthoringCapabilities {
    let has = |capability| inspection.facts.capabilities.contains(&capability);
    GeometryAuthoringCapabilities {
        topology: has(GeometryInspectionCapability::Topology),
        persistent_ids: has(GeometryInspectionCapability::PersistentIds),
        parameters: has(GeometryInspectionCapability::Parameters),
        engineering_properties: has(GeometryInspectionCapability::EngineeringProperties),
        material_candidates: has(GeometryInspectionCapability::MaterialCandidates),
        regeneration: has(GeometryInspectionCapability::Regeneration),
    }
}

fn assignment<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a CheckedExpression> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        (target.as_slice() == [name]).then_some(value)
    })
}

fn imported_file(statements: &[CheckedFeaStatement]) -> Option<&CheckedExpression> {
    assignment(statements, "file")
}

fn source_format(path: &str) -> Option<GeometrySourceFormat> {
    let extension = path.rsplit_once('.')?.1.to_ascii_lowercase();
    Some(match extension.as_str() {
        "step" | "stp" => GeometrySourceFormat::Step,
        "iges" | "igs" => GeometrySourceFormat::Iges,
        "brep" => GeometrySourceFormat::Brep,
        _ => return None,
    })
}

fn region_role(value: &str) -> Option<GeometryRegionRole> {
    Some(match value {
        "vertex" | "vertices" => GeometryRegionRole::Vertex,
        "edge" | "edges" => GeometryRegionRole::Edge,
        "face" | "faces" => GeometryRegionRole::Face,
        "solid" | "solids" => GeometryRegionRole::Solid,
        "cell" | "cells" => GeometryRegionRole::Cell,
        _ => return None,
    })
}

fn parameter_type(value: &str) -> Option<CadParameterPhysicalType> {
    Some(match value {
        "real" | "dimensionless" => CadParameterPhysicalType::Dimensionless,
        "length" => CadParameterPhysicalType::Length,
        "angle" => CadParameterPhysicalType::Angle,
        "area" => CadParameterPhysicalType::Area,
        "volume" => CadParameterPhysicalType::Volume,
        "mass" => CadParameterPhysicalType::Mass,
        "temperature" => CadParameterPhysicalType::Temperature,
        _ => return None,
    })
}

fn error(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.geometry",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}
