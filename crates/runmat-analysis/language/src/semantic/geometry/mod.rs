mod command;

use super::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, CheckedGeometryInspection,
};
use crate::{
    FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey, FeaSymbolKind,
    GeometryInspectionCapability, LogicalSourcePath,
};
use runmat_geometry_core::{
    CadParameterAccess, CadParameterDefinition, CadParameterPhysicalType, CadParameterSelector,
    GeometryAuthoringCapabilities, GeometryDefinition, GeometryDefinitionSource, GeometryDigest,
    GeometryRegionCardinality, GeometryRegionDefinition, GeometryRegionRole,
    GeometryRegionSelector, GeometrySourceFormat, ImportedGeometryDefinition, UnitSystem,
};
use std::collections::BTreeMap;

pub fn lower_geometry_definitions(
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    inspections: &BTreeMap<(LogicalSourcePath, String), CheckedGeometryInspection>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition> {
    let mut result = BTreeMap::new();
    for (identity, declaration) in declarations {
        if identity.1.kind != FeaSymbolKind::Geometry {
            continue;
        }
        let Some(inspection) = inspections.get(&(identity.0.clone(), identity.1.name.clone()))
        else {
            if imported_file(&declaration.statements).is_some() {
                error(
                    declaration,
                    declaration.span,
                    "imported geometry has no inspection facts",
                    diagnostics,
                );
            }
            continue;
        };
        let initial_errors = diagnostics.len();
        let definition = imported_definition(declaration, inspection, diagnostics);
        if diagnostics.len() == initial_errors {
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
    result
}

fn imported_definition(
    declaration: &CheckedFeaDeclaration,
    inspection: &CheckedGeometryInspection,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> GeometryDefinition {
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
    let regions = declaration
        .statements
        .iter()
        .filter_map(|statement| lower_region(declaration, statement, diagnostics))
        .collect();
    let cad_parameters = declaration
        .statements
        .iter()
        .filter_map(|statement| lower_cad_parameter(declaration, statement, diagnostics))
        .collect();
    GeometryDefinition {
        geometry_id: declaration.key.name.clone(),
        source: GeometryDefinitionSource::Imported(ImportedGeometryDefinition {
            logical_path: inspection.source_path.as_str().to_string(),
            format,
            units,
            content_digest: GeometryDigest::from_bytes(inspection.facts.content_digest.0),
            topology_digest: GeometryDigest::from_bytes(inspection.facts.topology_digest.0),
            capabilities: capabilities(inspection),
        }),
        regions,
        cad_parameters,
    }
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
    Some(CadParameterDefinition {
        parameter_id: name.trim().to_string(),
        physical_type,
        selector,
        access,
    })
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
