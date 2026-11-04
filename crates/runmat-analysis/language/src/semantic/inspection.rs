use crate::{
    resolve_logical_path, BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation,
    FeaDiagnosticPhase, GeometryInspection, GeometryInspectionRequest, GeometryInspectionSource,
    LogicalSourcePath,
};
use runmat_design_syntax::{
    FeaDeclarationKind, FeaExpressionKind, FeaModuleItem, FeaStatementKind,
};
use runmat_types::Span;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct CheckedGeometryInspection {
    pub module: LogicalSourcePath,
    pub geometry: String,
    pub source_path: LogicalSourcePath,
    pub facts: GeometryInspection,
}

pub fn inspect_geometry_declarations<H: GeometryInspectionSource + ?Sized>(
    host: &H,
    bound: &BoundFeaProgram,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, String), CheckedGeometryInspection> {
    let mut inspections = BTreeMap::new();
    for (module_path, module) in &bound.graph.modules {
        let Some(syntax) = &module.syntax else {
            continue;
        };
        for item in &syntax.items {
            let FeaModuleItem::Declaration(declaration) = item else {
                continue;
            };
            if declaration.kind != FeaDeclarationKind::Geometry {
                continue;
            }
            let Some((relative_path, source_units, span)) = imported_geometry_source(declaration)
            else {
                continue;
            };
            let shared_interfaces = declaration.body.statements.iter().any(|statement| {
                matches!(&statement.kind, FeaStatementKind::Assignment { target, value, .. }
                    if target.segments.len() == 1
                        && target.segments[0].text == "interfaces"
                        && matches!(&value.kind, FeaExpressionKind::Reference(reference)
                            if reference.segments.len() == 1
                                && reference.segments[0].text == "shared"))
            });
            let source_path =
                match resolve_logical_path(&bound.graph.source_root, module_path, &relative_path) {
                    Ok(path) => path,
                    Err(error) => {
                        diagnostics.push(FeaDiagnostic::error(
                            "fea.inspection.invalid_geometry_path",
                            format!("cannot resolve geometry source `{relative_path}`: {error}"),
                            FeaDiagnosticPhase::Inspection,
                            Some(location(module_path, span)),
                        ));
                        continue;
                    }
                };
            match host.inspect(&GeometryInspectionRequest {
                referrer: module_path.clone(),
                logical_path: source_path.clone(),
                source_units,
                expected_content_digest: None,
                shared_interfaces,
            }) {
                Ok(facts) => {
                    inspections.insert(
                        (module_path.clone(), declaration.name.text.clone()),
                        CheckedGeometryInspection {
                            module: module_path.clone(),
                            geometry: declaration.name.text.clone(),
                            source_path,
                            facts,
                        },
                    );
                }
                Err(error) => {
                    let mut diagnostic = FeaDiagnostic::error(
                        error.code,
                        error.message,
                        FeaDiagnosticPhase::Inspection,
                        Some(location(module_path, span)),
                    );
                    diagnostic.availability = error.availability;
                    diagnostics.push(diagnostic);
                }
            }
        }
    }
    inspections
}

fn imported_geometry_source(
    declaration: &runmat_design_syntax::FeaDeclaration,
) -> Option<(String, Option<runmat_geometry_core::LengthUnit>, Span)> {
    // Omitted `units` asks the host for the file's own declared length unit.
    let authored_units = declaration.body.statements.iter().any(|statement| {
        matches!(&statement.kind, FeaStatementKind::Assignment { target, .. }
            if target.segments.len() == 1 && target.segments[0].text == "units")
    });
    let units = if !authored_units {
        None
    } else {
        declaration.body.statements.iter().find_map(|statement| {
            let FeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
                return None;
            };
            if target.segments.len() != 1 || target.segments[0].text != "units" {
                return None;
            }
            let FeaExpressionKind::Reference(unit) = &value.kind else {
                return None;
            };
            let [unit] = unit.segments.as_slice() else {
                return None;
            };
            match unit.text.as_str() {
                "m" | "meter" | "meters" => Some(runmat_geometry_core::LengthUnit::Meter),
                "mm" | "millimeter" | "millimeters" => {
                    Some(runmat_geometry_core::LengthUnit::Millimeter)
                }
                "in" | "inch" | "inches" => Some(runmat_geometry_core::LengthUnit::Inch),
                _ => None,
            }
        })
    };
    if authored_units && units.is_none() {
        return None;
    }
    declaration.body.statements.iter().find_map(|statement| {
        let FeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        if target.segments.len() != 1 || target.segments[0].text != "file" {
            return None;
        }
        let FeaExpressionKind::String(path) = &value.kind else {
            return None;
        };
        Some((path.clone(), units, value.span))
    })
}

fn location(path: &LogicalSourcePath, span: Span) -> FeaDiagnosticLocation {
    FeaDiagnosticLocation {
        path: path.clone(),
        span,
    }
}
