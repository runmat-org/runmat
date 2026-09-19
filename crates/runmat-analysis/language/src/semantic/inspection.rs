use crate::{
    resolve_logical_path, BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation,
    FeaDiagnosticPhase, GeometryInspection, GeometryInspectionRequest, GeometryInspectionSource,
    LogicalSourcePath,
};
use runmat_parser::fea::{FeaDeclarationKind, FeaExpressionKind, FeaModuleItem, FeaStatementKind};
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
            let Some((relative_path, span)) = imported_geometry_source(declaration) else {
                continue;
            };
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
                expected_content_digest: None,
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
    declaration: &runmat_parser::fea::FeaDeclaration,
) -> Option<(String, Span)> {
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
        Some((path.clone(), value.span))
    })
}

fn location(path: &LogicalSourcePath, span: Span) -> FeaDiagnosticLocation {
    FeaDiagnosticLocation {
        path: path.clone(),
        span,
    }
}
