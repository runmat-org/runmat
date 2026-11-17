//! `select name` regions on imported geometry, checked against the names the CAD file carries.

use super::error;
use crate::semantic::{CheckedFeaDeclaration, CheckedGeometryInspection};
use crate::{
    FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaDiagnosticSeverity,
    GeometryInspectionCapability,
};
use runmat_geometry_core::{
    GeometryRegionCardinality, GeometryRegionDefinition, GeometryRegionRole, GeometryRegionSelector,
};

/// Refuses a `select name` that matches no entity of its role, or several for an exactly_one
/// region. Without readable names each such region gets a located warning that the name cannot
/// be confirmed.
pub(super) fn check(
    declaration: &CheckedFeaDeclaration,
    regions: &[GeometryRegionDefinition],
    inspection: &CheckedGeometryInspection,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let names_known = inspection
        .facts
        .capabilities
        .contains(&GeometryInspectionCapability::AuthoredNames);
    for region in regions {
        let GeometryRegionSelector::Name(name) = &region.selector else {
            continue;
        };
        if !names_known {
            diagnostics.push(FeaDiagnostic {
                severity: FeaDiagnosticSeverity::Warning,
                ..FeaDiagnostic::error(
                    "fea.semantic.geometry_names_unconfirmed",
                    format!(
                        "region `{}` selects name \"{name}\", but the names in `{}` cannot be confirmed without the exact geometry backend",
                        region.region_id, inspection.source_path
                    ),
                    FeaDiagnosticPhase::Lowering,
                    Some(FeaDiagnosticLocation {
                        path: declaration.module.clone(),
                        span: declaration.span,
                    }),
                )
            });
            continue;
        }
        let role = match region.role {
            GeometryRegionRole::Vertex => "vertex",
            GeometryRegionRole::Edge => "edge",
            GeometryRegionRole::Face => "face",
            GeometryRegionRole::Solid | GeometryRegionRole::Cell => "solid",
        };
        let roles = if role == "vertex" {
            "vertices".to_string()
        } else {
            format!("{role}s")
        };
        let available = inspection
            .facts
            .authored_names
            .get(role)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let matches = available
            .iter()
            .filter(|candidate| *candidate == name)
            .count();
        if matches > 1 && region.cardinality == GeometryRegionCardinality::ExactlyOne {
            error(
                declaration,
                declaration.span,
                format!(
                    "region `{}` selects {role} name \"{name}\", which {matches} {roles} in `{}` carry; it expects exactly_one (name each entity uniquely in the CAD file, or use expect one_or_more)",
                    region.region_id, inspection.source_path
                ),
                diagnostics,
            );
            continue;
        }
        if matches > 0 {
            continue;
        }
        let mut available = available.to_vec();
        available.dedup();
        let message = if available.is_empty() {
            format!(
                "region `{}` selects {role} name \"{name}\", but `{}` names no {roles}",
                region.region_id, inspection.source_path
            )
        } else {
            format!(
                "region `{}` selects {role} name \"{name}\", which `{}` does not carry; named {roles}: {}",
                region.region_id,
                inspection.source_path,
                available
                    .iter()
                    .map(|candidate| format!("\"{candidate}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        error(declaration, declaration.span, message, diagnostics);
    }
}
