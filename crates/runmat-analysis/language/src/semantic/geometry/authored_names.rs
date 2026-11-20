//! `select name` regions on imported geometry, checked against the names the CAD file carries.

use super::error;
use crate::semantic::CheckedFeaStatementKind;
use crate::semantic::{CheckedFeaDeclaration, CheckedGeometryInspection};
use crate::{
    FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaDiagnosticSeverity,
    GeometryInspectionCapability,
};
use runmat_geometry_core::{
    GeometryRegionCardinality, GeometryRegionDefinition, GeometryRegionRole, GeometryRegionSelector,
};

/// The span of `region <id> : …` inside the geometry declaration, so a refusal points at the
/// selector the author wrote; the declaration's span when the block can't be found.
fn region_span(declaration: &CheckedFeaDeclaration, region_id: &str) -> runmat_types::Span {
    declaration
        .statements
        .iter()
        .find(|statement| match &statement.kind {
            CheckedFeaStatementKind::Block { header, .. } => header
                .trim()
                .strip_prefix("region ")
                .and_then(|rest| rest.split_once(':'))
                .is_some_and(|(name, _)| name.trim() == region_id),
            _ => false,
        })
        .map_or(declaration.span, |statement| statement.span)
}

/// Names a refusal lists: the names the author wrote in the design or CAD file. Generated
/// topology references (`feature.<name>.<kind>.<role>…`) still match a `select name` that spells
/// them, but they are not offered as suggestions.
fn listed(name: &str) -> bool {
    !name.starts_with("feature.")
}

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
                        span: region_span(declaration, &region.region_id),
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
                region_span(declaration, &region.region_id),
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
        let mut available = available
            .iter()
            .filter(|candidate| listed(candidate))
            .cloned()
            .collect::<Vec<_>>();
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
        error(
            declaration,
            region_span(declaration, &region.region_id),
            message,
            diagnostics,
        );
    }
}
