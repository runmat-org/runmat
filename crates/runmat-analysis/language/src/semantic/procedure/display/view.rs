//! The solved bodies a field view draws and how far it displaces them:
//! `display <output> { hide = [geometry.part.region.payload]; context = [geometry.part.region.base];
//! deformation = 200 }`. Regions are solid regions of the study geometry, recorded by their region
//! name (the CAD name); a region may be hidden or context, not both.
use super::super::values;
use super::error;
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, FeaDiagnostic,
    FeaTopologyRole, FeaType,
};

const REGION_CODE: &str = "fea.semantic.display_region";

/// The region names of `hide` and `context`, each in authored order without repeats.
#[derive(Default)]
pub(super) struct DisplayRegions {
    pub hide: Vec<String>,
    pub context: Vec<String>,
}

pub(super) fn display_regions(
    declaration: &CheckedFeaDeclaration,
    hide: Option<&CheckedExpression>,
    context: Option<&CheckedExpression>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> DisplayRegions {
    let geometry = study_geometry(declaration);
    let mut lower = |value: Option<&CheckedExpression>, member: &str| {
        value
            .map(|value| region_list(declaration, geometry, value, member, diagnostics))
            .unwrap_or_default()
    };
    let regions = DisplayRegions {
        hide: lower(hide, "hide"),
        context: lower(context, "context"),
    };
    if let (Some(value), Some(both)) = (
        context,
        regions
            .context
            .iter()
            .find(|region| regions.hide.contains(region)),
    ) {
        error(
            declaration,
            REGION_CODE,
            value.span,
            format!("region `{both}` is both hidden and context; list it in one of them"),
            diagnostics,
        );
    }
    regions
}

/// The study's geometry as its `geometry = geometry.<name>` assignment exposes it.
fn study_geometry(declaration: &CheckedFeaDeclaration) -> Option<&str> {
    match &values::assignment(&declaration.statements, "geometry")?.kind {
        CheckedExpressionKind::Reference(path) => match path.as_slice() {
            [namespace, name] if namespace == "geometry" => Some(name.as_str()),
            _ => None,
        },
        _ => None,
    }
}

/// `[geometry.part.region.a, geometry.part.region.b]` (or one region without brackets): the
/// region names, each a solid region of the study geometry.
fn region_list(
    declaration: &CheckedFeaDeclaration,
    geometry: Option<&str>,
    value: &CheckedExpression,
    member: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<String> {
    let items = match &value.kind {
        CheckedExpressionKind::List(items) => items.iter().collect::<Vec<_>>(),
        _ => vec![value],
    };
    let mut names = Vec::new();
    for item in items {
        match study_region(geometry, item) {
            Some(name) if names.iter().any(|known| known == name) => error(
                declaration,
                REGION_CODE,
                item.span,
                format!("display {member} lists region `{name}` twice"),
                diagnostics,
            ),
            Some(name) => names.push(name.to_string()),
            None => error(
                declaration,
                REGION_CODE,
                item.span,
                format!(
                    "display {member} names solid regions of the study geometry, such as \
                     `{member} = [geometry.{}.region.payload]`",
                    geometry.unwrap_or("part")
                ),
                diagnostics,
            ),
        }
    }
    names
}

/// The region name of a reference to a solid region of the study geometry.
fn study_region<'a>(geometry: Option<&str>, value: &'a CheckedExpression) -> Option<&'a str> {
    let CheckedExpressionKind::Reference(path) = &value.kind else {
        return None;
    };
    let [namespace, owner, region_namespace, region] = path.as_slice() else {
        return None;
    };
    (namespace == "geometry"
        && Some(owner.as_str()) == geometry
        && region_namespace == "region"
        && value.value_type == FeaType::Region(FeaTopologyRole::Solid))
    .then_some(region.as_str())
}

/// `deformation = 200`: a positive, finite, dimensionless scale on the displacement.
pub(super) fn display_deformation(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let factor = match (&value.kind, &value.value_type) {
        (CheckedExpressionKind::Number(source), FeaType::Real | FeaType::Integer) => {
            source.parse::<f64>().ok()
        }
        _ => None,
    }
    .filter(|factor| factor.is_finite() && *factor > 0.0);
    if factor.is_none() {
        error(
            declaration,
            "fea.semantic.display_deformation",
            value.span,
            "display deformation is a positive dimensionless scale on the displacement, such as \
             `deformation = 200`",
            diagnostics,
        );
    }
    factor
}
