//! `assembly.feature`: a cut, holes or a weld bead in the assembly
//! ([`runmat_cad::assembly::operations`]), and Delete and Suppress on their Outline rows
//! (`operation:<name>`).
//!
//! The form's fields: `type` (`cut`, `hole`, `weld`), `profile` (the sketch: its Outline row
//! `feature:<sketch>`, or a curve or point of it), `end` (`blind`, `symmetric`, `through_all`,
//! `through_all_both`) with `distance`, `reverse`; a hole's `size` and `fit` (its clearance
//! diameter unless `diameter` is given) and `hole_type` (`simple`, `counterbore`,
//! `countersink`, sized from the fastener's standard row unless `cbore_diameter`/`cbore_depth`
//! or `csk_diameter`/`csk_angle` are given); a weld's `weld_size` and `flip`; `components`
//! (none: every part component). `name` names an existing feature to change in place.

use super::mate::field;
use super::{model, parse_pick, Compiled};
use crate::features::picks_of;
use runmat_cad::assembly::model::AssemblyDocument;
use runmat_cad::assembly::operations::{
    AssemblyOperation, Counterbore, Countersink, OperationKind,
};
use runmat_cad::model::feature::{Extent, ProfileRef};
use runmat_cad::model::hole_tables::{fastener, HoleFit, HoleSpec, HoleStandard};
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

type Values = Map<String, Value>;

fn text<'a>(values: &'a Values, id: &str) -> &'a str {
    values.get(id).and_then(Value::as_str).unwrap_or("").trim()
}

fn flag(values: &Values, id: &str) -> bool {
    match values.get(id) {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s == "true",
        _ => false,
    }
}

fn given(values: &Values, id: &str) -> bool {
    !matches!(values.get(id), None | Some(Value::Null))
        && values.get(id).and_then(Value::as_str) != Some("")
}

/// The sketch the `profile` pick names: its row (`feature:<sketch>`) or one of its curves or
/// points (`sketch.<sketch>.<entity>`).
fn sketch_of(document: &AssemblyDocument, values: &Values) -> Result<String, EditFailure> {
    let pick = picks_of(values, "profile")
        .into_iter()
        .next()
        .ok_or_else(|| model("pick the sketch the feature uses"))?;
    let name = pick
        .strip_prefix("feature:")
        .or_else(|| {
            pick.strip_prefix("sketch.")
                .and_then(|r| r.split('.').next())
        })
        .unwrap_or(&pick)
        .to_owned();
    if !document.features.iter().any(|f| f.id.0 == name) {
        return Err(model(format!("the assembly has no sketch `{name}`")));
    }
    Ok(name)
}

fn extent(values: &Values, allow_symmetric: bool) -> Result<(Extent, bool), EditFailure> {
    Ok(match text(values, "end") {
        "blind" => (
            Extent::Blind {
                distance: field(values, "distance", "mm")?,
            },
            false,
        ),
        "symmetric" if allow_symmetric => (
            Extent::Symmetric {
                distance: field(values, "distance", "mm")?,
            },
            false,
        ),
        "through_all_both" if allow_symmetric => (Extent::ThroughAll, true),
        "" | "through_all" => (Extent::ThroughAll, false),
        other => {
            return Err(model(format!(
                "an assembly {} can't end `{other}`",
                if allow_symmetric { "cut" } else { "hole" }
            )))
        }
    })
}

fn mm(v: f64) -> ValueExpr {
    ValueExpr::qty((v * 1e6).round() / 1e3, "mm")
}

/// The fastener row the `size` field names (ISO metric, or unified inch for `1/4-20`, `#10-32`).
fn standard_row(size: &str) -> Result<runmat_cad::model::hole_tables::FastenerHoles, EditFailure> {
    let standard = if size.contains('-') || size.starts_with('#') || size.contains('/') {
        HoleStandard::AnsiInch
    } else {
        HoleStandard::Iso
    };
    fastener(&HoleSpec {
        standard,
        size: size.to_owned(),
        fit: HoleFit::Normal,
        class: None,
    })
    .map_err(model)
}

fn hole(values: &Values, sketch: String) -> Result<OperationKind, EditFailure> {
    let (extent, _) = extent(values, false)?;
    let size = Some(text(values, "size").to_owned()).filter(|s| !s.is_empty());
    let row = size.as_deref().map(standard_row).transpose()?;
    let fit = match text(values, "fit") {
        "close" => HoleFit::Close,
        "loose" => HoleFit::Loose,
        _ => HoleFit::Normal,
    };
    let diameter = if given(values, "diameter") {
        field(values, "diameter", "mm")?
    } else {
        let row = row
            .as_ref()
            .ok_or_else(|| model("give the fastener size or the hole diameter"))?;
        mm(row.clearance(fit))
    };
    let from_row = |which: &str| {
        model(format!(
            "give the {which}'s sizes, or a standard fastener size"
        ))
    };
    let (counterbore, countersink) = match text(values, "hole_type") {
        "counterbore" => {
            let (d, h) = match (given(values, "cbore_diameter"), &row) {
                (true, _) => (
                    field(values, "cbore_diameter", "mm")?,
                    field(values, "cbore_depth", "mm")?,
                ),
                (false, Some(r)) => {
                    let (d, h) = r.counterbore.ok_or_else(|| from_row("counterbore"))?;
                    (mm(d), mm(h))
                }
                (false, None) => return Err(from_row("counterbore")),
            };
            (
                Some(Counterbore {
                    diameter: d,
                    depth: h,
                }),
                None,
            )
        }
        "countersink" => {
            let (d, a) = match (given(values, "csk_diameter"), &row) {
                (true, _) => (
                    field(values, "csk_diameter", "mm")?,
                    if given(values, "csk_angle") {
                        field(values, "csk_angle", "deg")?
                    } else {
                        ValueExpr::qty(90.0, "deg")
                    },
                ),
                (false, Some(r)) => {
                    let (d, a) = r.countersink.ok_or_else(|| from_row("countersink"))?;
                    (
                        mm(d),
                        ValueExpr::qty((a.to_degrees() * 1e3).round() / 1e3, "deg"),
                    )
                }
                (false, None) => return Err(from_row("countersink")),
            };
            (
                None,
                Some(Countersink {
                    diameter: d,
                    angle: a,
                }),
            )
        }
        _ => (None, None),
    };
    Ok(OperationKind::Hole {
        profile: ProfileRef::sketch(&sketch),
        diameter,
        extent,
        reverse: flag(values, "reverse"),
        size,
        counterbore,
        countersink,
    })
}

/// The feature the form describes.
fn kind(document: &AssemblyDocument, values: &Values) -> Result<OperationKind, EditFailure> {
    let sketch = sketch_of(document, values)?;
    Ok(match text(values, "type") {
        "" | "cut" => {
            let (extent, both) = extent(values, true)?;
            OperationKind::Cut {
                profile: ProfileRef::sketch(&sketch),
                extent,
                reverse: flag(values, "reverse"),
                both,
            }
        }
        "hole" => hole(values, sketch)?,
        "weld" => OperationKind::Weld {
            path: ProfileRef::sketch(&sketch),
            size: field(values, "weld_size", "mm")?,
            flip: flag(values, "flip"),
        },
        other => {
            return Err(model(format!(
                "an assembly feature is a cut, hole or weld, not `{other}`"
            )))
        }
    })
}

/// The components the feature names (none: every part component).
fn components(compiled: &Compiled, values: &Values) -> Result<Vec<String>, EditFailure> {
    let mut names = Vec::new();
    for id in picks_of(values, "components") {
        let name = match parse_pick(&id) {
            Some(pick) => compiled.occurrence_name(&pick.uuid)?,
            // A name as the row's values carry it.
            None => id,
        };
        if !names.contains(&name) {
            names.push(name);
        }
    }
    Ok(names)
}

/// A free name `<type><n>`.
fn fresh(document: &AssemblyDocument, word: &str) -> String {
    (1..)
        .map(|n| format!("{word}{n}"))
        .find(|n| !document.operations.iter().any(|o| &o.name == n))
        .unwrap()
}

/// `assembly.feature`: adds the feature, or changes the one `name` names.
pub(super) fn apply(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    values: &Values,
) -> Result<String, EditFailure> {
    let op = kind(document, values)?;
    let components = components(compiled, values)?;
    for c in &components {
        if !document.occurrences.iter().any(|o| &o.name == c) {
            return Err(model(format!("the assembly has no component `{c}`")));
        }
    }
    let word = op.word();
    let label = match &op {
        OperationKind::Cut { .. } => "cut",
        OperationKind::Hole { .. } => "hole",
        OperationKind::Weld { .. } => "weld bead",
    };
    let name = text(values, "name");
    if let Some(existing) = document.operations.iter_mut().find(|o| o.name == name) {
        existing.op = op;
        existing.components = components;
        return Ok(format!("Changed assembly {label} {name}"));
    }
    let name = if name.is_empty() {
        fresh(document, word)
    } else {
        name.to_owned()
    };
    document.operations.push(AssemblyOperation {
        name: name.clone(),
        op,
        components,
        suppressed: false,
    });
    Ok(format!("Added assembly {label} {name}"))
}

/// The selected feature rows (`operation:<name>`).
pub(super) fn selected(request: &super::EditRequest) -> impl Iterator<Item = &str> {
    request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("operation:"))
}

/// `assembly.delete` and `assembly.suppress` on selected feature rows.
pub(super) fn delete_or_suppress(
    document: &mut AssemblyDocument,
    request: &super::EditRequest,
) -> Result<String, EditFailure> {
    let names: Vec<String> = selected(request).map(str::to_owned).collect();
    let delete = request.command == "assembly.delete";
    for name in &names {
        let at = document
            .operations
            .iter()
            .position(|o| &o.name == name)
            .ok_or_else(|| model(format!("the assembly has no feature `{name}`")))?;
        if delete {
            document.operations.remove(at);
        } else {
            document.operations[at].suppressed = true;
        }
    }
    let verb = if delete { "Deleted" } else { "Suppressed" };
    Ok(format!("{verb} {}", names.join(", ")))
}
