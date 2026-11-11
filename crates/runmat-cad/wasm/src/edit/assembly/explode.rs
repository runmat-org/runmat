//! Exploded views from the canvas: `explode.new` adds a view, `explode.step` and
//! `explode.radial` add a step to it, and `explode.lines` adds trail lines.
//!
//! A step goes into the exploded view the selection names (`exploded:<view>`, or a step's
//! `explode_step:<view>/<id>`), else into the assembly's last exploded view.
//!
//! - `explode.step`: the components move `distance` mm along the picked direction (a face's
//!   normal, an edge's or a cylinder's axis, or an origin axis).
//! - `explode.radial`: the components spread `distance` mm out from the picked axis.
//! - `explode.lines`: one trail line per component the picked steps move (all the view's steps
//!   when none is picked), skipping components that already have one.

use super::mate::field;
use super::mate::quantity;
use super::place::direction;
use super::{model, Compiled};
use crate::features::picks_of;
use runmat_cad::assembly::explode::{
    ExplodeDirection, ExplodeLine, ExplodeMotion, ExplodeStep, ExplodedView, RadialCenter,
};
use runmat_cad::assembly::model::{AssemblyDocument, RelPath};
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{Map, Value};

type Values = Map<String, Value>;

/// An identifier from a display name: `Exploded 1` → `exploded_1`.
fn identifier(name: &str) -> String {
    let mut id: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    while id.contains("__") {
        id = id.replace("__", "_");
    }
    let id = id.trim_matches('_').to_owned();
    if id.is_empty() || id.starts_with(|c: char| c.is_ascii_digit()) {
        format!("exploded_{id}").trim_end_matches('_').to_owned()
    } else {
        id
    }
}

/// `explode.new`: an empty exploded view named from `name` (made unique).
pub(super) fn new_view(
    document: &mut AssemblyDocument,
    values: &Values,
) -> Result<String, EditFailure> {
    let base = identifier(
        values
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("exploded"),
    );
    let name = std::iter::once(base.clone())
        .chain((2..).map(|n| format!("{base}_{n}")))
        .find(|n| !document.exploded.iter().any(|v| &v.name == n))
        .expect("an unused name");
    document.exploded.push(ExplodedView::new(&name));
    Ok(format!("Added exploded view {name}"))
}

/// The exploded view a step or line goes into: the one the selection names, else the last.
fn target_view<'a>(
    document: &'a mut AssemblyDocument,
    selection: &[String],
) -> Result<&'a mut ExplodedView, EditFailure> {
    let named = selection.iter().find_map(|id| {
        id.strip_prefix("exploded:")
            .or_else(|| {
                id.strip_prefix("explode_step:")
                    .and_then(|r| r.split_once('/').map(|(v, _)| v))
            })
            .map(str::to_owned)
    });
    if document.exploded.is_empty() {
        return Err(model("create an exploded view first"));
    }
    let at = match named {
        Some(name) => document
            .exploded
            .iter()
            .position(|v| v.name == name)
            .ok_or_else(|| model(format!("the assembly has no exploded view `{name}`")))?,
        None => document.exploded.len() - 1,
    };
    Ok(&mut document.exploded[at])
}

fn next_id(taken: impl Iterator<Item = String>, stem: &str) -> String {
    let taken: Vec<String> = taken.collect();
    (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|n| !taken.contains(n))
        .expect("an unused id")
}

fn round(v: [f64; 3]) -> [f64; 3] {
    v.map(|c| {
        let r = (c * 1e6).round() / 1e6;
        if r == 0.0 {
            0.0
        } else {
            r
        }
    })
}

/// `explode.step` (`radial` false) or `explode.radial` (`radial` true) on components `names`.
pub(super) fn step(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    names: &[String],
    values: &Values,
    selection: &[String],
    radial: bool,
) -> Result<String, EditFailure> {
    let pick = picks_of(values, if radial { "axis" } else { "direction" })
        .into_iter()
        .next();
    let (axis, point) = match &pick {
        Some(id) => direction(compiled, id)?,
        None => ([0.0, 0.0, 1.0], [0.0; 3]),
    };
    let distance = field(values, "distance", "mm")?;
    let motion = if radial {
        ExplodeMotion::Radial {
            center: RadialCenter::Axis {
                point: round(point).map(|c| quantity(c, "mm")),
                direction: round(axis),
            },
            distance,
        }
    } else {
        ExplodeMotion::Translate {
            direction: ExplodeDirection::Vector {
                vector: round(axis),
            },
            distance,
            spacing: None,
        }
    };
    let view = target_view(document, selection)?;
    let id = next_id(view.steps.iter().map(|s| s.id.clone()), "step");
    view.steps.push(ExplodeStep {
        id: id.clone(),
        occurrences: names.iter().map(|n| RelPath::new(&[n])).collect(),
        motion,
    });
    Ok(format!(
        "Added {} step {id} to {} ({})",
        if radial { "radial" } else { "explode" },
        view.name,
        names.join(", ")
    ))
}

/// `explode.lines`: a trail line for each component the picked steps move.
pub(super) fn lines(
    document: &mut AssemblyDocument,
    values: &Values,
    selection: &[String],
) -> Result<String, EditFailure> {
    let picked: Vec<String> = picks_of(values, "steps")
        .into_iter()
        .filter_map(|id| {
            id.strip_prefix("explode_step:")
                .and_then(|r| r.split_once('/'))
                .map(|(_, s)| s.to_owned())
        })
        .collect();
    let mut scope = selection.to_vec();
    scope.extend(picks_of(values, "steps"));
    let view = target_view(document, &scope)?;
    let mut added = Vec::new();
    let occurrences: Vec<RelPath> = view
        .steps
        .iter()
        .filter(|s| picked.is_empty() || picked.contains(&s.id))
        .flat_map(|s| s.occurrences.iter().cloned())
        .collect();
    for occurrence in occurrences {
        if view.lines.iter().any(|l| l.occurrence == occurrence) {
            continue;
        }
        let id = next_id(view.lines.iter().map(|l| l.id.clone()), "line");
        added.push(occurrence.to_string());
        view.lines.push(ExplodeLine {
            id,
            occurrence,
            anchor: None,
            jogs: Vec::new(),
        });
    }
    if added.is_empty() {
        return Err(model(
            "every component these steps move already has an explode line",
        ));
    }
    Ok(format!(
        "Added explode lines to {} ({})",
        view.name,
        added.join(", ")
    ))
}
