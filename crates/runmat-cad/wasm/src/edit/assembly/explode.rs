//! Exploded views from the canvas: `explode.new` adds a view, `explode.step` and
//! `explode.radial` add a step to it, and `explode.lines` adds trail lines.
//!
//! A step goes into the exploded view the selection names (`exploded:<view>`, or a step's
//! `explode_step:<view>/<id>`), else into the assembly's last exploded view.
//!
//! - `explode.step`: the components move `distance` mm along the picked direction (a face's
//!   normal, an edge's or a cylinder's axis, or an origin axis).
//! - `explode.radial`: the components spread `distance` mm out from the picked axis.
//! - `explode.auto`: a new view whose steps pull the components apart along their mates,
//!   outermost first ([`auto`]).
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
        patterns: Vec::new(),
        mirrors: Vec::new(),
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

/// A component's assembled box in assembly coordinates (mm): its part's bounds under its
/// placement, all eight corners.
fn corners(compiled: &Compiled, name: &str) -> Option<Vec<[f64; 3]>> {
    let (prototype, t) = compiled.placed_named(name)?;
    let b = &compiled.display["prototypes"][prototype]["part"]["bounds"];
    let lo: Vec<f64> = (0..3).filter_map(|k| b["min"][k].as_f64()).collect();
    let hi: Vec<f64> = (0..3).filter_map(|k| b["max"][k].as_f64()).collect();
    if lo.len() != 3 || hi.len() != 3 || t.len() < 12 {
        return None;
    }
    Some(
        (0..8)
            .map(|i| {
                let p = [0, 1, 2].map(|k| if i >> k & 1 == 0 { lo[k] } else { hi[k] });
                [0, 1, 2].map(|r| {
                    (t[3 * r] * p[0] + t[3 * r + 1] * p[1] + t[3 * r + 2] * p[2] + t[9 + r])
                        * 1000.0
                })
            })
            .collect(),
    )
}

fn centre(corners: &[[f64; 3]]) -> [f64; 3] {
    let n = corners.len().max(1) as f64;
    [0, 1, 2].map(|k| corners.iter().map(|c| c[k]).sum::<f64>() / n)
}

/// The Z axis of mate connector `name` in assembly coordinates, with the top-level component
/// it is fixed in: a face connector's normal, a cylinder's or edge's axis.
fn connector_z(
    document: &AssemblyDocument,
    compiled: &Compiled,
    name: &str,
) -> Option<(String, [f64; 3])> {
    use runmat_cad::assembly::model::{AnchorRef, OffsetExpr, RotationExpr};
    let c = document.connector(name)?;
    let segment = c.occurrence.0.first()?;
    let (prototype, t) = compiled.placed_named(&segment.name)?;
    let local = match &c.anchor {
        AnchorRef::Origin => [0.0, 0.0, 1.0],
        AnchorRef::Explicit {
            offset:
                OffsetExpr {
                    rotation: RotationExpr::Basis { z, .. },
                    ..
                },
        } => *z,
        AnchorRef::Explicit {
            offset:
                OffsetExpr {
                    rotation: RotationExpr::Identity,
                    ..
                },
        } => [0.0, 0.0, 1.0],
        AnchorRef::Published { name } => compiled.published_frame(prototype, name.as_str())?.z,
        _ => return None,
    };
    if t.len() < 9 {
        return None;
    }
    let z =
        [0, 1, 2].map(|i| t[3 * i] * local[0] + t[3 * i + 1] * local[1] + t[3 * i + 2] * local[2]);
    let n = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    (n > 1e-9).then(|| (segment.name.clone(), z.map(|v| v / n)))
}

/// `explode.auto`: a new exploded view whose steps pull the assembly apart along its mates.
///
/// The method:
/// 1. The mates join the top-level components into a graph. In each connected group the first
///    fixed component (else the first in assembly order) stays put; every other component's
///    parent is the component it is mated to on the shortest mate path from there.
/// 2. A component leaves its parent along the Z axis of the mate connector between them (a
///    face's normal for a planar mate, the axis for a concentric, revolute or slider mate),
///    pointed from the parent's centre toward the component's.
/// 3. Its distance is the smallest move along that direction that puts its assembled box
///    `spacing` clear of its parent's, and everything mated beyond it rides along, so each level
///    clears the one it came off.
/// 4. Steps are written outermost first (deepest in the mate graph, then last in assembly
///    order): one translate step per component, which can be edited, reordered or deleted like
///    any other step. Components with no mate stay where they are.
pub(super) fn auto(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    values: &Values,
) -> Result<String, EditFailure> {
    let spacing = match values.get("spacing") {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(20.0),
        Some(Value::Object(o)) => o.get("value").and_then(Value::as_f64).unwrap_or(20.0),
        Some(Value::String(s)) => s
            .trim()
            .trim_end_matches("mm")
            .trim()
            .parse()
            .unwrap_or(20.0),
        _ => 20.0,
    };
    if spacing.is_nan() || spacing < 0.0 {
        return Err(model("the spacing can't be negative"));
    }
    let order: Vec<String> = document
        .occurrences
        .iter()
        .filter(|o| !o.suppressed)
        .map(|o| o.name.clone())
        .collect();
    // Mate edges between top-level components, each with the connector Z on both sides.
    let mut edges: Vec<(String, [f64; 3], String, [f64; 3])> = Vec::new();
    for mate in document.mates.iter().filter(|m| !m.suppressed) {
        let (Some((a, za)), Some((b, zb))) = (
            connector_z(document, compiled, &mate.a),
            connector_z(document, compiled, &mate.b),
        ) else {
            continue;
        };
        if a != b && order.contains(&a) && order.contains(&b) {
            edges.push((a, za, b, zb));
        }
    }
    if edges.is_empty() {
        return Err(model(
            "no two components are mated: auto-explode moves components along their mates",
        ));
    }
    // Breadth-first from each group's root: parent, depth and the direction off the parent.
    let fixed = |n: &str| document.occurrences.iter().any(|o| o.name == n && o.fixed);
    let mated = |n: &str| edges.iter().any(|(a, _, b, _)| a == n || b == n);
    let mut parent: std::collections::BTreeMap<String, (String, usize, [f64; 3])> =
        Default::default();
    let mut roots: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut starts: Vec<&String> = order.iter().filter(|n| fixed(n)).collect();
    starts.extend(order.iter().filter(|n| !fixed(n)));
    for root in starts {
        if seen.contains(root) || !mated(root) {
            continue;
        }
        roots.push(root.clone());
        seen.push(root.clone());
        let mut queue = std::collections::VecDeque::from([(root.clone(), 0usize)]);
        while let Some((at, depth)) = queue.pop_front() {
            for (a, za, b, zb) in &edges {
                let (other, z) = if a == &at {
                    (b, zb)
                } else if b == &at {
                    (a, za)
                } else {
                    continue;
                };
                if seen.contains(other) {
                    continue;
                }
                seen.push(other.clone());
                parent.insert(other.clone(), (at.clone(), depth + 1, *z));
                queue.push_back((other.clone(), depth + 1));
            }
        }
    }
    let moved_by = |name: &str| -> Vec<String> {
        // The component and everything whose parent chain passes through it.
        let mut out = vec![name.to_owned()];
        for n in &order {
            let mut at = n.as_str();
            while let Some((p, ..)) = parent.get(at) {
                if p == name {
                    out.push(n.clone());
                    break;
                }
                at = p;
            }
        }
        out
    };
    let mut movers: Vec<&String> = order.iter().filter(|n| parent.contains_key(*n)).collect();
    // Outermost first: deepest, then last in assembly order.
    movers.sort_by_key(|n| {
        let depth = parent[*n].1;
        let at = order.iter().position(|o| o == *n).unwrap_or(0);
        (std::cmp::Reverse(depth), std::cmp::Reverse(at))
    });
    let base = identifier(
        values
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("auto explode"),
    );
    let name = std::iter::once(base.clone())
        .chain((2..).map(|n| format!("{base}_{n}")))
        .find(|n| !document.exploded.iter().any(|v| &v.name == n))
        .expect("an unused name");
    let mut view = ExplodedView::new(&name);
    for m in &movers {
        let (p, _, z) = &parent[*m];
        let (Some(own), Some(theirs)) = (corners(compiled, m), corners(compiled, p)) else {
            continue;
        };
        let away = sub3(centre(&own), centre(&theirs));
        let along = dot3(away, *z);
        // Pointed away from the parent; on a tie (a pin centred in its hole), along the
        // component's own connector axis.
        let n = if along < -1e-6 { z.map(|v| -v) } else { *z };
        let lo = own
            .iter()
            .map(|c| dot3(*c, n))
            .fold(f64::INFINITY, f64::min);
        let hi = theirs
            .iter()
            .map(|c| dot3(*c, n))
            .fold(f64::NEG_INFINITY, f64::max);
        let distance = ((hi - lo).max(0.0) + spacing).max(1.0);
        let distance = (distance * 10.0).round() / 10.0;
        view.steps.push(ExplodeStep {
            id: format!("{}_out", identifier(m)),
            occurrences: moved_by(m)
                .iter()
                .map(|n| RelPath::new(&[n.as_str()]))
                .collect(),
            patterns: Vec::new(),
            mirrors: Vec::new(),
            motion: ExplodeMotion::Translate {
                direction: ExplodeDirection::Vector { vector: round(n) },
                distance: quantity(distance, "mm"),
                spacing: None,
            },
        });
        if values.get("lines").and_then(Value::as_bool).unwrap_or(true) {
            view.lines.push(ExplodeLine {
                id: format!("{}_line", identifier(m)),
                occurrence: RelPath::new(&[m.as_str()]),
                anchor: None,
                jogs: Vec::new(),
            });
        }
    }
    if view.steps.is_empty() {
        return Err(model("no mated component has a box to move clear of"));
    }
    let count = view.steps.len();
    document.exploded.push(view);
    let still: Vec<&String> = order
        .iter()
        .filter(|n| !parent.contains_key(*n) && !roots.contains(n))
        .collect();
    let mut summary = format!(
        "Added exploded view {name}: {count} step{} from the mates, outermost first ({} stay{})",
        if count == 1 { "" } else { "s" },
        roots.join(", "),
        if roots.len() == 1 { "s" } else { "" }
    );
    if !still.is_empty() {
        summary.push_str(&format!(
            "; {} {} no mate and stay{}",
            still
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            if still.len() == 1 { "has" } else { "have" },
            if still.len() == 1 { "s" } else { "" }
        ));
    }
    Ok(summary)
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Whether the selection names an exploded view or an explode step.
pub(super) fn selected(selection: &[String]) -> bool {
    selection
        .iter()
        .any(|id| id.starts_with("exploded:") || id.starts_with("explode_step:"))
}

/// `assembly.delete` on exploded views (`exploded:<view>`) and explode steps
/// (`explode_step:<view>/<id>`). A step's trail lines go with it when no step left in its view
/// moves their component. A view that another view builds on (`base`) stays.
pub(super) fn delete(
    document: &mut AssemblyDocument,
    selection: &[String],
) -> Result<String, EditFailure> {
    let mut gone = Vec::new();
    for id in selection {
        if let Some(name) = id.strip_prefix("exploded:") {
            if let Some(user) = document
                .exploded
                .iter()
                .find(|v| v.base.as_deref() == Some(name))
            {
                return Err(model(format!(
                    "{} builds on {name}: delete it or change its base first",
                    user.name
                )));
            }
            let at = document
                .exploded
                .iter()
                .position(|v| v.name == name)
                .ok_or_else(|| model(format!("the assembly has no exploded view `{name}`")))?;
            document.exploded.remove(at);
            gone.push(format!("exploded view {name}"));
        }
    }
    for id in selection {
        let Some((view, step)) = id
            .strip_prefix("explode_step:")
            .and_then(|r| r.split_once('/'))
        else {
            continue;
        };
        // A step of a view deleted above is gone already.
        let Some(v) = document.exploded.iter_mut().find(|v| v.name == view) else {
            continue;
        };
        let at = v
            .steps
            .iter()
            .position(|s| s.id == step)
            .ok_or_else(|| model(format!("{view} has no step `{step}`")))?;
        v.steps.remove(at);
        let moved: Vec<RelPath> = v
            .steps
            .iter()
            .flat_map(|s| s.occurrences.iter().cloned())
            .collect();
        v.lines.retain(|l| moved.contains(&l.occurrence));
        gone.push(format!("step {step}"));
    }
    if gone.is_empty() {
        return Err(model("select the exploded views or steps to delete"));
    }
    Ok(format!("Deleted {}", gone.join(", ")))
}
