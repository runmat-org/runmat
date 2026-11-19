//! `motion.drag`, and Move component's Free drag mode: a component dragged through the freedom
//! its mates leave.
//!
//! The canvas's drag handle sits at the component's box centre; its arrows write `dx`, `dy` and
//! `dz` (mm), how far the pointer pulled that point. The mate solve pulls the point toward its
//! target while every mate holds ([`crate::motion::drag`]), and each component the drag moved
//! keeps its new pose as its placement. With `collisions` on, a drag that would push the
//! component into one it was clear of stops at the last clear pose, found by bisecting the pull.

use super::{model, parse_pick, Compiled, EditRequest};
use crate::features::picks_of;
use crate::runtime::{resolve_assembly_in, AssemblyResolution, Files};
use runmat_cad::assembly::model::AssemblyDocument;
use runmat_cad::assembly::motion::drag::Dragged;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::Value;

/// Bisection steps on a colliding drag: the stop is within 1/256 of the pull.
const STEPS: usize = 8;

fn number(request: &EditRequest, id: &str) -> f64 {
    match request.values.get(id) {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// The dragged component's name: the `component` pick, else the first of `components`, else
/// the selection.
fn component(compiled: &Compiled, request: &EditRequest) -> Result<String, EditFailure> {
    let mut ids = picks_of(&request.values, "component");
    if ids.is_empty() {
        ids = picks_of(&request.values, "components");
    }
    if ids.is_empty() {
        ids = request.selection.clone();
    }
    let pick = ids
        .iter()
        .find_map(|id| parse_pick(id))
        .ok_or_else(|| model("pick the component to drag"))?;
    compiled.occurrence_name(&pick.uuid)
}

/// Pairs in contact after `dragged` that were apart before it.
fn new_contacts(
    resolution: &AssemblyResolution,
    files: &Files,
    dragged: &Dragged,
    before: &[(String, String)],
) -> Result<Vec<(String, String)>, EditFailure> {
    let moved: Vec<String> = dragged.placements.iter().map(|(n, _)| n.clone()).collect();
    let found = crate::evaluate::clearance_in(resolution, &dragged.tree, files, 0.0, Some(&moved))
        .map_err(model)?;
    Ok(found
        .overlaps
        .into_iter()
        .map(|o| (o.a, o.b))
        .filter(|p| !before.contains(p))
        .collect())
}

pub(super) fn drag(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Result<String, EditFailure> {
    let name = component(compiled, request)?;
    let delta = ["dx", "dy", "dz"].map(|id| number(request, id) / 1000.0);
    if delta == [0.0; 3] {
        return Err(model(
            "drag the handle, or type how far to pull the component",
        ));
    }
    let resolution = resolve_assembly_in(path, text, files, None)
        .map_err(|_| model("the assembly doesn't compile"))?;
    let base = &resolution.resolved;
    let solve = |f: f64| {
        crate::motion::drag(base, &resolution.source, &name, delta.map(|d| d * f)).map_err(model)
    };
    let mut dragged = solve(1.0)?;
    if dragged.placements.is_empty() {
        return Err(model(format!(
            "`{name}` can't move that way: its mates leave it no freedom in that direction"
        )));
    }
    let mut stopped = None;
    if request.values.get("collisions").and_then(Value::as_bool) == Some(true) {
        let moved: Vec<String> = dragged.placements.iter().map(|(n, _)| n.clone()).collect();
        let before: Vec<(String, String)> =
            crate::evaluate::clearance_in(&resolution, base, files, 0.0, Some(&moved))
                .map_err(model)?
                .overlaps
                .into_iter()
                .map(|o| (o.a, o.b))
                .collect();
        let hit = new_contacts(&resolution, files, &dragged, &before)?;
        if let Some(first) = hit.first().cloned() {
            // The largest clear fraction of the pull.
            let (mut lo, mut hi) = (0.0, 1.0);
            let mut clear: Option<Dragged> = None;
            for _ in 0..STEPS {
                let mid = (lo + hi) / 2.0;
                let d = solve(mid)?;
                if new_contacts(&resolution, files, &d, &before)?.is_empty() {
                    lo = mid;
                    clear = Some(d);
                } else {
                    hi = mid;
                }
            }
            match clear {
                Some(d) if !d.placements.is_empty() => {
                    dragged = d;
                    stopped = Some(first);
                }
                _ => {
                    return Err(model(format!(
                        "`{}` meets `{}` as soon as it moves that way",
                        first.0, first.1
                    )))
                }
            }
        }
    }
    for (moved, placement) in dragged.placements {
        if let Some(o) = document.occurrences.iter_mut().find(|o| o.name == moved) {
            o.placement = placement;
        }
    }
    Ok(match stopped {
        Some((a, b)) => format!("Dragged {name}; stopped where {a} meets {b}"),
        None => format!("Dragged {name}"),
    })
}
