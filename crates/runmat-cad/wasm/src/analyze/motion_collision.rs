//! `motion.collision`: interference over a motion study. Each solved frame of the study
//! ([`crate::motion::run`]) places the moving components, and the exact-body interference check
//! ([`crate::evaluate::clearance_in`]) runs on the pairs they are in (or the picked components').
//! Bodies are regenerated once and placed per frame; the check keeps its pair answers by
//! placement, so a frame where nothing moved relative to a neighbour costs nothing.
//!
//! - `values.study`: the study's name (or a `motion:<name>` row in the selection; the only study
//!   when there is one).
//! - `values.mode`: `stop` reports the first frame where components that were apart at the start
//!   meet, and stops there (as the motion stops at a hard contact); `highlight` runs every frame
//!   and lists each pair's contact, from when to when, with the most overlap.
//! - `values.components`: components (`occurrence:<uuid>`) whose pairs are checked; every moving
//!   component when empty.

use super::{Analysis, Row};
use crate::runtime::{resolve_assembly_in, Files};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// A pair's contact over the run.
struct Contact {
    a: String,
    b: String,
    items: [String; 2],
    from: f64,
    to: f64,
    volume: f64,
}

fn study_name(
    values: &Map<String, Value>,
    selection: &[String],
    studies: &[String],
) -> Result<String, String> {
    let named = values
        .get("study")
        .and_then(|v| match v {
            Value::Array(a) => a.first().and_then(Value::as_str),
            other => other.as_str(),
        })
        .map(|s| s.strip_prefix("motion:").unwrap_or(s).to_owned())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            selection
                .iter()
                .find_map(|s| s.strip_prefix("motion:").map(str::to_owned))
        });
    match named {
        Some(n) if studies.contains(&n) => Ok(n),
        Some(n) => Err(format!("the assembly has no motion study `{n}`")),
        None if studies.len() == 1 => Ok(studies[0].clone()),
        None if studies.is_empty() => {
            Err("the assembly has no motion study; add one with Animation study first".into())
        }
        None => Err("pick the motion study to check".into()),
    }
}

pub fn collision(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
    values: &Map<String, Value>,
    selection: &[String],
) -> Result<Analysis, String> {
    let resolution = resolve_assembly_in(path, text, files, configuration)
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let base = &resolution.resolved;
    let studies: Vec<String> =
        runmat_cad::assembly::source::DesignSource::assembly(&resolution.source, &base.top, None)
            .map(|d| d.motions.iter().map(|m| m.name.clone()).collect())
            .unwrap_or_default();
    let name = study_name(values, selection, &studies)?;
    let stop = values.get("mode").and_then(Value::as_str) != Some("highlight");
    let run = crate::motion::run(base, &resolution.source, &name)
        .ok_or("the assembly doesn't compile")?;
    if let Some(p) = run.problems.first() {
        return Err(format!("motion study `{name}`: {p}"));
    }
    // The components whose pairs are checked: the picked ones, else every moving one.
    let picked: Vec<String> = values
        .get("components")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .filter_map(|id| id.strip_prefix("occurrence:"))
                .filter_map(|uuid| {
                    (0..base.nodes.len() as u32)
                        .find(|&n| base.node(n).uuid.to_string() == uuid)
                        .map(|n| base.path(n))
                })
                .collect()
        })
        .unwrap_or_default();
    let changed: Vec<String> = if picked.is_empty() {
        run.moving.iter().map(|&n| base.path(n)).collect()
    } else {
        picked
    };
    if changed.is_empty() {
        return Err(format!(
            "nothing moves in motion study `{name}`, so nothing can collide"
        ));
    }
    let mut frame_tree = base.clone();
    let mut contacts: BTreeMap<(String, String), Contact> = BTreeMap::new();
    let mut at_start: Vec<(String, String)> = Vec::new();
    let mut stopped: Option<(f64, String, String)> = None;
    let mut checked = 0usize;
    let mut problems = Vec::new();
    for (k, (t, frame)) in run.times.iter().zip(&run.frames).enumerate() {
        for (node, world) in frame_tree.nodes.iter_mut().zip(frame) {
            node.world = *world;
        }
        let found =
            crate::evaluate::clearance_in(&resolution, &frame_tree, files, 0.0, Some(&changed))?;
        checked += 1;
        for e in found.errors {
            if !problems.contains(&e) {
                problems.push(e);
            }
        }
        for o in &found.overlaps {
            let key = (o.a.clone(), o.b.clone());
            if k == 0 {
                at_start.push(key.clone());
            }
            let c = contacts.entry(key).or_insert_with(|| Contact {
                a: o.a.clone(),
                b: o.b.clone(),
                items: o.items.clone(),
                from: *t,
                to: *t,
                volume: 0.0,
            });
            c.to = *t;
            c.volume = c.volume.max(o.volume);
            if stop && k > 0 && stopped.is_none() && !at_start.contains(&(o.a.clone(), o.b.clone()))
            {
                stopped = Some((*t, o.a.clone(), o.b.clone()));
            }
        }
        if stopped.is_some() {
            break;
        }
    }
    let s = |t: f64| format!("{t:.2} s");
    let rows: Vec<Row> = contacts
        .values()
        .map(|c| Row {
            cells: vec![
                c.a.clone(),
                c.b.clone(),
                if (c.to - c.from).abs() < 1e-12 {
                    s(c.from)
                } else {
                    format!("{} to {}", s(c.from), s(c.to))
                },
                format!("{:.3} cm³", c.volume * 1e6),
            ],
            items: c.items.to_vec(),
            severity: Some(if at_start.contains(&(c.a.clone(), c.b.clone())) {
                "warning"
            } else {
                "error"
            }),
            legend: None,
            reference: None,
        })
        .collect();
    let summary = match &stopped {
        Some((t, a, b)) => format!("The motion stops at {}: {a} meets {b}", s(*t)),
        None if contacts.is_empty() => format!(
            "No collisions over {} solved frame{} ({} s)",
            checked,
            if checked == 1 { "" } else { "s" },
            run.duration
        ),
        None => format!(
            "{} pair{} in contact over {} solved frames",
            contacts.len(),
            if contacts.len() == 1 { "" } else { "s" },
            checked
        ),
    };
    if let Some((t, why)) = &run.stopped {
        problems.push(format!("the study's solve ends at {}: {why}", s(*t)));
    }
    Ok(Analysis {
        title: "Collision detection".into(),
        summary,
        columns: ["Part", "With", "When", "Most overlap"]
            .map(String::from)
            .to_vec(),
        rows,
        problems,
        data: json!({
            "study": name,
            "mode": if stop { "stop" } else { "highlight" },
            "frames": checked,
            "stopped": stopped.as_ref().map(|(t, a, b)| json!({ "time": t, "a": a, "b": b })),
            "contacts": contacts.values().map(|c| json!({ "a": c.a, "b": c.b, "items": c.items, "from": c.from, "to": c.to, "volume": c.volume })).collect::<Vec<_>>(),
        }),
        settings: vec![
            format!("Study: {name}"),
            format!("On contact: {}", if stop { "Stop" } else { "Highlight" }),
        ],
        ..Default::default()
    })
}
