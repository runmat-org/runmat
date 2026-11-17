//! `manage.recognize-features` on a part: an imported body ([`graph`] recognizes its faces)
//! becomes a sketch and extrude, cuts, holes, rounds and chamfers, written as normal features
//! in place of the `import` that made it.
//!
//! A body is converted whole or not at all: an imported face is named only by the file (when
//! the file names it at all), so new features can't be built on the parts of an import that
//! would stay. The rebuilt
//! features are regenerated and compared with the import before anything is written: the volumes
//! must agree within the tolerance times the body's area, and every imported face must have a
//! rebuilt face of the same kind whose centroid lies within the tolerance and whose area agrees
//! within the tolerance times its perimeter. When a face fits no pattern, or the comparison
//! fails, the import stays as it is and the command lists what was left and why.
//!
//! References to the imported body elsewhere in the part (an output's `body`, a feature's
//! target) move to the new body. The STEP file stays in the project, so undo restores the
//! import exactly.

pub(super) mod graph;

use super::super::{model, EditRequest};
use super::replace::{replace, Target};
use crate::features::picks_of;
use crate::picks::{lower_text, Picks};
use graph::{HolePlan, Plan, Seg};
use runmat_cad::evaluate::{AnalysisRequest, AnalysisResult};
use runmat_cad::model::FeatureKind;
use runmat_cad::refs::history::{EdgeCurve, SurfaceKind, Topology};
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use runmat_design_syntax::edit::TextEdit;
use runmat_types::Span;
use serde_json::Value;
use std::fmt::Write as _;

/// Millimeters to the micrometer (finer than any match tolerance the form takes), trailing
/// zeros dropped.
fn mm(meters: f64) -> String {
    let s = format!("{:.3}", meters * 1e3);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0 mm".into()
    } else {
        format!("{s} mm")
    }
}

fn point2(p: [f64; 2]) -> String {
    format!("[{}, {}]", mm(p[0]), mm(p[1]))
}

/// A point in the part from sketch coordinates `p` at `level` along axis `k`.
fn world(p: [f64; 2], level: f64, k: usize) -> [f64; 3] {
    let mut out = [0.0; 3];
    out[k] = level;
    out[(k + 1) % 3] = p[0];
    out[(k + 2) % 3] = p[1];
    out
}

const ORIGIN: [&str; 3] = ["origin.yz", "origin.zx", "origin.xy"];

/// Feature names not yet in the part.
struct Names(Vec<String>);

impl Names {
    fn fresh(&mut self, stem: &str) -> String {
        let id = (1..)
            .map(|n| {
                if n == 1 {
                    stem.to_owned()
                } else {
                    format!("{stem}{n}")
                }
            })
            .find(|id| !self.0.contains(id))
            .expect("an unused name");
        self.0.push(id.clone());
        id
    }
}

/// The sketch statement for `loops`, its entities named `l<n>`, `a<n>`, `c<n>`; returns the
/// entity names of the first loop in order.
fn sketch(out: &mut String, id: &str, plane: &str, loops: &[&[Seg]]) -> Vec<String> {
    let _ = writeln!(out, "    sketch {id} on {plane} {{");
    let (mut l, mut a, mut c) = (0, 0, 0);
    let mut first = Vec::new();
    for (i, lp) in loops.iter().enumerate() {
        for seg in lp.iter() {
            let (name, body) = match seg {
                Seg::Line { a: s, b: e } => {
                    l += 1;
                    (
                        format!("l{l}"),
                        format!("line(start = {}, end = {})", point2(*s), point2(*e)),
                    )
                }
                Seg::Arc { c: m, a: s, b: e } => {
                    a += 1;
                    (
                        format!("a{a}"),
                        format!(
                            "arc(center = {}, start = {}, end = {})",
                            point2(*m),
                            point2(*s),
                            point2(*e)
                        ),
                    )
                }
                Seg::Circle { c: m, r } => {
                    c += 1;
                    (
                        format!("c{c}"),
                        format!("circle(center = {}, radius = {})", point2(*m), mm(*r)),
                    )
                }
            };
            let _ = writeln!(out, "        {name} = {body}");
            if i == 0 {
                first.push(name);
            }
        }
    }
    out.push_str("    }\n");
    first
}

/// What a plan writes, and how it reads back.
struct Written {
    text: String,
    base: String,
    counts: Vec<String>,
}

fn write(plan: &Plan, names: &mut Names) -> Written {
    let k = plan.axis;
    let mut out = String::new();
    let mut counts = Vec::new();
    // Sketches lie on the origin plane across the axis, or a plane offset from it at the level.
    let plane_at = |out: &mut String, names: &mut Names, level: f64| -> String {
        if level.abs() < graph::EPS {
            return ORIGIN[k].to_owned();
        }
        let id = names.fresh("level");
        let _ = writeln!(
            out,
            "    feature {id} = plane.offset(base = {}, distance = {})",
            ORIGIN[k],
            mm(level)
        );
        format!("plane.{id}")
    };
    let plane = plane_at(&mut out, names, plan.lo);
    let sk = names.fresh("outline");
    let mut loops: Vec<&[Seg]> = vec![&plan.outline];
    loops.extend(plan.openings.iter().map(Vec::as_slice));
    let entities = sketch(&mut out, &sk, &plane, &loops);
    let base = names.fresh("base");
    let _ = writeln!(
        out,
        "    feature {base} = extrude(profile = sketch.{sk}, distance = {})",
        mm(plan.hi - plan.lo)
    );
    counts.push(format!(
        "an extrude {} long{}",
        mm(plan.hi - plan.lo),
        if plan.openings.is_empty() {
            String::new()
        } else {
            format!(" with {} opening(s) through it", plan.openings.len())
        }
    ));
    // Rounds and chamfers on the outline's corners, one feature per kind and size.
    let side = |i: usize| format!("feature.{base}.face.side({sk}.{})", entities[i]);
    let mut done = vec![false; plan.corners.len()];
    for (i, c) in plan.corners.iter().enumerate() {
        if done[i] {
            continue;
        }
        let same: Vec<usize> = (i..plan.corners.len())
            .filter(|j| {
                plan.corners[*j].chamfer == c.chamfer
                    && (plan.corners[*j].size - c.size).abs() < graph::EPS
            })
            .collect();
        let edges: Vec<String> = same
            .iter()
            .map(|j| {
                done[*j] = true;
                let (a, b) = plan.corners[*j].lines;
                format!("edges_between({}, {})", side(a), side(b))
            })
            .collect();
        let (stem, word, param) = if c.chamfer {
            ("corner_chamfer", "chamfer", "distance")
        } else {
            ("corner_round", "fillet", "radius")
        };
        let id = names.fresh(stem);
        let _ = writeln!(
            out,
            "    feature {id} = {word}(edges = [{}], {param} = {})",
            edges.join(", "),
            mm(c.size)
        );
        counts.push(format!(
            "{} {} {}",
            edges.len(),
            if c.chamfer { "chamfered" } else { "rounded" },
            if edges.len() == 1 {
                "corner"
            } else {
                "corners"
            }
        ));
    }
    // Pockets: a cut from the cap they open at.
    for p in &plan.pockets {
        let at = if p.from_top { plan.hi } else { plan.lo };
        let plane = plane_at(&mut out, names, at);
        let id = names.fresh("pocket_sketch");
        let loops: Vec<&[Seg]> = p.loops.iter().map(Vec::as_slice).collect();
        sketch(&mut out, &id, &plane, &loops);
        let cut = names.fresh("pocket");
        let depth = if p.from_top {
            plan.hi - p.level
        } else {
            p.level - plan.lo
        };
        let reverse = if p.from_top { ", reverse" } else { "" };
        let _ = writeln!(out, "    feature {cut} = extrude(profile = sketch.{id}, distance = {}{reverse}, mode = cut)", mm(depth));
        counts.push(format!("a pocket {} deep", mm(depth)));
    }
    // Holes, one feature per cap, size, depth and bottom.
    let mut done = vec![false; plan.holes.len()];
    for (i, h) in plan.holes.iter().enumerate() {
        if done[i] {
            continue;
        }
        let same = |o: &HolePlan| {
            o.from_top == h.from_top
                && (o.r - h.r).abs() < graph::EPS
                && match (o.depth, h.depth) {
                    (None, None) => true,
                    (Some(a), Some(b)) => (a - b).abs() < graph::EPS,
                    _ => false,
                }
                && match (o.point_angle, h.point_angle) {
                    (None, None) => true,
                    (Some(a), Some(b)) => (a - b).abs() < 1e-6,
                    _ => false,
                }
        };
        let level = if h.from_top { plan.hi } else { plan.lo };
        let mut at = Vec::new();
        for (j, o) in plan.holes.iter().enumerate().skip(i) {
            if !done[j] && same(o) {
                done[j] = true;
                let p = world(o.c, level, k);
                at.push(format!(
                    "h{} = point({}, {}, {})",
                    at.len() + 1,
                    mm(p[0]),
                    mm(p[1]),
                    mm(p[2])
                ));
            }
        }
        let id = names.fresh("holes");
        let face = if h.from_top { "end" } else { "start" };
        let end = match h.depth {
            None => "through_all".to_owned(),
            Some(d) => format!("depth = {}", mm(d)),
        };
        let point = h.point_angle.map_or(String::new(), |a| {
            format!(
                ", drill_point = {} deg",
                format!("{a:.3}")
                    .trim_end_matches('0')
                    .trim_end_matches('.')
            )
        });
        let _ = writeln!(
            out,
            "    feature {id} = hole(face = feature.{base}.face.{face}, at = {{ {} }}, diameter = {}, {end}{point})",
            at.join(", "),
            mm(2.0 * h.r)
        );
        counts.push(format!(
            "{} {} hole{} of {}",
            at.len(),
            match (h.depth, h.point_angle) {
                (None, _) => "through",
                (Some(_), Some(_)) => "blind drilled",
                (Some(_), None) => "flat-bottomed",
            },
            if at.len() == 1 { "" } else { "s" },
            mm(2.0 * h.r)
        ));
    }
    Written {
        text: out,
        base,
        counts,
    }
}

/// The byte range of the statement `feature <id> = import(…)`, from its line start to the end
/// of its line.
fn statement(text: &str, id: &str) -> Option<(usize, usize)> {
    let head = format!("feature {id} = import(");
    let at = text.find(&head)?;
    let start = text[..at].rfind('\n').map_or(0, |n| n + 1);
    let mut depth = 0i32;
    for (i, ch) in text[at..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let close = at + i + 1;
                    let end = text[close..]
                        .find('\n')
                        .map_or(text.len(), |n| close + n + 1);
                    return Some((start, end));
                }
            }
            _ => {}
        }
    }
    None
}

/// One edit from the common prefix and suffix of `old` and `new`.
fn change(old: &str, new: String) -> SourceChange {
    let pre = old
        .bytes()
        .zip(new.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    let max = old.len().min(new.len()) - pre;
    let suf = old
        .bytes()
        .rev()
        .zip(new.bytes().rev())
        .take(max)
        .take_while(|(a, b)| a == b)
        .count();
    let (mut s, mut oe, mut ne) = (pre, old.len() - suf, new.len() - suf);
    while !old.is_char_boundary(s) || !new.is_char_boundary(s) {
        s -= 1;
    }
    while !old.is_char_boundary(oe) || !new.is_char_boundary(ne) {
        oe += 1;
        ne += 1;
    }
    SourceChange {
        edits: vec![TextEdit::replace(
            Span { start: s, end: oe },
            new[s..ne].to_owned(),
        )],
        source: new,
        reprinted: false,
    }
}

/// The body `index` of a regeneration: its topology and volume.
fn body(
    part: &crate::runtime::RegeneratedPart,
    index: usize,
) -> Result<(Topology, f64), EditFailure> {
    let b = part
        .result
        .state
        .bodies
        .get(index)
        .ok_or_else(|| model(format!("the part has no body {index}")))?;
    let volume = match part.kernel.analyze(
        &AnalysisRequest::MassProperties { density: 1.0 },
        &[&b.handle],
    ) {
        Ok(AnalysisResult::MassProperties(m)) => m.volume,
        _ => return Err(model("the kernel can't measure the body's volume")),
    };
    Ok((b.topology.clone(), volume))
}

/// Display bodies whose reference names feature `id`, as (index, reference text).
fn bodies_of(part: &crate::runtime::RegeneratedPart, id: &str) -> Vec<(usize, String)> {
    let prefix = format!("feature.{id}.");
    part.display
        .bodies
        .iter()
        .filter_map(|b| {
            let r = b.reference.as_deref()?;
            r.starts_with(&prefix)
                .then(|| (b.index as usize, r.to_owned()))
        })
        .collect()
}

fn perimeter(t: &Topology, f: usize) -> f64 {
    t.edges
        .iter()
        .filter(|e| e.faces.contains(&(f as u32)))
        .map(|e| e.geom.length)
        .sum()
}

fn same_surface(a: &SurfaceKind, b: &SurfaceKind) -> bool {
    match (a, b) {
        (SurfaceKind::Plane { normal: m }, SurfaceKind::Plane { normal: n }) => {
            (0..3).all(|i| (m[i] - n[i]).abs() < 1e-6)
        }
        (
            SurfaceKind::Cylinder {
                radius: r,
                axis_direction: d,
                ..
            },
            SurfaceKind::Cylinder {
                radius: s,
                axis_direction: e,
                ..
            },
        ) => (r - s).abs() < 1e-6 && (d[0] * e[0] + d[1] * e[1] + d[2] * e[2]).abs() > 1.0 - 1e-6,
        (SurfaceKind::Other, SurfaceKind::Other) => true,
        _ => false,
    }
}

/// Imported faces with no rebuilt face of the same kind within `tol`.
fn unmatched(old: &Topology, new: &Topology, tol: f64) -> Vec<usize> {
    let mut used = vec![false; new.faces.len()];
    let mut out = Vec::new();
    for (f, face) in old.faces.iter().enumerate() {
        let slack = tol * perimeter(old, f).max(tol);
        let hit = new.faces.iter().enumerate().position(|(g, other)| {
            let c = (0..3)
                .map(|i| (face.geom.centroid[i] - other.geom.centroid[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            !used[g]
                && same_surface(&face.geom.surface, &other.geom.surface)
                && c < tol
                && (face.geom.area - other.geom.area).abs() < slack
        });
        match hit {
            Some(g) => used[g] = true,
            None => out.push(f),
        }
    }
    out
}

fn face_word(t: &Topology, f: usize) -> &'static str {
    match t.faces[f].geom.surface {
        SurfaceKind::Plane { .. } => "plane",
        SurfaceKind::Cylinder { .. } => "cylinder",
        SurfaceKind::Other => {
            let curved = t
                .edges
                .iter()
                .any(|e| e.faces.contains(&(f as u32)) && !matches!(e.geom.curve, EdgeCurve::Line));
            if curved {
                "curved face"
            } else {
                "face"
            }
        }
    }
}

/// A part that doesn't parse, by its first error.
fn syntax(diagnostics: Vec<crate::runtime::Diagnostic>) -> EditFailure {
    model(format!(
        "the part has errors: {}",
        diagnostics.first().map_or("", |d| d.message.as_str())
    ))
}

pub(in crate::edit) fn recognize_features(
    path: &str,
    text: &str,
    request: &EditRequest,
    picks: &Picks,
) -> Result<(SourceChange, String), EditFailure> {
    let tol = match request.values.get("tolerance") {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.01) * 1e-3,
        Some(Value::String(s)) => {
            s.trim()
                .trim_end_matches("mm")
                .trim()
                .parse::<f64>()
                .unwrap_or(0.01)
                * 1e-3
        }
        _ => 1e-5,
    };
    if tol <= 0.0 {
        return Err(model("the tolerance must be more than 0 mm"));
    }
    let mut chosen = picks_of(&request.values, "body");
    if chosen.is_empty() {
        chosen = request
            .selection
            .iter()
            .filter(|s| s.starts_with("body:"))
            .cloned()
            .collect();
    }
    let pick = match chosen.as_slice() {
        [one] => one.clone(),
        [] => return Err(model("pick the imported body to recognize")),
        many => return Err(model(format!("pick one body, not {}", many.len()))),
    };
    let part = crate::runtime::regenerate_part(path, text).map_err(syntax)?;
    // The body, and the import that made it.
    let (index, reference) = if let Some(at) = pick.strip_prefix("body:") {
        let i: usize = at
            .parse()
            .map_err(|_| model(format!("`{pick}` isn't a body")))?;
        let r = part
            .display
            .bodies
            .iter()
            .find(|b| b.index as usize == i)
            .and_then(|b| b.reference.clone())
            .ok_or_else(|| model(format!("body {i} has no name to refer to")))?;
        (i, r)
    } else {
        let id = pick.strip_prefix("feature:").unwrap_or(&pick);
        match bodies_of(&part, id).as_slice() {
            [one] => one.clone(),
            _ => return Err(model(format!("`{id}` makes no single body to recognize"))),
        }
    };
    let import = reference
        .strip_prefix("feature.")
        .and_then(|r| r.split('.').next())
        .filter(|id| {
            picks
                .document
                .features
                .iter()
                .any(|f| f.id.0 == *id && matches!(f.kind, FeatureKind::Import(_)))
        })
        .ok_or_else(|| {
            model(format!(
                "{reference} isn't an imported body; recognize works on a body an import made"
            ))
        })?
        .to_owned();
    if bodies_of(&part, &import).len() != 1 {
        return Err(model(format!(
            "the import `{import}` makes {} bodies; split them to files first, then recognize each part",
            bodies_of(&part, &import).len()
        )));
    }
    let (topology, volume) = body(&part, index)?;
    let plan = graph::recognize(&topology).map_err(|left| {
        let reasons: Vec<String> = left
            .iter()
            .take(6)
            .map(|l| format!("face {} ({}): {}", l.face, face_word(&topology, l.face), l.why))
            .collect();
        let more = if left.len() > 6 { format!("; and {} more", left.len() - 6) } else { String::new() };
        model(format!(
            "{reference} stays imported: {} of its {} faces fit no feature this command recognizes. {}{more}",
            left.len(),
            topology.faces.len(),
            reasons.join("; ")
        ))
    })?;

    // The features in place of the import, then the body's other users moved to the new body.
    let mut names = Names(
        picks
            .document
            .features
            .iter()
            .map(|f| f.id.0.clone())
            .collect(),
    );
    let written = write(&plan, &mut names);
    let (start, end) = statement(text, &import).ok_or_else(|| {
        model(format!(
            "the import `{import}` isn't written as `feature {import} = import(…)`"
        ))
    })?;
    let spliced = format!("{}{}{}", &text[..start], written.text, &text[end..]);
    let was = lower_text(&reference)?;
    let now = lower_text(&format!("feature.{}.body", written.base))?;
    let target = Target::Reference(was, now);
    let moved = edit_part(&spliced, path, |document| {
        for feature in document.features.iter_mut() {
            let mut kind = serde_json::to_value(&feature.kind).map_err(|e| model(e.to_string()))?;
            let mut uses = Vec::new();
            replace(&mut kind, &target, "", &mut uses);
            if !uses.is_empty() {
                feature.kind = serde_json::from_value(kind).map_err(|e| model(e.to_string()))?;
            }
        }
        for output in document.parts.iter_mut() {
            let mut body = serde_json::to_value(&output.body).map_err(|e| model(e.to_string()))?;
            let mut uses = Vec::new();
            replace(&mut body, &target, "", &mut uses);
            if !uses.is_empty() {
                output.body = serde_json::from_value(body).map_err(|e| model(e.to_string()))?;
            }
        }
        Ok(())
    })?
    .source;

    // The rebuilt body must be the import, within the tolerance.
    let rebuilt = crate::runtime::regenerate_part(path, &moved).map_err(syntax)?;
    let errors: Vec<String> = rebuilt
        .result
        .diagnostics
        .diagnostics
        .iter()
        .filter(|d| d.severity == runmat_cad::diagnostics::Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    let failed = |why: String| {
        model(format!(
            "{reference} stays imported: the recognized features {why}"
        ))
    };
    if !errors.is_empty() {
        return Err(failed(format!("don't rebuild: {}", errors.join("; "))));
    }
    let new_index = match bodies_of(&rebuilt, &written.base).as_slice() {
        [(i, _)] => *i,
        _ => return Err(failed("don't make one body".into())),
    };
    let (new_topology, new_volume) = body(&rebuilt, new_index)?;
    let total_area: f64 = topology.faces.iter().map(|f| f.geom.area).sum();
    if (new_volume - volume).abs() > tol * total_area {
        return Err(failed(format!(
            "make a body of {:.6} cm³, not the import's {:.6} cm³",
            new_volume * 1e6,
            volume * 1e6
        )));
    }
    let missing = unmatched(&topology, &new_topology, tol);
    if !missing.is_empty() || new_topology.faces.len() != topology.faces.len() {
        let listed: Vec<String> = missing
            .iter()
            .take(6)
            .map(|f| format!("face {f} ({})", face_word(&topology, *f)))
            .collect();
        return Err(failed(format!(
            "make {} faces for the import's {}; unmatched within {}: {}",
            new_topology.faces.len(),
            topology.faces.len(),
            mm(tol),
            if listed.is_empty() {
                "none".into()
            } else {
                listed.join(", ")
            }
        )));
    }
    let summary = format!(
        "Recognized {reference} as {}: {}. It matches the import within {}: volume {:.3} cm³, all {} faces. The STEP file stays in the project; undo restores the import",
        written.base,
        written.counts.join(", "),
        mm(tol),
        volume * 1e6,
        topology.faces.len()
    );
    Ok((change(text, moved), summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn millimeters_read_short() {
        assert_eq!(mm(0.04), "40 mm");
        assert_eq!(mm(0.0125), "12.5 mm");
        assert_eq!(mm(-0.0000000001), "0 mm");
        assert_eq!(mm(0.0031234567), "3.123 mm");
    }

    #[test]
    fn the_import_statement_is_found_by_its_parentheses() {
        let text = "part p {\n    feature step = import(source = \"a (1).step\", digest = sha256_x)\n    output p = { body = feature.step.body }\n}\n";
        let (s, e) = statement(text, "step").unwrap();
        assert_eq!(
            &text[s..e],
            "    feature step = import(source = \"a (1).step\", digest = sha256_x)\n"
        );
    }

    #[test]
    fn one_edit_spans_what_changed() {
        let c = change("abcXYZdef", "abcQdef".into());
        assert_eq!(c.edits[0].span, Span { start: 3, end: 6 });
        assert_eq!(c.edits[0].text, "Q");
    }
}
