//! The part's 3D annotations placed in the model for the canvas, and listed for the Outline.
//!
//! Each annotation's attachment point comes from its canonical references on this rebuild's
//! display (a face's center, a hole's axis, the middle of an edge between two faces), so it
//! follows the geometry. Its label sits at the offset the file gives (`at`, mm), else off the
//! part along the direction from the model's center, staggered when several share a point.
//! Text is the drawing's: the same characteristic symbols, zones, tolerances and finishes.

use super::form;
use crate::drawing::point_of;
use crate::drawing::values::{self, length_mm, modifier, number, symbol, tolerance_text};
use crate::drawing_edits::pmi::characteristic_words;
use runmat_cad::display::wire::numbers;
use runmat_cad::display::{AnnotationDisplay, PartDisplay};
use runmat_cad::drawing::{AnnotationKind, DimensionStyle, WeldSide};
use runmat_cad::index::AnnotationRow;
use runmat_cad::model::PartDocument;
use runmat_cad::pmi::{ModelAnnotation, PmiSet, Removal};
use runmat_cad::refs::name::{EntityName, TopoKind};
use runmat_cad::refs::reference::Reference;
use runmat_cad::regen::engine::RegenResult;
use runmat_cad::syntax::refs::print_reference;
use serde_json::{json, Value};

type P3 = [f64; 3];

/// A point on a flat or free-form face (mm): the triangle nearest the face's area-weighted
/// center, at that triangle's center. A face with a hole or a boss through it has its center
/// off the face, and its mesh's vertices crowd round the hole, so neither stands in for it.
/// None for a cylinder (its axis is the anchor) and for a face the display doesn't hold.
fn on_face(display: &Value, text: &str) -> Option<P3> {
    face_point(display, text).map(|(p, _)| p)
}

/// The point [`on_face`] gives, with the face's normal when the face is flat.
fn face_point(display: &Value, text: &str) -> Option<(P3, Option<P3>)> {
    let face = display["data"]["bodies"]
        .as_array()?
        .iter()
        .flat_map(|b| b["faces"].as_array().into_iter().flatten())
        .find(|f| {
            f["reference"] == text
                || f["names"]
                    .as_array()
                    .is_some_and(|n| n.iter().any(|x| x == text))
        })?;
    if values::cylinder(&face["surface"]).is_some() {
        return None;
    }
    let pos: Vec<f64> = numbers(&face["positions"]);
    let at = |i: usize| -> Option<P3> {
        Some([
            *pos.get(3 * i)? * 1000.0,
            *pos.get(3 * i + 1)? * 1000.0,
            *pos.get(3 * i + 2)? * 1000.0,
        ])
    };
    let tris: Vec<(P3, f64, P3)> = numbers(&face["triangles"])
        .into_iter()
        .map(|x| x as usize)
        .collect::<Vec<_>>()
        .chunks_exact(3)
        .filter_map(|c| {
            let (a, b, d) = (at(c[0])?, at(c[1])?, at(c[2])?);
            let u = [0, 1, 2].map(|i| b[i] - a[i]);
            let v = [0, 1, 2].map(|i| d[i] - a[i]);
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let area = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() / 2.0;
            Some(([0, 1, 2].map(|i| (a[i] + b[i] + d[i]) / 3.0), area, n))
        })
        .collect();
    // Flat when every triangle's normal lies along the first one's.
    let normal = tris.iter().find(|t| t.1 > 0.0).and_then(|t| unit(t.2));
    let flat = normal.filter(|n| {
        tris.iter()
            .filter(|t| t.1 > 0.0)
            .filter_map(|t| unit(t.2))
            .all(|m| (m[0] * n[0] + m[1] * n[1] + m[2] * n[2]).abs() > 0.9999)
    });
    let total: f64 = tris.iter().map(|t| t.1).sum();
    if total <= 0.0 {
        return None;
    }
    let center = [0, 1, 2].map(|i| tris.iter().map(|(c, a, _)| c[i] * a).sum::<f64>() / total);
    let gap = |c: &P3| (0..3).map(|i| (c[i] - center[i]).powi(2)).sum::<f64>();
    tris.iter()
        .map(|t| t.0)
        .min_by(|a, b| gap(a).total_cmp(&gap(b)))
        .map(|p| (p, flat))
}

/// The middle of an edge (mm), halfway along its drawn polyline.
fn edge_middle(display: &Value, text: &str) -> Option<P3> {
    let edge = display["data"]["bodies"]
        .as_array()?
        .iter()
        .flat_map(|b| b["edges"].as_array().into_iter().flatten())
        .find(|e| {
            e["reference"] == text
                || e["names"]
                    .as_array()
                    .is_some_and(|n| n.iter().any(|x| x == text))
        })?;
    let pts: Vec<P3> = numbers(&edge["points"])
        .chunks_exact(3)
        .map(|c| [c[0] * 1000.0, c[1] * 1000.0, c[2] * 1000.0])
        .collect();
    let len = |a: &P3, b: &P3| (0..3).map(|i| (b[i] - a[i]).powi(2)).sum::<f64>().sqrt();
    let total: f64 = pts.windows(2).map(|w| len(&w[0], &w[1])).sum();
    let mut left = total / 2.0;
    for w in pts.windows(2) {
        let l = len(&w[0], &w[1]);
        if l >= left && l > 0.0 {
            let t = left / l;
            return Some([0, 1, 2].map(|i| w[0][i] + (w[1][i] - w[0][i]) * t));
        }
        left -= l;
    }
    pts.first().copied()
}

/// The foot of `p` on the plane through `on` with normal `n`.
fn foot(p: P3, on: P3, n: P3) -> P3 {
    let d = (0..3).map(|i| (p[i] - on[i]) * n[i]).sum::<f64>();
    [0, 1, 2].map(|i| p[i] - d * n[i])
}

/// A flat face's plane (a point on it and its normal), for a reference that names one face.
fn plane(display: &Value, r: &Reference) -> Option<(P3, P3)> {
    if matches!(r, Reference::EdgesBetween { .. }) {
        return None;
    }
    let (p, n) = face_point(display, &print_reference(r)?)?;
    Some((p, n?))
}

/// Where a reference sits on the display (mm): a point on a face, a hole's axis, an edge's
/// middle; else the first face it resolved to on this rebuild.
fn point(display: &Value, r: &Reference, resolved: &[Vec<EntityName>]) -> Option<P3> {
    let own = || match r {
        // The display may name the edge with its two faces either way round.
        Reference::EdgesBetween { a, b } => {
            edge_middle(display, &print_reference(r)?).or_else(|| {
                let swapped = Reference::EdgesBetween {
                    a: b.clone(),
                    b: a.clone(),
                };
                edge_middle(display, &print_reference(&swapped)?)
            })
        }
        _ => on_face(display, &print_reference(r)?),
    };
    own().or_else(|| point_of(display, r)).or_else(|| {
        let name = resolved.first()?.first()?;
        let text = print_reference(&Reference::Name {
            kind: TopoKind::Face,
            name: name.clone(),
        })?;
        on_face(display, &text)
            .or_else(|| values::face(display, &text).map(|(pts, _)| values::centroid(&pts)))
    })
}

/// The references an annotation attaches to, with what each resolved to on this rebuild.
fn attachments<'a>(
    pmi: &'a PmiSet,
    result: &'a RegenResult,
    a: &'a ModelAnnotation,
) -> Vec<(&'a Reference, &'a [Vec<EntityName>])> {
    let r = &result.pmi;
    let none: &[Vec<EntityName>] = &[];
    let own = r.annotations.get(&a.id).map(Vec::as_slice).unwrap_or(none);
    match &a.kind {
        AnnotationKind::DatumSymbol { label } => pmi
            .datums
            .iter()
            .filter(|d| &d.label == label)
            .map(|d| {
                (
                    &d.on,
                    r.datums.get(label).map(Vec::as_slice).unwrap_or(none),
                )
            })
            .collect(),
        AnnotationKind::Gdt { tolerance } => pmi
            .tolerances
            .iter()
            .filter(|t| &t.name == tolerance)
            .flat_map(|t| t.on.iter())
            .map(|on| {
                (
                    on,
                    r.tolerances
                        .get(tolerance)
                        .map(Vec::as_slice)
                        .unwrap_or(none),
                )
            })
            .collect(),
        AnnotationKind::Coating { treatment } => pmi
            .treatments
            .iter()
            .filter(|t| &t.name == treatment)
            .flat_map(|t| t.on.iter())
            .map(|on| (on, none))
            .collect(),
        _ => a.references().into_iter().map(|on| (on, own)).collect(),
    }
}

/// The label's cells: a frame's characteristic, zone and datums; a dimension's value with its
/// tolerance; a finish's removal requirement, Ra (µm), method and lay; a weld's symbol, size,
/// side, all-around and field marks, length-pitch and tail; a note's lines.
fn cells(
    pmi: &PmiSet,
    a: &ModelAnnotation,
    measured: Option<(DimensionStyle, f64)>,
) -> Vec<String> {
    match &a.kind {
        AnnotationKind::DatumSymbol { label } => vec![label.clone()],
        AnnotationKind::Gdt { tolerance } => {
            let Some(t) = pmi.tolerances.iter().find(|t| &t.name == tolerance) else {
                return vec![tolerance.clone()];
            };
            let mut zone = format!(
                "{}{}{}",
                if t.diametral { "⌀" } else { "" },
                length_mm(&t.tolerance).map(number).unwrap_or_default(),
                modifier(t.modifier)
            );
            if let Some(p) = t.projected.as_ref().and_then(length_mm) {
                zone.push_str(&format!(" Ⓟ{}", number(p)));
            }
            let mut out = vec![symbol(t.characteristic).to_owned(), zone];
            out.extend(
                t.datums
                    .iter()
                    .map(|d| format!("{}{}", d.label, modifier(d.modifier))),
            );
            out
        }
        AnnotationKind::Dimension { pmi: name, .. } => {
            let value = match measured {
                Some((DimensionStyle::Diameter, v)) => format!("⌀{}", number(v)),
                Some((_, v)) => number(v),
                None => name.clone().unwrap_or_else(|| "Dimension".into()),
            };
            let tolerance = name.as_deref().and_then(|n| {
                if let Some(s) = pmi.sizes.iter().find(|s| s.name == n) {
                    return s.fit.clone().or_else(|| tolerance_text(&s.size));
                }
                tolerance_text(&pmi.locations.iter().find(|l| l.name == n)?.distance)
            });
            vec![match tolerance {
                Some(t) => format!("{value} {t}"),
                None => value,
            }]
        }
        AnnotationKind::SurfaceFinish { on } => {
            let Some(f) = pmi.finishes.iter().find(|f| f.on.contains(on)) else {
                return vec!["any".into(), "Ra".into()];
            };
            let removal = match f.removal {
                Removal::Any => "any",
                Removal::Required => "required",
                Removal::Prohibited => "prohibited",
            };
            let ra = length_mm(&f.ra)
                .map(|r| number(r * 1000.0))
                .unwrap_or_default();
            let mut out = vec![removal.to_owned(), format!("Ra {ra}")];
            out.push(f.process.clone().unwrap_or_default());
            out.push(f.lay.clone().unwrap_or_default());
            out
        }
        AnnotationKind::Coating { treatment } => vec![pmi
            .treatments
            .iter()
            .find(|t| &t.name == treatment)
            .map(|t| t.specification.clone())
            .unwrap_or_else(|| treatment.clone())],
        AnnotationKind::Weld {
            symbol,
            size,
            all_around,
            field,
            side,
            length,
            pitch,
            tail,
            ..
        } => {
            let side = match side {
                WeldSide::Arrow => "arrow",
                WeldSide::Other => "other",
                WeldSide::Both => "both",
            };
            let lp = match (
                length.as_ref().and_then(length_mm),
                pitch.as_ref().and_then(length_mm),
            ) {
                (Some(l), Some(p)) => format!("{}-{}", number(l), number(p)),
                _ => String::new(),
            };
            vec![
                symbol.clone(),
                size.as_ref()
                    .and_then(length_mm)
                    .map(number)
                    .unwrap_or_default(),
                side.to_owned(),
                if *all_around { "all_around" } else { "" }.to_owned(),
                if *field { "field" } else { "" }.to_owned(),
                lp,
                tail.clone().unwrap_or_default(),
            ]
        }
        AnnotationKind::Note { text, .. } => text.split('\n').map(str::to_owned).collect(),
        _ => Vec::new(),
    }
}

fn kind_word(kind: &AnnotationKind) -> &'static str {
    match kind {
        AnnotationKind::DatumSymbol { .. } => "datum",
        AnnotationKind::Gdt { .. } => "gdt",
        AnnotationKind::Dimension { .. } => "dimension",
        AnnotationKind::SurfaceFinish { .. } => "finish",
        AnnotationKind::Coating { .. } => "coating",
        AnnotationKind::Weld { .. } => "weld",
        _ => "note",
    }
}

fn unit(v: P3) -> Option<P3> {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (l > 1e-9).then(|| v.map(|x| x / l))
}

fn metres(p: P3) -> [f32; 3] {
    p.map(|x| (x / 1000.0) as f32)
}

/// The part's annotations placed for the canvas.
pub fn annotation_display(
    document: &PartDocument,
    result: &RegenResult,
    display: &PartDisplay,
) -> Vec<AnnotationDisplay> {
    let pmi = &document.pmi;
    if pmi.annotations.is_empty() {
        return Vec::new();
    }
    let json = json!({ "kind": "part", "data": { "bodies": &display.bodies } });
    let (lo, hi) = display
        .bounds
        .as_ref()
        .map(|b| {
            (
                b.min.map(|x| x as f64 * 1000.0),
                b.max.map(|x| x as f64 * 1000.0),
            )
        })
        .unwrap_or(([0.0; 3], [0.0; 3]));
    let center: P3 = [0, 1, 2].map(|i| (lo[i] + hi[i]) / 2.0);
    let size = (0..3)
        .map(|i| (hi[i] - lo[i]).powi(2))
        .sum::<f64>()
        .sqrt()
        .max(10.0);
    let mut placed: Vec<P3> = Vec::new();
    pmi.annotations
        .iter()
        .map(|a| {
            let refs = attachments(pmi, result, a);
            let points: Vec<Option<P3>> = refs.iter().map(|(r, e)| point(&json, r, e)).collect();
            let ok = !points.is_empty() && points.iter().all(Option::is_some)
                || matches!(&a.kind, AnnotationKind::Note { leader: None, .. })
                || matches!(&a.kind, AnnotationKind::Coating { .. }) && refs.is_empty();
            let mut found: Vec<P3> = points.iter().flatten().copied().collect();
            // A dimension to a flat face measures along its normal: its end on the face is the
            // foot of the other end, so the dimension line runs the way it measures.
            if matches!(&a.kind, AnnotationKind::Dimension { .. })
                && found.len() == 2
                && refs.len() == 2
            {
                if let Some((on, n)) = plane(&json, refs[0].0) {
                    found[0] = foot(found[1], on, n);
                }
                if let Some((on, n)) = plane(&json, refs[1].0) {
                    found[1] = foot(found[0], on, n);
                }
            }
            let anchor = found.first().copied().unwrap_or(center);
            let to = match &a.kind {
                AnnotationKind::Dimension { .. } => found.get(1).copied(),
                _ => None,
            };
            let measured = match &a.kind {
                AnnotationKind::Dimension { references, .. } => {
                    super::build::measure(&json, references).ok()
                }
                _ => None,
            };
            let mid = match to {
                Some(b) => [0, 1, 2].map(|i| (anchor[i] + b[i]) / 2.0),
                None => anchor,
            };
            let at = match a.at {
                Some(off) => [0, 1, 2].map(|i| mid[i] + off[i]),
                None => {
                    // A dimension's value sits on its line; any other label off the part, away
                    // from the model's center and up.
                    let mut p = if to.is_some() {
                        mid
                    } else {
                        let out = [0, 1, 2].map(|i| mid[i] - center[i]);
                        let dir =
                            unit([out[0], out[1], out[2] + 0.35 * size]).unwrap_or([0.0, 0.0, 1.0]);
                        [0, 1, 2].map(|i| mid[i] + dir[i] * 0.3 * size)
                    };
                    // Labels that would land on one another step aside, along a slant that
                    // separates them in the standard views (Top, Front, Right and isometric).
                    // Two labels clash when they would overlap seen along any axis.
                    let clash = |p: &P3| {
                        placed.iter().any(|q| {
                            (0..3).any(|skip| {
                                (0..3)
                                    .filter(|&i| i != skip)
                                    .map(|i| (q[i] - p[i]).powi(2))
                                    .sum::<f64>()
                                    .sqrt()
                                    < 0.06 * size
                            })
                        })
                    };
                    let step = [0.45, -0.75, 0.5];
                    for _ in 0..16 {
                        if !clash(&p) {
                            break;
                        }
                        for i in 0..3 {
                            p[i] += 0.07 * size * step[i];
                        }
                    }
                    p
                }
            };
            placed.push(at);
            AnnotationDisplay {
                id: a.id.clone(),
                kind: kind_word(&a.kind).to_owned(),
                cells: cells(pmi, a, measured),
                anchor: metres(anchor),
                at: metres(at),
                to: to.map(metres),
                view: a.view.clone(),
                ok,
                free: found.is_empty() && ok,
            }
        })
        .collect()
}

/// The Outline's rows: each annotation's command and values (for Properties), a label and a
/// short detail.
pub fn annotation_rows(document: &PartDocument, shown: &[AnnotationDisplay]) -> Vec<AnnotationRow> {
    let pmi = &document.pmi;
    pmi.annotations
        .iter()
        .map(|a| {
            let d = shown.iter().find(|d| d.id == a.id);
            let cells: Vec<String> = d.map(|d| d.cells.clone()).unwrap_or_default();
            let first = cells.first().cloned().unwrap_or_default();
            let (label, detail) = match &a.kind {
                AnnotationKind::DatumSymbol { label } => (format!("Datum {label}"), String::new()),
                AnnotationKind::Gdt { tolerance } => {
                    let words = pmi
                        .tolerances
                        .iter()
                        .find(|t| &t.name == tolerance)
                        .map(|t| characteristic_words(t.characteristic))
                        .unwrap_or("GD&T frame");
                    (
                        words.to_owned(),
                        cells.get(1..).map(|c| c.join(" | ")).unwrap_or_default(),
                    )
                }
                AnnotationKind::Dimension { .. } => ("Dimension".into(), first),
                AnnotationKind::SurfaceFinish { .. } => (
                    "Surface finish".into(),
                    cells.get(1).cloned().unwrap_or_default(),
                ),
                AnnotationKind::Coating { .. } => ("Coating".into(), first),
                AnnotationKind::Weld { symbol, .. } => {
                    let mut w = symbol.clone();
                    if let Some(c) = w.get_mut(0..1) {
                        c.make_ascii_uppercase();
                    }
                    (
                        format!("{w} weld"),
                        cells.get(1).cloned().unwrap_or_default(),
                    )
                }
                _ => {
                    let line: String = first.chars().take(40).collect();
                    ("Note".into(), line)
                }
            };
            let (command, values) = form::values(document, a)
                .map(|(c, v)| (c.to_owned(), v))
                .unwrap_or_default();
            AnnotationRow {
                id: a.id.clone(),
                command,
                label,
                detail,
                view: a.view.clone(),
                values,
                ok: d.is_none_or(|d| d.ok),
            }
        })
        .collect()
}
