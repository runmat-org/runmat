//! `annotate.auto-dimension`: dimension and tolerance a prismatic part from its datums.
//!
//! The datum reference frame is the picked faces (labelled A, B, C in order), else the datums
//! the part declares, else the part's largest planar face (A), the largest face square to it
//! (B) and the largest face square to both (C). Then each feature not yet toleranced gets:
//!
//! - **Plus and minus:** each hole its diameter and its location from every datum plane
//!   parallel to its axis; each planar face parallel to a datum plane its distance from it;
//!   all at the given ± tolerance.
//! - **Geometric:** each hole its diameter (±) and a position frame (⌀ twice the tolerance at
//!   maximum material condition) to the datum frame; each other planar face a profile of a
//!   surface (twice the tolerance) to it; the primary datum face a flatness of half the
//!   tolerance.
//!
//! Running it again adds only what the features added since lack.

use super::{build, fresh_id, Edit};
use crate::drawing::values;
use crate::drawing_edits::pmi::{self as shared, Frame, Tolerance};
use crate::drawing_edits::{number, text, Values};
use crate::edit::model;
use crate::picks::Picks;
use runmat_cad::drawing::{AnnotationKind, ArcCondition, DimensionStyle};
use runmat_cad::model::PartDocument;
use runmat_cad::pmi::{Characteristic, DatumReference, MaterialCondition, ModelAnnotation, PmiSet};
use runmat_cad::refs::reference::Reference;
use runmat_cad::syntax::edit::edit_part;
use serde_json::Value;

type P3 = [f64; 3];

/// A face of the part as the scheme reads it.
struct Face {
    text: String,
    reference: Reference,
    /// Plane: its normal. Cylinder: axis direction and point (mm), radius.
    shape: Shape,
    area: f64,
}

enum Shape {
    Plane { normal: P3 },
    Cylinder { axis: P3, point: P3, radius: f64 },
}

fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn area(points: &[P3], triangles: &[f64]) -> f64 {
    triangles
        .chunks_exact(3)
        .filter_map(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| points.get(i as usize).copied());
            let (a, b, c) = (a?, b?, c?);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            Some(dot(n, n).sqrt() / 2.0)
        })
        .sum()
}

fn faces(display: &Value, picks: &Picks) -> Vec<Face> {
    let mut out = Vec::new();
    for body in display["data"]["bodies"].as_array().into_iter().flatten() {
        if body["flat_pattern"] == true {
            continue;
        }
        for f in body["faces"].as_array().into_iter().flatten() {
            let Some(text) = f["reference"].as_str() else {
                continue;
            };
            let Some(reference) = picks
                .reference(text)
                .ok()
                .and_then(|v| serde_json::from_value::<Reference>(v).ok())
            else {
                continue;
            };
            let Some((points, surface)) = values::face(display, text) else {
                continue;
            };
            let triangles = runmat_cad::display::wire::numbers(&f["triangles"]);
            let shape = if let Some((point, axis, radius)) = values::cylinder(surface) {
                Shape::Cylinder {
                    axis,
                    point,
                    radius,
                }
            } else if surface["surface"] == "plane" {
                let Some(normal) = values::vec3(&surface["normal"]) else {
                    continue;
                };
                Shape::Plane { normal }
            } else {
                continue;
            };
            out.push(Face {
                text: text.to_owned(),
                reference,
                area: area(&points, &triangles),
                shape,
            });
        }
    }
    out
}

fn normal(f: &Face) -> Option<P3> {
    match f.shape {
        Shape::Plane { normal, .. } => Some(normal),
        Shape::Cylinder { .. } => None,
    }
}

/// Whether the part's PMI already tolerances the face (a size, location or frame on it).
fn toleranced(pmi: &PmiSet, r: &Reference) -> bool {
    pmi.sizes.iter().any(|s| s.on.contains(r))
        || pmi.locations.iter().any(|l| &l.to == r)
        || pmi.tolerances.iter().any(|t| t.on.contains(r))
}

fn next_label(pmi: &PmiSet) -> String {
    (b'A'..=b'Z')
        .map(|c| (c as char).to_string())
        .find(|l| {
            !["I", "O", "Q"].contains(&l.as_str()) && !pmi.datums.iter().any(|d| &d.label == l)
        })
        .unwrap_or_else(|| "Z".into())
}

pub fn run(path: &str, text_: &str, _doc: &PartDocument, values: &Values, picks: &Picks) -> Edit {
    let display = build::display(picks);
    let all = faces(&display, picks);
    if all.is_empty() {
        return Err(model("the part has no faces to dimension yet"));
    }
    let geometric = text(values, "scheme", "plus_minus") == "geometric";
    let tol = number(values, "tolerance", 0.1);
    if tol <= 0.0 {
        return Err(model("the tolerance must be more than zero"));
    }
    let picked = build::references(picks, values, "datums")?;
    let mut added = 0usize;
    let change = edit_part(text_, path, |doc| {
        let pmi = &mut doc.pmi;
        let mut push = |pmi: &mut PmiSet, kind: AnnotationKind, stem: &str| {
            let id = fresh_id(pmi, stem);
            pmi.annotations.push(ModelAnnotation {
                id,
                kind,
                view: None,
                at: None,
            });
            added += 1;
        };
        // The datum frame: picked faces, else declared datums, else inferred from the faces.
        let mut frame: Vec<(String, usize)> = Vec::new();
        let face_of = |r: &Reference| all.iter().position(|f| &f.reference == r);
        let mut declare = |pmi: &mut PmiSet,
                           at: usize,
                           frame: &mut Vec<(String, usize)>|
         -> Result<(), runmat_cad::syntax::edit::EditFailure> {
            let r = &all[at].reference;
            if let Some(d) = pmi.datums.iter().find(|d| &d.on == r) {
                frame.push((d.label.clone(), at));
                return Ok(());
            }
            let label = next_label(pmi);
            shared::declare_datum(pmi, &label, r)?;
            frame.push((label.clone(), at));
            push(pmi, AnnotationKind::DatumSymbol { label }, "datum");
            Ok(())
        };
        if !picked.is_empty() {
            for r in &picked {
                let at =
                    face_of(r).ok_or_else(|| model("a datum here is a planar face of the part"))?;
                if normal(&all[at]).is_none() {
                    return Err(model("pick planar faces as datums"));
                }
                declare(pmi, at, &mut frame)?;
            }
        } else {
            for d in pmi.datums.clone() {
                if let Some(at) = face_of(&d.on).filter(|&at| normal(&all[at]).is_some()) {
                    frame.push((d.label.clone(), at));
                }
            }
            if frame.is_empty() {
                let mut planes: Vec<usize> = (0..all.len())
                    .filter(|&i| normal(&all[i]).is_some())
                    .collect();
                planes.sort_by(|&a, &b| all[b].area.total_cmp(&all[a].area));
                let mut chosen: Vec<usize> = Vec::new();
                for i in planes {
                    let n = normal(&all[i]).expect("plane");
                    if chosen
                        .iter()
                        .all(|&c| dot(n, normal(&all[c]).expect("plane")).abs() < 0.02)
                    {
                        chosen.push(i);
                    }
                    if chosen.len() == 3 {
                        break;
                    }
                }
                for at in chosen {
                    declare(pmi, at, &mut frame)?;
                }
            }
        }
        if frame.is_empty() {
            return Err(model("the part has no planar face to use as datum A"));
        }
        let datum_faces: Vec<usize> = frame.iter().map(|(_, at)| *at).collect();
        let labels: Vec<DatumReference> = frame
            .iter()
            .take(3)
            .map(|(label, _)| DatumReference {
                label: label.clone(),
                modifier: MaterialCondition::Rfs,
            })
            .collect();
        let pm = Tolerance::Deviation {
            plus: tol,
            minus: tol,
        };
        let dimension =
            |pmi: &mut PmiSet, refs: Vec<Reference>, style: DimensionStyle, nominal: f64| {
                shared::declare_dimension(pmi, &refs, nominal, Some(&pm), None).map(|name| {
                    AnnotationKind::Dimension {
                        style,
                        references: refs,
                        pmi: Some(name),
                        group: None,
                        reference: false,
                        condition: ArcCondition::Center,
                        occurrences: Vec::new(),
                    }
                })
            };
        // Holes and bosses: one per axis and radius (a seam can split one wall in two).
        let mut seen: Vec<(P3, P3, f64)> = Vec::new();
        for f in &all {
            let Shape::Cylinder {
                axis,
                point,
                radius,
            } = f.shape
            else {
                continue;
            };
            let same = seen.iter().any(|(a, p, r)| {
                let v = [point[0] - p[0], point[1] - p[1], point[2] - p[2]];
                let along = dot(v, *a);
                let off = (0..3)
                    .map(|k| (v[k] - along * a[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (r - radius).abs() < 1e-6 && dot(*a, axis).abs() > 0.9999 && off < 1e-6
            });
            if same || toleranced(pmi, &f.reference) {
                seen.push((axis, point, radius));
                continue;
            }
            seen.push((axis, point, radius));
            let size = dimension(
                pmi,
                vec![f.reference.clone()],
                DimensionStyle::Diameter,
                2.0 * radius,
            )?;
            push(pmi, size, "dim");
            if geometric {
                let tolerance = shared::declare_tolerance(
                    pmi,
                    Frame {
                        characteristic: Characteristic::Position,
                        zone: 2.0 * tol,
                        diametral: true,
                        modifier: MaterialCondition::Mmc,
                        datums: labels.clone(),
                        projected: None,
                    },
                    vec![f.reference.clone()],
                    None,
                )?;
                push(pmi, AnnotationKind::Gdt { tolerance }, "gdt");
            } else {
                for &d in &datum_faces {
                    let n = normal(&all[d]).expect("plane");
                    if dot(n, axis).abs() > 0.02 {
                        continue;
                    }
                    let a = values::face(&display, &all[d].text);
                    let b = values::face(&display, &f.text);
                    let Some(dist) = a.zip(b).and_then(|(a, b)| values::linear(&a, &b)) else {
                        continue;
                    };
                    let loc = dimension(
                        pmi,
                        vec![all[d].reference.clone(), f.reference.clone()],
                        DimensionStyle::Linear,
                        dist,
                    )?;
                    push(pmi, loc, "dim");
                }
            }
        }
        // Planar faces parallel to a datum plane: their distance, or a profile.
        let mut offsets: Vec<(usize, i64)> = Vec::new();
        for (i, f) in all.iter().enumerate() {
            let Some(n) = normal(f) else { continue };
            if datum_faces.contains(&i) || toleranced(pmi, &f.reference) {
                continue;
            }
            if geometric {
                let tolerance = shared::declare_tolerance(
                    pmi,
                    Frame {
                        characteristic: Characteristic::ProfileOfSurface,
                        zone: 2.0 * tol,
                        diametral: false,
                        modifier: MaterialCondition::Rfs,
                        datums: labels.clone(),
                        projected: None,
                    },
                    vec![f.reference.clone()],
                    None,
                )?;
                push(pmi, AnnotationKind::Gdt { tolerance }, "gdt");
                continue;
            }
            let Some(&d) = datum_faces
                .iter()
                .find(|&&d| dot(n, normal(&all[d]).expect("plane")).abs() > 0.9999)
            else {
                continue;
            };
            let a = values::face(&display, &all[d].text);
            let b = values::face(&display, &f.text);
            let Some(dist) = a.zip(b).and_then(|(a, b)| values::linear(&a, &b)) else {
                continue;
            };
            // Coplanar with the datum, or a second piece at the same offset: nothing new.
            let key = (d, (dist * 1000.0).round() as i64);
            if dist < 1e-3 || offsets.contains(&key) {
                continue;
            }
            offsets.push(key);
            let loc = dimension(
                pmi,
                vec![all[d].reference.clone(), f.reference.clone()],
                DimensionStyle::Linear,
                dist,
            )?;
            push(pmi, loc, "dim");
        }
        if geometric {
            let primary = &all[datum_faces[0]].reference;
            if !pmi.tolerances.iter().any(|t| t.on.contains(primary)) {
                let tolerance = shared::declare_tolerance(
                    pmi,
                    Frame {
                        characteristic: Characteristic::Flatness,
                        zone: tol / 2.0,
                        diametral: false,
                        modifier: MaterialCondition::Rfs,
                        datums: Vec::new(),
                        projected: None,
                    },
                    vec![primary.clone()],
                    None,
                )?;
                push(pmi, AnnotationKind::Gdt { tolerance }, "gdt");
            }
        }
        Ok(())
    })?;
    if added == 0 {
        return Err(model("every feature is dimensioned already"));
    }
    Ok((change, format!("Added {added} annotations")))
}
