//! What a drawing's annotations read from the drawn parts: dimensions measured on the model's
//! faces (the distance between two parallel planar faces, a cylinder's diameter), hole callouts
//! from the hole's surface (`⌀4 THRU`), and GD&T feature control frames from the part's PMI
//! (`∥|0.05|A`: the characteristic's symbol, the zone and the datums, one cell each, as ASME
//! Y14.5 and ISO 1101 write them).
//!
//! Values print as drawings print them: in the part's millimetres with no unit, trailing zeros
//! dropped (`80`, `12.5`).

use runmat_cad::drawing::index::AnnotationValues;
use runmat_cad::drawing::{DimensionStyle, ModelRef};
use runmat_cad::model::PartDocument;
use runmat_cad::pmi::{Characteristic, MaterialCondition};
use runmat_cad::refs::name::{EntityName, TopoKind};
use runmat_cad::refs::reference::Reference;
use runmat_cad::syntax::refs::print_reference;
use runmat_cad::value::ValueExpr;
use serde_json::Value;
use std::collections::BTreeMap;

type P3 = [f64; 3];

/// The drawn parts' displays (by the file name the views wrote) and documents.
pub struct Values<'a> {
    pub displays: &'a serde_json::Map<String, Value>,
    pub documents: &'a BTreeMap<String, PartDocument>,
}

/// A number as a drawing prints it: two decimals at most, trailing zeros dropped.
pub fn number(v: f64) -> String {
    let s = format!("{:.2}", (v * 100.0).round() / 100.0);
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

/// A part face's mesh points (mm) and its surface, by the reference text it carries.
pub fn face<'v>(display: &'v Value, text: &str) -> Option<(Vec<P3>, &'v Value)> {
    for body in display["data"]["bodies"].as_array()? {
        for f in body["faces"].as_array().into_iter().flatten() {
            let named = f["reference"] == text
                || f["names"]
                    .as_array()
                    .is_some_and(|n| n.iter().any(|x| x == text));
            if named {
                let pos = floats(&f["positions"]);
                let points = pos
                    .chunks_exact(3)
                    .map(|c| [c[0] * 1000.0, c[1] * 1000.0, c[2] * 1000.0])
                    .collect();
                return Some((points, &f["surface"]));
            }
        }
    }
    None
}

fn centroid(points: &[P3]) -> P3 {
    let n = points.len().max(1) as f64;
    let mut c = [0.0; 3];
    for p in points {
        for i in 0..3 {
            c[i] += p[i] / n;
        }
    }
    c
}

fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3(v: &Value) -> Option<P3> {
    let f = floats(v);
    (f.len() == 3).then(|| [f[0], f[1], f[2]])
}

fn text_of(name: &EntityName) -> Option<String> {
    print_reference(&Reference::Name {
        kind: TopoKind::Face,
        name: name.clone(),
    })
}

/// A cylinder's axis point (mm), unit direction and radius (mm).
pub fn cylinder(surface: &Value) -> Option<(P3, P3, f64)> {
    if surface["surface"] != "cylinder" {
        return None;
    }
    let p = vec3(&surface["axis_point"])?.map(|x| x * 1000.0);
    let d = vec3(&surface["axis_direction"])?;
    let l = dot(d, d).sqrt().max(1e-12);
    Some((p, d.map(|x| x / l), surface["radius"].as_f64()? * 1000.0))
}

fn symbol(c: Characteristic) -> &'static str {
    match c {
        Characteristic::Straightness => "⏤",
        Characteristic::Flatness => "⏥",
        Characteristic::Circularity => "○",
        Characteristic::Cylindricity => "⌭",
        Characteristic::ProfileOfLine => "⌒",
        Characteristic::ProfileOfSurface => "⌓",
        Characteristic::Parallelism => "∥",
        Characteristic::Perpendicularity => "⊥",
        Characteristic::Angularity => "∠",
        Characteristic::Position => "⌖",
        Characteristic::Concentricity => "◎",
        Characteristic::Symmetry => "⌯",
        Characteristic::CircularRunout => "↗",
        Characteristic::TotalRunout => "⌰",
    }
}

fn modifier(m: MaterialCondition) -> &'static str {
    match m {
        MaterialCondition::Rfs => "",
        MaterialCondition::Mmc => " Ⓜ",
        MaterialCondition::Lmc => " Ⓛ",
    }
}

/// A length written in the part: millimetres, or the unit converted.
fn length_mm(v: &ValueExpr) -> Option<f64> {
    match v {
        ValueExpr::Quantity { value, unit } => Some(match unit.as_str() {
            "mm" => *value,
            "m" => value * 1000.0,
            "cm" => value * 10.0,
            "um" | "µm" => value / 1000.0,
            "in" => value * 25.4,
            _ => return None,
        }),
        ValueExpr::Number { value } => Some(*value),
        _ => None,
    }
}

impl AnnotationValues for Values<'_> {
    fn dimension(
        &self,
        model: &ModelRef,
        _style: DimensionStyle,
        entities: &[Vec<EntityName>],
    ) -> Option<String> {
        let display = self.displays.get(model.file.as_str())?;
        let faces: Vec<(Vec<P3>, &Value)> = entities
            .iter()
            .filter_map(|e| e.first())
            .filter_map(text_of)
            .filter_map(|t| face(display, &t))
            .collect();
        match faces.as_slice() {
            [(_, surface)] => cylinder(surface).map(|(_, _, r)| format!("⌀{}", number(2.0 * r))),
            [(a, sa), (b, _)] => {
                // Two parallel planes: the distance along the first one's normal.
                let n = vec3(&sa["normal"]).unwrap_or([0.0, 0.0, 1.0]);
                let (ca, cb) = (centroid(a), centroid(b));
                let d = (0..3).map(|i| (cb[i] - ca[i]) * n[i]).sum::<f64>().abs();
                Some(number(d))
            }
            _ => None,
        }
    }

    fn tolerance_frame(&self, model: &ModelRef, tolerance: &str) -> Option<String> {
        let doc = self.documents.get(model.file.as_str())?;
        let t = doc.pmi.tolerances.iter().find(|t| t.name == tolerance)?;
        let zone = format!(
            "{}{}{}",
            if t.diametral { "⌀" } else { "" },
            number(length_mm(&t.tolerance)?),
            modifier(t.modifier)
        );
        let mut cells = vec![symbol(t.characteristic).to_owned(), zone];
        cells.extend(
            t.datums
                .iter()
                .map(|d| format!("{}{}", d.label, modifier(d.modifier))),
        );
        Some(cells.join("|"))
    }

    fn hole_callout(&self, model: &ModelRef, hole: &Reference) -> Option<String> {
        let display = self.displays.get(model.file.as_str())?;
        let (points, surface) = face(display, &print_reference(hole)?)?;
        let (p, d, r) = cylinder(surface)?;
        let along: Vec<f64> = points
            .iter()
            .map(|q| dot([q[0] - p[0], q[1] - p[1], q[2] - p[2]], d))
            .collect();
        let depth = along.iter().cloned().fold(f64::MIN, f64::max)
            - along.iter().cloned().fold(f64::MAX, f64::min);
        // Through when the wall spans the body along its axis.
        let bounds = &display["data"]["bounds"];
        let (lo, hi) = (vec3(&bounds["min"])?, vec3(&bounds["max"])?);
        let span: f64 = (0..3).map(|i| (hi[i] - lo[i]) * 1000.0 * d[i].abs()).sum();
        let end = if depth >= span - 1e-3 {
            "THRU".to_owned()
        } else {
            format!("↧{}", number(depth))
        };
        Some(format!("⌀{} {end}", number(2.0 * r)))
    }
}
