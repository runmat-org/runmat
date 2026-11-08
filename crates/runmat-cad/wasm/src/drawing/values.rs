//! What a drawing's annotations read from the drawn parts: dimensions measured on the model's
//! faces (the distance between two parallel planar faces, a cylinder's diameter), hole callouts
//! from the hole feature (`4× ⌀2.2 THRU`) or else the hole's surface (`⌀4 THRU`), size and
//! location tolerances from the PMI after the value (`⌀4.1 +0.012/0`), surface finishes
//! (`Ra 0.8`), and GD&T feature control frames from the part's PMI
//! (`∥|0.05|A`: the characteristic's symbol, the zone and the datums, one cell each, as ASME
//! Y14.5 and ISO 1101 write them).
//!
//! Values print as drawings print them: in the part's millimetres with no unit, trailing zeros
//! dropped (`80`, `12.5`).

use runmat_cad::drawing::index::AnnotationValues;
use runmat_cad::drawing::{ArcCondition, DimensionStyle, ModelRef};
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

/// The distance a linear dimension between two faces measures: between two parallel axes, or
/// along a plane's normal (to a hole's axis).
pub(crate) fn linear((a, sa): &(Vec<P3>, &Value), (b, sb): &(Vec<P3>, &Value)) -> Option<f64> {
    // Two holes with parallel axes: the distance between the axes.
    if let (Some((pa, da, _)), Some((pb, _, _))) = (cylinder(sa), cylinder(sb)) {
        let v = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let along = dot(v, da);
        return Some((0..3).map(|i| (v[i] - along * da[i]).powi(2)).sum::<f64>().sqrt());
    }
    // Two parallel planes, or a plane and a hole: the distance along the plane's normal, to the
    // hole's axis (a mesh's points are not spread evenly round a cylinder, so their centroid is
    // off the axis).
    let n = vec3(&sa["normal"]).or_else(|| vec3(&sb["normal"])).unwrap_or([0.0, 0.0, 1.0]);
    let anchor = |points: &[P3], surface: &Value| cylinder(surface).map_or_else(|| centroid(points), |(p, _, _)| p);
    let (ca, cb) = (anchor(a, sa), anchor(b, sb));
    Some((0..3).map(|i| (cb[i] - ca[i]) * n[i]).sum::<f64>().abs())
}

/// The acute angle (degrees, 0 to 90) between two planes, as an angular dimension between their
/// edges reads it, or none when either isn't a plane.
pub(crate) fn angle((_, sa): &(Vec<P3>, &Value), (_, sb): &(Vec<P3>, &Value)) -> Option<f64> {
    if sa["surface"] != "plane" || sb["surface"] != "plane" {
        return None;
    }
    let (na, nb) = (vec3(&sa["normal"])?, vec3(&sb["normal"])?);
    let c = dot(na, nb).clamp(-1.0, 1.0);
    let a = c.acos().to_degrees();
    Some(a.min(180.0 - a))
}

/// A linear dimension's anchors moved from its cylinders' axes to the near (`min`) or far
/// (`max`) sides of their outlines, along the line the dimension measures, so its extension
/// lines meet the outline where the value is measured to.
pub fn arc_anchors(display: &Value, references: &[Reference], points: &mut [P3], condition: ArcCondition) {
    let sign = match condition {
        ArcCondition::Center => return,
        ArcCondition::Min => -1.0,
        ArcCondition::Max => 1.0,
    };
    let [pa, pb] = points else {
        return;
    };
    let surfaces: Vec<Option<&Value>> = references
        .iter()
        .map(|r| print_reference(r).and_then(|t| face(display, &t)).map(|(_, s)| s))
        .collect();
    let cyl = |i: usize| surfaces.get(i).copied().flatten().and_then(cylinder);
    let mut v = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    // Across a cylinder's axis, or along a plane's normal.
    if let Some((_, d, _)) = cyl(0).or_else(|| cyl(1)) {
        let along = dot(v, d);
        v = [0, 1, 2].map(|i| v[i] - along * d[i]);
    }
    if let Some(n) = surfaces.iter().flatten().find_map(|s| vec3(&s["normal"])) {
        let k = dot(v, n);
        v = n.map(|x| x * k);
    }
    let l = dot(v, v).sqrt();
    if l < 1e-9 {
        return;
    }
    let u = v.map(|x| x / l);
    if let Some((_, _, r)) = cyl(0) {
        *pa = [0, 1, 2].map(|i| pa[i] - sign * r * u[i]);
    }
    if let Some((_, _, r)) = cyl(1) {
        *pb = [0, 1, 2].map(|i| pb[i] + sign * r * u[i]);
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
        style: DimensionStyle,
        entities: &[Vec<EntityName>],
    ) -> Option<String> {
        let display = self.displays.get(model.file.as_str())?;
        let faces: Vec<(Vec<P3>, &Value)> = entities
            .iter()
            .filter_map(|e| e.first())
            .filter_map(text_of)
            .filter_map(|t| face(display, &t))
            .collect();
        match (style, faces.as_slice()) {
            (DimensionStyle::Radius, [(_, surface)]) => cylinder(surface).map(|(_, _, r)| format!("R{}", number(r))),
            (_, [(_, surface)]) => cylinder(surface).map(|(_, _, r)| format!("⌀{}", number(2.0 * r))),
            (DimensionStyle::Angular, [a, b]) => angle(a, b).map(|deg| format!("{}°", number(deg))),
            // Baseline: each feature from the first; chain: each from the one before. One value
            // per pair, in order, joined by ` · ` (the sheet draws one dimension line each).
            (DimensionStyle::Baseline, [first, rest @ ..]) if !rest.is_empty() => {
                let v: Option<Vec<String>> = rest.iter().map(|f| linear(first, f).map(number)).collect();
                v.map(|v| v.join(" · "))
            }
            (DimensionStyle::Chain, all) if all.len() >= 2 => {
                let v: Option<Vec<String>> = all.windows(2).map(|w| linear(&w[0], &w[1]).map(number)).collect();
                v.map(|v| v.join(" · "))
            }
            (_, [a, b]) => linear(a, b).map(number),
            _ => None,
        }
    }

    fn dimension_to(
        &self,
        model: &ModelRef,
        style: DimensionStyle,
        condition: ArcCondition,
        entities: &[Vec<EntityName>],
    ) -> Option<String> {
        let sign = match condition {
            ArcCondition::Center => return self.dimension(model, style, entities),
            ArcCondition::Min => -1.0,
            ArcCondition::Max => 1.0,
        };
        let display = self.displays.get(model.file.as_str())?;
        let faces: Vec<(Vec<P3>, &Value)> = entities
            .iter()
            .filter_map(|e| e.first())
            .filter_map(text_of)
            .filter_map(|t| face(display, &t))
            .collect();
        let [a, b] = faces.as_slice() else {
            return None;
        };
        let radii: f64 = [a, b].iter().filter_map(|(_, s)| cylinder(s)).map(|(_, _, r)| r).sum();
        Some(number(linear(a, b)? + sign * radii))
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
        if let Some(text) = self.documents.get(model.file.as_str()).and_then(|d| hole_feature_callout(d, hole)) {
            return Some(text);
        }
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

    fn pmi_tolerance(&self, model: &ModelRef, name: &str) -> Option<String> {
        let pmi = &self.documents.get(model.file.as_str())?.pmi;
        if let Some(size) = pmi.sizes.iter().find(|t| t.name == name) {
            return size.fit.clone().or_else(|| tolerance_text(&size.size));
        }
        let location = pmi.locations.iter().find(|t| t.name == name)?;
        tolerance_text(&location.distance)
    }

    fn surface_finish(&self, model: &ModelRef, on: &Reference) -> Option<String> {
        let pmi = &self.documents.get(model.file.as_str())?.pmi;
        let finish = pmi.finishes.iter().find(|f| f.on.contains(on))?;
        // Roughness reads in micrometres (`Ra 0.8`).
        let ra = length_mm(&finish.ra)? * 1000.0;
        let mut text = format!("Ra {}", number(ra));
        if let Some(process) = &finish.process {
            text.push_str(&format!(" {process}"));
        }
        Some(text)
    }
}

/// A tolerance as a drawing writes it after the nominal: `±0.1` when symmetric, else
/// `+0.012/0` (upper over lower, a zero deviation without a sign).
fn tolerance_text(value: &ValueExpr) -> Option<String> {
    let ValueExpr::Toleranced { minus, plus, .. } = value else {
        return None;
    };
    let (p, m) = (length_mm(plus)?, length_mm(minus)?);
    if (p - m).abs() < 1e-9 {
        return Some(format!("±{}", number(p)));
    }
    let signed = |v: f64, sign: char| if v.abs() < 1e-9 { "0".to_owned() } else { format!("{sign}{}", number_fine(v)) };
    Some(format!("{}/{}", signed(p, '+'), signed(m, '-')))
}

/// Three decimals at most, for deviations (`0.012`).
fn number_fine(v: f64) -> String {
    let s = format!("{:.3}", (v * 1000.0).round() / 1000.0);
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// A hole feature's callout as ASME Y14.5 writes it: the count, the size and the end
/// (`4× ⌀2.2 THRU`, `2× M3x0.5-6H ↧6`), read from the feature the reference names.
fn hole_feature_callout(document: &PartDocument, hole: &Reference) -> Option<String> {
    use runmat_cad::model::feature::{FeatureKind, HoleEnd, HoleForm};
    let Reference::Name { name: runmat_cad::refs::name::EntityName::Generated(g), .. } = hole else {
        return None;
    };
    let feature = document.features.iter().find(|f| f.id == g.feature)?;
    let FeatureKind::Hole(h) = &feature.kind else {
        return None;
    };
    let count = h.positions.len();
    let mut size = match (&h.diameter, &h.spec) {
        (Some(d), _) => format!("⌀{}", number(length_mm(d)?)),
        (None, Some(spec)) => {
            let tapped = matches!(h.form, HoleForm::Tapped { .. });
            if tapped {
                let t = runmat_cad::model::hole_tables::thread(spec).ok()?;
                format!("{}-{}", t.designation, t.class)
            } else {
                let holes = runmat_cad::model::hole_tables::fastener(spec).ok()?;
                format!("⌀{}", number(holes.clearance(spec.fit) * 1000.0))
            }
        }
        (None, None) => return None,
    };
    let end = match &h.end {
        HoleEnd::Blind { depth } => format!("↧{}", number(length_mm(depth)?)),
        HoleEnd::ThroughAll | HoleEnd::UpToNext | HoleEnd::UpToFace { .. } => "THRU".to_owned(),
    };
    size.push(' ');
    size.push_str(&end);
    // The form after the hole (ASME Y14.5 symbols): counterbore ⌴⌀ ↧, countersink ⌵⌀ × angle.
    let mm = |v: &Option<runmat_cad::value::ValueExpr>| v.as_ref().and_then(length_mm).map(number);
    match &h.form {
        HoleForm::Counterbore { diameter, depth } => {
            if let Some(d) = mm(diameter) {
                size.push_str(&format!(" ⌴⌀{d}"));
                if let Some(t) = mm(depth) {
                    size.push_str(&format!(" ↧{t}"));
                }
            }
        }
        HoleForm::Countersink { diameter, angle } => {
            if let Some(d) = mm(diameter) {
                size.push_str(&format!(" ⌵⌀{d}"));
                if let Some(runmat_cad::value::ValueExpr::Quantity { value, unit }) = angle {
                    if unit == "deg" {
                        size.push_str(&format!(" × {}°", number(*value)));
                    }
                }
            }
        }
        _ => {}
    }
    Some(if count > 1 { format!("{count}× {size}") } else { size })
}

/// The model entity a leadered annotation points at: a GD&T frame's toleranced feature and a
/// datum symbol's datum feature (both from the part's PMI), a finish mark's face and a note's
/// leader.
pub fn leader_target(
    kind: &runmat_cad::drawing::AnnotationKind,
    document: Option<&PartDocument>,
) -> Option<runmat_cad::refs::reference::Reference> {
    use runmat_cad::drawing::AnnotationKind as K;
    match kind {
        K::Gdt { tolerance } => document?
            .pmi
            .tolerances
            .iter()
            .find(|t| &t.name == tolerance)?
            .on
            .first()
            .cloned(),
        K::DatumSymbol { label } => document?
            .pmi
            .datums
            .iter()
            .find(|d| &d.label == label)
            .map(|d| d.on.clone()),
        K::SurfaceFinish { on } | K::CenterMark { on } | K::Weld { on, .. } => Some(on.clone()),
        // A hole callout points at the hole: its axis, which a view along it shows as the centre.
        K::HoleCallout { hole } => Some(hole.clone()),
        K::DatumTarget { label } => {
            let split = label.find(|c: char| c.is_ascii_digit())?;
            let (datum, index) = label.split_at(split);
            let index: u32 = index.parse().ok()?;
            document?.pmi.targets.iter().find(|t| t.datum == datum && t.index == index).map(|t| t.on.clone())
        }
        K::Note { leader, .. } => leader.clone(),
        _ => None,
    }
}

/// A cylindrical reference's radius (mm), or none for other surfaces.
pub fn radius_of(display: &Value, reference: &Reference) -> Option<f64> {
    let (_, surface) = face(display, &print_reference(reference)?)?;
    cylinder(surface).map(|(_, _, r)| r)
}
