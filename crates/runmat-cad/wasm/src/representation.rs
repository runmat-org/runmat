//! A part's analysis representations, built from its regeneration on demand. Each is the
//! part's document with derived features appended in memory and regenerated, so the design's
//! source is never changed:
//!
//! - **Defeatured**: faces of fillets, holes and chamfers at or below the representation's
//!   thresholds go in one `direct.delete_faces` per body (the kernel extends the neighboring
//!   faces over the gap). Each candidate is a feature's faces on one body: a fillet's blends
//!   (sized by its radius), a chamfer's faces (by its distance), a hole feature's faces, and any
//!   other full concave cylinder (a cut or imported hole, by its diameter). Faces in a region stay
//!   when `keep_regions`. A candidate the kernel can't remove stays and is listed with the reason.
//! - **Mid-surface**: the kernel's mid-surface (`surface.mid_surface`) of the body, from every
//!   pair of opposite parallel planar faces closer than half their overlap's width, with each
//!   pair's wall thickness.
//! - **Beam**: a prismatic body (two parallel end faces of equal area, every other face parallel
//!   to the line between them) as its centerline and its end section's properties. Any other body
//!   is refused, naming the face that breaks the rule.

use crate::runtime::{regenerate_part_in, RegeneratedPart};
use runmat_cad::assembly::model::Representation;
use runmat_cad::display::{BodyDisplay, EdgeDisplay, FaceDisplay, PartDisplay};
use runmat_cad::model::{DirectEdit, Feature, FeatureKind, PartDocument, PartRepresentation, SurfaceFeature};
use runmat_cad::refs::Reference;
use runmat_cad::regen::kernel::TessellationOptions;
use runmat_cad::syntax::edit::edit_part;
use runmat_cad::units::{Dimension, Units};
use runmat_cad::value::{Evaluator, NoDynamicValues, ValueExpr};
use runmat_cad::refs::history::SurfaceKind;
use serde::Serialize;
use serde_json::Value;

/// What a representation built: its geometry to draw and what a person and a study need to
/// know about it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Built {
    pub name: String,
    pub kind: String,
    #[serde(skip)]
    pub display: PartDisplay,
    /// Defeatured: what was removed, one line each (`fillet round, r 1.5 mm`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub removed: Vec<String>,
    /// Defeatured: candidates that stay, with why.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub kept: Vec<String>,
    /// Mid-surface: each face pair's wall thickness, metres.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub thicknesses: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beam: Option<BeamSection>,
}

/// A beam's centerline (metres) and its end section's properties (SI), about the section's
/// centroid in the end face's plane.
#[derive(Clone, Debug, Default, Serialize)]
pub struct BeamSection {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub length: f64,
    pub area: f64,
    /// Principal second moments of area, larger first.
    pub i_major: f64,
    pub i_minor: f64,
    /// The polar second moment (`i_major + i_minor`).
    pub polar: f64,
}

impl Built {
    /// One line for the bottom strip.
    pub fn summary(&self) -> String {
        match self.kind.as_str() {
            "defeatured" => {
                let removed = if self.removed.is_empty() {
                    "nothing at or below the thresholds to remove".to_owned()
                } else {
                    format!("removed {}", self.removed.join("; "))
                };
                let kept = if self.kept.is_empty() {
                    String::new()
                } else {
                    format!("; kept {}", self.kept.join("; "))
                };
                format!("{} (defeatured): {removed}{kept}", self.name)
            }
            "midsurface" => format!(
                "{} (mid-surface): {} wall{}, thickness {}",
                self.name,
                self.thicknesses.len(),
                if self.thicknesses.len() == 1 { "" } else { "s" },
                self.thicknesses.iter().map(|t| mm(*t)).collect::<Vec<_>>().join(", ")
            ),
            _ => match &self.beam {
                Some(b) => format!(
                    "{} (beam): length {}, section area {} mm², I {} / {} mm⁴",
                    self.name,
                    mm(b.length),
                    round(b.area * 1e6),
                    round(b.i_major * 1e12),
                    round(b.i_minor * 1e12)
                ),
                None => self.name.clone(),
            },
        }
    }
}

fn round(x: f64) -> String {
    let s = format!("{:.2}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

fn mm(m: f64) -> String {
    format!("{} mm", round(m * 1000.0))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n == 0.0 {
        a
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}

fn points(face: &FaceDisplay) -> Vec<[f64; 3]> {
    face.positions
        .chunks_exact(3)
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect()
}

/// Two unit vectors spanning the plane normal to `n`.
fn basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let seed = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = unit(cross(n, seed));
    (u, cross(n, u))
}

fn face_name(face: &FaceDisplay) -> String {
    face.label
        .clone()
        .or_else(|| face.reference.clone())
        .unwrap_or_else(|| format!("face {}", face.index))
}

fn reference_of(text: &str) -> Result<Reference, String> {
    let v = crate::picks::lower_text(text).map_err(|e| e.to_string())?;
    serde_json::from_value(v).map_err(|e| format!("`{text}`: {e}"))
}

/// The representation `name` of the regenerated part `full` (from `text` at `path`).
pub fn build(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    full: &RegeneratedPart,
    name: &str,
    tessellation: &TessellationOptions,
) -> Result<Built, String> {
    let rep = full
        .document
        .representations
        .iter()
        .find(|r| r.name == name)
        .ok_or_else(|| format!("the part has no representation `{name}`"))?;
    match rep.kind {
        Representation::Defeatured => defeatured(path, text, configuration, full, rep, tessellation),
        Representation::Midsurface => midsurface(path, text, configuration, full, rep, tessellation),
        Representation::Beam => beam(full, rep),
        other => Err(format!(
            "a part's representation is defeatured, midsurface or beam, not `{}`",
            other.key()
        )),
    }
}

/// The body `rep` names, else the part's only body.
fn body_of<'a>(full: &'a RegeneratedPart, rep: &PartRepresentation) -> Result<&'a BodyDisplay, String> {
    let bodies: Vec<&BodyDisplay> = full.display.bodies.iter().filter(|b| !b.flat_pattern).collect();
    match &rep.body {
        Some(r) => {
            let text = runmat_cad::syntax::refs::print_reference(r).unwrap_or_default();
            bodies
                .into_iter()
                .find(|b| b.reference.as_deref() == Some(text.as_str()))
                .ok_or_else(|| format!("`{text}` isn't a body of the part"))
        }
        None => match bodies.as_slice() {
            [one] => Ok(one),
            [] => Err("the part has no body".into()),
            _ => Err(format!(
                "the part has {} bodies: pick the one the {} is made from",
                bodies.len(),
                rep.kind.label().to_lowercase()
            )),
        },
    }
}

/// The part's text with `features` appended, regenerated.
fn regenerate_with(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    features: Vec<Feature>,
    tessellation: &TessellationOptions,
) -> Result<RegeneratedPart, String> {
    let change = edit_part(text, path, |document| {
        document.features.extend(features.iter().cloned());
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    regenerate_part_in(path, &change.source, configuration, tessellation).map_err(|d| {
        d.first()
            .map(|d| d.message.clone())
            .unwrap_or_else(|| "the representation doesn't regenerate".into())
    })
}

/// The error messages a regeneration reports on the features `ids`.
fn failures(part: &RegeneratedPart, ids: &[String]) -> Vec<String> {
    use runmat_cad::diagnostics::Severity;
    part.result
        .diagnostics
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .filter(|d| ids.iter().any(|id| d.message.contains(id.as_str()) || format!("{:?}", d.site).contains(id.as_str())))
        .map(|d| d.message.clone())
        .collect()
}

fn fresh(document: &PartDocument, stem: &str, taken: &[String]) -> String {
    (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|id| !document.features.iter().any(|f| f.id.0 == *id) && !taken.contains(id))
        .expect("an unused name")
}

/// One removable group: a feature's faces on one body, or one concave cylinder.
struct Candidate {
    body: u32,
    what: String,
    faces: Vec<String>,
    regions: Vec<String>,
}

fn length(ev: &Evaluator<'_>, full: &RegeneratedPart, v: &Option<ValueExpr>) -> Result<Option<f64>, String> {
    v.as_ref()
        .map(|v| ev.eval_as(v, &full.result.values, Dimension::LENGTH).map_err(|e| e.to_string()))
        .transpose()
}

/// Whether `face` is a full concave cylinder (a hole's wall), and its radius.
fn concave_cylinder(face: &FaceDisplay) -> Option<f64> {
    let Some(SurfaceKind::Cylinder { axis_point, axis_direction, radius }) = face.surface else {
        return None;
    };
    let a = unit(axis_direction);
    let pts = points(face);
    let normals: Vec<[f64; 3]> = face
        .normals
        .chunks_exact(3)
        .map(|n| [n[0] as f64, n[1] as f64, n[2] as f64])
        .collect();
    let (u, v) = basis(a);
    let mut angles = Vec::new();
    let mut inward = 0i32;
    for (k, p) in pts.iter().enumerate() {
        let d = sub(*p, axis_point);
        let radial = sub(d, [a[0] * dot(d, a), a[1] * dot(d, a), a[2] * dot(d, a)]);
        if let Some(n) = normals.get(k) {
            inward += if dot(*n, radial) < 0.0 { 1 } else { -1 };
        }
        angles.push(dot(radial, v).atan2(dot(radial, u)));
    }
    if inward <= 0 || angles.is_empty() {
        return None;
    }
    angles.sort_by(|x, y| x.total_cmp(y));
    let mut gap = angles[0] + std::f64::consts::TAU - angles[angles.len() - 1];
    for w in angles.windows(2) {
        gap = gap.max(w[1] - w[0]);
    }
    (gap < 0.6).then_some(radius)
}

fn defeatured(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    full: &RegeneratedPart,
    rep: &PartRepresentation,
    tessellation: &TessellationOptions,
) -> Result<Built, String> {
    let ev = Evaluator { units: &Units, dynamic: &NoDynamicValues };
    let max_fillet = length(&ev, full, &rep.max_fillet)?;
    let max_hole = length(&ev, full, &rep.max_hole)?;
    let max_chamfer = length(&ev, full, &rep.max_chamfer)?;
    if max_fillet.is_none() && max_hole.is_none() && max_chamfer.is_none() {
        return Err(format!("`{}` sets no threshold: set fillets, holes or chamfers up to a size", rep.name));
    }
    let doc = &full.document;
    let eps = 1e-9;
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut too_big: Vec<String> = Vec::new();
    for body in full.display.bodies.iter().filter(|b| !b.flat_pattern) {
        // Faces by the latest fillet, chamfer or hole feature that made them.
        let mut groups: Vec<(String, Vec<&FaceDisplay>)> = Vec::new();
        let mut loose: Vec<&FaceDisplay> = Vec::new();
        for face in &body.faces {
            // The feature that made the face: the one its name starts with (`feature.round.face…`).
            let maker = face
                .reference
                .as_deref()
                .and_then(|r| r.strip_prefix("feature."))
                .and_then(|r| r.split('.').next())
                .map(str::to_owned)
                .filter(|id| {
                    matches!(
                        doc.features.iter().find(|f| f.id.0 == *id).map(|f| &f.kind),
                        Some(FeatureKind::Fillet(_) | FeatureKind::FaceFillet(_) | FeatureKind::Chamfer(_) | FeatureKind::Hole(_))
                    )
                });
            match maker {
                Some(id) => match groups.iter_mut().find(|(g, _)| *g == id) {
                    Some((_, faces)) => faces.push(face),
                    None => groups.push((id, vec![face])),
                },
                None => loose.push(face),
            }
        }
        for (id, faces) in groups {
            let feature = doc.features.iter().find(|f| f.id.0 == id).expect("found above");
            let cylinder = faces
                .iter()
                .filter_map(|f| match f.surface {
                    Some(SurfaceKind::Cylinder { radius, .. }) => Some(radius),
                    _ => None,
                })
                .fold(None, |m: Option<f64>, r| Some(m.map_or(r, |m| m.max(r))));
            let (what, size, limit) = match &feature.kind {
                FeatureKind::Fillet(f) => {
                    let r = ev.eval_as(&f.radius, &full.result.values, Dimension::LENGTH).ok().or(cylinder);
                    (format!("fillet {id}"), r.map(|r| (format!("r {}", mm(r)), r)), max_fillet)
                }
                FeatureKind::FaceFillet(_) => (format!("fillet {id}"), cylinder.map(|r| (format!("r {}", mm(r)), r)), max_fillet),
                FeatureKind::Chamfer(c) => {
                    let form = serde_json::to_value(&c.form).unwrap_or(Value::Null);
                    let d = form
                        .as_object()
                        .into_iter()
                        .flatten()
                        .filter(|(k, _)| k.starts_with("distance"))
                        .filter_map(|(_, v)| serde_json::from_value::<ValueExpr>(v.clone()).ok())
                        .filter_map(|v| ev.eval_as(&v, &full.result.values, Dimension::LENGTH).ok())
                        .fold(None, |m: Option<f64>, d| Some(m.map_or(d, |m| m.max(d))));
                    (format!("chamfer {id}"), d.map(|d| (mm(d), d)), max_chamfer)
                }
                _ => (format!("hole {id}"), cylinder.map(|r| (format!("⌀ {}", mm(2.0 * r)), 2.0 * r)), max_hole),
            };
            let Some(limit) = limit else { continue };
            let Some((size_text, size)) = size else {
                too_big.push(format!("{what}: its size isn't known"));
                continue;
            };
            if size > limit + eps {
                continue;
            }
            candidates.push(Candidate {
                body: body.index,
                what: format!("{what}, {size_text}"),
                faces: faces.iter().filter_map(|f| f.reference.clone()).collect(),
                regions: faces.iter().flat_map(|f| f.regions.clone()).collect(),
            });
        }
        if let Some(limit) = max_hole {
            for face in loose {
                let Some(r) = concave_cylinder(face) else { continue };
                if 2.0 * r > limit + eps {
                    continue;
                }
                let Some(reference) = face.reference.clone() else {
                    too_big.push(format!("hole {}: the face has no name to remove it by", face_name(face)));
                    continue;
                };
                candidates.push(Candidate {
                    body: body.index,
                    what: format!("hole {}, ⌀ {}", face_name(face), mm(2.0 * r)),
                    faces: vec![reference],
                    regions: face.regions.clone(),
                });
            }
        }
    }
    let mut kept = too_big;
    if rep.keep_regions {
        candidates.retain(|c| {
            if c.regions.is_empty() {
                return true;
            }
            kept.push(format!("{}: it holds region {}", c.what, c.regions[0]));
            false
        });
    }
    let build = |chosen: &[&Candidate]| -> Result<(RegeneratedPart, Vec<String>), String> {
        let mut features = Vec::new();
        let mut ids = Vec::new();
        let mut bodies: Vec<u32> = chosen.iter().map(|c| c.body).collect();
        bodies.sort_unstable();
        bodies.dedup();
        for body in bodies {
            let mut faces = Vec::new();
            for c in chosen.iter().filter(|c| c.body == body) {
                for f in &c.faces {
                    faces.push(reference_of(f)?);
                }
            }
            let id = fresh(doc, "defeature", &ids);
            ids.push(id.clone());
            features.push(Feature::new(&id, FeatureKind::Direct(DirectEdit::DeleteFaces { faces, heal: true })));
        }
        let part = regenerate_with(path, text, configuration, features, tessellation)?;
        Ok((part, ids))
    };
    let all: Vec<&Candidate> = candidates.iter().collect();
    let mut accepted: Vec<&Candidate> = Vec::new();
    let mut part = None;
    if !all.is_empty() {
        let (first, ids) = build(&all)?;
        if failures(&first, &ids).is_empty() {
            accepted = all.clone();
            part = Some(first);
        } else {
            // One at a time: keep each the kernel removes with the ones before it.
            for c in &all {
                let mut trial = accepted.clone();
                trial.push(c);
                let (p, ids) = build(&trial)?;
                let failed = failures(&p, &ids);
                if failed.is_empty() {
                    accepted = trial;
                    part = Some(p);
                } else {
                    kept.push(format!("{}: {}", c.what, failed[0]));
                }
            }
        }
    }
    Ok(Built {
        name: rep.name.clone(),
        kind: "defeatured".into(),
        display: part.map(|p| p.display).unwrap_or_else(|| full.display.clone()),
        removed: accepted.iter().map(|c| c.what.clone()).collect(),
        kept,
        ..Default::default()
    })
}

fn plane_normal(face: &FaceDisplay) -> Option<[f64; 3]> {
    match face.surface {
        Some(SurfaceKind::Plane { normal }) => Some(unit(normal)),
        _ => None,
    }
}

fn midsurface(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    full: &RegeneratedPart,
    rep: &PartRepresentation,
    tessellation: &TessellationOptions,
) -> Result<Built, String> {
    let body = body_of(full, rep)?;
    let body_text = body.reference.clone().ok_or("the body has no name to refer to")?;
    let planes: Vec<(&FaceDisplay, [f64; 3], Vec<[f64; 3]>)> = body
        .faces
        .iter()
        .filter_map(|f| plane_normal(f).map(|n| (f, n, points(f))))
        .filter(|(f, _, p)| !p.is_empty() && f.reference.is_some())
        .collect();
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (i, (_, na, pa)) in planes.iter().enumerate() {
        let (u, v) = basis(*na);
        let rect = |p: &[[f64; 3]]| {
            let us = p.iter().map(|q| dot(*q, u));
            let vs = p.iter().map(|q| dot(*q, v));
            (
                us.clone().fold(f64::MAX, f64::min),
                us.fold(f64::MIN, f64::max),
                vs.clone().fold(f64::MAX, f64::min),
                vs.fold(f64::MIN, f64::max),
            )
        };
        let ra = rect(pa);
        let mut best: Option<(usize, f64)> = None;
        for (j, (_, nb, pb)) in planes.iter().enumerate() {
            if i == j || dot(*na, *nb) > -0.9999 {
                continue;
            }
            // B lies behind A, inside the material under A's outward normal.
            let t = dot(pa[0], *na) - dot(pb[0], *na);
            if t <= 1e-9 {
                continue;
            }
            let rb = rect(pb);
            let w = ra.1.min(rb.1) - ra.0.max(rb.0);
            let h = ra.3.min(rb.3) - ra.2.max(rb.2);
            if w <= 0.0 || h <= 0.0 || t > 0.5 * w.min(h) {
                continue;
            }
            if best.is_none_or(|(_, bt)| t < bt) {
                best = Some((j, t));
            }
        }
        if let Some((j, t)) = best {
            if !pairs.iter().any(|(a, b, _)| (*a == j && *b == i) || (*a == i && *b == j)) {
                pairs.push((i, j, t));
            }
        }
    }
    if pairs.is_empty() {
        return Err(format!(
            "`{body_text}` isn't a thin wall: it has no pair of opposite parallel faces closer than half their width"
        ));
    }
    let mut refs = Vec::new();
    for (a, b, _) in &pairs {
        refs.push((
            reference_of(planes[*a].0.reference.as_deref().unwrap())?,
            reference_of(planes[*b].0.reference.as_deref().unwrap())?,
        ));
    }
    let id = fresh(&full.document, "midsurface", &[]);
    let feature = Feature::new(
        &id,
        FeatureKind::Surface(SurfaceFeature::MidSurface { body: reference_of(&body_text)?, pairs: refs }),
    );
    let part = regenerate_with(path, text, configuration, vec![feature], tessellation)?;
    if let Some(error) = failures(&part, std::slice::from_ref(&id)).first() {
        return Err(format!("the mid-surface of `{body_text}` failed: {error}"));
    }
    let mut display = part.display.clone();
    display.bodies.retain(|b| b.reference.as_deref().is_some_and(|r| r.contains(&id)));
    if display.bodies.is_empty() {
        display.bodies = part
            .display
            .bodies
            .iter()
            .filter(|b| !full.display.bodies.iter().any(|f| f.reference == b.reference))
            .cloned()
            .collect();
    }
    if display.bodies.is_empty() {
        return Err(format!("the mid-surface of `{body_text}` made no surface"));
    }
    Ok(Built {
        name: rep.name.clone(),
        kind: "midsurface".into(),
        display,
        thicknesses: pairs.iter().map(|(_, _, t)| *t).collect(),
        ..Default::default()
    })
}

/// Area, centroid and second moments (about the centroid, in the plane basis `u`, `v`) of a
/// planar face's triangles.
fn section(face: &FaceDisplay, n: [f64; 3]) -> (f64, [f64; 3], f64, f64, f64) {
    let (u, v) = basis(n);
    let p = points(face);
    let (mut area, mut cu, mut cv) = (0.0, 0.0, 0.0);
    let (mut iuu, mut ivv, mut iuv) = (0.0, 0.0, 0.0);
    for t in face.triangles.chunks_exact(3) {
        let q: Vec<(f64, f64)> = t.iter().map(|k| (dot(p[*k as usize], u), dot(p[*k as usize], v))).collect();
        let a = 0.5 * ((q[1].0 - q[0].0) * (q[2].1 - q[0].1) - (q[2].0 - q[0].0) * (q[1].1 - q[0].1));
        let a = a.abs();
        area += a;
        cu += a * (q[0].0 + q[1].0 + q[2].0) / 3.0;
        cv += a * (q[0].1 + q[1].1 + q[2].1) / 3.0;
        let (x, y) = ([q[0].0, q[1].0, q[2].0], [q[0].1, q[1].1, q[2].1]);
        // About the origin: ∫v², ∫u², ∫uv over the triangle.
        ivv += a / 6.0 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[0] * x[1] + x[1] * x[2] + x[2] * x[0]);
        iuu += a / 6.0 * (y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[0] * y[1] + y[1] * y[2] + y[2] * y[0]);
        iuv += a / 12.0
            * (2.0 * (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]) + x[0] * y[1] + x[1] * y[0] + x[1] * y[2] + x[2] * y[1] + x[2] * y[0] + x[0] * y[2]);
    }
    if area == 0.0 {
        return (0.0, [0.0; 3], 0.0, 0.0, 0.0);
    }
    cu /= area;
    cv /= area;
    let iuu = iuu - area * cv * cv;
    let ivv = ivv - area * cu * cu;
    let iuv = iuv - area * cu * cv;
    let off = dot(p[0], n);
    let centroid = [
        u[0] * cu + v[0] * cv + n[0] * off,
        u[1] * cu + v[1] * cv + n[1] * off,
        u[2] * cu + v[2] * cv + n[2] * off,
    ];
    (area, centroid, iuu, ivv, iuv)
}

fn beam(full: &RegeneratedPart, rep: &PartRepresentation) -> Result<Built, String> {
    let body = body_of(full, rep)?;
    let body_text = body.reference.clone().unwrap_or_else(|| format!("body {}", body.index));
    // End-face candidates: pairs of opposite planar faces, longest first.
    let mut ends: Vec<(usize, usize, f64)> = Vec::new();
    for (i, a) in body.faces.iter().enumerate() {
        let Some(na) = plane_normal(a) else { continue };
        for (j, b) in body.faces.iter().enumerate().skip(i + 1) {
            let Some(nb) = plane_normal(b) else { continue };
            if dot(na, nb) > -0.9999 {
                continue;
            }
            let (pa, pb) = (points(a), points(b));
            if pa.is_empty() || pb.is_empty() {
                continue;
            }
            ends.push((i, j, (dot(pa[0], na) - dot(pb[0], na)).abs()));
        }
    }
    ends.sort_by(|x, y| y.2.total_cmp(&x.2));
    let Some(longest) = ends.first().copied() else {
        return Err(format!("`{body_text}` isn't prismatic: it has no two parallel end faces"));
    };
    let mut first_breach: Option<String> = None;
    for (i, j, len) in ends {
        let axis = plane_normal(&body.faces[i]).unwrap();
        let breach = body.faces.iter().enumerate().find(|(k, f)| {
            *k != i
                && *k != j
                && match f.surface {
                    Some(SurfaceKind::Plane { normal }) => dot(unit(normal), axis).abs() > 1e-6,
                    Some(SurfaceKind::Cylinder { axis_direction, .. }) => dot(unit(axis_direction), axis).abs() < 1.0 - 1e-6,
                    _ => true,
                }
        });
        if let Some((_, f)) = breach {
            if (i, j) == (longest.0, longest.1) {
                first_breach = Some(face_name(f));
            }
            continue;
        }
        let (area_a, ca, iuu, ivv, iuv) = section(&body.faces[i], axis);
        let (area_b, cb, ..) = section(&body.faces[j], plane_normal(&body.faces[j]).unwrap());
        if area_a <= 0.0 || (area_a - area_b).abs() > 0.02 * area_a {
            continue;
        }
        let half = 0.5 * (iuu + ivv);
        let r = (0.25 * (iuu - ivv) * (iuu - ivv) + iuv * iuv).sqrt();
        let section = BeamSection {
            start: ca,
            end: cb,
            length: len,
            area: area_a,
            i_major: half + r,
            i_minor: half - r,
            polar: iuu + ivv,
        };
        // Drawn as its centerline and the section at both ends.
        let line = EdgeDisplay {
            index: 0,
            reference: Some(format!("representation.{}", rep.name)),
            faces: [0, 1],
            points: [ca, cb].iter().flat_map(|p| p.iter().map(|x| *x as f32)).collect(),
        };
        let mut caps = vec![body.faces[i].clone(), body.faces[j].clone()];
        for (k, c) in caps.iter_mut().enumerate() {
            c.index = k as u32;
            c.regions.clear();
        }
        let display = PartDisplay {
            bodies: vec![BodyDisplay {
                index: 0,
                reference: Some(format!("representation.{}", rep.name)),
                triangle_count: caps.iter().map(|c| c.triangles.len() as u32 / 3).sum(),
                faces: caps,
                edges: vec![line],
                ..Default::default()
            }],
            bounds: full.display.bounds.clone(),
            appearance: full.display.appearance,
            ..Default::default()
        };
        return Ok(Built {
            name: rep.name.clone(),
            kind: "beam".into(),
            display,
            beam: Some(section),
            ..Default::default()
        });
    }
    Err(match first_breach {
        Some(face) => format!(
            "`{body_text}` isn't prismatic: face `{face}` isn't parallel to the line between its end faces"
        ),
        None => format!("`{body_text}` isn't prismatic: its end faces differ in area"),
    })
}

thread_local! {
    static REQUESTED: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// The representation a compile's `options` ask the part to draw (`{ "representation": "simple" }`);
/// empty or `"full"` is the design itself.
pub fn requested_in(options: &str) -> Option<String> {
    let value: Value = serde_json::from_str(options).ok()?;
    value
        .get("representation")
        .and_then(Value::as_str)
        .filter(|r| !r.is_empty() && *r != "full")
        .map(str::to_owned)
}

/// Runs `f` with `representation` as the one a part compile draws.
pub fn with_requested<T>(representation: Option<String>, f: impl FnOnce() -> T) -> T {
    let before = REQUESTED.with(|r| r.replace(representation));
    let out = f();
    REQUESTED.with(|r| *r.borrow_mut() = before);
    out
}

/// The representation the current compile draws, if any.
pub fn requested() -> Option<String> {
    REQUESTED.with(|r| r.borrow().clone())
}

/// A part compile's output drawing representation `name`: the display is the representation's,
/// and `representation` reports what it is (`{ name, kind, removed, kept, thicknesses, beam,
/// summary }`). When it can't be built, the design draws and `representation` carries `error`.
pub fn apply(path: &str, text: &str, configuration: Option<&str>, name: &str, out: &mut serde_json::Map<String, Value>) {
    let tessellation = TessellationOptions::default();
    let built = regenerate_part_in(path, text, configuration, &tessellation)
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())
        .and_then(|full| build(path, text, configuration, &full, name, &tessellation));
    match built {
        Ok(built) => {
            let mut report = serde_json::to_value(&built).unwrap_or(Value::Null);
            report["summary"] = Value::String(built.summary());
            out.insert("display".into(), serde_json::json!({ "kind": "part", "data": built.display }));
            out.insert("representation".into(), report);
        }
        Err(error) => {
            out.insert(
                "representation".into(),
                serde_json::json!({ "name": name, "error": error }),
            );
        }
    }
}

/// For each part component at an analysis representation (`representation = midsurface`), the
/// display of the first representation of that kind its part declares, in `operated` by node,
/// so the component draws it. Returns, per component that can't draw it, its name and why (it
/// draws the full part).
pub fn assembly_displays(
    resolved: &runmat_cad::assembly::resolve::ResolvedAssembly,
    documents: &std::collections::BTreeMap<String, PartDocument>,
    files: &crate::runtime::Files,
    operated: &mut crate::operated::Operated,
) -> Vec<(String, String)> {
    use runmat_cad::assembly::model::DefinitionKind;
    let mut problems = Vec::new();
    for (id, node) in resolved.nodes.iter().enumerate() {
        if node.definition.kind != DefinitionKind::Part
            || node.represented_by_ancestor
            || !PartRepresentation::KINDS.contains(&node.representation)
            || operated.displays.contains_key(&(id as u32))
        {
            continue;
        }
        let key = node.source_key();
        let (Some(document), Some(text)) = (documents.get(&key), files.get(&key)) else {
            continue;
        };
        let Some(rep) = document.representations.iter().find(|r| r.kind == node.representation) else {
            problems.push((
                node.segment.name.to_string(),
                format!("`{key}` declares no {} representation", node.representation.key()),
            ));
            continue;
        };
        let configuration = node.configuration.name.as_ref().map(|c| c.to_string());
        let tessellation = crate::runtime::ASSEMBLY_TESSELLATION;
        let built = regenerate_part_in(&key, text, configuration.as_deref(), &tessellation)
            .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())
            .and_then(|full| build(&key, text, configuration.as_deref(), &full, &rep.name, &tessellation));
        match built {
            Ok(built) => {
                operated.displays.insert(id as u32, built.display);
                operated
                    .digests
                    .insert(id as u32, format!("representation:{}:{}:{}", key, rep.name, text.len()));
            }
            Err(why) => problems.push((node.segment.name.to_string(), why)),
        }
    }
    problems
}
