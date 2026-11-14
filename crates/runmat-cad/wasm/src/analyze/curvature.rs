//! `evaluate.curvature`: the curvature over every face at each point of the face's mesh, from
//! the exact surface's principal curvatures. Convex is positive (red), concave negative (blue),
//! flat zero; each face is a row with its least and greatest curvature and its smallest radius.
//!
//! Values: `measure`: `largest` (the principal curvature of larger size, signed; 1/mm),
//! `mean` (the average of the two; 1/mm) or `gaussian` (their product; 1/mm²).

use super::field::{field_face, field_meshes, measure, model_size, SurfaceField, RAMP_SIGNED};
use super::{Analysis, Row};
use runmat_cad::evaluate::{FieldPoint, FieldQuantity};
use serde_json::{json, Map, Value};

/// Curvature is drawn with triangles no longer than this share of the model's size, whose
/// corner normals turn by no more than [`TURN`].
const EDGE_SHARE: f64 = 1.0 / 60.0;
const TURN: f64 = 10.0 * std::f64::consts::PI / 180.0;

#[derive(Clone, Copy, PartialEq)]
enum Measure {
    Largest,
    Mean,
    Gaussian,
}

impl Measure {
    fn parse(v: Option<&str>) -> Self {
        match v {
            Some("mean") => Self::Mean,
            Some("gaussian") => Self::Gaussian,
            _ => Self::Largest,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Largest => "Largest curvature",
            Self::Mean => "Mean curvature",
            Self::Gaussian => "Gaussian curvature",
        }
    }

    fn unit(self) -> &'static str {
        match self {
            Self::Gaussian => "1/mm²",
            _ => "1/mm",
        }
    }

    /// The value at a point in the unit (k in 1/m).
    fn value(self, p: &FieldPoint) -> Option<f64> {
        let (k1, k2) = (p.a?, p.b?);
        Some(match self {
            Self::Largest => {
                if k1.abs() > k2.abs() {
                    k1 / 1000.0
                } else {
                    k2 / 1000.0
                }
            }
            Self::Mean => (k1 + k2) / 2.0 / 1000.0,
            Self::Gaussian => k1 * k2 / 1e6,
        })
    }
}

fn shown(v: f64, unit: &str) -> String {
    if v == 0.0 {
        format!("0 {unit}")
    } else if v.abs() >= 0.01 {
        format!("{v:.4} {unit}")
    } else {
        format!("{v:.3e} {unit}")
    }
}

pub(super) fn curvature(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let what = Measure::parse(values.get("measure").and_then(Value::as_str));
    let bodies: Vec<_> = part
        .display
        .bodies
        .iter()
        .filter(|b| !b.flat_pattern)
        .collect();
    if bodies.is_empty() {
        return Err("the part has no bodies to measure".into());
    }
    let meshes = field_meshes(&bodies, model_size(&part.display) * EDGE_SHARE, Some(TURN));
    let points = measure(&part, &meshes, FieldQuantity::Curvature)?;
    let mut faces = Vec::new();
    let mut rows = Vec::new();
    let mut found = Vec::new();
    let mut tightest: Option<(f64, String)> = None;
    for (mesh, points) in meshes.iter().zip(&points) {
        let Some(face) = field_face(mesh, points, |p| what.value(p)) else {
            continue;
        };
        let measured: Vec<f64> = face
            .values
            .iter()
            .flatten()
            .map(|v| f64::from(*v))
            .collect();
        faces.push(face);
        let label = crate::face_labels::label(mesh.body.index, mesh.face);
        // The smallest radius: the inverse of the largest principal curvature's size (m).
        let kmax = points
            .iter()
            .filter_map(|p| Some(p.a?.abs().max(p.b?.abs())))
            .fold(0.0f64, f64::max);
        let radius = (kmax > 1e-9).then(|| 1.0 / kmax);
        if let Some(r) = radius {
            if tightest.as_ref().is_none_or(|(t, _)| r < *t) {
                tightest = Some((r, label.clone()));
            }
        }
        let (lo, hi) = measured
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(*v), hi.max(*v))
            });
        found.push(json!({
            "face": mesh.face.reference,
            "min": lo.is_finite().then_some(lo),
            "max": hi.is_finite().then_some(hi),
            "smallest_radius": radius,
        }));
        rows.push(Row {
            cells: vec![
                label,
                if lo.is_finite() {
                    shown(lo, what.unit())
                } else {
                    "—".into()
                },
                if hi.is_finite() {
                    shown(hi, what.unit())
                } else {
                    "—".into()
                },
                radius
                    .map(|r| format!("{:.3} mm", r * 1000.0))
                    .unwrap_or_else(|| "Flat".into()),
            ],
            items: mesh.face.reference.iter().cloned().collect(),
            severity: None,
            legend: None,
            reference: mesh.face.reference.clone(),
        });
    }
    if faces.is_empty() {
        return Err("no face of the part could be measured".into());
    }
    // Zero sits in the middle of the ramp: flat faces read neutral, convex red, concave blue.
    let reach = faces
        .iter()
        .flat_map(|f| f.values.iter().flatten())
        .fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
    let curved = rows.iter().filter(|r| r.cells[3] != "Flat").count();
    let summary = match &tightest {
        Some((r, at)) => format!(
            "{} faces · {curved} curved · smallest radius {:.3} mm on {at}",
            rows.len(),
            r * 1000.0
        ),
        None => format!("{} faces · all flat", rows.len()),
    };
    Ok(Analysis {
        title: "Curvature".into(),
        summary,
        columns: ["Face", "Least", "Greatest", "Smallest radius"]
            .map(String::from)
            .to_vec(),
        rows,
        problems: Vec::new(),
        data: json!({ "measure": what.label(), "unit": what.unit(), "faces": found }),
        colors: None,
        settings: vec![format!("Show: {} ({})", what.label(), what.unit())],
        arrow: None,
        dimensions: Vec::new(),
        field: Some(SurfaceField {
            kind: "spectrum",
            label: what.label().into(),
            unit: what.unit().into(),
            range: [-reach, reach],
            ramp: RAMP_SIGNED.to_vec(),
            faces,
        }),
    })
}
