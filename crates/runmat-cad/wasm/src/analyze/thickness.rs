//! `evaluate.thickness`: the wall thickness over every face, measured from each point of the
//! face's mesh into the body along the inward normal to the next face, on the exact geometry.
//! The model is colored by it (thin red to thick blue) and each face is a row with its thinnest
//! and thickest point.
//!
//! Values: `target` (mm, the thickness the part is designed to), `show` (`thin`: faces with a
//! point thinner than the target are flagged; `thick`: faces with a point thicker are).

use super::field::{
    field_face, field_meshes, measure, model_size, value_range, SurfaceField, RAMP_LOW_RED,
};
use super::{Analysis, Row};
use runmat_cad::evaluate::FieldQuantity;
use serde_json::{json, Map, Value};

/// Thickness is drawn with triangles no longer than this share of the model's size.
const EDGE_SHARE: f64 = 1.0 / 60.0;

pub(super) fn mm(m: f64) -> String {
    format!("{:.3} mm", m * 1000.0)
}

pub(super) fn thickness(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let target = values.get("target").and_then(Value::as_f64).unwrap_or(2.0) / 1000.0;
    let thick = values.get("show").and_then(Value::as_str) == Some("thick");
    let bodies: Vec<_> = part
        .display
        .bodies
        .iter()
        .filter(|b| !b.flat_pattern)
        .collect();
    if bodies.is_empty() {
        return Err("the part has no bodies to measure".into());
    }
    let meshes = field_meshes(&bodies, model_size(&part.display) * EDGE_SHARE, None);
    let points = measure(&part, &meshes, FieldQuantity::Thickness)?;
    let mut faces = Vec::new();
    let mut rows = Vec::new();
    let mut flagged = 0usize;
    let mut found: Vec<Value> = Vec::new();
    for (mesh, points) in meshes.iter().zip(&points) {
        let measured: Vec<f64> = points.iter().filter_map(|p| p.a).collect();
        let Some(face) = field_face(mesh, points, |p| p.a.map(|t| t * 1000.0)) else {
            continue;
        };
        faces.push(face);
        let label = crate::face_labels::label(mesh.body.index, mesh.face);
        let (lo, hi) = measured
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), t| {
                (lo.min(*t), hi.max(*t))
            });
        let (verdict, severity) = if measured.is_empty() {
            ("Not measured".to_string(), Some("info"))
        } else if !thick && lo < target {
            flagged += 1;
            (format!("Thinner than {}", mm(target)), Some("warning"))
        } else if thick && hi > target {
            flagged += 1;
            (format!("Thicker than {}", mm(target)), Some("warning"))
        } else {
            ("Within target".to_string(), None)
        };
        found.push(json!({
            "face": mesh.face.reference,
            "min": (!measured.is_empty()).then_some(lo),
            "max": (!measured.is_empty()).then_some(hi),
        }));
        rows.push(Row {
            cells: vec![
                label,
                if measured.is_empty() {
                    "—".into()
                } else {
                    mm(lo)
                },
                if measured.is_empty() {
                    "—".into()
                } else {
                    mm(hi)
                },
                verdict,
            ],
            items: mesh.face.reference.iter().cloned().collect(),
            severity,
            legend: None,
            reference: mesh.face.reference.clone(),
        });
    }
    let range = value_range(&faces).ok_or("no point of the part could be measured")?;
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let summary = format!(
        "{} faces · thinnest {} · thickest {} · {flagged} face{} {} than {}",
        rows.len(),
        mm(range[0] / 1000.0),
        mm(range[1] / 1000.0),
        plural(flagged),
        if thick { "thicker" } else { "thinner" },
        mm(target)
    );
    Ok(Analysis {
        title: "Thickness".into(),
        summary,
        columns: ["Face", "Thinnest", "Thickest", "Against target"]
            .map(String::from)
            .to_vec(),
        rows,
        problems: Vec::new(),
        data: json!({ "target": target, "show": if thick { "thick" } else { "thin" }, "faces": found }),
        colors: None,
        settings: vec![
            format!("Target: {}", mm(target)),
            format!("Show: {} regions", if thick { "thick" } else { "thin" }),
        ],
        arrow: None,
        dimensions: Vec::new(),
        field: Some(SurfaceField {
            kind: "spectrum",
            label: "Thickness".into(),
            unit: "mm".into(),
            range,
            ramp: RAMP_LOW_RED.to_vec(),
            faces,
        }),
    })
}
