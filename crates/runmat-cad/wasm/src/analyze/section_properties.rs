//! `evaluate.section-properties`: the area properties of picked planar faces (one plane, as one
//! region, from the kernel's exact faces) or of a sketch's closed regions (one region pick, or the
//! whole sketch), as an engineer sizes a beam or a plate by them.
//!
//! Values: `section` (picks: faces, a sketch, or a sketch region). Results: one row per
//! property, in mm, mm² and mm⁴, with the plane axes the moments are taken along in the header;
//! the faces picked take the result's color on the model.

use super::{Analysis, FaceColor, FaceColors, LegendEntry, Row};
use runmat_cad::evaluate::section::{face_section, sketch_section, PlaneSection};
use runmat_cad::evaluate::{AnalysisRequest, AnalysisResult};
use runmat_cad::model::{FeatureKind, PartDocument};
use serde_json::{json, Map, Value};

const SECTION_RGBA: [f32; 4] = [0.24, 0.48, 0.86, 1.0];

/// The sketch a pick names (`sketch.s1`, `sketch:s1`, `feature:s1`, a region pick), with the
/// region's loops when it names one.
fn sketch_pick<'a>(
    document: &'a PartDocument,
    pick: &str,
) -> Option<(&'a runmat_cad::model::Feature, Option<Vec<Vec<String>>>)> {
    let (id, loops) = match crate::features::region_pick(pick) {
        Some((sketch, loops)) => (sketch, Some(loops)),
        None => (
            pick.strip_prefix("sketch.")
                .or_else(|| pick.strip_prefix("sketch:"))
                .or_else(|| pick.strip_prefix("feature:"))?
                .to_owned(),
            None,
        ),
    };
    let feature = document.features.iter().find(|f| f.id.0 == id)?;
    matches!(feature.kind, FeatureKind::Sketch(_)).then_some((feature, loops))
}

fn mm(m: f64, power: i32) -> String {
    let v = m * 1000f64.powi(power);
    let unit = match power {
        1 => "mm",
        2 => "mm²",
        _ => "mm⁴",
    };
    if v.abs() >= 1e5 {
        format!("{v:.4e} {unit}")
    } else {
        format!("{v:.3} {unit}")
    }
}

fn point(p: [f64; 3]) -> String {
    format!(
        "({:.3}, {:.3}, {:.3}) mm",
        p[0] * 1000.0,
        p[1] * 1000.0,
        p[2] * 1000.0
    )
}

fn axis(v: [f64; 3]) -> String {
    let r = |c: f64| (c * 1e4).round() / 1e4 + 0.0;
    format!("({}, {}, {})", r(v[0]), r(v[1]), r(v[2]))
}

fn rows(s: &PlaneSection) -> Vec<Row> {
    let row = |name: &str, value: String| Row {
        cells: vec![name.into(), value],
        items: Vec::new(),
        severity: None,
        legend: None,
        reference: None,
    };
    vec![
        row("Area", mm(s.area, 2)),
        row("Centroid", point(s.centroid)),
        row("Ixx (about x through the centroid)", mm(s.ixx, 4)),
        row("Iyy (about y through the centroid)", mm(s.iyy, 4)),
        row("Ixy", mm(s.ixy, 4)),
        row("Polar moment J", mm(s.polar, 4)),
        row("Principal moment I1", mm(s.i1, 4)),
        row("Principal moment I2", mm(s.i2, 4)),
        row(
            "Angle from x to I1's axis",
            format!("{:.2}°", s.angle.to_degrees()),
        ),
    ]
}

pub(super) fn section_properties(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let picks = crate::features::picks_of(values, "section");
    if picks.is_empty() {
        return Err("pick planar faces, a sketch or a sketch region".into());
    }
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let (section, what, colored) = if let Some((feature, loops)) =
        sketch_pick(&part.document, &picks[0])
    {
        if picks.len() > 1 {
            return Err("pick one sketch, or planar faces".into());
        }
        let FeatureKind::Sketch(sketch) = &feature.kind else {
            unreachable!("sketch_pick returns sketches")
        };
        let (solved, frame) = part
            .result
            .state
            .sketches
            .get(&feature.id)
            .ok_or_else(|| format!("sketch `{}` didn't solve", feature.id.0))?;
        let section = match &loops {
            Some(loops) => {
                runmat_cad::evaluate::section::region_section(sketch, solved, frame, loops)
            }
            None => sketch_section(sketch, solved, frame),
        }
        .ok_or_else(|| format!("sketch `{}` has no closed region", feature.id.0))?;
        let what = match loops {
            Some(_) => format!("Region of sketch {}", feature.id.0),
            None => format!("Sketch {}", feature.id.0),
        };
        (section, what, Vec::new())
    } else {
        // Faces: all on one body.
        let mut body_at = None;
        let mut indices = Vec::new();
        let mut labels = Vec::new();
        for pick in &picks {
            let (b, f) = part
                .display
                .bodies
                .iter()
                .find_map(|b| {
                    b.faces
                        .iter()
                        .find(|f| {
                            f.reference.as_deref() == Some(pick.as_str()) || f.names.contains(pick)
                        })
                        .map(|f| (b, f))
                })
                .ok_or_else(|| format!("`{pick}` isn't a face or a sketch of the part"))?;
            if body_at.is_some_and(|at| at != b.index) {
                return Err("pick faces of one body".into());
            }
            body_at = Some(b.index);
            indices.push(f.index);
            labels.push(crate::face_labels::label(b.index, f));
        }
        let state = body_at
            .and_then(|i| part.result.state.bodies.get(i as usize))
            .ok_or("the face's body didn't regenerate")?;
        let face = match part.kernel.analyze(
            &AnalysisRequest::FaceArea { faces: indices },
            &[&state.handle],
        ) {
            Ok(AnalysisResult::FaceArea(face)) => face,
            Ok(other) => return Err(format!("the kernel answered {other:?}")),
            Err(e) => return Err(format!("{e:?}")),
        };
        let what = if labels.len() == 1 {
            labels.remove(0)
        } else {
            format!("{} faces", labels.len())
        };
        (face_section(&face), what, picks.clone())
    };
    Ok(Analysis {
        title: "Section properties".into(),
        summary: format!(
            "{what} · area {} · J {}",
            mm(section.area, 2),
            mm(section.polar, 4)
        ),
        columns: ["Property", "Value"].map(String::from).to_vec(),
        rows: rows(&section),
        problems: Vec::new(),
        data: json!({ "section": section, "of": what }),
        colors: (!colored.is_empty()).then(|| FaceColors {
            legend: vec![LegendEntry {
                label: "Section".into(),
                color: SECTION_RGBA,
            }],
            faces: colored
                .iter()
                .map(|item| FaceColor {
                    item: item.clone(),
                    entry: 0,
                })
                .collect(),
        }),
        settings: vec![
            format!("Section: {what}"),
            format!("x axis {}", axis(section.x)),
            format!("y axis {}", axis(section.y)),
        ],
        arrow: None,
        dimensions: Vec::new(),
        field: None,
    })
}
