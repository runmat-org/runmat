//! The Evaluate tab's commands on parts and assemblies, as results the Desktop lists under the
//! canvas: `analyze(path, text, files, request)` → `{ ok: true, title, summary, columns, rows,
//! problems }` or `{ ok: false, message }`.
//!
//! - `evaluate.mass-properties`: mass, volume, centre of gravity and inertia about it, from
//!   each part's material or `density` property ([`crate::mass`]); for an assembly, one row
//!   per part occurrence.
//! - `evaluate.interference`, `evaluate.clearance` (`values.minimum`, mm): part occurrences that
//!   share material, and those closer than the clearance, on their exact bodies
//!   ([`crate::evaluate::assembly_clearance`]).
//!
//! - `weldment.cut-list`: a weldment part's members grouped by profile, length and end angles.
//! - `evaluate.draft`: each face's draft against a pull direction (`values.direction`, a pick;
//!   `values.angle`, degrees), banded ([`crate::draft_analysis`]).
//! - `evaluate.undercut`: faces the part blocks along every way they can release from a
//!   two-part mould along the pull (`values.direction`), by rays along it ([`crate::undercut`]).
//! - `simulate.check`: the fit and mass checks a study runs before it meshes
//!   ([`crate::study_checks`]; `values.gap`, mm, for open contacts).
//!
//! The assembly Evaluate tab's `assembly.mass-properties`, `assembly.interference` and
//! `assembly.clearance` run the same checks.
//!
//! `values.configuration` names the active configuration. Rows name occurrences by path in
//! `items`, so selecting a row selects them.

use crate::mass::{assembly_mass, part_mass, MassReport};
use crate::runtime::Files;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

#[derive(Clone, Debug, Default, Deserialize)]
pub struct AnalyzeRequest {
    pub command: String,
    #[serde(default)]
    pub values: Map<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Row {
    pub cells: Vec<String>,
    /// Outline IDs of what the row is about (`occurrence:<path>`).
    pub items: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<&'static str>,
    /// The row's entry in the result's color legend: the row's marker takes that color, as the
    /// model does (draft bands). Rows without one show their severity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legend: Option<usize>,
    /// The canonical reference the row is about, when it names one entity (a face): shown on
    /// hover and copied from the row, while the first cell reads as a label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Analysis {
    pub title: String,
    pub summary: String,
    pub columns: Vec<String>,
    pub rows: Vec<Row>,
    pub problems: Vec<String>,
    /// The typed result (the mass report, the interference list), for agents and exports.
    pub data: Value,
    /// Colors the model takes while the results are open (draft bands), with their legend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub colors: Option<FaceColors>,
    /// The settings the result was computed with (`Pull: Top plane (+Z)`, `Draft angle: 2°`),
    /// shown in the results header.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub settings: Vec<String>,
    /// A direction to draw on the model while the results are open (the pull).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arrow: Option<ModelArrow>,
}

/// An arrow on the model: from `origin` along the unit `direction`, `length` long (model
/// units, metres), with a short label at its tip.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelArrow {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
    pub length: f64,
    pub label: String,
}

/// An analysis's colors on the model: a legend, and each face's entry in it. Faces are named as
/// the rows name them (canonical references).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FaceColors {
    pub legend: Vec<LegendEntry>,
    pub faces: Vec<FaceColor>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LegendEntry {
    pub label: String,
    /// Linear RGBA, 0 to 1.
    pub color: [f32; 4],
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FaceColor {
    pub item: String,
    /// The index of the face's legend entry.
    pub entry: usize,
}

/// Draft bands on the model, in legend order: positive green, negative blue, needs draft red,
/// straddles amber.
const DRAFT_BANDS: [(crate::draft_analysis::DraftBand, &str, [f32; 4]); 4] = {
    use crate::draft_analysis::DraftBand;
    [
        (DraftBand::Positive, "Positive", [0.22, 0.66, 0.36, 1.0]),
        (DraftBand::Negative, "Negative", [0.24, 0.48, 0.86, 1.0]),
        (DraftBand::NeedsDraft, "Needs draft", [0.86, 0.24, 0.22, 1.0]),
        (DraftBand::Straddle, "Straddles", [0.93, 0.66, 0.16, 1.0]),
    ]
};

/// A band's entry in [`DRAFT_BANDS`].
fn draft_entry(band: crate::draft_analysis::DraftBand) -> usize {
    DRAFT_BANDS.iter().position(|(b, ..)| *b == band).unwrap_or(0)
}

fn draft_colors(faces: &[crate::draft_analysis::FaceDraft]) -> FaceColors {
    FaceColors {
        legend: DRAFT_BANDS.iter().map(|(_, label, color)| LegendEntry { label: (*label).into(), color: *color }).collect(),
        faces: faces.iter().map(|f| FaceColor { item: f.face.clone(), entry: draft_entry(f.band) }).collect(),
    }
}

/// The pull drawn on the model: from the top of the part's bounds along the pull, a third of
/// its size long.
fn pull_arrow(display: &runmat_cad::display::PartDisplay, pull: [f64; 3]) -> Option<ModelArrow> {
    display.bounds.as_ref().map(|b| {
        let lo = b.min.map(f64::from);
        let hi = b.max.map(f64::from);
        let centre = [0, 1, 2].map(|i| (lo[i] + hi[i]) / 2.0);
        let size = (0..3).map(|i| hi[i] - lo[i]).fold(0.0, f64::max);
        let reach = (0..3).map(|i| (hi[i] - lo[i]) / 2.0 * pull[i].abs()).sum::<f64>();
        ModelArrow { origin: [0, 1, 2].map(|i| centre[i] + pull[i] * reach), direction: pull, length: size / 3.0, label: "Pull".into() }
    })
}

/// The pull pick as the results header names it.
fn pull_label(pick: &str, pull: [f64; 3], label: Option<String>) -> String {
    let axis = match pull.map(|c| (c * 1e6).round() / 1e6) {
        [x, y, z] if (x.abs() - 1.0).abs() < 1e-6 && y == 0.0 && z == 0.0 => Some(if x > 0.0 { "+X" } else { "−X" }),
        [x, y, z] if (y.abs() - 1.0).abs() < 1e-6 && x == 0.0 && z == 0.0 => Some(if y > 0.0 { "+Y" } else { "−Y" }),
        [x, y, z] if (z.abs() - 1.0).abs() < 1e-6 && x == 0.0 && y == 0.0 => Some(if z > 0.0 { "+Z" } else { "−Z" }),
        _ => None,
    };
    let what = match pick {
        "origin:top" | "origin.xy" => "Top plane".to_string(),
        "origin:front" | "origin.zx" => "Front plane".to_string(),
        "origin:right" | "origin.yz" => "Right plane".to_string(),
        "origin:x" | "origin.x" => "X axis".to_string(),
        "origin:y" | "origin.y" => "Y axis".to_string(),
        "origin:z" | "origin.z" => "Z axis".to_string(),
        _ => label.unwrap_or_else(|| "a face".into()),
    };
    match axis {
        Some(axis) => format!("{what} ({axis})"),
        None => format!("{what} ({:.2}, {:.2}, {:.2})", pull[0], pull[1], pull[2]),
    }
}

fn g(kg: f64) -> String {
    if kg >= 1.0 {
        format!("{kg:.3} kg")
    } else {
        format!("{:.1} g", kg * 1000.0)
    }
}

fn mm(m: f64) -> String {
    format!("{:.2}", m * 1000.0)
}

fn mass_analysis(report: &MassReport) -> Analysis {
    let t = &report.total;
    let c = t.center;
    // kg·mm², the unit drawings and data sheets quote.
    let i = |r: usize, k: usize| format!("{:.1}", t.inertia[r][k] * 1e6);
    let mut summary = format!(
        "{} · {:.1} cm³ · CG ({}, {}, {}) mm · Ixx {} Iyy {} Izz {} kg·mm² about the CG",
        g(t.mass),
        t.volume * 1e6,
        mm(c[0]),
        mm(c[1]),
        mm(c[2]),
        i(0, 0),
        i(1, 1),
        i(2, 2)
    );
    if !report.complete() {
        summary += &format!(" · incomplete: {} without mass", report.problems.len());
    }
    let rows = report
        .parts
        .iter()
        .map(|p| Row {
            cells: vec![
                p.path.clone(),
                p.configuration.clone(),
                p.material
                    .as_deref()
                    .map(runmat_cad::drawing::bom_table::material_name)
                    .unwrap_or_else(|| "—".into()),
                p.density
                    .map(|d| format!("{d:.0} kg/m³"))
                    .unwrap_or_else(|| "—".into()),
                format!("{:.2} cm³", p.volume * 1e6),
                p.mass.map(g).unwrap_or_else(|| "—".into()),
            ],
            items: p.item.iter().cloned().collect(),
            severity: p.mass.is_none().then_some("warning"),
            legend: None,
            reference: None,
        })
        .collect();
    Analysis {
        title: "Mass properties".into(),
        summary,
        columns: [
            "Part",
            "Configuration",
            "Material",
            "Density",
            "Volume",
            "Mass",
        ]
        .map(String::from)
        .to_vec(),
        rows,
        problems: report.problems.clone(),
        data: serde_json::to_value(report).unwrap_or(Value::Null),
        colors: None,
        settings: Vec::new(),
        arrow: None,
    }
}

/// Runs an Evaluate command on the part or assembly `path`.
pub fn analyze(
    path: &str,
    text: &str,
    files: &Files,
    request: &AnalyzeRequest,
) -> Result<Analysis, String> {
    let configuration = request
        .values
        .get("configuration")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty() && *c != "default");
    let assembly = path.ends_with(".assembly");
    match request.command.as_str() {
        "evaluate.mass-properties" | "assembly.mass-properties" => {
            let report = if assembly {
                assembly_mass(path, text, files, configuration)?
            } else {
                part_mass(path, text, configuration)?
            };
            Ok(mass_analysis(&report))
        }
        "evaluate.interference" | "evaluate.clearance" | "assembly.interference" | "assembly.clearance" if assembly => {
            let clearance = if request.command.ends_with(".clearance") {
                // The catalog field is in millimetres.
                request
                    .values
                    .get("minimum")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0)
                    / 1000.0
            } else {
                0.0
            };
            let found =
                crate::evaluate::assembly_clearance(path, text, files, configuration, clearance)?;
            let mut rows: Vec<Row> = found
                .overlaps
                .iter()
                .map(|o| Row {
                    cells: vec![
                        o.a.clone(),
                        o.b.clone(),
                        format!("overlap {:.3} cm³", o.volume * 1e6),
                    ],
                    items: o.items.to_vec(),
                    severity: Some("error"),
                    legend: None,
                    reference: None,
                })
                .collect();
            rows.extend(found.near.iter().map(|n| Row {
                cells: vec![
                    n.a.clone(),
                    n.b.clone(),
                    format!("gap {} mm", mm(n.distance)),
                ],
                items: n.items.to_vec(),
                severity: Some("warning"),
                legend: None,
                reference: None,
            }));
            let plural = |n: usize| if n == 1 { "" } else { "s" };
            let mut summary = format!(
                "{} part{} checked · {} overlapping pair{}",
                found.parts,
                plural(found.parts),
                found.overlaps.len(),
                plural(found.overlaps.len())
            );
            if clearance > 0.0 {
                summary += &format!(
                    " · {} pair{} closer than {} mm",
                    found.near.len(),
                    plural(found.near.len()),
                    mm(clearance)
                );
            }
            Ok(Analysis {
                title: if clearance > 0.0 {
                    "Clearance"
                } else {
                    "Interference"
                }
                .into(),
                summary,
                columns: ["Part", "With", "Finding"].map(String::from).to_vec(),
                rows,
                problems: found.errors.clone(),
                data: serde_json::to_value(&found).unwrap_or(Value::Null),
                colors: None,
                settings: Vec::new(),
                arrow: None,
            })
        }
        "evaluate.interference" | "evaluate.clearance" | "assembly.interference" | "assembly.clearance" => {
            Err("interference and clearance are checked between an assembly's parts".into())
        }
        "weldment.cut-list" if !assembly => {
            let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
                .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
            let mut rows = Vec::new();
            let mut members = 0;
            for (feature, record) in &part.result.state.records {
                let runmat_cad::regen::records::FeatureRecord::Weldment { cut_list, .. } = record else {
                    continue;
                };
                for r in cut_list {
                    members += r.quantity;
                    rows.push(Row {
                        cells: vec![
                            r.item.to_string(),
                            r.quantity.to_string(),
                            r.description.clone(),
                            r.material
                                .as_deref()
                                .map(runmat_cad::drawing::bom_table::material_name)
                                .unwrap_or_else(|| "—".into()),
                            format!("{} mm", mm(r.length)),
                            format!("{:.1}°", r.start_angle.to_degrees()),
                            format!("{:.1}°", r.end_angle.to_degrees()),
                            r.mass_per_member.map(g).unwrap_or_else(|| "—".into()),
                        ],
                        items: vec![format!("feature:{feature}")],
                        severity: None,
                        legend: None,
                        reference: None,
                    });
                }
            }
            if rows.is_empty() {
                return Err("the part has no structural members".into());
            }
            Ok(Analysis {
                title: "Cut list".into(),
                summary: format!("{} item{} · {members} members", rows.len(), if rows.len() == 1 { "" } else { "s" }),
                columns: ["Item", "Qty", "Description", "Material", "Length", "Start angle", "End angle", "Mass each"]
                    .map(String::from)
                    .to_vec(),
                rows,
                problems: Vec::new(),
                data: Value::Null,
                colors: None,
                settings: Vec::new(),
                arrow: None,
            })
        }
        "evaluate.draft" if !assembly => {
            use crate::draft_analysis::{draft_faces, pull_direction, DraftBand};
            let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
                .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
            let pick = crate::features::picks_of(&request.values, "direction")
                .into_iter()
                .next()
                .ok_or("pick the pull direction: an origin plane or axis, or a face")?;
            let pull = pull_direction(&part.display, &pick)?;
            let angle = request.values.get("angle").and_then(Value::as_f64).unwrap_or(1.0);
            let label = crate::face_labels::labeller(&part);
            let faces = draft_faces(&part.display, pull, angle, &label);
            let picked = part.display.bodies.iter().find_map(|b| b.faces.iter().find(|f| f.reference.as_deref() == Some(pick.as_str())).map(|f| label(b.index, f)));
            let pull_text = pull_label(&pick, pull, picked);
            let arrow = pull_arrow(&part.display, pull);
            let count = |b: DraftBand| faces.iter().filter(|f| f.band == b).count();
            let (needs, straddle) = (count(DraftBand::NeedsDraft), count(DraftBand::Straddle));
            let summary = format!(
                "{} faces · {} positive · {} negative · {needs} need draft · {straddle} straddle the parting line",
                faces.len(),
                count(DraftBand::Positive),
                count(DraftBand::Negative),
            );
            let rows = faces
                .iter()
                .map(|f| Row {
                    cells: vec![
                        f.label.clone(),
                        format!("{:.1}°", f.min),
                        format!("{:.1}°", f.max),
                        match f.band {
                            DraftBand::Positive => "Positive",
                            DraftBand::Negative => "Negative",
                            DraftBand::NeedsDraft => "Needs draft",
                            DraftBand::Straddle => "Straddles",
                        }
                        .into(),
                    ],
                    items: vec![f.face.clone()],
                    severity: match f.band {
                        DraftBand::NeedsDraft => Some("warning"),
                        DraftBand::Straddle => Some("info"),
                        _ => None,
                    },
                    legend: Some(draft_entry(f.band)),
                    reference: Some(f.face.clone()),
                })
                .collect();
            Ok(Analysis {
                title: "Draft analysis".into(),
                summary,
                columns: ["Face", "Least draft", "Greatest draft", "Band"].map(String::from).to_vec(),
                rows,
                problems: Vec::new(),
                data: serde_json::json!({ "pull": pull, "angle": angle, "faces": faces }),
                colors: Some(draft_colors(&faces)),
                settings: vec![format!("Pull: {pull_text}"), format!("Draft angle: {angle}°")],
                arrow,
            })
        }
        "evaluate.undercut" if !assembly => {
            use crate::draft_analysis::pull_direction;
            use crate::undercut::{undercut_faces, UndercutBand};
            let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
                .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
            let pick = crate::features::picks_of(&request.values, "direction")
                .into_iter()
                .next()
                .ok_or("pick the pull direction: an origin plane or axis, or a face")?;
            let pull = pull_direction(&part.display, &pick)?;
            let label = crate::face_labels::labeller(&part);
            let faces = undercut_faces(&part.display, pull, &label);
            let picked = part.display.bodies.iter().find_map(|b| b.faces.iter().find(|f| f.reference.as_deref() == Some(pick.as_str())).map(|f| label(b.index, f)));
            let bands = [
                (UndercutBand::Cavity, "Releases with the cavity", [0.22, 0.66, 0.36, 1.0]),
                (UndercutBand::Core, "Releases with the core", [0.24, 0.48, 0.86, 1.0]),
                (UndercutBand::Undercut, "Undercut", [0.86, 0.24, 0.22, 1.0]),
            ];
            let entry = |b: UndercutBand| bands.iter().position(|(x, ..)| *x == b).unwrap_or(0);
            let count = |b: UndercutBand| faces.iter().filter(|f| f.band == b).count();
            let undercut = count(UndercutBand::Undercut);
            let summary = format!(
                "{} faces · {} release with the cavity · {} with the core · {undercut} undercut{}",
                faces.len(),
                count(UndercutBand::Cavity),
                count(UndercutBand::Core),
                if undercut > 0 { " (a side action, a lifter or a part change)" } else { "" }
            );
            let rows = faces
                .iter()
                .map(|f| Row {
                    cells: vec![f.label.clone(), format!("{:.0} %", f.share * 100.0), bands[entry(f.band)].1.into()],
                    items: vec![f.face.clone()],
                    severity: (f.band == UndercutBand::Undercut).then_some("warning"),
                    legend: Some(entry(f.band)),
                    reference: Some(f.face.clone()),
                })
                .collect();
            Ok(Analysis {
                title: "Undercut".into(),
                summary,
                columns: ["Face", "Undercut area", "Release"].map(String::from).to_vec(),
                rows,
                problems: Vec::new(),
                data: serde_json::json!({ "pull": pull, "faces": faces }),
                colors: Some(FaceColors {
                    legend: bands.iter().map(|(_, l, c)| LegendEntry { label: (*l).into(), color: *c }).collect(),
                    faces: faces.iter().map(|f| FaceColor { item: f.face.clone(), entry: entry(f.band) }).collect(),
                }),
                settings: vec![format!("Pull: {}", pull_label(&pick, pull, picked))],
                arrow: pull_arrow(&part.display, pull),
            })
        }
        "simulate.check" => {
            // values.gap: the open-contact gap in mm.
            let gap = request.values.get("gap").and_then(Value::as_f64).map(|mm| mm / 1000.0);
            let found = crate::study_checks::study_checks(path, text, files, configuration, gap)?;
            let errors = found.checks.iter().filter(|c| c.severity == "error").count();
            let warnings = found.checks.iter().filter(|c| c.severity == "warning").count();
            let plural = |n: usize| if n == 1 { "" } else { "s" };
            let mut summary = format!("{} · {} part{}", g(found.mass.total.mass), found.mass.parts.len(), plural(found.mass.parts.len()));
            summary += &if found.ready() {
                format!(" · ready to mesh · {warnings} warning{}", plural(warnings))
            } else {
                format!(" · {errors} error{} to fix before the study meshes · {warnings} warning{}", plural(errors), plural(warnings))
            };
            let rows = found
                .checks
                .iter()
                .map(|c| Row {
                    cells: vec![if c.check == "fit" { "Fit" } else { "Mass" }.into(), c.message.clone()],
                    items: c.items.clone(),
                    severity: Some(c.severity),
                    legend: None,
                    reference: None,
                })
                .collect();
            Ok(Analysis {
                title: "Study checks".into(),
                summary,
                columns: ["Check", "Finding"].map(String::from).to_vec(),
                rows,
                problems: Vec::new(),
                data: serde_json::to_value(&found).unwrap_or(Value::Null),
                colors: None,
                settings: Vec::new(),
                arrow: None,
            })
        }
        other => Err(format!(
            "`{other}` runs from the agent or the CLI in this build"
        )),
    }
}

/// `analyze` as JSON.
pub fn analyze_json(path: &str, text: &str, files: &Files, request: &str) -> String {
    let request: AnalyzeRequest = match serde_json::from_str(request) {
        Ok(r) => r,
        Err(e) => {
            return json!({ "ok": false, "message": format!("the request isn't valid: {e}") })
                .to_string()
        }
    };
    match analyze(path, text, files, &request) {
        Ok(a) => {
            let mut v = serde_json::to_value(&a).unwrap_or(Value::Null);
            v["ok"] = Value::Bool(true);
            v.to_string()
        }
        Err(message) => json!({ "ok": false, "message": message }).to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn mass_properties_of_a_part_list_its_material_and_mass() {
        let text = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 10 mm])\n        l3 = line(start = [100 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output block = { body = feature.body.body, material = \"material.aluminum\" }\n}\n";
        let out: Value = serde_json::from_str(&analyze_json(
            "/p/block.part",
            text,
            &Files::new(),
            &json!({ "command": "evaluate.mass-properties" }).to_string(),
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{out}");
        assert!(
            out["summary"].as_str().unwrap().starts_with("27.0 g"),
            "{out}"
        );
        assert_eq!(out["rows"][0]["cells"][2], "Aluminum");
        let refused: Value = serde_json::from_str(&analyze_json(
            "/p/block.part",
            text,
            &Files::new(),
            &json!({ "command": "evaluate.interference" }).to_string(),
        ))
        .unwrap();
        assert_eq!(refused["ok"], false);
    }

    /// A 200 × 100 mm frame of 20 mm square tube, mitred, written by the printer.
    pub(crate) fn frame() -> String {
        use runmat_cad::ids::{FeatureId, SketchEntityId};
        use runmat_cad::model::weldment::*;
        use runmat_cad::model::*;
        use runmat_cad::value::ValueExpr;
        let mut d = PartDocument::new("/p/frame.part");
        let mut s = Sketch {
            plane: PlaneRef::origin_xy(),
            entities: Vec::new(),
            constraints: Vec::new(),
            solved: Default::default(),
        };
        let p = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)];
        let names = ["bottom", "right", "top", "left"];
        for k in 0..4 {
            s.entities.push(SketchEntity {
                id: SketchEntityId::new(names[k]),
                geometry: SketchGeometry::Line {
                    start: PointExpr::mm(p[k].0, p[k].1),
                    end: PointExpr::mm(p[(k + 1) % 4].0, p[(k + 1) % 4].1),
                },
                construction: false,
            });
        }
        d.features.push(Feature::new("layout", FeatureKind::Sketch(s)));
        d.features.push(Feature::new(
            "members",
            FeatureKind::Weldment(WeldmentFeature::StructuralMember {
                groups: vec![MemberGroup {
                    sketch: FeatureId::new("layout"),
                    segments: names.iter().map(|n| SketchEntityId::new(*n)).collect(),
                    profile: MemberProfile::RectTube { width: ValueExpr::mm(20.0), height: ValueExpr::mm(20.0), wall: ValueExpr::mm(2.0) },
                    material: Some("material.steel".into()),
                    corner: CornerTreatment::Miter,
                    rotation: ValueExpr::deg(0.0),
                }],
            }),
        ));
        runmat_cad::syntax::print_part(&d, &[]).unwrap()
    }

    #[test]
    fn draft_analysis_bands_a_blocks_faces_against_the_pull() {
        // A 100 x 10 x 10 mm block pulled along +Z: the top releases, the bottom from the
        // other half, the four walls need draft.
        let text = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 10 mm])\n        l3 = line(start = [100 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output block = { body = feature.body.body }\n}\n";
        let out: Value = serde_json::from_str(&analyze_json(
            "/p/block.part",
            text,
            &Files::new(),
            &json!({ "command": "evaluate.draft", "values": { "direction": ["origin:top"], "angle": 2 } }).to_string(),
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{out}");
        assert_eq!(out["summary"], "6 faces · 1 positive · 1 negative · 4 need draft · 0 straddle the parting line", "{out}");
        assert_eq!(out["settings"], json!(["Pull: Top plane (+Z)", "Draft angle: 2°"]), "{out}");
        // The pull arrow stands on the block's top, along +Z.
        assert_eq!(out["arrow"]["direction"], json!([0.0, 0.0, 1.0]), "{out}");
        assert!((out["arrow"]["origin"][2].as_f64().unwrap() - 0.01).abs() < 1e-6, "{out}");
        // Faces read as labels; the canonical reference rides along for the tooltip and copy.
        let top = out["rows"].as_array().unwrap().iter().find(|r| r["reference"] == "feature.body.face.end").expect("the top face by name");
        assert_eq!(top["cells"][0], "Body end face");
        assert_eq!(top["cells"][1], "90.0°");
        let wall = out["rows"].as_array().unwrap().iter().find(|r| r["reference"] == "feature.body.face.side(s1.l2)").unwrap();
        assert_eq!(wall["cells"][0], "Body side 2");
        assert_eq!(wall["cells"][3], "Needs draft");
        assert_eq!(wall["severity"], "warning");
        // Each row's marker takes its band's legend entry, as the model does.
        for row in out["rows"].as_array().unwrap() {
            let entry = row["legend"].as_u64().unwrap() as usize;
            assert_eq!(out["colors"]["legend"][entry]["label"], row["cells"][3], "{row}");
        }
        // The model takes the band colors: every face, the top green, with a four-entry legend.
        let colors = &out["colors"];
        assert_eq!(colors["legend"].as_array().map(Vec::len), Some(4), "{colors}");
        assert_eq!(colors["faces"].as_array().map(Vec::len), Some(6), "{colors}");
        let entry = colors["faces"].as_array().unwrap().iter().find(|f| f["item"] == "feature.body.face.end").unwrap()["entry"].as_u64().unwrap();
        assert_eq!(colors["legend"][entry as usize]["label"], "Positive");
        // A face picks the pull too: the top's normal is +Z.
        let by_face: Value = serde_json::from_str(&analyze_json(
            "/p/block.part",
            text,
            &Files::new(),
            &json!({ "command": "evaluate.draft", "values": { "direction": ["feature.body.face.end"], "angle": 2 } }).to_string(),
        ))
        .unwrap();
        assert_eq!(by_face["summary"], out["summary"], "{by_face}");
    }

    #[test]
    fn a_weldment_lists_its_members_by_length_and_end_angle() {
        let text = frame();
        let out: Value = serde_json::from_str(&analyze_json(
            "/p/frame.part",
            &text,
            &Files::new(),
            &json!({ "command": "weldment.cut-list" }).to_string(),
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{out}\n{text}");
        assert_eq!(out["rows"].as_array().unwrap().len(), 2, "{out}");
        assert_eq!(out["summary"], "2 items · 4 members");
        // The long side: centre line plus half the tube's depth at each mitre.
        assert_eq!(out["rows"][0]["cells"][4], "220.00 mm");
        assert_eq!(out["rows"][0]["cells"][5], "45.0°");
    }
}
