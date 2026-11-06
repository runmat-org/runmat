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

fn mass_analysis(report: &MassReport, assembly: bool) -> Analysis {
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
            items: if assembly {
                vec![format!("occurrence:{}", p.path)]
            } else {
                Vec::new()
            },
            severity: p.mass.is_none().then_some("warning"),
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
            Ok(mass_analysis(&report, assembly))
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
                    items: vec![format!("occurrence:{}", o.a), format!("occurrence:{}", o.b)],
                    severity: Some("error"),
                })
                .collect();
            rows.extend(found.near.iter().map(|n| Row {
                cells: vec![
                    n.a.clone(),
                    n.b.clone(),
                    format!("gap {} mm", mm(n.distance)),
                ],
                items: vec![format!("occurrence:{}", n.a), format!("occurrence:{}", n.b)],
                severity: Some("warning"),
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
            })
        }
        "simulate.check" => {
            // values.gap: the open-contact gap in mm.
            let gap = request.values.get("gap").and_then(Value::as_f64).map(|mm| mm / 1000.0);
            let found = crate::study_checks::study_checks(path, text, files, configuration, gap)?;
            let errors = found.checks.iter().filter(|c| c.severity == "error").count();
            let warnings = found.checks.len() - errors;
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
                })
                .collect();
            Ok(Analysis {
                title: "Study checks".into(),
                summary,
                columns: ["Check", "Finding"].map(String::from).to_vec(),
                rows,
                problems: Vec::new(),
                data: serde_json::to_value(&found).unwrap_or(Value::Null),
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
                    profile: MemberProfile::RectTube { width: 0.02, height: 0.02, wall: 0.002 },
                    material: Some("material.steel".into()),
                    corner: CornerTreatment::Miter,
                    rotation: ValueExpr::deg(0.0),
                }],
            }),
        ));
        runmat_cad::syntax::print_part(&d, &[]).unwrap()
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
