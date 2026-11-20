//! `evaluate.check-geometry`: the problems in a part's bodies that make later features, exports
//! and meshes fail, found by the kernel on the exact geometry ([`CheckFinding`]).
//!
//! Values: `bodies` (picks; every body when empty), `strict` (also find self-intersecting
//! faces, slower), `short_edges` with `short_edge` (mm), `gaps` with `gap` (mm), `min_radius`.
//! Each finding is a row naming its body and its face, edge or vertex by canonical reference, so
//! selecting the row selects it; faces with an error take the result's color on the model.

use super::{Analysis, FaceColor, FaceColors, LegendEntry, Row};
use runmat_cad::display::{BodyDisplay, PartDisplay};
use runmat_cad::evaluate::{
    AnalysisRequest, AnalysisResult, CheckEntity, CheckFinding, CheckProblem,
};
use serde_json::{json, Map, Value};

/// Defaults the catalog fields carry: edges under 0.01 mm are short; tolerances over 0.01 mm
/// are gaps.
const SHORT_EDGE_MM: f64 = 0.01;
const GAP_MM: f64 = 0.01;

const ERROR_RGBA: [f32; 4] = [0.86, 0.24, 0.22, 1.0];
const WARNING_RGBA: [f32; 4] = [0.93, 0.66, 0.16, 1.0];

fn on(values: &Map<String, Value>, key: &str, default: bool) -> bool {
    values.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn number(values: &Map<String, Value>, key: &str, default: f64) -> f64 {
    values.get(key).and_then(Value::as_f64).unwrap_or(default)
}

/// How a problem reads in the table, and its severity.
fn problem_text(p: CheckProblem) -> (&'static str, Option<&'static str>) {
    use CheckProblem as P;
    match p {
        P::InvalidBody => ("Invalid shell or solid", Some("error")),
        P::InvalidFace => ("Invalid face", Some("error")),
        P::InvalidEdge => ("Invalid edge", Some("error")),
        P::InvalidVertex => ("Invalid vertex", Some("error")),
        P::SelfIntersectingFace => ("Self-intersecting face", Some("error")),
        P::OpenEdge => ("Open edge (one face)", Some("error")),
        P::ShortEdge => ("Short edge", Some("warning")),
        P::EdgeGap => ("Edge gap", Some("warning")),
        P::VertexGap => ("Vertex gap", Some("warning")),
        P::SmallestRadius => ("Smallest radius", Some("info")),
        P::LargestEdgeGap => ("Largest edge gap", Some("info")),
        P::LargestVertexGap => ("Largest vertex gap", Some("info")),
    }
}

/// Millimetres, with enough places for a tolerance.
fn mm(m: f64) -> String {
    let v = m * 1000.0;
    if v.abs() >= 0.1 {
        format!("{v:.3} mm")
    } else {
        format!("{v:.5} mm")
    }
}

/// The entity's canonical reference and its label for people.
fn entity_names(body: &BodyDisplay, entity: CheckEntity) -> (Option<String>, String) {
    match entity {
        CheckEntity::Body => (body.reference.clone(), "Body".into()),
        CheckEntity::Face(i) => body
            .faces
            .iter()
            .find(|f| f.index == i)
            .map(|f| {
                (
                    f.reference.clone(),
                    crate::face_labels::label(body.index, f),
                )
            })
            .unwrap_or((None, format!("Face {}", i + 1))),
        CheckEntity::Edge(i) => (
            body.edges
                .iter()
                .find(|e| e.index == i)
                .and_then(|e| e.reference.clone()),
            format!("Edge {}", i + 1),
        ),
        CheckEntity::Vertex(i) => (
            body.vertices
                .iter()
                .find(|v| v.index == i)
                .and_then(|v| v.reference.clone()),
            format!("Vertex {}", i + 1),
        ),
    }
}

fn body_label(body: &BodyDisplay) -> String {
    body.reference
        .as_deref()
        .map(|r| r.trim_start_matches("feature.").replace(".body", ""))
        .unwrap_or_else(|| format!("Body {}", body.index + 1))
}

/// The bodies to check: the picked ones, else every model body (not flat patterns).
fn checked_bodies<'a>(display: &'a PartDisplay, picks: &[String]) -> Vec<&'a BodyDisplay> {
    display
        .bodies
        .iter()
        .filter(|b| !b.flat_pattern)
        .filter(|b| picks.is_empty() || b.reference.as_ref().is_some_and(|r| picks.contains(r)))
        .collect()
}

pub(super) fn check_geometry(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let strict = on(values, "strict", false);
    let short = on(values, "short_edges", true);
    let gaps = on(values, "gaps", true);
    let radius = on(values, "min_radius", true);
    let short_edge = number(values, "short_edge", SHORT_EDGE_MM) / 1000.0;
    let gap = number(values, "gap", GAP_MM) / 1000.0;
    let picks = crate::features::picks_of(values, "bodies");
    let bodies = checked_bodies(&part.display, &picks);
    if bodies.is_empty() {
        return Err(if picks.is_empty() {
            "the part has no bodies to check".into()
        } else {
            "the picked bodies aren't bodies of this part".into()
        });
    }
    let request = AnalysisRequest::Check {
        // Off: no edge is short, no tolerance is a gap.
        short_edge: if short { short_edge } else { 0.0 },
        gap: if gaps { gap } else { f64::INFINITY },
        strict,
    };
    let mut rows = Vec::new();
    let mut found: Vec<Value> = Vec::new();
    let mut colored: Vec<FaceColor> = Vec::new();
    let mut problems = Vec::new();
    let (mut errors, mut warnings) = (0usize, 0usize);
    for body in &bodies {
        let Some(state) = part.result.state.bodies.get(body.index as usize) else {
            continue;
        };
        let findings: Vec<CheckFinding> = match part.kernel.analyze(&request, &[&state.handle]) {
            Ok(AnalysisResult::Check { findings }) => findings,
            Ok(other) => return Err(format!("the kernel answered {other:?}")),
            Err(e) => {
                problems.push(format!(
                    "{}: the kernel could not check it: {e:?}",
                    body_label(body)
                ));
                continue;
            }
        };
        for f in findings {
            use CheckProblem as P;
            let wanted = match f.problem {
                P::SmallestRadius => radius,
                P::LargestEdgeGap | P::LargestVertexGap => gaps,
                _ => true,
            };
            if !wanted {
                continue;
            }
            let (text, severity) = problem_text(f.problem);
            match severity {
                Some("error") => errors += 1,
                Some("warning") => warnings += 1,
                _ => {}
            }
            let (reference, label) = entity_names(body, f.entity);
            let value = match f.problem {
                P::SmallestRadius => f.value.map(|r| format!("R {}", mm(r))),
                _ => f.value.map(mm),
            };
            if let (CheckEntity::Face(_), Some(r), Some(s @ ("error" | "warning"))) =
                (f.entity, reference.as_ref(), severity)
            {
                if !colored.iter().any(|c| &c.item == r) {
                    colored.push(FaceColor {
                        item: r.clone(),
                        entry: usize::from(s == "warning"),
                    });
                }
            }
            found.push(json!({
                "body": body.reference,
                "problem": f.problem,
                "entity": f.entity,
                "reference": reference,
                "value": f.value,
            }));
            rows.push(Row {
                cells: vec![
                    body_label(body),
                    label,
                    text.into(),
                    value.unwrap_or_else(|| "—".into()),
                ],
                items: reference.iter().cloned().collect(),
                severity,
                legend: None,
                reference,
            });
        }
    }
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let summary = if errors + warnings == 0 {
        format!(
            "{} bod{} checked · no problems found",
            bodies.len(),
            if bodies.len() == 1 { "y" } else { "ies" }
        )
    } else {
        format!(
            "{} bod{} checked · {errors} error{} · {warnings} warning{}",
            bodies.len(),
            if bodies.len() == 1 { "y" } else { "ies" },
            plural(errors),
            plural(warnings)
        )
    };
    let mut settings = vec![format!(
        "Check: {}",
        if strict { "strict" } else { "standard" }
    )];
    if short {
        settings.push(format!("Short edges under {}", mm(short_edge)));
    }
    if gaps {
        settings.push(format!("Gaps over {}", mm(gap)));
    }
    Ok(Analysis {
        title: "Check geometry".into(),
        summary,
        columns: ["Body", "Item", "Finding", "Value"]
            .map(String::from)
            .to_vec(),
        rows,
        problems,
        data: json!({ "strict": strict, "short_edge": short_edge, "gap": gap, "findings": found }),
        colors: (!colored.is_empty()).then(|| FaceColors {
            legend: vec![
                LegendEntry {
                    label: "Error".into(),
                    color: ERROR_RGBA,
                },
                LegendEntry {
                    label: "Warning".into(),
                    color: WARNING_RGBA,
                },
            ],
            faces: colored,
        }),
        settings,
        arrow: None,
        dimensions: Vec::new(),
        field: None,
    })
}
