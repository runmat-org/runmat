//! `sketch.check`: a part's or an assembly's sketch checked for the feature it will make
//! ([`runmat_cad::evaluate::sketch_check`]): open ends, branches, overlaps, crossings, loops
//! and Revolve's axis. Each finding is a row naming its curves, so selecting the row selects
//! them in the sketch.
//!
//! Values: `sketch` (the open sketch; else the one selected in the Outline), `for`
//! (`extrude`, `revolve`, `sweep_path`, `loft_profile`).

use super::{Analysis, Row};
use runmat_cad::evaluate::sketch_check::{check_sketch, SketchUse};
use runmat_cad::model::FeatureKind;
use serde_json::{json, Map, Value};

/// The sketch the check is for: `values.sketch`, else the one selected (`feature:<id>`,
/// `sketch:<id>`, `sketch.<id>`).
fn sketch_id(values: &Map<String, Value>, selection: &[String]) -> Result<String, String> {
    if let Some(id) = values.get("sketch").and_then(Value::as_str) {
        return Ok(id.to_owned());
    }
    selection
        .iter()
        .find_map(|s| {
            s.strip_prefix("feature:")
                .or_else(|| s.strip_prefix("sketch:"))
                .or_else(|| s.strip_prefix("sketch."))
                .map(|id| id.split('.').next().unwrap_or(id).to_owned())
        })
        .ok_or_else(|| "open a sketch, or select one in the Outline".into())
}

pub(super) fn sketch_check(
    path: &str,
    text: &str,
    files: &crate::runtime::Files,
    configuration: Option<&str>,
    values: &Map<String, Value>,
    selection: &[String],
) -> Result<Analysis, String> {
    let id = sketch_id(values, selection)?;
    let use_ = SketchUse::parse(values.get("for").and_then(Value::as_str));
    let check = if path.ends_with(".assembly") {
        let resolution = crate::runtime::resolve_assembly_in(path, text, files, configuration)
            .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
        let sketch = resolution
            .resolved
            .sketch(&id)
            .ok_or_else(|| format!("the assembly has no solved sketch `{id}`"))?;
        check_sketch(&sketch.sketch, &sketch.solved, use_)
    } else {
        let part =
            crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
                .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
        let feature = part
            .document
            .features
            .iter()
            .find(|f| f.id.0 == id)
            .ok_or_else(|| format!("the part has no sketch `{id}`"))?;
        let FeatureKind::Sketch(sketch) = &feature.kind else {
            return Err(format!("`{id}` isn't a sketch"));
        };
        let (solved, _) = part
            .result
            .state
            .sketches
            .get(&feature.id)
            .ok_or_else(|| format!("sketch `{id}` didn't solve"))?;
        check_sketch(sketch, solved, use_)
    };
    let mm = |p: [f64; 2]| format!("({:.3}, {:.3}) mm", p[0] * 1000.0, p[1] * 1000.0);
    let rows: Vec<Row> = check
        .findings
        .iter()
        .map(|f| Row {
            cells: vec![
                match f.problem {
                    runmat_cad::evaluate::sketch_check::SketchProblem::OpenEnd => "Open end",
                    runmat_cad::evaluate::sketch_check::SketchProblem::Branch => "Branch",
                    runmat_cad::evaluate::sketch_check::SketchProblem::Overlap => "Overlap",
                    runmat_cad::evaluate::sketch_check::SketchProblem::Crossing => "Crossing",
                    runmat_cad::evaluate::sketch_check::SketchProblem::LoopCount => "Loops",
                    runmat_cad::evaluate::sketch_check::SketchProblem::Pieces => "Pieces",
                    runmat_cad::evaluate::sketch_check::SketchProblem::NoAxis => "No axis",
                    runmat_cad::evaluate::sketch_check::SketchProblem::CrossesAxis => {
                        "Crosses the axis"
                    }
                }
                .into(),
                f.message.clone(),
                f.at.map(mm).unwrap_or_else(|| "—".into()),
            ],
            items: f
                .curves
                .iter()
                .map(|c| format!("sketch.{id}.{c}"))
                .collect(),
            severity: Some(f.severity),
            legend: None,
            reference: None,
        })
        .collect();
    let errors = check.errors();
    let warnings = check.findings.len() - errors;
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let mut summary = format!(
        "Sketch {id} · {} curve{} · {} closed loop{}",
        check.curves,
        plural(check.curves),
        check.loops,
        plural(check.loops)
    );
    summary += &if errors == 0 {
        format!(" · ready for {}", use_.label().to_lowercase())
    } else {
        format!(" · {errors} problem{} to fix", plural(errors))
    };
    if warnings > 0 {
        summary += &format!(" · {warnings} warning{}", plural(warnings));
    }
    let mut settings = vec![format!("Check for: {}", use_.label())];
    if let Some(axis) = &check.axis {
        settings.push(format!("Axis: {axis}"));
    }
    Ok(Analysis {
        title: "Check sketch".into(),
        summary,
        columns: ["Finding", "Detail", "Where"].map(String::from).to_vec(),
        rows,
        problems: Vec::new(),
        data: json!({ "sketch": id, "check": check }),
        colors: None,
        settings,
        arrow: None,
        dimensions: Vec::new(),
        field: None,
    })
}
