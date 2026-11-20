//! `evaluate.dfm`: the part checked for a manufacturing process by the built-in rule pack for
//! it ([`runmat_cad::dfm::pack`]) and, when `values.rules` names one, a company pack in the
//! project (a JSON check pack, run as well). The part's manufacturing features are measured
//! ([`DfmGraph::measure`]) and each rule is a row per item it applies to: the rule, the item,
//! the limit with its unit, the measured value and the outcome.
//!
//! Values: `process` (`machining`, `sheet_metal`, `molding`, `additive`), `rules` (a project
//! file, `company/dfm` or `company/dfm.json`), `direction` (the mold's pull and the additive
//! build direction; the Top plane's normal when not picked).

use super::{Analysis, Row};
use runmat_cad::dfm::{self, DfmGraph, DfmOptions, Process};
use runmat_cad::props::{CheckPack, Gate, ItemOutcome};
use runmat_cad::units::Units;
use runmat_cad::value::{Evaluator, NoDynamicValues};
use serde_json::{json, Map, Value};

fn process_of(v: Option<&str>) -> Process {
    match v {
        Some("sheet_metal") => Process::SheetMetal,
        Some("molding") => Process::Molding,
        Some("additive") => Process::Additive,
        _ => Process::Machining,
    }
}

fn process_label(p: Process) -> &'static str {
    match p {
        Process::Machining => "Machining",
        Process::SheetMetal => "Sheet metal",
        Process::Molding => "Molding",
        Process::Additive => "Additive",
    }
}

/// The company pack `name` from the project's files: the path as given, from the root, or with
/// `.json`, beside the part or at the root.
fn company_pack(
    path: &str,
    name: &str,
    files: &crate::runtime::Files,
) -> Result<CheckPack, String> {
    let dir = path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    let name = name.trim_start_matches('/');
    let candidates = [
        format!("{dir}/{name}"),
        format!("{dir}/{name}.json"),
        format!("/{name}"),
        format!("/{name}.json"),
        name.to_string(),
    ];
    let (found, text) = candidates
        .iter()
        .find_map(|c| files.get(c).map(|t| (c.clone(), t)))
        .ok_or_else(|| format!("no rule pack `{name}` in the project (looked for {name}.json beside the part and at the root)"))?;
    serde_json::from_str(text).map_err(|e| format!("{found} isn't a check pack: {e}"))
}

pub(super) fn dfm(
    path: &str,
    text: &str,
    files: &crate::runtime::Files,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let process = process_of(values.get("process").and_then(Value::as_str));
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let pick = crate::features::picks_of(values, "direction")
        .into_iter()
        .next()
        .unwrap_or_else(|| "origin.xy".into());
    let direction = crate::draft_analysis::pull_direction(&part.display, &pick)?;
    let opts = DfmOptions {
        pull: direction,
        build: direction,
        ..DfmOptions::default()
    };
    let ev = Evaluator {
        units: &Units,
        dynamic: &NoDynamicValues,
    };
    let analyzer: &dyn runmat_cad::evaluate::GeometryAnalyzer = part.kernel.as_ref();
    let graph = DfmGraph::measure(&part.document, &part.result, &ev, Some(analyzer), &opts);
    let mut packs = vec![dfm::pack(process)];
    let company = values
        .get("rules")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty());
    if let Some(name) = company {
        packs.push(company_pack(path, name, files)?);
    }
    let (results, _) = dfm::run(&packs, &graph, &Units);
    let mut rows = Vec::new();
    let (mut failed, mut passed, mut unknown) = (0usize, 0usize, 0usize);
    let mut problems = Vec::new();
    for r in &results {
        let pack = packs.iter().find(|p| p.name == r.pack);
        let def = pack.and_then(|p| p.checks.iter().find(|c| c.name == r.check));
        let rule = dfm::rule_text(&r.pack, &r.check)
            .map(str::to_owned)
            .unwrap_or_else(|| {
                format!(
                    "{} ({})",
                    r.check,
                    def.map(|d| d.attribute.as_str()).unwrap_or("")
                )
            });
        let limit = def.map(dfm::limit_text).unwrap_or_else(|| "—".into());
        if let Some(e) = &r.error {
            problems.push(format!("{} · {}: {e}", r.pack, r.check));
            continue;
        }
        for (item, outcome) in &r.items {
            let measured = def
                .and_then(|d| graph.measured(item, &d.attribute))
                .or_else(|| match outcome {
                    ItemOutcome::Fail { found } => Some(found.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "not measured".into());
            let (result, severity) = match outcome {
                ItemOutcome::Pass => {
                    passed += 1;
                    ("Passes", None)
                }
                ItemOutcome::Fail { .. } => {
                    failed += 1;
                    (
                        "Fails",
                        Some(match r.gate {
                            Gate::Release => "error",
                            Gate::Publish => "warning",
                            Gate::None => "info",
                        }),
                    )
                }
                ItemOutcome::Unknown => {
                    unknown += 1;
                    ("Not measured", Some("info"))
                }
            };
            rows.push(Row {
                cells: vec![
                    rule.clone(),
                    item.clone(),
                    limit.clone(),
                    measured,
                    result.into(),
                ],
                items: graph
                    .feature_of(item)
                    .map(|f| vec![format!("feature:{}", f.0)])
                    .unwrap_or_default(),
                severity,
                legend: None,
                reference: None,
            });
        }
    }
    // Failures first, then the rest in pack order.
    rows.sort_by_key(|r| match r.severity {
        Some("error") => 0,
        Some("warning") => 1,
        Some("info") if r.cells[4] == "Fails" => 2,
        _ => 3,
    });
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let summary = format!(
        "{} · {} check{} · {failed} fail{} · {passed} pass · {unknown} not measured",
        process_label(process),
        rows.len(),
        plural(rows.len()),
        if failed == 1 { "s" } else { "" },
    );
    let mut settings = vec![
        format!("Process: {}", process_label(process)),
        format!(
            "Rules: {}",
            packs
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    ];
    if matches!(process, Process::Molding | Process::Additive) {
        settings.push(format!(
            "{}: {}",
            if process == Process::Molding {
                "Pull"
            } else {
                "Build direction"
            },
            super::pull_label(&pick, direction, None)
        ));
    }
    Ok(Analysis {
        title: "DFM check".into(),
        summary,
        columns: ["Rule", "Item", "Limit", "Measured", "Result"]
            .map(String::from)
            .to_vec(),
        rows,
        problems,
        data: json!({ "process": process, "results": results }),
        colors: None,
        settings,
        arrow: None,
        dimensions: Vec::new(),
        field: None,
    })
}
