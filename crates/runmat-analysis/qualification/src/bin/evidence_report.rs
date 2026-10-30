//! Generates the evidence catalog: runs every tagged witness test through cargo (one cargo
//! invocation per package) and writes markdown and JSON. Output is regenerated, never edited.
//!
//! evidence_report [--out-dir DIR] [--cargo "prefix words"] [--package NAME]... [--no-run]
//!
//! `--cargo` is the command prefix, e.g. `/path/gate build -- cargo` so each invocation is a
//! single gated cargo command. Default: `cargo`.
use runmat_analysis_qualification::{
    catalog::{cells, Cell, TestResult, Totals, Verdict},
    requirement::Requirement,
    requirements,
    witness::{repository_root, scan, validate_tags, Tag, Target, TestLocator},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::PathBuf,
    process::Command,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out_dir = repository_root().join("docs/simulation/qualification/generated");
    let mut cargo = vec!["cargo".to_string()];
    let mut only: BTreeSet<String> = BTreeSet::new();
    let mut run = true;
    let mut reuse: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out-dir" => out_dir = PathBuf::from(args.next().ok_or("--out-dir needs a path")?),
            "--cargo" => {
                cargo = args
                    .next()
                    .ok_or("--cargo needs a command prefix")?
                    .split_whitespace()
                    .map(String::from)
                    .collect()
            }
            "--package" => {
                only.insert(args.next().ok_or("--package needs a name")?);
            }
            "--no-run" => run = false,
            "--reuse" => {
                reuse = Some(PathBuf::from(
                    args.next().ok_or("--reuse needs a catalog json")?,
                ))
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let registry: Vec<&'static Requirement> = requirements().collect();
    let (tags, mut problems) = scan(&repository_root());
    problems.extend(validate_tags(&tags, &registry));
    if !problems.is_empty() {
        for p in &problems {
            eprintln!("{p}");
        }
        return Err("invalid witness tags; fix them before reporting".into());
    }
    let results = if run {
        // `--reuse` keeps the results of an earlier catalog for tests whose locator is
        // unchanged and runs only the tests it has no result for.
        let mut known = BTreeMap::new();
        if let Some(path) = &reuse {
            known = load_results(path)?;
        }
        let missing: Vec<Tag> = tags
            .iter()
            .filter(|t| !known.contains_key(&t.test.describe()))
            .cloned()
            .collect();
        let mut fresh = if missing.is_empty() {
            BTreeMap::new()
        } else {
            run_tests(&missing, &cargo, &only)?
        };
        known.append(&mut fresh);
        Some(known)
    } else {
        None
    };
    let all = cells(&registry, &tags, results.as_ref());
    fs::create_dir_all(out_dir.join("requirements"))?;
    write_markdown(&out_dir, &registry, &all)?;
    write_json(&out_dir, &registry, &all, run)?;
    write_scope(&out_dir, &all)?;
    let mut totals = Totals::default();
    for c in &all {
        totals.add(c.verdict);
    }
    println!(
        "cells {} pass {} partial {} fail {} open {} tagged-not-run {}",
        totals.total(),
        totals.pass,
        totals.partial,
        totals.fail,
        totals.open,
        totals.tagged
    );
    Ok(())
}

fn run_tests(
    tags: &[Tag],
    cargo: &[String],
    only: &BTreeSet<String>,
) -> Result<BTreeMap<String, TestResult>, Box<dyn std::error::Error>> {
    let mut tests: BTreeMap<(&str, String), BTreeSet<&TestLocator>> = BTreeMap::new();
    for tag in tags {
        if only.is_empty() || only.contains(&tag.test.package) {
            tests
                .entry((&tag.test.package, tag.features.join(",")))
                .or_default()
                .insert(&tag.test);
        }
    }
    let mut results = BTreeMap::new();
    for ((package, features), all_locators) in tests {
        // Harness-less integration binaries (`fn main`) are one witness each: run the whole
        // target and take its exit status.
        let (custom, locators): (Vec<_>, Vec<_>) =
            all_locators.into_iter().partition(|l| l.name == "main");
        for l in &custom {
            let Target::Integration(target) = &l.target else {
                continue;
            };
            let mut command = Command::new("sh");
            command.args(["-c", "\"$@\" 2>&1", "sh"]).args(cargo);
            command.args(["test", "--offline", "-p", package, "--test", target]);
            if !features.is_empty() {
                command.args(["--features", &features]);
            }
            command.env("RUST_MIN_STACK", "67108864");
            eprintln!("running {package} test target {target} as one witness");
            let output = command.output()?;
            results.insert(
                l.describe(),
                if output.status.success() {
                    TestResult::Passed
                } else {
                    TestResult::Failed("harness-less test target exited nonzero".into())
                },
            );
        }
        if locators.is_empty() {
            continue;
        }
        // One shell merges stderr into stdout so cargo's `Running ...` headers stay ordered
        // with the test lines that follow them.
        let mut command = Command::new("sh");
        command.args(["-c", "\"$@\" 2>&1", "sh"]).args(cargo);
        command.args(["test", "--offline", "-p", package]);
        if !features.is_empty() {
            command.args(["--features", &features]);
        }
        let mut libs = false;
        let mut integrations = BTreeSet::new();
        for l in &locators {
            match &l.target {
                Target::Library => libs = true,
                Target::Integration(t) => {
                    integrations.insert(t.clone());
                }
            }
        }
        if libs {
            command.arg("--lib");
        }
        for t in &integrations {
            command.args(["--test", t]);
        }
        command.arg("--no-fail-fast");
        command.arg("--").args(["--exact", "--include-ignored"]);
        let names: BTreeSet<&str> = locators.iter().map(|l| l.name.as_str()).collect();
        command.args(&names);
        command.env("RUST_MIN_STACK", "67108864");
        eprintln!("running {package}: {} tests", locators.len());
        let output = command.output()?;
        let text = String::from_utf8_lossy(&output.stdout).into_owned();
        let mut seen: BTreeMap<(Target, String), TestResult> = BTreeMap::new();
        let mut section: Option<Target> = None;
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("Running ") {
                section = Some(if rest.starts_with("unittests") {
                    Target::Library
                } else if let Some(file) = rest.strip_prefix("tests/") {
                    Target::Integration(
                        file.split(".rs")
                            .next()
                            .unwrap_or("")
                            .split('/')
                            .next()
                            .unwrap_or("")
                            .to_string(),
                    )
                } else {
                    Target::Integration(String::new())
                });
            } else if let (Some(rest), Some(section)) = (line.strip_prefix("test "), &section) {
                if let Some((name, verdict)) = rest.rsplit_once(" ... ") {
                    let result = match verdict.trim() {
                        "ok" => TestResult::Passed,
                        "FAILED" => TestResult::Failed("test failed".into()),
                        other => TestResult::Failed(format!("not passed: {other}")),
                    };
                    seen.insert((section.clone(), name.to_string()), result);
                }
            }
        }
        let build_failed = seen.is_empty() && !output.status.success();
        for l in locators {
            let result = match seen.get(&(l.target.clone(), l.name.clone())) {
                Some(r) => r.clone(),
                None if build_failed => {
                    TestResult::Failed("cargo test did not build or run".into())
                }
                None => TestResult::Failed("test not found in cargo test run".into()),
            };
            results.insert(l.describe(), result);
        }
    }
    Ok(results)
}

fn short(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        format!("{}...", flat.chars().take(max).collect::<String>())
    }
}

fn totals_of<'a>(cells: impl Iterator<Item = &'a Cell>) -> Totals {
    let mut t = Totals::default();
    for c in cells {
        t.add(c.verdict);
    }
    t
}

fn write_markdown(
    dir: &std::path::Path,
    registry: &[&'static Requirement],
    all: &[Cell],
) -> std::io::Result<()> {
    let total = totals_of(all.iter());
    let mut index = String::new();
    writeln!(index, "# FEA evidence catalog\n").unwrap();
    writeln!(index, "Generated by `evidence_report` from the register and the `// witness:` tags at the tests. Never edit by hand; regenerate.\n").unwrap();
    writeln!(
        index,
        "Cells: {} total, {} PASS (complete coverage, tests passed), {} PARTIAL (tests passed, narrower than the criterion), {} FAIL, {} OPEN, {} TAGGED (not run).\n",
        total.total(),
        total.pass,
        total.partial,
        total.fail,
        total.open,
        total.tagged
    )
    .unwrap();
    writeln!(
        index,
        "| Requirement | Title | PASS | PARTIAL | FAIL | OPEN | TAGGED |\n|---|---|---:|---:|---:|---:|---:|"
    )
    .unwrap();
    for r in registry {
        let t = totals_of(all.iter().filter(|c| c.requirement == r.id));
        writeln!(
            index,
            "| [{}](requirements/{}.md) | {} | {} | {} | {} | {} | {} |",
            r.id,
            r.id,
            short(r.title, 80),
            t.pass,
            t.partial,
            t.fail,
            t.open,
            t.tagged
        )
        .unwrap();
        let mut page = String::new();
        writeln!(page, "# {}: {}\n", r.id, r.title).unwrap();
        writeln!(
            page,
            "{} PASS, {} PARTIAL, {} FAIL, {} OPEN, {} TAGGED. Generated; do not edit.\n",
            t.pass, t.partial, t.fail, t.open, t.tagged
        )
        .unwrap();
        for o in r.obligations {
            writeln!(page, "## {}\n", o.id).unwrap();
            writeln!(
                page,
                "Evidence kind: {:?}. Criterion: {}\n",
                o.evidence_kind,
                short(o.criterion, 400)
            )
            .unwrap();
            writeln!(page, "| Formulation | Mesh | Execution | Status | Tests (boundary) |\n|---|---|---|---|---|").unwrap();
            for c in all
                .iter()
                .filter(|c| c.requirement == r.id && c.obligation == o.id)
            {
                let tests = c
                    .tests
                    .iter()
                    .map(|t| {
                        format!(
                            "`{}` ({}:{}) - {}",
                            t.locator.describe(),
                            t.locator.source,
                            t.locator.line,
                            short(&t.boundary, 160)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("<br>");
                writeln!(
                    page,
                    "| {} | {} | {} | {} | {} |",
                    c.envelope.formulation.wire_name(),
                    c.envelope.mesh.wire_name(),
                    c.envelope.execution.wire_name(),
                    c.verdict.label(),
                    tests
                )
                .unwrap();
            }
            writeln!(page).unwrap();
        }
        fs::write(dir.join("requirements").join(format!("{}.md", r.id)), page)?;
    }
    fs::write(dir.join("evidence-catalog.md"), index)
}

fn write_json(
    dir: &std::path::Path,
    registry: &[&'static Requirement],
    all: &[Cell],
    ran: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;
    let total = totals_of(all.iter());
    let requirements: Vec<_> = registry
        .iter()
        .map(|r| {
            let obligations: Vec<_> = r
                .obligations
                .iter()
                .map(|o| {
                    let cells: Vec<_> = all
                        .iter()
                        .filter(|c| c.requirement == r.id && c.obligation == o.id)
                        .map(|c| {
                            json!({
                                "formulation": c.envelope.formulation.wire_name(),
                                "mesh": c.envelope.mesh.wire_name(),
                                "execution": c.envelope.execution.wire_name(),
                                "status": c.verdict.label(),
                                "tests": c.tests.iter().map(|t| json!({
                                    "package": t.locator.package,
                                    "target": format!("{:?}", t.locator.target),
                                    "name": t.locator.name,
                                    "source": t.locator.source,
                                    "line": t.locator.line,
                                    "boundary": t.boundary,
                                    "result": match &t.result { None => json!("not_run"), Some(TestResult::Passed) => json!("passed"), Some(TestResult::Failed(m)) => json!({"failed": m}) },
                                })).collect::<Vec<_>>(),
                            })
                        })
                        .collect();
                    json!({"id": o.id, "criterion": o.criterion, "evidence_kind": format!("{:?}", o.evidence_kind), "cells": cells})
                })
                .collect();
            let t = totals_of(all.iter().filter(|c| c.requirement == r.id));
            json!({"id": r.id, "title": r.title, "pass": t.pass, "partial": t.partial, "fail": t.fail, "open": t.open, "tagged": t.tagged, "obligations": obligations})
        })
        .collect();
    let doc = json!({
        "schema_version": 1,
        "tests_run": ran,
        "totals": {"cells": total.total(), "pass": total.pass, "partial": total.partial, "fail": total.fail, "open": total.open, "tagged": total.tagged},
        "requirements": requirements,
    });
    let mut bytes = serde_json::to_vec_pretty(&doc)?;
    bytes.push(b'\n');
    fs::write(dir.join("evidence-catalog.json"), bytes)?;
    let _ = Verdict::Open;
    Ok(())
}

fn write_scope(dir: &std::path::Path, all: &[Cell]) -> Result<(), Box<dyn std::error::Error>> {
    use runmat_analysis_qualification::scope::{rows, trace, Mapping, TraceStatus};
    let rows = rows().map_err(|p| format!("scope inventory: {p:?}"))?;
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut md = String::new();
    let mut json_rows = Vec::new();
    let mut table = String::new();
    for row in &rows {
        let t = trace(row, all);
        *counts.entry(t.status.label()).or_default() += 1;
        let maps = row
            .maps
            .iter()
            .map(|m| match m {
                Mapping::Obligation {
                    requirement,
                    obligation,
                } => {
                    format!("{}/{}", requirement, obligation.as_deref().unwrap_or("*"))
                }
                Mapping::Excluded(r) => format!("excluded: {r}"),
                Mapping::External { owner, reason } => format!("external ({owner}): {reason}"),
            })
            .collect::<Vec<_>>();
        writeln!(
            table,
            "| {} | {} | {} | {} | {} | {}/{}/{}/{}/{} |",
            row.id,
            row.source,
            short(&row.capability, 110),
            short(&maps.join("; "), 240),
            t.status.label(),
            t.pass,
            t.partial,
            t.tagged,
            t.fail,
            t.cells
        )
        .unwrap();
        json_rows.push(serde_json::json!({
            "id": row.id, "source": row.source, "capability": row.capability,
            "maps": maps, "status": t.status.label(),
            "cells": t.cells, "pass": t.pass, "partial": t.partial, "tagged": t.tagged, "fail": t.fail,
        }));
    }
    writeln!(md, "# FEA scope traceability\n").unwrap();
    writeln!(md, "Generated by `evidence_report` from the ledger claims (docs/simulation/ledger.json) and the evidence catalog. Never edit by hand.\n").unwrap();
    writeln!(
        md,
        "Scope rows: {}. Unmapped: 0 (enforced by tests). Status counts: {}.\n",
        rows.len(),
        counts
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(md, "Status: `pass` every mapped cell is PASS; `fail` any mapped cell failed; `tied` some mapped cell has a witness (PASS, PARTIAL or tagged), none failed; `open` no mapped cell has a witness; `excluded` and `external` carry no FEA obligation. Cells column is pass/partial/tagged/fail/total.\n").unwrap();
    writeln!(md, "| Scope row | Source | Capability | Obligations | Status | Cells |\n|---|---|---|---|---|---|").unwrap();
    md.push_str(&table);
    fs::write(dir.join("scope-traceability.md"), md)?;
    let mut bytes =
        serde_json::to_vec_pretty(&serde_json::json!({"schema_version": 1, "rows": json_rows}))?;
    bytes.push(b'\n');
    fs::write(dir.join("scope-traceability.json"), bytes)?;
    let _ = TraceStatus::Open;
    Ok(())
}

fn load_results(
    path: &std::path::Path,
) -> Result<BTreeMap<String, TestResult>, Box<dyn std::error::Error>> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let mut out = BTreeMap::new();
    for requirement in value["requirements"].as_array().into_iter().flatten() {
        for obligation in requirement["obligations"].as_array().into_iter().flatten() {
            for cell in obligation["cells"].as_array().into_iter().flatten() {
                for test in cell["tests"].as_array().into_iter().flatten() {
                    let target = test["target"].as_str().unwrap_or("");
                    let target = match target.strip_prefix("Integration(\"") {
                        Some(rest) => format!("test:{}", rest.trim_end_matches("\")")),
                        None => "lib".to_string(),
                    };
                    let key = format!(
                        "{}[{}]::{}",
                        test["package"].as_str().unwrap_or(""),
                        target,
                        test["name"].as_str().unwrap_or("")
                    );
                    match &test["result"] {
                        serde_json::Value::String(s) if s == "passed" => {
                            out.insert(key, TestResult::Passed);
                        }
                        // Failures are never reused: they are run again.
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(out)
}
