//! `runmat-design`: the design tools from the command line, for scripts and batch jobs.
//!
//! ```text
//! runmat-design export <file> --format <step|3mf|stl|gltf|dxf|dwg|pdf|bom|kicad|fabrication>
//!     [--all-configurations] [--configuration <name>] [--sheet <name>] [--out <dir>]
//!     [--preset <fab_house|assembly|full|custom>]
//!     [--outputs <gerber,drill,drill-map,drawing,ipc356,pick-place,bom,assembly-drawing,ipc2581,odb>]
//!     [--layers <F.Cu,B.Cu,…>] [--variant <name>]... [--include-through-hole] [--zip|--no-zip]
//!     [--remember]
//! ```
//!
//!
//! ```text
//! runmat-design check <file>... [--json] [--no-simulate] [--root <dir>]
//! ```
//!
//! `check` runs the circuit checks on each `.circuit` or `.circuit.layout` (compile, ERC, the
//! design-review suites; with a board DRC, unrouted connections and pending schematic changes;
//! every simulation section's measurements against their limits). It prints each finding,
//! or the reports as JSON with `--json`, and exits 1 when any file has an error, so a CI job
//! or a coding agent can gate on it.
//!
//! `--format fabrication` writes a board's fabrication set into folders (`gerber/`, `drill/`,
//! `drawing/`, `assembly/`, `ipc2581/`, `odb/`, `netlist/`) under `--out`, by default the
//! folder the board's settings name beside the board. It runs with the settings the board's
//! layout keeps (`fabrication = { … }`, the ones the Desktop's Fabrication outputs saved), so a
//! re-export makes the same files. Each flag overrides them: `--preset` picks a preset's
//! outputs, `--outputs` lists them, `--layers` the Gerber layers, `--variant` the variants that
//! get assembly files (`--variant ""` is the default variant). `--remember` saves the result
//! into the layout.
//!
//! The workspace is the file's folder and everything under it (`--root <dir>` names another).
//! `--all-configurations` exports a product family: a part's STEP, or every sheet of a
//! drawing as DXF, once per configuration. Files are written to `--out` (default: the current
//! folder) and listed on stdout.

use base64::Engine;
use runmat_design_tools::call;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!("usage: runmat-design check <file>... [--json] [--no-simulate] [--root <dir>]");
    eprintln!("usage: runmat-design export <file> --format <step|3mf|stl|gltf|dxf|dwg|pdf|bom|kicad|fabrication> [--all-configurations] [--configuration <name>] [--sheet <name>] [--root <dir>] [--out <dir>] [--preset <name>] [--outputs <list>] [--layers <list>] [--variant <name>]... [--include-through-hole] [--zip|--no-zip] [--remember]");
    ExitCode::from(2)
}

fn check(args: &[String]) -> ExitCode {
    let (mut json_out, mut simulate, mut root, mut paths) = (false, true, None, Vec::new());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--json" => json_out = true,
            "--no-simulate" => simulate = false,
            "--root" => root = it.next().map(PathBuf::from),
            p if !p.starts_with("--") => paths.push(PathBuf::from(p)),
            _ => return usage(),
        }
    }
    if paths.is_empty() {
        return usage();
    }
    let mut reports = Vec::new();
    let mut failed = false;
    for p in paths {
        let file = match std::fs::canonicalize(&p) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{}: {e}", p.display());
                return ExitCode::FAILURE;
            }
        };
        let base = root.clone().and_then(|r| std::fs::canonicalize(r).ok()).unwrap_or_else(|| file.parent().unwrap_or(Path::new("/")).to_path_buf());
        let files = runmat_design_tools::workspace::read_workspace(&base);
        match runmat_design_tools::ci::check_circuit(&file.to_string_lossy(), &files, simulate) {
            Ok(r) => {
                failed |= !r.passed;
                if !json_out {
                    for line in runmat_design_tools::ci::report_lines(&r, &base.to_string_lossy()) {
                        println!("{line}");
                    }
                }
                reports.push(r);
            }
            Err(e) => {
                eprintln!("{}: {e}", p.display());
                return ExitCode::FAILURE;
            }
        }
    }
    if json_out {
        println!("{}", serde_json::to_string_pretty(&reports).unwrap_or_default());
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("check") {
        return check(&args[1..]);
    }
    if args.first().map(String::as_str) != Some("export") || args.len() < 2 {
        return usage();
    }
    let mut format = None;
    let (mut out, mut root, mut configuration, mut sheet): (Option<PathBuf>, Option<PathBuf>, Option<String>, Option<String>) = (None, None, None, None);
    let mut all = false;
    let (mut outputs, mut variants, mut tht): (Option<String>, Vec<String>, Option<bool>) = (None, Vec::new(), None);
    let (mut preset, mut layers, mut zip, mut remember): (Option<String>, Option<String>, Option<bool>, bool) = (None, None, None, false);
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--format" => format = it.next().cloned(),
            "--out" => out = it.next().map(PathBuf::from),
            "--root" => root = it.next().map(PathBuf::from),
            "--configuration" => configuration = it.next().cloned(),
            "--sheet" => sheet = it.next().cloned(),
            "--all-configurations" => all = true,
            "--outputs" => outputs = it.next().cloned(),
            "--variant" => variants.extend(it.next().cloned()),
            "--include-through-hole" => tht = Some(true),
            "--preset" => preset = it.next().cloned(),
            "--layers" => layers = it.next().cloned(),
            "--zip" => zip = Some(true),
            "--no-zip" => zip = Some(false),
            "--remember" => remember = true,
            _ => return usage(),
        }
    }
    let Some(format) = format else { return usage() };
    let file = match std::fs::canonicalize(&args[1]) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{}: {e}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let root = root.and_then(|r| std::fs::canonicalize(r).ok()).unwrap_or_else(|| file.parent().unwrap_or(Path::new("/")).to_path_buf());
    let files = runmat_design_tools::workspace::read_workspace(&root);
    let mut input = json!({
        "path": file.to_string_lossy(),
        "format": format,
        "allConfigurations": all,
        "configuration": configuration,
        "sheet": sheet,
    });
    if format == "fabrication" {
        // The board's saved settings apply; each flag given overrides them.
        const ALL: [(&str, &str); 10] = [
            ("gerber", "gerber"),
            ("drill", "drill"),
            ("drill-map", "drillMap"),
            ("drawing", "drawing"),
            ("ipc356", "ipc356"),
            ("pick-place", "pickPlace"),
            ("bom", "bom"),
            ("assembly-drawing", "assemblyDrawing"),
            ("ipc2581", "ipc2581"),
            ("odb", "odb"),
        ];
        let mut fab = serde_json::Map::new();
        if let Some(p) = &preset {
            fab.insert("preset".into(), json!(p));
        }
        if let Some(list) = &outputs {
            let want: Vec<String> = list.split(',').map(|w| w.trim().replace('_', "-")).collect();
            if let Some(bad) = want.iter().find(|w| !ALL.iter().any(|(k, _)| k == w)) {
                eprintln!("--outputs: unknown output {bad}");
                return usage();
            }
            for (k, field) in ALL {
                fab.insert(field.into(), Value::Bool(want.iter().any(|w| w == k)));
            }
        }
        if !variants.is_empty() {
            fab.insert("variants".into(), json!(variants));
        }
        if let Some(l) = &layers {
            fab.insert("layers".into(), json!(l.split(',').map(str::trim).collect::<Vec<_>>()));
        }
        if let Some(t) = tht {
            fab.insert("includeThroughHole".into(), Value::Bool(t));
        }
        if let Some(z) = zip {
            fab.insert("zip".into(), Value::Bool(z));
        }
        fab.insert("remember".into(), Value::Bool(remember));
        input["fabrication"] = Value::Object(fab);
    }
    let result: Value = match call("design.export", input, &files) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    // A fabrication set goes to its settings' folder beside the board unless --out names one.
    let out = out.unwrap_or_else(|| match result["fabrication"]["folder"].as_str() {
        Some(folder) => file.parent().unwrap_or(Path::new(".")).join(folder),
        None => PathBuf::from("."),
    });
    if let Some(update) = result["fabrication"]["layout"].as_object() {
        let (path, text) = (update["path"].as_str().unwrap_or(""), update["text"].as_str().unwrap_or(""));
        if let Err(e) = std::fs::write(path, text) {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("saved the settings in {path}");
    }
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("{}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    for f in result["files"].as_array().into_iter().flatten() {
        let name = f["name"].as_str().unwrap_or("export");
        let content = f["content"].as_str().unwrap_or("");
        let bytes = if f["encoding"] == "base64" {
            base64::engine::general_purpose::STANDARD.decode(content).unwrap_or_default()
        } else {
            content.as_bytes().to_vec()
        };
        let path = out.join(name);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Err(e) = std::fs::write(&path, bytes) {
            eprintln!("{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
        println!("{}", path.display());
    }
    for n in result["notes"].as_array().into_iter().flatten() {
        eprintln!("note: {}", n.as_str().unwrap_or(""));
    }
    ExitCode::SUCCESS
}
