//! `runmat-design`: the design tools from the command line, for scripts and batch jobs.
//!
//! ```text
//! runmat-design export <file> --format <step|3mf|stl|gltf|dxf|dwg|pdf|bom|kicad|fabrication>
//!     [--all-configurations] [--configuration <name>] [--sheet <name>] [--out <dir>]
//!     [--outputs <gerber,drill,drill-map,drawing,pick-place,bom,ipc2581,odb,ipc356>]
//!     [--variant <name>]... [--include-through-hole]
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
//! `--format fabrication` writes a board's fabrication set into folders under `--out`
//! (`gerber/`, `drill/`, `drawing/`, `assembly/`, `ipc2581/`, `odb/`, `netlist/`).
//! `--outputs` limits it to the listed outputs, and `--variant` to the named variants' assembly
//! files (`--variant ""` is the default variant).
//!
//! The workspace is the file's folder and everything under it (`--root <dir>` names another).
//! `--all-configurations` exports a product family: a part's STEP, or every sheet of a
//! drawing as DXF, once per configuration. Files are written to `--out` (default: the current
//! folder) and listed on stdout.

use base64::Engine;
use runmat_design_tools::{call, Files};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// Model libraries (`.lib`, `.cir`, `.sp`) are read for the simulation sections that include them.
const DESIGN: &[&str] = &[".part", ".assembly", ".drawing", ".circuit", ".circuit.layout", ".circuit.lock", ".step", ".stp", ".lib", ".cir", ".sp"];

fn walk(dir: &Path, files: &mut Files, depth: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        if p.is_dir() && depth < 12 {
            walk(&p, files, depth + 1);
        } else if DESIGN.iter().any(|x| name.ends_with(x)) {
            if let Ok(text) = std::fs::read_to_string(&p) {
                files.insert(p.to_string_lossy().into_owned(), text);
            }
        }
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: runmat-design check <file>... [--json] [--no-simulate] [--root <dir>]");
    eprintln!("usage: runmat-design export <file> --format <step|3mf|stl|gltf|dxf|dwg|pdf|bom|kicad|fabrication> [--all-configurations] [--configuration <name>] [--sheet <name>] [--root <dir>] [--out <dir>] [--outputs <list>] [--variant <name>]... [--include-through-hole]");
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
        let mut files = Files::new();
        walk(&base, &mut files, 0);
        match runmat_design_tools::ci::check_circuit(&file.to_string_lossy(), &files, simulate) {
            Ok(r) => {
                failed |= !r.passed;
                if !json_out {
                    let shown = r.path.strip_prefix(&format!("{}/", base.display())).unwrap_or(&r.path).to_owned();
                    for f in &r.findings {
                        let items = if f.items.is_empty() { String::new() } else { format!(" [{}]", f.items.join(", ")) };
                        println!("{shown}: {} {} {}: {}{items}", f.severity, f.check, f.code, f.message);
                    }
                    for m in &r.measurements {
                        let verdict = match m.pass {
                            Some(true) => "pass",
                            Some(false) => "FAIL",
                            None => "no limits",
                        };
                        let value = m.value.map(|v| format!("{v:.6} {}", m.unit)).unwrap_or_else(|| "-".into());
                        println!("{shown}: measure {}.{} = {value} ({verdict})", m.simulation, m.name);
                    }
                    println!(
                        "{shown}: {} ({} errors, {} warnings; ran {})",
                        if r.passed { "passed" } else { "FAILED" },
                        r.errors,
                        r.warnings,
                        r.ran.join(", ")
                    );
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
    let (mut out, mut root, mut configuration, mut sheet) = (PathBuf::from("."), None, None, None);
    let mut all = false;
    let (mut outputs, mut variants, mut tht): (Option<String>, Vec<String>, bool) = (None, Vec::new(), false);
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--format" => format = it.next().cloned(),
            "--out" => out = it.next().map(PathBuf::from).unwrap_or(out),
            "--root" => root = it.next().map(PathBuf::from),
            "--configuration" => configuration = it.next().cloned(),
            "--sheet" => sheet = it.next().cloned(),
            "--all-configurations" => all = true,
            "--outputs" => outputs = it.next().cloned(),
            "--variant" => variants.extend(it.next().cloned()),
            "--include-through-hole" => tht = true,
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
    let mut files = Files::new();
    walk(&root, &mut files, 0);
    let mut input = json!({
        "path": file.to_string_lossy(),
        "format": format,
        "allConfigurations": all,
        "configuration": configuration,
        "sheet": sheet,
    });
    if format == "fabrication" {
        const ALL: [(&str, &str); 9] = [
            ("gerber", "gerber"),
            ("drill", "drill"),
            ("drill-map", "drillMap"),
            ("drawing", "drawing"),
            ("pick-place", "pickPlace"),
            ("bom", "bom"),
            ("ipc2581", "ipc2581"),
            ("odb", "odb"),
            ("ipc356", "ipc356"),
        ];
        let mut fab = serde_json::Map::new();
        if let Some(list) = &outputs {
            let want: Vec<&str> = list.split(',').map(str::trim).collect();
            if let Some(bad) = want.iter().find(|w| !ALL.iter().any(|(k, _)| k == *w)) {
                eprintln!("--outputs: unknown output {bad}");
                return usage();
            }
            for (k, field) in ALL {
                fab.insert(field.into(), Value::Bool(want.contains(&k)));
            }
        }
        if !variants.is_empty() {
            fab.insert("variants".into(), json!(variants));
        }
        fab.insert("includeThroughHole".into(), Value::Bool(tht));
        input["fabrication"] = Value::Object(fab);
    }
    let result: Value = match call("design.export", input, &files) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
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
