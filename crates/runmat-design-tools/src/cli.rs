//! The design commands of the `runmat` command line, for scripts and batch jobs: `runmat check`
//! on design files and `runmat design run|export`.
//!
//! ```text
//! runmat design export <file> --format <step|3mf|stl|gltf|dxf|dwg|pdf|svg|bom|kicad|fabrication>
//!     [--all-configurations] [--configuration <name>] [--sheet <name>] [--out <dir>]
//!     [--preset <fab_house|assembly|full|custom>]
//!     [--outputs <gerber,drill,drill-map,drawing,ipc356,pick-place,bom,assembly-drawing,ipc2581,odb>]
//!     [--layers <F.Cu,B.Cu,…>] [--variant <name>]... [--include-through-hole] [--zip|--no-zip]
//!     [--remember] [--copper] [--no-mask]
//! runmat design run <file> <command> [--values <json>] [--select <outline id>]... [--root <dir>]
//!     [--out <dir>] [--dry-run] [--json]
//! ```
//!
//! `run` runs one catalog command (`assembly.bom`, `part.fillet`, `drawing.export`) through the
//! command registry, the dispatcher the Desktop and the agent use: an edit is written to its
//! files (`--dry-run` prints the change instead), a results table prints as columns, and an
//! export writes its files beside the source (or into `--out`). `--json` prints the answer.
//!
//!
//! ```text
//! runmat check <file>... [--json] [--no-simulate] [--interference] [--root <dir>]
//! ```
//!
//! On a `.part`, `.assembly` or `.drawing`, `check` runs `design.check` (regenerate, resolve
//! occurrences and mates, compile views) and prints each diagnostic; `--interference` also
//! checks an assembly's parts for overlapping material and fails on any overlap.
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
//! get assembly files (`--variant ""` is the default variant), `--gerber-units` and
//! `--gerber-decimals` the Gerber format, `--pick-place-sides` and `--pick-place-units` the
//! pick-and-place, `--bom-columns` and `--bom-ungrouped` the bill of materials, `--revision` the board
//! revision. `--remember`
//! saves the result into the layout.
//!
//! `--format step` on a `.circuit` or `.circuit.layout` writes the board as STEP AP242: the board,
//! its solder mask and each component (its package's STEP model, else its body box), each a
//! product under its name. `--copper` adds the outer copper as thin solids; `--no-mask` leaves the
//! mask out.
//!
//! The workspace is the file's project (the nearest folder up holding a `runmat.toml`, else the
//! file's folder) and everything under it (`--root <dir>` names another).
//! `--all-configurations` exports a product family: a part's STEP, or every sheet of a
//! drawing as DXF, once per configuration. Files are written to `--out` (default: the current
//! folder) and listed on stdout.

use crate::call;
use crate::workspace::{output_path, physical_path, source_path};
use base64::Engine;
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Where a command prints: its report (`out`) and its errors and notes (`err`). The `runmat`
/// binary passes stdout and stderr; tests pass buffers.
pub struct Io<'a> {
    pub out: &'a mut dyn Write,
    pub err: &'a mut dyn Write,
}

impl Io<'_> {
    /// Runs `f` with stdout and stderr.
    pub fn std<R>(f: impl FnOnce(&mut Io) -> R) -> R {
        let (mut out, mut err) = (std::io::stdout().lock(), std::io::stderr().lock());
        f(&mut Io {
            out: &mut out,
            err: &mut err,
        })
    }
}

macro_rules! say {
    ($io:expr, $($arg:tt)*) => {{ let _ = writeln!($io.out, $($arg)*); }};
}
macro_rules! tell {
    ($io:expr, $($arg:tt)*) => {{ let _ = writeln!($io.err, $($arg)*); }};
}

fn usage(io: &mut Io) -> ExitCode {
    tell!(
        io,
        "usage: runmat check <file>... [--json] [--no-simulate] [--interference] [--root <dir>]"
    );
    tell!(io, "usage: runmat design run <file> <command> [--values <json>] [--select <outline id>]... [--root <dir>] [--out <dir>] [--dry-run] [--json]");
    tell!(io, "usage: runmat design export <file> --format <step|3mf|stl|gltf|dxf|dwg|pdf|svg|bom|kicad|fabrication> [--all-configurations] [--configuration <name>] [--sheet <name>] [--root <dir>] [--out <dir>] [--preset <name>] [--outputs <list>] [--layers <list>] [--variant <name>]... [--include-through-hole] [--zip|--no-zip] [--gerber-units mm|inch] [--gerber-decimals 5|6] [--pick-place-sides both|separate|top|bottom] [--pick-place-units mm|inch] [--bom-columns <list>] [--bom-ungrouped] [--revision <text>] [--remember] [--copper] [--no-mask]");
    ExitCode::from(2)
}

/// `runmat check` on design files: `args` are the files and the flags.
pub fn check(args: &[String], io: &mut Io) -> ExitCode {
    crate::kernel::install();
    let (mut json_out, mut simulate, mut root, mut paths) = (false, true, None, Vec::new());
    let mut interference = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--json" => json_out = true,
            "--no-simulate" => simulate = false,
            "--interference" => interference = true,
            "--root" => root = it.next().map(PathBuf::from),
            p if !p.starts_with("--") => paths.push(PathBuf::from(p)),
            _ => return usage(io),
        }
    }
    if paths.is_empty() {
        return usage(io);
    }
    let mut reports = Vec::new();
    let mut failed = false;
    for p in paths {
        let file = match std::fs::canonicalize(&p) {
            Ok(f) => f,
            Err(e) => {
                tell!(io, "{}: {e}", p.display());
                return ExitCode::FAILURE;
            }
        };
        let base = root
            .clone()
            .and_then(|r| std::fs::canonicalize(r).ok())
            .unwrap_or_else(|| crate::workspace::project_root(&file));
        let files = crate::workspace::read_workspace(&base);
        let path = source_path(&file);
        if matches!(crate::DesignKind::of(&path), Some(k) if !k.is_circuit()) {
            match crate::circuit_values::with_native_values(&files, || {
                check_mechanical(io, &path, &files, interference, &base, json_out)
            }) {
                Ok((passed, report)) => {
                    failed |= !passed;
                    reports.push(report);
                }
                Err(e) => {
                    tell!(io, "{}: {e}", p.display());
                    return ExitCode::FAILURE;
                }
            }
            continue;
        }
        match crate::circuit_values::with_native_values(&files, || {
            crate::ci::check_circuit(&source_path(&file), &files, simulate)
        }) {
            Ok(mut r) => {
                r.path = file.to_string_lossy().into_owned();
                failed |= !r.passed;
                if !json_out {
                    for line in crate::ci::report_lines(&r, &base.to_string_lossy()) {
                        say!(io, "{line}");
                    }
                }
                reports.push(serde_json::to_value(&r).unwrap_or_default());
            }
            Err(e) => {
                tell!(io, "{}: {e}", p.display());
                return ExitCode::FAILURE;
            }
        }
    }
    if json_out {
        say!(
            io,
            "{}",
            serde_json::to_string_pretty(&reports).unwrap_or_default()
        );
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// `design.check` on a part, assembly or drawing: every diagnostic, one line each. With
/// `--interference`, an assembly also runs the Evaluate tab's interference check (exact bodies
/// that share material; parts that only touch don't count) and fails on any overlap.
fn check_mechanical(
    io: &mut Io,
    path: &str,
    files: &crate::Files,
    interference: bool,
    base: &Path,
    json_out: bool,
) -> Result<(bool, Value), String> {
    let shown_base = source_path(base);
    let shown = |p: &str| {
        p.strip_prefix(&shown_base)
            .map(|r| r.trim_start_matches('/').to_owned())
            .unwrap_or_else(|| p.to_owned())
    };
    let mut report: Value =
        call("design.check", json!({ "path": path }), files).map_err(|e| e.to_string())?;
    let mut passed = report["ok"].as_bool().unwrap_or(false);
    if !json_out {
        say!(io, "{}", shown(report["summary"].as_str().unwrap_or(path)));
        for d in report["diagnostics"].as_array().into_iter().flatten() {
            let at = d["range"]["start"]["line"]
                .as_u64()
                .map(|l| format!(":{}", l + 1))
                .unwrap_or_default();
            let subject = d["subject"]
                .as_str()
                .map(|s| format!(" {s}:"))
                .unwrap_or_default();
            say!(
                io,
                "  {}{at}{subject} {}",
                d["severity"].as_str().unwrap_or("info"),
                d["message"].as_str().unwrap_or("")
            );
        }
    }
    if interference && crate::DesignKind::of(path) == Some(crate::DesignKind::Assembly) {
        let inspected: Value = call(
            "design.inspect",
            json!({ "path": path, "sections": ["interference"] }),
            files,
        )
        .map_err(|e| e.to_string())?;
        let value = inspected["sections"]["interference"].clone();
        let overlaps = value
            .pointer("/result/overlaps")
            .or_else(|| value.pointer("/overlaps"))
            .and_then(Value::as_array)
            .cloned();
        match &overlaps {
            Some(rows) => {
                passed &= rows.is_empty();
                if !json_out {
                    say!(io, "  interference: {} overlap(s)", rows.len());
                    for r in rows {
                        say!(
                            io,
                            "    {} x {}: {} mm3",
                            r["a"].as_str().unwrap_or("?"),
                            r["b"].as_str().unwrap_or("?"),
                            r["volume_mm3"]
                        );
                    }
                }
            }
            None => {
                passed = false;
                if !json_out {
                    say!(io, "  interference: not run ({})", inspected["notes"]);
                }
            }
        }
        report["interference"] = value;
    }
    Ok((passed, report))
}

/// `runmat design run`: one catalog command through `crate::commands::run`.
fn run(args: &[String], io: &mut Io) -> ExitCode {
    let (Some(file), Some(command)) = (args.first(), args.get(1)) else {
        return usage(io);
    };
    let (mut values, mut selection, mut root, mut out, mut dry, mut json_out) =
        (json!({}), Vec::new(), None, None, false, false);
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--values" => match it.next().map(|v| serde_json::from_str::<Value>(v)) {
                Some(Ok(v @ Value::Object(_))) => values = v,
                _ => {
                    tell!(io, "--values takes a JSON object");
                    return ExitCode::FAILURE;
                }
            },
            "--select" => selection.extend(it.next().cloned()),
            "--root" => root = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            "--dry-run" => dry = true,
            "--json" => json_out = true,
            _ => return usage(io),
        }
    }
    let file = match std::fs::canonicalize(file) {
        Ok(f) => f,
        Err(e) => {
            tell!(io, "{file}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let base = root
        .and_then(|r| std::fs::canonicalize(r).ok())
        .unwrap_or_else(|| crate::workspace::project_root(&file));
    let files = crate::workspace::read_workspace(&base);
    let path = source_path(&file);
    let Some(text) = files
        .get(&path)
        .cloned()
        .or_else(|| std::fs::read_to_string(&file).ok())
    else {
        tell!(io, "{path}: can't read it");
        return ExitCode::FAILURE;
    };
    crate::kernel::install();
    let request = crate::commands::RunRequest {
        command: command.clone(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection,
    };
    let answer = crate::circuit_values::with_native_values(&files, || {
        crate::commands::run(&path, &text, &files, &request)
    });
    if json_out {
        say!(
            io,
            "{}",
            serde_json::to_string_pretty(&answer).unwrap_or_default()
        );
    }
    if answer["ok"] != true {
        tell!(
            io,
            "{command}: {}",
            answer["message"].as_str().unwrap_or("failed")
        );
        return ExitCode::FAILURE;
    }
    if json_out && dry {
        return ExitCode::SUCCESS;
    }
    match answer["kind"].as_str() {
        Some("edit") => {
            for change in answer["files"].as_array().into_iter().flatten() {
                let (Some(p), Some(t)) = (change["path"].as_str(), change["text"].as_str()) else {
                    continue;
                };
                if dry {
                    say!(io, "--- {p}\n{t}");
                } else {
                    let target = match physical_path(&base, p) {
                        Ok(target) => target,
                        Err(e) => {
                            tell!(io, "{p}: {e}");
                            return ExitCode::FAILURE;
                        }
                    };
                    if let Err(e) = std::fs::write(&target, t) {
                        tell!(io, "{}: {e}", target.display());
                        return ExitCode::FAILURE;
                    }
                    if !json_out {
                        say!(io, "wrote {}", target.display());
                    }
                }
            }
            if !json_out {
                say!(io, "{}", answer["summary"].as_str().unwrap_or(""));
            }
        }
        Some("table") if !json_out => {
            let cells = |row: &Value| -> Vec<String> {
                row.as_array()
                    .into_iter()
                    .flatten()
                    .map(|c| {
                        c.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| c.to_string())
                    })
                    .collect()
            };
            say!(
                io,
                "{}: {}",
                answer["title"].as_str().unwrap_or(""),
                answer["summary"].as_str().unwrap_or("")
            );
            say!(io, "{}", cells(&answer["columns"]).join("\t"));
            for row in answer["rows"].as_array().into_iter().flatten() {
                say!(io, "{}", cells(&row["cells"]).join("\t"));
            }
        }
        Some("file") if !dry => {
            let dir = out.unwrap_or_else(|| file.parent().unwrap_or(Path::new(".")).to_path_buf());
            let files = match answer.get("files") {
                Some(Value::Array(files)) if !files.is_empty() => files.as_slice(),
                _ => {
                    tell!(io, "{command}: file result has no valid output inventory");
                    return ExitCode::FAILURE;
                }
            };
            for f in files {
                let bytes = match (f["base64"].as_str(), f["text"].as_str()) {
                    (Some(b), _) => match base64::engine::general_purpose::STANDARD.decode(b) {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            tell!(io, "{command}: invalid output encoding: {e}");
                            return ExitCode::FAILURE;
                        }
                    },
                    (None, Some(t)) => t.as_bytes().to_vec(),
                    _ => {
                        tell!(io, "{command}: output has no file contents");
                        return ExitCode::FAILURE;
                    }
                };
                let Some(name) = f["name"].as_str() else {
                    tell!(io, "{command}: output has no filename");
                    return ExitCode::FAILURE;
                };
                let target = match output_path(&dir, Path::new(name)) {
                    Ok(target) => target,
                    Err(e) => {
                        tell!(io, "export: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                if let Some(parent) = target.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        tell!(io, "{}: {e}", parent.display());
                        return ExitCode::FAILURE;
                    }
                }
                if let Err(e) = std::fs::write(&target, bytes) {
                    tell!(io, "{}: {e}", target.display());
                    return ExitCode::FAILURE;
                }
                if !json_out {
                    say!(io, "{}", target.display());
                }
            }
        }
        _ => {}
    }
    ExitCode::SUCCESS
}

/// `runmat design <run|export> …`: `args` starts with the subcommand.
pub fn design(args: &[String], io: &mut Io) -> ExitCode {
    // Every command regenerates on the Desktop's kernel (OCCT, with the `occt` feature), including
    // the parts an assembly check or export regenerates through the runtime.
    crate::kernel::install();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..], io),
        Some("export") if args.len() >= 2 => export(&args[1..], io),
        _ => usage(io),
    }
}

fn export(args: &[String], io: &mut Io) -> ExitCode {
    let mut format = None;
    let (mut out, mut root, mut configuration, mut sheet): (
        Option<PathBuf>,
        Option<PathBuf>,
        Option<String>,
        Option<String>,
    ) = (None, None, None, None);
    let mut all = false;
    let (mut outputs, mut variants, mut tht): (Option<String>, Vec<String>, Option<bool>) =
        (None, Vec::new(), None);
    let (mut preset, mut layers, mut zip, mut remember): (
        Option<String>,
        Option<String>,
        Option<bool>,
        bool,
    ) = (None, None, None, false);
    // Output formats (fabrication): passed through as design.export's fabrication fields.
    let mut formats = serde_json::Map::new();
    // A board's STEP: `--copper` adds the outer copper, `--no-mask` leaves the mask out.
    let mut board = serde_json::Map::new();
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--gerber-units" => drop(formats.insert("gerberUnits".into(), json!(it.next()))),
            "--gerber-decimals" => drop(formats.insert(
                "gerberDecimals".into(),
                json!(it.next().and_then(|d| d.parse::<u8>().ok())),
            )),
            "--pick-place-sides" => drop(formats.insert("pickPlaceSides".into(), json!(it.next()))),
            "--pick-place-units" => drop(formats.insert("pickPlaceUnits".into(), json!(it.next()))),
            "--bom-columns" => drop(formats.insert(
                "bomColumns".into(),
                json!(it.next().map(|l| l.split(',').map(str::trim).collect::<Vec<_>>())),
            )),
            "--revision" => drop(formats.insert("revision".into(), json!(it.next()))),
            "--bom-ungrouped" => drop(formats.insert("bomGroupByValue".into(), json!(false))),
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
            "--copper" => drop(board.insert("copper".into(), Value::Bool(true))),
            "--no-mask" => drop(board.insert("mask".into(), Value::Bool(false))),
            _ => return usage(io),
        }
    }
    let Some(format) = format else {
        return usage(io);
    };
    let file = match std::fs::canonicalize(&args[0]) {
        Ok(f) => f,
        Err(e) => {
            tell!(io, "{}: {e}", args[0]);
            return ExitCode::FAILURE;
        }
    };
    let root = root
        .and_then(|r| std::fs::canonicalize(r).ok())
        .unwrap_or_else(|| crate::workspace::project_root(&file));
    let files = crate::workspace::read_workspace(&root);
    let mut input = json!({
        "path": source_path(&file),
        "format": format,
        "allConfigurations": all,
        "configuration": configuration,
        "sheet": sheet,
    });
    if !board.is_empty() {
        input["board"] = Value::Object(board);
    }
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
            let want: Vec<String> = list
                .split(',')
                .map(|w| w.trim().replace('_', "-"))
                .collect();
            if let Some(bad) = want.iter().find(|w| !ALL.iter().any(|(k, _)| k == w)) {
                tell!(io, "--outputs: unknown output {bad}");
                return usage(io);
            }
            for (k, field) in ALL {
                fab.insert(field.into(), Value::Bool(want.iter().any(|w| w == k)));
            }
        }
        if !variants.is_empty() {
            fab.insert("variants".into(), json!(variants));
        }
        if let Some(l) = &layers {
            fab.insert(
                "layers".into(),
                json!(l.split(',').map(str::trim).collect::<Vec<_>>()),
            );
        }
        if let Some(t) = tht {
            fab.insert("includeThroughHole".into(), Value::Bool(t));
        }
        if let Some(z) = zip {
            fab.insert("zip".into(), Value::Bool(z));
        }
        fab.extend(formats);
        fab.insert("remember".into(), Value::Bool(remember));
        input["fabrication"] = Value::Object(fab);
    }
    let result: Value = match crate::circuit_values::with_native_values(&files, || {
        call("design.export", input, &files)
    }) {
        Ok(v) => v,
        Err(e) => {
            tell!(io, "{e}");
            return ExitCode::FAILURE;
        }
    };
    // A fabrication set goes to its settings' folder beside the board unless --out names one.
    let out = out.unwrap_or_else(|| match result["fabrication"]["folder"].as_str() {
        Some(folder) => file.parent().unwrap_or(Path::new(".")).join(folder),
        None => PathBuf::from("."),
    });
    if let Some(update) = result["fabrication"]["layout"].as_object() {
        let (path, text) = (
            update["path"].as_str().unwrap_or(""),
            update["text"].as_str().unwrap_or(""),
        );
        let physical = match physical_path(&root, path) {
            Ok(path) => path,
            Err(e) => {
                tell!(io, "{path}: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = std::fs::write(physical, text) {
            tell!(io, "{path}: {e}");
            return ExitCode::FAILURE;
        }
        tell!(io, "saved the settings in {path}");
    }
    if let Err(e) = std::fs::create_dir_all(&out) {
        tell!(io, "{}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    for f in result["files"].as_array().into_iter().flatten() {
        let name = f["name"].as_str().unwrap_or("export");
        let content = f["content"].as_str().unwrap_or("");
        let bytes = if f["encoding"] == "base64" {
            base64::engine::general_purpose::STANDARD
                .decode(content)
                .unwrap_or_default()
        } else {
            content.as_bytes().to_vec()
        };
        let path = match output_path(&out, Path::new(name)) {
            Ok(path) => path,
            Err(e) => {
                tell!(io, "{name}: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Err(e) = std::fs::write(&path, bytes) {
            tell!(io, "{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
        say!(io, "{}", path.display());
    }
    for n in result["notes"].as_array().into_iter().flatten() {
        tell!(io, "note: {}", n.as_str().unwrap_or(""));
    }
    ExitCode::SUCCESS
}
