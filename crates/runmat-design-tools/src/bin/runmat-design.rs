//! `runmat-design`: the design tools from the command line, for scripts and batch jobs.
//!
//! ```text
//! runmat-design export <file> --format <step|3mf|stl|gltf|dxf|dwg|bom|kicad>
//!     [--all-configurations] [--configuration <name>] [--sheet <name>] [--out <dir>]
//! ```
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

const DESIGN: &[&str] = &[".part", ".assembly", ".drawing", ".circuit", ".circuit.layout", ".circuit.lock", ".step", ".stp"];

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
    eprintln!("usage: runmat-design export <file> --format <step|3mf|stl|gltf|dxf|dwg|bom|kicad> [--all-configurations] [--configuration <name>] [--sheet <name>] [--root <dir>] [--out <dir>]");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("export") || args.len() < 2 {
        return usage();
    }
    let mut format = None;
    let (mut out, mut root, mut configuration, mut sheet) = (PathBuf::from("."), None, None, None);
    let mut all = false;
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--format" => format = it.next().cloned(),
            "--out" => out = it.next().map(PathBuf::from).unwrap_or(out),
            "--root" => root = it.next().map(PathBuf::from),
            "--configuration" => configuration = it.next().cloned(),
            "--sheet" => sheet = it.next().cloned(),
            "--all-configurations" => all = true,
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
    let input = json!({
        "path": file.to_string_lossy(),
        "format": format,
        "allConfigurations": all,
        "configuration": configuration,
        "sheet": sheet,
    });
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
