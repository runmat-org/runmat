//! How a `.circuit` or `.circuit.layout` names the circuits it uses: by symbol, through the
//! folder scope, as parts and assemblies name designs (report 12 §3).
//!
//! - `use circuit esc as board`: the circuit `esc` declared by a `.circuit` file in this folder;
//! - `use circuit power from "./power"`: one declared in the folder `./power`;
//! - `use sheet buck from "./power" as reg`: the sheet `buck` of whichever circuit in that
//!   folder declares it (an interface or net class is found the same way);
//! - `use circuit esc from hardware`: a `+hardware` package in this folder or one above it.
//!
//! A design named by its file (`from "./esc.circuit"`) is an error that says how to name it.
//! Library folders (`use library passives from "lib/passives"`) keep their paths.

use super::{resolve_path, Files, CIRCUIT_SUFFIX};
use crate::syntax::parse_circuit;

/// The folder holding `path` (`""` for a top-level relative file).
fn folder_of(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some(("", _)) if path.starts_with('/') => "/",
        Some((d, _)) => d,
        None => "",
    }
}

fn join(folder: &str, rel: &str) -> String {
    match folder {
        "" => resolve_path("x", rel),
        "/" => resolve_path("/x", rel),
        f => resolve_path(&format!("{f}/x"), rel),
    }
}

/// Whether `from` names a folder by path rather than an alias or package.
fn is_path(from: &str) -> bool {
    from.starts_with('.') || from.starts_with('/') || from.contains('/')
}

/// Whether `from` names one design file (the refused file form).
pub fn is_file_form(from: &str) -> bool {
    let lower = from.to_ascii_lowercase();
    [
        CIRCUIT_SUFFIX,
        ".circuit.layout",
        ".part",
        ".assembly",
        ".drawing",
    ]
    .iter()
    .any(|ext| lower.ends_with(ext))
}

/// The error for a `use <kind> <name> from "<file>"` line: the symbol form to write instead.
pub fn file_form_message(kind: &str, name: &str, from: &str) -> String {
    let folder = folder_of(from).trim_start_matches("./");
    let module = if folder.is_empty() || folder == "." {
        String::new()
    } else {
        format!(" from \"./{folder}\"")
    };
    format!("name the design: write `use {kind} {name}{module} as …`, not the file \"{from}\"")
}

/// The folders, nearest first, where `from` (as written in the file `doc`) looks: the file's
/// own folder for `""`, the folder a path names, or each `+pkg` folder from `doc`'s folder up.
pub fn folders(doc: &str, from: &str) -> Vec<String> {
    let here = folder_of(doc);
    if from.is_empty() {
        return vec![here.to_owned()];
    }
    if is_path(from) {
        let target = if from.starts_with('/') {
            resolve_path("/x", from)
        } else {
            join(here, from)
        };
        return vec![target];
    }
    let package: Vec<String> = from.split('.').map(|s| format!("+{s}")).collect();
    let package = package.join("/");
    let mut out = Vec::new();
    let mut at = here.to_owned();
    loop {
        out.push(if at.is_empty() {
            package.clone()
        } else {
            join(&at, &package)
        });
        if at.is_empty() || at == "/" {
            break;
        }
        at = folder_of(&at).to_owned();
    }
    out
}

/// The folder listings (`<folder>/`) a host reads so the uses of `doc` can resolve.
pub fn folder_dependencies(doc: &str, from: &str) -> Vec<String> {
    folders(doc, from)
        .into_iter()
        .map(|f| if f.ends_with('/') { f } else { format!("{f}/") })
        .collect()
}

/// Whether the circuit file `text` declares `circuit name`: a line-start scan, so a large
/// circuit isn't parsed on every compile of its layout.
fn declares_circuit(text: &str, name: &str) -> bool {
    text.lines().any(|line| {
        line.trim_start()
            .strip_prefix("circuit")
            .and_then(|rest| rest.strip_prefix(char::is_whitespace))
            .map(str::trim_start)
            .and_then(|rest| rest.strip_prefix(name))
            .is_some_and(|rest| {
                rest.is_empty() || rest.starts_with(|c: char| c.is_whitespace() || c == '{')
            })
    })
}

/// Whether the circuit file `path` declares the `kind name` a `use` line asks for.
fn declares(path: &str, text: &str, kind: &str, name: &str) -> bool {
    if kind == "circuit" {
        return declares_circuit(text, name);
    }
    let Some(design) = parse_circuit(path, text).design else {
        return false;
    };
    match kind {
        "sheet" => design.sheets.contains_key(name),
        "interface" => design.interfaces.contains_key(name),
        "net_class" => design.net_classes.contains_key(name),
        _ => false,
    }
}

/// The `.circuit` file that declares `kind name`, seen from `doc` through `from` (`""` for the
/// file's own folder), among `files`.
pub fn locate(
    doc: &str,
    kind: &str,
    name: &str,
    from: &str,
    files: &Files,
) -> Result<String, String> {
    if is_file_form(from) {
        return Err(file_form_message(kind, name, from));
    }
    for folder in folders(doc, from) {
        let found: Vec<&String> = files
            .iter()
            .filter(|(p, _)| p.as_str() != doc)
            .filter(|(p, _)| {
                p.to_ascii_lowercase().ends_with(CIRCUIT_SUFFIX) && folder_of(p) == folder
            })
            .filter(|(p, t)| declares(p, t, kind, name))
            .map(|(p, _)| p)
            .collect();
        match found.as_slice() {
            [one] => return Ok((*one).clone()),
            [a, b, ..] => {
                return Err(format!(
                    "`{kind} {name}` is declared twice in one folder: {a} and {b}"
                ))
            }
            [] => {}
        }
    }
    let place = if from.is_empty() {
        "this folder".to_owned()
    } else {
        format!("`{from}`")
    };
    Err(format!("no `{kind} {name}` in {place}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(entries: &[(&str, &str)]) -> Files {
        entries
            .iter()
            .map(|(p, t)| (p.to_string(), t.to_string()))
            .collect()
    }

    #[test]
    fn a_circuit_is_found_by_its_declaration_in_the_named_folder() {
        let f = files(&[
            (
                "/p/esc/power_stage.circuit",
                "runmat 0.7\n// circuit esc is the other file\ncircuit esc_power {\n}\n",
            ),
            (
                "/p/esc/main.circuit",
                "runmat 0.7\nstd 1\n\ncircuit esc {\n}\n",
            ),
            ("/p/+parts/filter.circuit", "circuit rc_filter {\n}\n"),
        ]);
        assert_eq!(
            locate("/p/esc/esc.circuit.layout", "circuit", "esc", "", &f),
            Ok("/p/esc/main.circuit".into())
        );
        assert_eq!(
            locate("/p/top.circuit", "circuit", "esc", "./esc", &f),
            Ok("/p/esc/main.circuit".into())
        );
        // A package is searched for from the file's folder up.
        assert_eq!(
            locate("/p/esc/main.circuit", "circuit", "rc_filter", "parts", &f),
            Ok("/p/+parts/filter.circuit".into())
        );
        assert_eq!(
            locate("/p/top.circuit", "circuit", "esc", "", &f),
            Err("no `circuit esc` in this folder".into())
        );
        assert_eq!(
            folder_dependencies("/p/esc/main.circuit", "parts"),
            ["/p/esc/+parts/", "/p/+parts/", "/+parts/"]
        );
        assert!(locate(
            "/p/a.circuit.layout",
            "circuit",
            "esc",
            "./esc/main.circuit",
            &f
        )
        .unwrap_err()
        .starts_with("name the design"));
    }
}
