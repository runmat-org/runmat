//! Library `.circuit` files and `use "<path>.circuit"`.
//!
//! A `.circuit` with no `circuit` block is a library of `footprint`, `symbol` and `part`
//! declarations. `use "lib/sensors.circuit"` brings its declarations into the using file's
//! scope by name (parts by MPN, footprints and symbols by identifier), and a library may use
//! other libraries. Resolution is depth-first in `use` order, each file read once, so the same
//! files always give the same scope.
//!
//! A name declared in two files is an error that names both; a library reached twice through
//! different paths is the same file, not a second declaration. A cycle of `use` lines is an
//! error listing the chain, and a file that can't be read is an error naming its path.

use super::{resolve_path, Files};
use crate::ir::{CircuitUse, Design};
use crate::library::catalog::{package_in, Declared};
use crate::library::checks;
use crate::syntax::parse_circuit;
use std::collections::BTreeMap;

/// A problem resolving libraries, reported in `file` at the byte range of the `use` line's
/// path (or at the start when it has none).
#[derive(Clone, Debug, PartialEq)]
pub struct LibraryError {
    pub file: String,
    pub start: usize,
    pub end: usize,
    pub message: String,
}

/// Where each declaration in scope comes from: `(kind, name)` → file.
pub type Origins = BTreeMap<(String, String), String>;

/// The libraries a file uses, transitively, each with its parsed declarations, in resolution
/// order (a library before the file that uses it).
fn gather(
    doc: &str,
    text: &str,
    design: &Design,
    files: &Files,
    stack: &mut Vec<String>,
    done: &mut Vec<String>,
    out: &mut Vec<(String, Design)>,
) -> Result<(), LibraryError> {
    for u in design.uses.iter().filter(|u| u.is_library_file()) {
        let path = resolve_path(doc, &u.from);
        let at = use_span(text, u);
        let err = |message: String| LibraryError {
            file: doc.to_owned(),
            start: at.0,
            end: at.1,
            message,
        };
        if let Some(i) = stack.iter().position(|p| *p == path) {
            let mut chain: Vec<&str> = stack[i..].iter().map(String::as_str).collect();
            chain.push(&path);
            return Err(err(format!(
                "these libraries use each other in a cycle: {}",
                chain.join(" → ")
            )));
        }
        if done.contains(&path) {
            continue;
        }
        let Some(lib_text) = files.get(&path) else {
            return Err(err(format!(
                "the library \"{}\" can't be read: no file {path}",
                u.from
            )));
        };
        let parsed = parse_circuit(&path, lib_text);
        let Some(lib) = parsed.design else {
            let first = parsed
                .diagnostics
                .first()
                .map(|d| d.message.clone())
                .unwrap_or_default();
            return Err(err(format!(
                "the library \"{}\" has errors: {first}",
                u.from
            )));
        };
        if !lib.library {
            return Err(err(format!(
                "\"{}\" is a design (it has a `circuit` block); `use \"….circuit\"` takes a library of footprints, symbols and parts",
                u.from
            )));
        }
        stack.push(path.clone());
        gather(&path, lib_text, &lib, files, stack, done, out)?;
        stack.pop();
        done.push(path.clone());
        out.push((path, lib));
    }
    Ok(())
}

/// The byte range of a `use` line's quoted path in `text`.
fn use_span(text: &str, u: &CircuitUse) -> (usize, usize) {
    let quoted = format!("\"{}\"", u.from);
    text.find(&quoted)
        .map(|i| (i, i + quoted.len()))
        .unwrap_or((0, 0))
}

/// `design` (the file `doc`, written `text`) with the declarations of every library it uses
/// brought into scope, and where each declaration comes from. `files` holds the libraries.
pub fn resolve(
    doc: &str,
    text: &str,
    design: &Design,
    files: &Files,
) -> Result<(Design, Origins), LibraryError> {
    let mut used = Vec::new();
    gather(
        doc,
        text,
        design,
        files,
        &mut vec![doc.to_owned()],
        &mut Vec::new(),
        &mut used,
    )?;
    let mut out = design.clone();
    let mut origins = Origins::new();
    let claim = |kind: &str, name: &str, file: &str, origins: &mut Origins| -> Result<(), String> {
        match origins.insert((kind.to_owned(), name.to_owned()), file.to_owned()) {
            Some(other) if other != file => Err(format!(
                "{kind} `{name}` is declared in both {other} and {file}; rename one or remove one"
            )),
            _ => Ok(()),
        }
    };
    let own_err = |message: String| LibraryError {
        file: doc.to_owned(),
        start: 0,
        end: 0,
        message,
    };
    for name in design.footprints.keys() {
        claim("footprint", name, doc, &mut origins).map_err(own_err)?;
    }
    for name in design.symbols.keys() {
        claim("symbol", name, doc, &mut origins).map_err(own_err)?;
    }
    for name in design.parts.keys() {
        claim("part", name, doc, &mut origins).map_err(own_err)?;
    }
    for (path, lib) in used {
        let at = |message: String| {
            let span = design
                .uses
                .iter()
                .find(|u| u.is_library_file())
                .map(|u| use_span(text, u))
                .unwrap_or((0, 0));
            LibraryError {
                file: doc.to_owned(),
                start: span.0,
                end: span.1,
                message,
            }
        };
        for (name, f) in lib.footprints {
            claim("footprint", &name, &path, &mut origins).map_err(at)?;
            out.footprints.insert(name, f);
        }
        for (name, s) in lib.symbols {
            claim("symbol", &name, &path, &mut origins).map_err(at)?;
            out.symbols.insert(name, s);
        }
        for (name, p) in lib.parts {
            claim("part", &name, &path, &mut origins).map_err(at)?;
            out.parts.insert(name, p);
        }
    }
    Ok((out, origins))
}

/// The checks a library passes on its own, for the declarations the file `doc` holds (those
/// it uses are checked in their own files): footprints, then parts on their footprints.
/// `scope` is the file's design with its used libraries resolved.
pub fn library_findings(doc: &str, own: &Design, scope: &Design, files: &Files) -> Vec<String> {
    let mut out = Vec::new();
    for (name, f) in &own.footprints {
        out.extend(checks::footprint_findings(name, f));
    }
    let file = |p: &str| files.get(&resolve_path(doc, p)).cloned();
    let declared = Declared {
        footprints: Some(&scope.footprints),
        symbols: Some(&scope.symbols),
    };
    for part in own.parts.values() {
        match package_in(part, declared, &file) {
            Ok(package) => out.extend(checks::part_findings(part, &package)),
            Err(e) => out.push(e),
        }
        if let Some(s) = &part.symbol {
            match scope.symbols.get(s) {
                Some(symbol) => out.extend(checks::pins_match_symbol(part, s, symbol).err()),
                None => out.push(format!(
                    "part `{}`: no symbol `{s}` is declared here or in a library this file uses",
                    part.mpn
                )),
            }
        }
    }
    out
}

/// The library files `doc`'s `use` lines name, resolved to paths, for the host to read.
pub fn library_dependencies(doc: &str, design: &Design) -> Vec<String> {
    design
        .uses
        .iter()
        .filter(|u| u.is_library_file())
        .map(|u| resolve_path(doc, &u.from))
        .collect()
}

#[cfg(test)]
#[path = "libraries_tests.rs"]
mod tests;
