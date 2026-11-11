//! A part or an assembly as a simulation study's geometry source.
//!
//! A study that names `file = "./arm.part"` (or an `.assembly`) gets its exact geometry from one
//! regeneration on the installed kernel: [`study_geometry`] regenerates the design and writes the
//! exact B-rep as STEP AP242 in memory, the same artifact the design's STEP export is. Its
//! regions travel as the entity names of that artifact: every face of a face region carries the
//! region's name (or `label`), and every output body is a product under its output's name. The
//! study reads those names exactly as it reads the names of an imported STEP file, so the
//! geometry and its regions come from the same evaluation and cannot disagree.
//!
//! The bytes are deterministic. The writer's timestamp is replaced by a fixed one, and the
//! header's description carries the SHA-256 of the design's source closure (the file and every
//! part, sub-assembly and imported STEP file it reads), so the study's geometry identity changes
//! when any of those sources changes and stays the same when none does.

use crate::runtime::{dependency_files, Files};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// The design file kinds a study can read as geometry.
pub fn is_study_design(path: &str) -> bool {
    path.ends_with(".part") || path.ends_with(".assembly")
}

/// The exact STEP AP242 bytes of the part or assembly at `path`, regenerated from `files` (which
/// hold it and every file it reads) in its named `configuration` (the default when `None`).
pub fn study_geometry(
    path: &str,
    files: &Files,
    configuration: Option<&str>,
) -> Result<Vec<u8>, String> {
    let text = files
        .get(path)
        .ok_or_else(|| format!("`{path}` is not among the design files read"))?;
    let step = if path.ends_with(".assembly") {
        super::assembly_step_in(path, text, files, configuration)?
    } else if path.ends_with(".part") {
        super::part_step_in(path, text, configuration)?
    } else {
        return Err(format!("`{path}` is not a .part or .assembly file"));
    };
    let step = step.ok_or_else(|| {
        if path.ends_with(".assembly") {
            format!("`{path}` places no part, or the installed design kernel writes no exact STEP")
        } else {
            format!("the installed design kernel writes no exact STEP for `{path}`")
        }
    })?;
    let text = String::from_utf8(step).map_err(|_| "the kernel's STEP file is not text")?;
    Ok(deterministic_header(&text, &source_closure_digest(path, files)).into_bytes())
}

/// Every file `path` reads, transitively, that `files` holds, `path` first.
pub fn source_closure(path: &str, files: &Files) -> Vec<String> {
    let mut seen = BTreeSet::from([path.to_owned()]);
    let mut order = vec![path.to_owned()];
    let mut next = 0;
    while next < order.len() {
        let current = order[next].clone();
        next += 1;
        let Some(text) = files.get(&current) else {
            continue;
        };
        if !(is_study_design(&current)) {
            continue;
        }
        for dependency in dependency_files(&current, text, files) {
            if files.contains_key(&dependency) && seen.insert(dependency.clone()) {
                order.push(dependency);
            }
        }
    }
    order
}

/// SHA-256 over the source closure: each file's path relative to `path`'s folder and the
/// SHA-256 of its text, in path order. Moving the project folder leaves it unchanged.
pub fn source_closure_digest(path: &str, files: &Files) -> String {
    let mut entries: Vec<(String, String)> = source_closure(path, files)
        .into_iter()
        .filter_map(|file| {
            let text = files.get(&file)?;
            Some((
                relative_to(folder(path), &file),
                hex::encode(Sha256::digest(text.as_bytes())),
            ))
        })
        .collect();
    entries.sort();
    let mut hasher = Sha256::new();
    for (file, digest) in entries {
        hasher.update(file.as_bytes());
        hasher.update([0]);
        hasher.update(digest.as_bytes());
        hasher.update([b'\n']);
    }
    hex::encode(hasher.finalize())
}

fn folder(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(folder, _)| folder)
}

/// `file` relative to `base` (both slash-separated, normalized).
fn relative_to(base: &str, file: &str) -> String {
    let base: Vec<&str> = base.split('/').filter(|s| !s.is_empty()).collect();
    let target: Vec<&str> = file.split('/').filter(|s| !s.is_empty()).collect();
    let common = base
        .iter()
        .zip(&target)
        .take_while(|(a, b)| a == b)
        .count();
    let mut out: Vec<&str> = vec![".."; base.len() - common];
    out.extend(&target[common..]);
    out.join("/")
}

/// The STEP text with a fixed `FILE_NAME` timestamp and a `FILE_DESCRIPTION` naming the source
/// closure's digest. The data section is untouched.
fn deterministic_header(step: &str, closure: &str) -> String {
    let Some(end) = step.find("ENDSEC;") else {
        return step.to_owned();
    };
    let (header, data) = step.split_at(end);
    let header = replace_statement(header, "FILE_DESCRIPTION(", |_| {
        format!("FILE_DESCRIPTION(('RunMat study geometry','sources sha256:{closure}'),'2;1');")
    });
    let header = replace_statement(&header, "FILE_NAME(", |statement| {
        replace_quoted(statement, 1, "1970-01-01T00:00:00")
    });
    format!("{header}{data}")
}

/// `text` with the statement starting at `start` (through its `;`) replaced by `with(statement)`.
fn replace_statement(text: &str, start: &str, with: impl Fn(&str) -> String) -> String {
    let Some(at) = text.find(start) else {
        return text.to_owned();
    };
    let Some(len) = statement_len(&text[at..]) else {
        return text.to_owned();
    };
    format!(
        "{}{}{}",
        &text[..at],
        with(&text[at..at + len]),
        &text[at + len..]
    )
}

/// The length of the statement at the start of `text` through its terminating `;`, skipping
/// quoted strings (STEP doubles a quote inside a string).
fn statement_len(text: &str) -> Option<usize> {
    let mut quoted = false;
    for (i, c) in text.char_indices() {
        match c {
            '\'' => quoted = !quoted,
            ';' if !quoted => return Some(i + 1),
            _ => {}
        }
    }
    None
}

/// `statement` with its `index`th quoted string (0-based) replaced by `value`.
fn replace_quoted(statement: &str, index: usize, value: &str) -> String {
    let mut seen = 0;
    let mut open = None;
    for (i, c) in statement.char_indices() {
        if c != '\'' {
            continue;
        }
        match open {
            None => open = Some(i),
            Some(start) => {
                // A doubled quote is an escaped quote inside the string.
                if statement[i + 1..].starts_with('\'') {
                    continue;
                }
                if seen == index {
                    return format!("{}'{value}'{}", &statement[..start], &statement[i + 1..]);
                }
                seen += 1;
                open = None;
            }
        }
    }
    statement.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('Open CASCADE Model'),'2;1');\nFILE_NAME('Open CASCADE Shape Model','2026-11-07T09:25:36',('Author'),(\n    'Open CASCADE'),'Open CASCADE STEP processor 7.8','Open CASCADE 7.8'\n  ,'Unknown');\nFILE_SCHEMA(('X'));\nENDSEC;\nDATA;\n#1 = PRODUCT('a;b','2026-11-07T09:25:36','',(#2));\nENDSEC;\n";

    #[test]
    fn header_timestamp_is_fixed_and_names_the_source_closure() {
        let out = deterministic_header(HEADER, "abc");
        assert!(out.contains("FILE_DESCRIPTION(('RunMat study geometry','sources sha256:abc'),'2;1');"));
        assert!(out.contains("FILE_NAME('Open CASCADE Shape Model','1970-01-01T00:00:00',"));
        // The data section is untouched, timestamp-like strings included.
        assert!(out.contains("#1 = PRODUCT('a;b','2026-11-07T09:25:36','',(#2));"));
        // A later write (the header's timestamp comes first) gives the same bytes.
        let later = HEADER.replacen("2026-11-07T09:25:36", "2026-11-08T11:11:11", 1);
        assert_ne!(later, HEADER);
        assert_eq!(deterministic_header(&later, "abc"), out);
    }

    #[test]
    fn the_source_closure_follows_an_assemblys_parts_and_a_parts_step_imports() {
        let mut files = Files::new();
        files.insert(
            "/p/sim/study.assembly".into(),
            "runmat 0.7\nstd 1\n\nassembly study {\n    use part arm from \"../mech/arm\" as arm\n}\n".into(),
        );
        files.insert(
            "/p/mech/arm.part".into(),
            "runmat 0.7\nstd 1\n\npart arm {\n    feature s = import(source = \"./arm.step\")\n}\n".into(),
        );
        files.insert("/p/mech/arm.step".into(), "ISO-10303-21;".into());
        files.insert("/p/mech/other.part".into(), "runmat 0.7\n".into());
        let closure = source_closure("/p/sim/study.assembly", &files);
        assert_eq!(closure[0], "/p/sim/study.assembly");
        assert!(closure.contains(&"/p/mech/arm.part".to_owned()), "{closure:?}");
        assert!(closure.contains(&"/p/mech/arm.step".to_owned()), "{closure:?}");
        assert!(!closure.contains(&"/p/mech/other.part".to_owned()));
        let before = source_closure_digest("/p/sim/study.assembly", &files);
        // Unrelated files leave the identity alone; a dependency's edit changes it.
        files.insert("/p/mech/other.part".into(), "runmat 0.7\n// edited\n".into());
        assert_eq!(before, source_closure_digest("/p/sim/study.assembly", &files));
        files.insert("/p/mech/arm.step".into(), "ISO-10303-21; edited".into());
        assert_ne!(before, source_closure_digest("/p/sim/study.assembly", &files));
        // Moving the project keeps the identity.
        let moved: Files = files
            .iter()
            .map(|(k, v)| (k.replacen("/p/", "/q/r/", 1), v.clone()))
            .collect();
        assert_eq!(
            source_closure_digest("/p/sim/study.assembly", &files),
            source_closure_digest("/q/r/sim/study.assembly", &moved)
        );
    }

    #[test]
    fn relative_paths_climb_out_of_the_study_folder() {
        assert_eq!(relative_to("/p/sim", "/p/mech/arm.part"), "../mech/arm.part");
        assert_eq!(relative_to("/p/sim", "/p/sim/a.part"), "a.part");
    }
}
