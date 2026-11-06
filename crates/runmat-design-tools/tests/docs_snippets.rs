//! Every design-language example in the design docs (`docs/design`) checks through the
//! compilers, the way `design.check` checks a file in the editor.
//!
//! A fence's language is the file extension: ```` ```part ````, ```` ```assembly ````,
//! ```` ```drawing ````, ```` ```circuit ````, ```` ```circuit.layout ````. Attributes after it:
//!
//! - `file=<name>`: the block is that file of the page's workspace (`/docs/<page>/<name>`), and
//!   every later block on the page sees it. An assembly can then place a part written above it.
//!   Without `file=`, a block is `<its declared name>.<ext>` in a folder of its own, seen only by
//!   itself.
//! - `fragment`: a few statements, not a whole file. The block is wrapped in the version header
//!   and a `<kind> fragment { … }` declaration (unless it has its own) and must parse into the
//!   model with no diagnostics; its references are not resolved. With `pending=<id>` it must not
//!   parse yet.
//! - `pending=<id>`: the example uses a feature not in the current release (an id in
//!   docs/design/ledger.json). It must fail its check; once it passes, remove the flag.
//! - `fails`: the example shows a diagnostic on purpose. It must fail its check.
//! - `m`: a value in the example runs `.m` code. It is checked when the test is built with
//!   `--features m-values` (a RunMat session runs the functions) and counted otherwise.
//! - `occt`: the geometry needs the editor's kernel (curved faces, blends of curved edges). It is
//!   checked when the test is built with `--features occt` and counted otherwise; without the
//!   feature, parts regenerate on the reference kernel, which builds planar geometry.
//!
//! The full check is `cargo test -p runmat-design-tools --features occt,m-values --test docs_snippets`.
//!
//! Any other fence with `file=<name>` (```` ```matlab file=camBaseRadius.m ````, a
//! `runmat.toml`) joins the page's workspace without being checked itself, so the designs that
//! read it can.
//!
//! Set `DESIGN_DOCS_ONLY=<path prefix>` to check one page, and `DESIGN_DOCS_VERBOSE=1` to print
//! every block's verdict and diagnostics.

use runmat_design_tools::Files;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const LANGUAGES: &[&str] = &["part", "assembly", "drawing", "circuit", "circuit.layout"];

struct Block {
    page: String,
    line: usize,
    language: String,
    file: Option<String>,
    fragment: bool,
    pending: Option<String>,
    fails: bool,
    m: bool,
    occt: bool,
    /// A non-design fence (`matlab`, `toml`) with `file=`: workspace content only.
    support: bool,
    code: String,
}

fn docs_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design")
}

fn markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            markdown_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "md") {
            out.push(path);
        }
    }
}

fn blocks(page: &str, text: &str) -> Vec<Block> {
    let mut found = Vec::new();
    let mut open: Option<(String, usize, Vec<String>)> = None;
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        let is_fence = trimmed.starts_with("```");
        match open.take() {
            None if is_fence => open = Some((trimmed[3..].trim().to_owned(), index + 1, Vec::new())),
            None => {}
            Some((info, start, code)) if is_fence && trimmed[3..].trim().is_empty() => {
                let mut words = info.split_whitespace();
                let language = words.next().unwrap_or_default().to_owned();
                let support = !LANGUAGES.contains(&language.as_str());
                if support && !info.split_whitespace().any(|w| w.starts_with("file=")) {
                    continue;
                }
                let mut block = Block {
                    page: page.to_owned(),
                    line: start,
                    language,
                    file: None,
                    fragment: false,
                    pending: None,
                    fails: false,
                    m: false,
                    occt: false,
                    support,
                    code: code.join("\n") + "\n",
                };
                for word in words {
                    match word.split_once('=') {
                        Some(("file", name)) => block.file = Some(name.to_owned()),
                        Some(("pending", id)) => block.pending = Some(id.to_owned()),
                        None if word == "fragment" => block.fragment = true,
                        None if word == "fails" => block.fails = true,
                        None if word == "m" => block.m = true,
                        None if word == "occt" => block.occt = true,
                        _ => panic!("{page}:{start}: unknown fence attribute `{word}`"),
                    }
                }
                found.push(block);
            }
            Some((info, start, mut code)) => {
                code.push(line.to_owned());
                open = Some((info, start, code));
            }
        }
    }
    assert!(open.is_none(), "{page}: unclosed code fence");
    found
}

/// `design.check`, with a RunMat session running `.m` values when the feature is on.
fn check(file: &str, files: &Files) -> Result<Value, runmat_design_tools::ToolError> {
    #[cfg(feature = "m-values")]
    return runmat_design_tools::inspect::m_values::call_with_m_values("design.check", json!({ "path": file }), files);
    #[cfg(not(feature = "m-values"))]
    runmat_design_tools::call("design.check", json!({ "path": file }), files)
}

fn errors(out: &Value) -> Vec<String> {
    out["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .map(|d| {
            let line = d["range"]["start"]["line"].as_u64().map_or(String::new(), |l| format!("line {}: ", l + 1));
            format!("{line}{}", d["message"].as_str().unwrap_or_default())
        })
        .collect()
}

/// The name a block's declaration gives (`part plate {` is `plate`); a file is named after it.
fn declared_name(block: &Block) -> String {
    let word = if block.language == "circuit.layout" { "layout" } else { block.language.as_str() };
    block
        .code
        .lines()
        .find_map(|line| line.trim_start().strip_prefix(&format!("{word} ")))
        .and_then(|rest| rest.split_whitespace().next())
        .unwrap_or("snippet")
        .to_owned()
}

/// A fragment as a whole file: the header and its declaration around it, unless it has them.
fn fragment_source(block: &Block) -> String {
    let code = block.code.trim_start();
    let declaration = match block.language.as_str() {
        "circuit.layout" => "layout",
        other => other,
    };
    if code.starts_with("runmat ") {
        return block.code.clone();
    }
    let body = if code.starts_with(&format!("{declaration} ")) {
        block.code.clone()
    } else {
        let indented: Vec<String> = block.code.lines().map(|l| if l.is_empty() { String::new() } else { format!("    {l}") }).collect();
        format!("{declaration} fragment {{\n{}\n}}\n", indented.join("\n"))
    };
    format!("runmat 0.7\nstd 1\n\n{body}")
}

fn parse_fragment(block: &Block, folder: &str, workspace: &Files) -> Vec<String> {
    let source = fragment_source(block);
    let path = format!("{folder}/fragment.{}", block.language);
    let messages = |d: &[runmat_design_syntax::FeaSyntaxDiagnostic]| -> Vec<String> {
        d.iter().map(|d| {
            let line = source[..d.span.start.min(source.len())].matches('\n').count() + 1;
            format!("line {line}: {}", d.message)
        }).collect()
    };
    let scope = || {
        runmat_cad::modules::ModuleScope::from_files(
            workspace.iter().map(|(p, t)| (p.as_str(), t.as_str())).chain(std::iter::once((path.as_str(), source.as_str()))),
        )
    };
    match block.language.as_str() {
        "part" => {
            let parsed = runmat_cad::syntax::parse_part(&path, &source);
            let mut out = messages(&parsed.diagnostics);
            if out.is_empty() && parsed.document.is_none() {
                out.push("no part model".into());
            }
            out
        }
        "assembly" => {
            let parsed = runmat_cad::syntax::parse_assembly_in(&path, &source, &scope());
            let mut out = messages(&parsed.diagnostics);
            if out.is_empty() && parsed.document.is_none() {
                out.push("no assembly model".into());
            }
            out
        }
        "drawing" => {
            let parsed = runmat_cad::syntax::parse_drawing_in(&path, &source, &scope());
            let mut out = messages(&parsed.diagnostics);
            if out.is_empty() && parsed.document.is_none() {
                out.push("no drawing model".into());
            }
            out
        }
        "circuit" => {
            let parsed = runmat_circuit::syntax::parse_circuit(&path, &source);
            let mut out = messages(&parsed.diagnostics);
            if out.is_empty() && parsed.design.is_none() {
                out.push("no circuit model".into());
            }
            out
        }
        _ => {
            let parsed = runmat_circuit::syntax::board::parse_board(&path, &source);
            messages(&parsed.diagnostics)
        }
    }
}

#[test]
fn every_design_docs_example_checks() {
    let root = docs_root();
    let mut pages = Vec::new();
    markdown_files(&root, &mut pages);
    let only = std::env::var("DESIGN_DOCS_ONLY").ok();
    let verbose = std::env::var("DESIGN_DOCS_VERBOSE").is_ok();
    let (mut checked, mut fragments, mut skipped, mut failures) = (0usize, 0usize, 0usize, Vec::new());
    for path in pages {
        let page = path.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
        if only.as_deref().is_some_and(|prefix| !page.starts_with(prefix)) {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let folder = format!("/docs/{}", page.trim_end_matches(".md"));
        let mut workspace = Files::new();
        for (n, block) in blocks(&page, &text).into_iter().enumerate() {
            if block.support {
                workspace.insert(format!("{folder}/{}", block.file.as_deref().unwrap()), block.code.clone());
                continue;
            }
            if block.fragment {
                fragments += 1;
                let problems = parse_fragment(&block, &folder, &workspace);
                if verbose {
                    println!("{}:{} {} fragment {}", block.page, block.line, block.language, if problems.is_empty() { "parses".into() } else { problems.join("; ") });
                }
                match (&block.pending, problems.is_empty()) {
                    (None, false) => failures.push(format!("{}:{}: fragment does not parse:\n    {}", block.page, block.line, problems.join("\n    "))),
                    (Some(id), true) => failures.push(format!("{}:{}: fragment parses now; remove pending={id} and mark the ledger row landed", block.page, block.line)),
                    _ => {}
                }
                continue;
            }
            let file = match &block.file {
                Some(name) => format!("{folder}/{name}"),
                None => format!("{folder}/snippet{n}/{}.{}", declared_name(&block), block.language),
            };
            let mut files = if block.file.is_some() { workspace.clone() } else { Files::new() };
            files.insert(file.clone(), block.code.clone());
            if block.file.is_some() {
                workspace.insert(file.clone(), block.code.clone());
            }
            if (block.m && !cfg!(feature = "m-values")) || (block.occt && !cfg!(feature = "occt")) {
                skipped += 1;
                continue;
            }
            let out = match check(&file, &files) {
                Ok(out) => out,
                Err(e) => json!({ "diagnostics": [{ "severity": "error", "message": e.to_string() }] }),
            };
            let errs = errors(&out);
            let expect_fail = block.pending.is_some() || block.fails;
            checked += 1;
            if verbose {
                println!("{}:{} {} {}", block.page, block.line, block.language, if errs.is_empty() { "clean".into() } else { errs.join("; ") });
            }
            match (expect_fail, errs.is_empty()) {
                (false, false) => failures.push(format!("{}:{}: does not check clean:\n    {}", block.page, block.line, errs.join("\n    "))),
                (true, true) if block.pending.is_some() => failures.push(format!(
                    "{}:{}: checks clean now; remove pending={} and mark the ledger row landed",
                    block.page,
                    block.line,
                    block.pending.as_deref().unwrap()
                )),
                (true, true) => failures.push(format!("{}:{}: marked `fails` but checks clean", block.page, block.line)),
                _ => {}
            }
        }
    }
    println!(
        "design docs: {checked} examples checked, {fragments} fragments, {skipped} left for --features occt,m-values, {} failures",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    if only.is_none() {
        assert!(checked > 0, "no design-language examples found under docs/design");
    }
}
