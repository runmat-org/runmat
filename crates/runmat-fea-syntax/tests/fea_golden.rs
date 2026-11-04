//! The `.fea` behaviour lock for the lossless-tree migration (M1).
//!
//! Every `.fea` source in the repository (example and fixture files, and every ```fea block in
//! the Markdown docs) is parsed and formatted, and three facts are compared with the snapshot in
//! `tests/golden/fea-sources.json`, which was recorded from the token parser before the
//! lossless tree replaced it:
//!
//! - the parsed module (its items and token stream), as a digest of their `Debug` forms;
//! - every syntax diagnostic (kind, span and message), as a digest, and their count;
//! - the formatter's output, as a digest, and its diagnostics.
//!
//! A source edited since the snapshot is reported and skipped (its old facts say nothing about
//! its new text); a source added since is listed too. Set `RUNMAT_FEA_GOLDEN_WRITE=1` to record
//! a new snapshot, which is only right when a `.fea` behaviour change is intended.

use runmat_fea_syntax::{format_fea, parse_fea, FeaFormatOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const SNAPSHOT: &str = "tests/golden/fea-sources.json";
const FORMAT: &str = "runmat.fea-syntax-golden";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Facts {
    source: String,
    module: String,
    diagnostic_count: usize,
    diagnostics: String,
    formatted: String,
    format_diagnostics: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Snapshot {
    format: String,
    version: u32,
    sources: BTreeMap<String, Facts>,
}

fn digest(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

fn facts(source: &str) -> Facts {
    let module = parse_fea(source);
    let formatted = format_fea(source, FeaFormatOptions::default());
    Facts {
        source: digest(source),
        module: digest(&format!("{:#?}\n{:#?}", module.tokens, module.items)),
        diagnostic_count: module.diagnostics.len(),
        diagnostics: digest(&format!("{:#?}", module.diagnostics)),
        formatted: digest(&formatted.source),
        format_diagnostics: digest(&format!("{:#?}", formatted.diagnostics)),
    }
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

const SKIPPED_DIRECTORIES: &[&str] = &[".git", "node_modules", ".artifacts", "dist"];

fn walk(directory: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().map(|entry| entry.path()).collect();
    entries.sort();
    for path in entries {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if SKIPPED_DIRECTORIES.contains(&name) || name.starts_with("target") {
                continue;
            }
            walk(&path, out);
        } else if name.ends_with(".fea") || name.ends_with(".md") {
            out.push(path);
        }
    }
}

/// The ```fea blocks of a Markdown file, in order.
fn markdown_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current: Option<String> = None;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        match current.as_mut() {
            None if trimmed.starts_with("```fea") && trimmed[6..].trim().is_empty() => {
                current = Some(String::new());
            }
            Some(_) if trimmed.trim_end() == "```" => blocks.push(current.take().unwrap()),
            Some(block) => block.push_str(line),
            None => {}
        }
    }
    blocks
}

/// Every `.fea` source in the repository, keyed by its path (and block index for Markdown).
fn sources() -> BTreeMap<String, String> {
    let root = repository_root();
    let mut files = Vec::new();
    for top in ["docs", "crates", "examples", "tests"] {
        walk(&root.join(top), &mut files);
    }
    let mut out = BTreeMap::new();
    for path in files {
        let relative = path
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if relative.ends_with(".fea") {
            out.insert(relative, text);
        } else {
            for (index, block) in markdown_blocks(&text).into_iter().enumerate() {
                out.insert(format!("{relative}#{index}"), block);
            }
        }
    }
    out
}

#[test]
fn every_repository_fea_source_parses_and_formats_as_recorded() {
    let current: BTreeMap<String, Facts> = sources()
        .iter()
        .map(|(key, source)| (key.clone(), facts(source)))
        .collect();
    let snapshot_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(SNAPSHOT);
    if std::env::var_os("RUNMAT_FEA_GOLDEN_WRITE").is_some() {
        let snapshot = Snapshot {
            format: FORMAT.into(),
            version: 1,
            sources: current,
        };
        let mut json = serde_json::to_string_pretty(&snapshot).unwrap();
        json.push('\n');
        std::fs::write(&snapshot_path, json).expect("write the golden snapshot");
        return;
    }
    let snapshot: Snapshot = serde_json::from_str(
        &std::fs::read_to_string(&snapshot_path).expect("the golden snapshot is committed"),
    )
    .expect("the golden snapshot parses");
    assert_eq!(snapshot.format, FORMAT);

    let mut checked = 0usize;
    let mut edited = Vec::new();
    let mut failures = Vec::new();
    for (key, recorded) in &snapshot.sources {
        let Some(now) = current.get(key) else {
            continue;
        };
        if now.source != recorded.source {
            edited.push(key.as_str());
            continue;
        }
        checked += 1;
        if now != recorded {
            failures.push(format!(
                "{key}:\n  recorded {recorded:#?}\n  now      {now:#?}"
            ));
        }
    }
    let added: Vec<&str> = current
        .keys()
        .filter(|key| !snapshot.sources.contains_key(*key))
        .map(String::as_str)
        .collect();
    eprintln!(
        "fea golden: {checked} sources identical; {} edited since the snapshot; {} added since: {:?} {:?}",
        edited.len(),
        added.len(),
        edited,
        added
    );
    assert!(
        failures.is_empty(),
        "{} of {checked} unchanged `.fea` sources parse or format differently:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        checked * 2 >= snapshot.sources.len(),
        "fewer than half of the recorded sources are unchanged ({checked} of {}); \
         re-record the snapshot from the token parser's tag before trusting this test",
        snapshot.sources.len()
    );
}

mod tree {
    use runmat_fea_syntax::tree::{NodeKind, SyntaxNode};
    use runmat_fea_syntax::*;
    use runmat_types::Span;

    struct Coverage<'a> {
        root: &'a SyntaxNode,
        spans: std::collections::BTreeSet<(usize, usize)>,
        missing: Vec<String>,
    }

    impl Coverage<'_> {
        fn expect(&mut self, what: &str, span: Span, kinds: &[NodeKind]) {
            let found = self
                .root
                .descendants()
                .into_iter()
                .any(|node| kinds.contains(&node.kind()) && node.trimmed_span() == span);
            if !found {
                self.missing
                    .push(format!("{what} at {}..{}", span.start, span.end));
            }
            self.spans.insert((span.start, span.end));
        }

        fn expression(&mut self, expression: &FeaExpression) {
            use FeaExpressionKind::*;
            let kind = match &expression.kind {
                Number(_) => NodeKind::NumberLiteral,
                String(_) => NodeKind::StringLiteral,
                Reference(_) => NodeKind::Reference,
                List(values) => {
                    values.iter().for_each(|value| self.expression(value));
                    NodeKind::List
                }
                Record(entries) => {
                    for entry in entries {
                        self.expect("record entry", entry.span, &[NodeKind::RecordEntry]);
                        self.expression(&entry.value);
                    }
                    NodeKind::Record
                }
                Group(inner) => {
                    self.expression(inner);
                    NodeKind::Group
                }
                Unary { operand, .. } => {
                    self.expression(operand);
                    NodeKind::Unary
                }
                Binary { left, right, .. } => {
                    self.expression(left);
                    self.expression(right);
                    NodeKind::Binary
                }
                Call { callee, arguments } => {
                    self.expression(callee);
                    for argument in arguments {
                        match argument {
                            FeaCallArgument::Positional(value) => self.expression(value),
                            FeaCallArgument::Named { value, span, .. } => {
                                self.expect("named argument", *span, &[NodeKind::NamedArgument]);
                                self.expression(value);
                            }
                        }
                    }
                    NodeKind::Call
                }
                PrefixApplication { callee, argument } => {
                    self.expression(callee);
                    self.expression(argument);
                    NodeKind::PrefixApplication
                }
                Quantity { value, unit } => {
                    self.expression(value);
                    self.expect("unit", unit.span, &[NodeKind::Unit]);
                    NodeKind::Quantity
                }
                Missing => return,
            };
            self.expect("expression", expression.span, &[kind]);
        }

        fn block(&mut self, block: &FeaBlock) {
            self.expect("block", block.span, &[NodeKind::Block]);
            for statement in &block.statements {
                self.statement(statement);
            }
        }

        fn statement(&mut self, statement: &FeaStatement) {
            use FeaStatementKind::*;
            self.spans
                .insert((statement.span.start, statement.span.end));
            let found =
                self.root.descendants().into_iter().any(|node| {
                    node.kind().is_statement() && node.trimmed_span() == statement.span
                });
            if !found {
                self.missing.push(format!(
                    "statement at {}..{}",
                    statement.span.start, statement.span.end
                ));
            }
            match &statement.kind {
                GeneratedByFunction { body, .. }
                | SignalDefinition { body, .. }
                | ModelDefinition { body, .. }
                | Block { body, .. }
                | OutputDefinition { body, .. } => self.block(body),
                UniformSample { start, end, rate } => {
                    self.expression(start);
                    self.expression(end);
                    self.expression(rate);
                }
                AppliedLoad { target, value, .. } => {
                    self.expression(target);
                    self.expression(value);
                }
                Input { default, .. } => default.iter().for_each(|value| self.expression(value)),
                Let { value, .. }
                | Feature { value, .. }
                | SourceParameterUpdate { value, .. }
                | Requirement { value, .. }
                | Assignment { value, .. }
                | Output { value, .. } => self.expression(value),
                PropertyAssignment { property, target } => {
                    self.expression(property);
                    self.expression(target);
                }
                Include(value) | Retain(value) => self.expression(value),
                Use(_) | Command { .. } => {}
            }
        }
    }

    #[test]
    fn the_tree_holds_every_byte_and_every_ast_node_of_every_repository_source() {
        let mut checked = 0;
        for (key, source) in super::sources() {
            let module = parse_fea(source.as_str());
            let root = module.syntax();
            assert_eq!(root.text(), source, "{key}: the tree's text is the source");
            assert_eq!(root.kind(), NodeKind::SourceFile);
            if !module.is_valid() {
                continue;
            }
            checked += 1;
            let mut coverage = Coverage {
                root: &root,
                spans: Default::default(),
                missing: Vec::new(),
            };
            for item in &module.items {
                match item {
                    FeaModuleItem::Import(import) => {
                        coverage.expect("import", import.span, &[NodeKind::Import]);
                    }
                    FeaModuleItem::Declaration(declaration) => {
                        coverage.expect("declaration", declaration.span, &[NodeKind::Declaration]);
                        coverage.expect("name", declaration.name.span, &[NodeKind::Name]);
                        if let Some(value) = &declaration.material_source {
                            coverage.expression(value);
                        }
                        coverage.block(&declaration.body);
                    }
                    FeaModuleItem::Error(_) => {}
                }
            }
            assert!(
                coverage.missing.is_empty(),
                "{key}: AST nodes without a tree node: {:#?}",
                coverage.missing
            );
        }
        assert!(checked > 50, "only {checked} valid sources were checked");
    }
}
