//! Designs a file names by symbol, resolved through the folder scope (report 12 §3.8):
//! go-to-definition on `use part seat_frame as seat` opens the declaration in whichever file of
//! the folder holds it, and hover says where the name resolved from.

use std::collections::BTreeMap;

use runmat_design_syntax::ast::{UseDeclaration, UseSource};
use runmat_design_syntax::tree::NodeKind;
use runmat_design_syntax::SourceSyntax;

use super::super::position::offset_to_position;
use super::symbols::SymbolTable;
use super::DesignFiles;

/// Where a `use` line's design resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub kind: String,
    pub name: String,
    pub file: String,
    /// The module as written (`./cabin`, `hardware`), `None` for the file's own folder.
    pub from: Option<String>,
    /// The declaration's name in `file`, when its text is at hand.
    pub at: Option<lsp_types::Range>,
}

impl Resolved {
    /// "resolved from: …" for hover.
    pub fn origin(&self) -> String {
        let place = match &self.from {
            None => "this folder".to_string(),
            Some(f) if f.starts_with('.') || f.starts_with('/') => format!("folder `{f}`"),
            Some(f) => format!(
                "`{f}` (a folder alias or `+{}` package)",
                f.replace('.', "/+")
            ),
        };
        format!(
            "`{} {}` resolved from {place}: `{}`",
            self.kind, self.name, self.file
        )
    }
}

/// The `kind name` (with its module) a `use` line names by symbol; `None` for the file form.
fn named(u: &UseDeclaration) -> Option<(String, String, Option<String>)> {
    match (u.symbol(), u.source()) {
        (Some(_), Some(UseSource::Path(p))) if runmat_cad::modules::is_file_path(&p) => None,
        (Some((k, n)), Some(UseSource::Path(p))) => {
            let p = if p.starts_with('.') || p.starts_with('/') {
                p
            } else {
                format!("./{p}")
            };
            Some((k.text(), n.text(), Some(p)))
        }
        (Some((k, n)), Some(UseSource::Package(s))) => {
            Some((k.text(), n.text(), Some(s.join("."))))
        }
        (Some((k, n)), None) => Some((k.text(), n.text(), None)),
        (None, Some(UseSource::Package(s))) if s.len() == 3 => {
            Some((s[1].clone(), s[2].clone(), Some(s[0].clone())))
        }
        _ => None,
    }
}

/// The byte range of `kind name` declared in `text`.
fn declaration_at(text: &str, kind: &str, name: &str) -> Option<(usize, usize)> {
    for pattern in [format!("{kind} {name} "), format!("output {name} ")] {
        if let Some(i) = text.find(&pattern) {
            let start = i + pattern.len() - name.len() - 1;
            return Some((start, start + name.len()));
        }
    }
    None
}

/// Each `use` symbol of the table that names a design by symbol, resolved.
pub fn resolve_uses(
    syntax: &SourceSyntax,
    table: &SymbolTable,
    path: &str,
    text: &str,
    files: &DesignFiles,
) -> BTreeMap<usize, Resolved> {
    let mut out = BTreeMap::new();
    let mut scope = None;
    for node in syntax.syntax().descendants() {
        if node.kind() != NodeKind::UseDeclaration {
            continue;
        }
        let Some(u) = UseDeclaration::cast(node.clone()) else {
            continue;
        };
        let Some((kind, name, from)) = named(&u) else {
            continue;
        };
        let span = node.trimmed_span();
        let Some(index) = table.symbols.iter().position(|s| s.span == span) else {
            continue;
        };
        let scope = scope.get_or_insert_with(|| super::compile::module_scope(path, text, files));
        if let Ok(file) = scope.locate(path, &kind, &name, from.as_deref()) {
            let at = files.get(&file).and_then(|t| {
                let (start, end) = declaration_at(t, &kind, &name)?;
                Some(lsp_types::Range::new(
                    offset_to_position(t, start),
                    offset_to_position(t, end),
                ))
            });
            out.insert(
                index,
                Resolved {
                    kind,
                    name,
                    file,
                    from,
                    at,
                },
            );
        }
    }
    out
}

/// The span of `name` in a `use <kind> <name> …` line (or a `<alias>.<kind>.<name>` one).
fn name_span(u: &UseDeclaration, source: &str, kind: &str, name: &str) -> Option<(usize, usize)> {
    let span = u.syntax().trimmed_span();
    let line = source.get(span.start..span.end)?;
    for pattern in [format!("{kind} {name}"), format!(".{kind}.{name}")] {
        if let Some(i) = line.find(&pattern) {
            let start = span.start + i + pattern.len() - name.len();
            return Some((start, start + name.len()));
        }
    }
    None
}

/// Each design file's spans naming the design `kind name` that `file` declares: its declaration
/// in `file`, and every `use` line of the other files that resolves to it, through their folder
/// scopes (report 12 §3.8: a rename follows the resolver, across folders).
pub fn design_references(
    kind: &str,
    name: &str,
    file: &str,
    files: &DesignFiles,
) -> BTreeMap<String, Vec<(usize, usize)>> {
    let mut out: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    if let Some(text) = files.get(file) {
        if let Some(at) = declaration_at(text, kind, name) {
            out.entry(file.to_string()).or_default().push(at);
        }
    }
    let scope = runmat_cad::modules::ModuleScope::from_files(
        files.iter().map(|(p, t)| (p.as_str(), t.as_str())),
    );
    for (path, text) in files {
        let Some(dialect) = runmat_design_syntax::Dialect::for_path(path) else {
            continue;
        };
        let syntax = runmat_design_syntax::parse_source(dialect, text);
        for node in syntax.syntax().descendants() {
            if node.kind() != NodeKind::UseDeclaration {
                continue;
            }
            let Some(u) = UseDeclaration::cast(node.clone()) else {
                continue;
            };
            let Some((k, n, from)) = named(&u) else {
                continue;
            };
            if k != kind
                || n != name
                || scope.locate(path, &k, &n, from.as_deref()).as_deref() != Ok(file)
            {
                continue;
            }
            if let Some(at) = name_span(&u, text, &k, &n) {
                out.entry(path.clone()).or_default().push(at);
            }
        }
    }
    out
}

/// The design `(kind, name, file)` the cursor at `offset` names: the design name on a `use`
/// line, or this file's own declaration name.
pub fn design_at(
    syntax: &SourceSyntax,
    path: &str,
    text: &str,
    files: &DesignFiles,
    offset: usize,
) -> Option<(String, String, String)> {
    for node in syntax.syntax().descendants() {
        if node.kind() != NodeKind::UseDeclaration {
            continue;
        }
        let span = node.trimmed_span();
        if offset < span.start || offset > span.end {
            continue;
        }
        let u = UseDeclaration::cast(node.clone())?;
        let (kind, name, from) = named(&u)?;
        let (start, end) = name_span(&u, text, &kind, &name)?;
        if offset < start || offset > end {
            return None;
        }
        let scope = super::compile::module_scope(path, text, files);
        let file = scope.locate(path, &kind, &name, from.as_deref()).ok()?;
        return Some((kind, name, file));
    }
    let kind = runmat_cad::modules::kind_of_file(path)?;
    for (k, n) in runmat_cad::modules::scan(path, text).declarations {
        if let Some((start, end)) = declaration_at(text, &k, &n) {
            if k == kind && start <= offset && offset <= end {
                return Some((k, n, path.to_string()));
            }
        }
    }
    None
}

/// Completions on a `use` line: after `use <kind> `, the designs of that kind in this folder;
/// after `from "`, the other folders as this file names them; after `from `, the packages.
pub fn use_completions(
    before: &str,
    path: &str,
    text: &str,
    files: &DesignFiles,
) -> Option<Vec<lsp_types::CompletionItem>> {
    let t = before.trim_start();
    let rest = t.strip_prefix("use ")?;
    let scope = super::compile::module_scope(path, text, files);
    let folder = runmat_cad::modules::folder_of(path).to_string();
    let item = |label: String, detail: String, kind| lsp_types::CompletionItem {
        label,
        detail: Some(detail),
        kind: Some(kind),
        ..Default::default()
    };
    let words: Vec<&str> = rest.split(' ').collect();
    match words.as_slice() {
        [kind, partial]
            if runmat_cad::modules::DESIGN_KINDS
                .iter()
                .any(|(k, _)| k == kind) =>
        {
            let row = scope.scope.folders.get(&folder)?;
            Some(
                row.declarations
                    .iter()
                    .filter(|((k, n), _)| k == kind && n.starts_with(partial))
                    .map(|((_, n), files)| {
                        item(
                            n.clone(),
                            files.first().map(|f| f.0.clone()).unwrap_or_default(),
                            lsp_types::CompletionItemKind::MODULE,
                        )
                    })
                    .collect(),
            )
        }
        [_, _, "from", partial] => {
            let quoted = partial.starts_with('"');
            let mut out = Vec::new();
            for (f, row) in &scope.scope.folders {
                if *f == folder || row.declarations.is_empty() {
                    continue;
                }
                let Some(module) = runmat_cad::modules::module_text(path, &format!("{f}/_")) else {
                    continue;
                };
                let is_path = module.starts_with('.');
                if is_path == quoted {
                    out.push(item(
                        module,
                        f.clone(),
                        lsp_types::CompletionItemKind::FOLDER,
                    ));
                }
            }
            Some(out)
        }
        _ => None,
    }
}
