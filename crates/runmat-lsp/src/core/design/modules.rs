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
            Some(f) => format!("`{f}` (a folder alias or `+{}` package)", f.replace('.', "/+")),
        };
        format!("`{} {}` resolved from {place}: `{}`", self.kind, self.name, self.file)
    }
}

/// The `kind name` (with its module) a `use` line names by symbol; `None` for the file form.
fn named(u: &UseDeclaration) -> Option<(String, String, Option<String>)> {
    match (u.symbol(), u.source()) {
        (Some(_), Some(UseSource::Path(p))) if runmat_cad::modules::is_file_path(&p) => None,
        (Some((k, n)), Some(UseSource::Path(p))) => {
            let p = if p.starts_with('.') || p.starts_with('/') { p } else { format!("./{p}") };
            Some((k.text(), n.text(), Some(p)))
        }
        (Some((k, n)), Some(UseSource::Package(s))) => Some((k.text(), n.text(), Some(s.join(".")))),
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
        let Some(u) = UseDeclaration::cast(node.clone()) else { continue };
        let Some((kind, name, from)) = named(&u) else { continue };
        let span = node.trimmed_span();
        let Some(index) = table.symbols.iter().position(|s| s.span == span) else { continue };
        let scope = scope.get_or_insert_with(|| super::compile::module_scope(path, text, files));
        if let Ok(file) = scope.locate(path, &kind, &name, from.as_deref()) {
            let at = files.get(&file).and_then(|t| {
                let (start, end) = declaration_at(t, &kind, &name)?;
                Some(lsp_types::Range::new(offset_to_position(t, start), offset_to_position(t, end)))
            });
            out.insert(index, Resolved { kind, name, file, from, at });
        }
    }
    out
}
