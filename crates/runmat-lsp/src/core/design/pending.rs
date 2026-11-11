//! A design document whose dependencies the host is still loading.
//!
//! The browser server reads no workspace: the host lists and reads the folders and files a
//! document names ([`super::design_dependencies`]) and supplies them. A cloud project that is
//! offline, or a provider that loads lazily, can leave some of them unread. The host then
//! names them as pending (`setDesignPending`), and the document's compile problems wait: a
//! `use` line whose folder is pending says it "isn't loaded yet" in place of the compiler's
//! "holds no designs", and the host supplies the files and re-checks once they arrive.

use lsp_types::DiagnosticSeverity;
use runmat_design_syntax::ast::{UseDeclaration, UseSource};
use runmat_design_syntax::tree::NodeKind;
use runmat_design_syntax::Dialect;
use runmat_types::Span;

use super::{DesignDocumentAnalysis, Problem};

/// Diagnostic code of a line whose files are still loading.
pub const PENDING_CODE: &str = "design.pending";

/// Whether the pending path `pending` (a folder listing `/a/b/` or a file) is in `listing` (a
/// folder listing) or is it.
fn under(pending: &str, listing: &str) -> bool {
    pending == listing || pending.starts_with(listing)
}

/// The folder listings a `use` line's `from` names, as the dialect's compiler resolves it.
fn listings(dialect: Dialect, path: &str, from: &str) -> Vec<String> {
    match dialect {
        Dialect::Circuit | Dialect::CircuitLayout => {
            runmat_circuit::runtime::modules::folder_dependencies(path, from)
        }
        _ => runmat_cad::modules::from_listings(path, from),
    }
}

/// The `from` of a `use` line as written: a path's text or a package's dotted name.
fn from_of(u: &UseDeclaration) -> Option<String> {
    match u.source()? {
        UseSource::Path(p) => Some(p),
        UseSource::Package(s) if u.symbol().is_some() => Some(s.join(".")),
        UseSource::Package(s) if s.len() == 3 => Some(s[0].clone()),
        UseSource::Package(_) => None,
    }
}

/// The span to mark on a `use` line: its `from` string, else the line's first line.
fn mark(source: &str, span: Span) -> Span {
    let line = source.get(span.start..span.end).unwrap_or_default();
    if let Some(open) = line.find('"') {
        if let Some(close) = line[open + 1..].find('"') {
            return Span {
                start: span.start + open,
                end: span.start + open + close + 2,
            };
        }
    }
    let end = line.find('\n').map_or(span.end, |i| span.start + i);
    Span { start: span.start, end }
}

impl DesignDocumentAnalysis {
    /// This analysis while `pending` (folder listings and files the host is still loading for
    /// this document, its dependencies' dependencies included) are unread. With none pending it
    /// is unchanged. Otherwise the compile problems wait (they would report the missing files),
    /// syntax problems stay, and each `use` line naming a pending folder says it isn't loaded
    /// yet; when no `use` line names one, the first line says which files are loading.
    pub fn with_pending(mut self, pending: &[String]) -> Self {
        if pending.is_empty() {
            return self;
        }
        let path = self.uri.path().to_string();
        let source = self.source().to_string();
        let mut marks: Vec<Problem> = Vec::new();
        for node in self.syntax.syntax().descendants() {
            if node.kind() != NodeKind::UseDeclaration {
                continue;
            }
            let Some(u) = UseDeclaration::cast(node.clone()) else { continue };
            let Some(from) = from_of(&u) else { continue };
            let waiting = listings(self.dialect, &path, &from)
                .iter()
                .any(|listing| pending.iter().any(|p| under(p, listing)));
            if waiting {
                marks.push(Problem {
                    span: mark(&source, node.trimmed_span()),
                    severity: DiagnosticSeverity::INFORMATION,
                    code: Some(PENDING_CODE.into()),
                    message: format!(
                        "`{from}` isn't loaded yet; this line is checked again when it arrives"
                    ),
                });
            }
        }
        if marks.is_empty() {
            let first = source.find('\n').unwrap_or(source.len());
            let more = if pending.len() > 1 {
                format!(" and {} more", pending.len() - 1)
            } else {
                String::new()
            };
            marks.push(Problem {
                span: Span { start: 0, end: first },
                severity: DiagnosticSeverity::INFORMATION,
                code: Some(PENDING_CODE.into()),
                message: format!(
                    "`{}`{more} isn't loaded yet; this file is checked again when it arrives",
                    pending[0]
                ),
            });
        }
        self.problems
            .retain(|p| p.code.as_deref() == Some("design.syntax"));
        self.problems.extend(marks);
        self
    }
}

#[cfg(test)]
#[path = "pending_tests.rs"]
mod tests;
