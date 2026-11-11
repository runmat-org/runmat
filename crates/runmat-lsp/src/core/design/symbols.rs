//! The names a design file declares and every place it uses them, read from the lossless tree.
//!
//! A declaration is a block or value statement with a name (`part bracket`, `sketch s1`,
//! `feature body = …`, `input width : length = …`, `net GND = …`, `footprint U1 = …`), a
//! sketch entity (`l1 = line(…)`), or a `use` line. A use is a reference path
//! (`feature.body.face.end`, `input.width`, `s1.l1`, `net.GND`, `U.A.OUT`) or, in circuits and
//! layouts, a bare name (`class = power`, `members = [C1, R1]`), a dotted pin string
//! (`"RS.1"`, `"net.+5V"`) or the net a track, arc, via or pour is named after.

use runmat_design_syntax::ast::{Expr, Headed};
use runmat_design_syntax::lexical::FeaTokenKind;
use runmat_design_syntax::tree::{NodeKind, SyntaxNode};
use runmat_design_syntax::{Dialect, SourceSyntax};
use runmat_types::Span;

/// One declared name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    /// The declaring word (`feature`, `input`, `sketch`, `net`, `footprint`), the callee of a
    /// sketch entity (`line`), or the kind a `use` line names (`material`, `circuit`).
    pub kind: String,
    pub name: String,
    /// Where the name is written (inside the quotes of a quoted name).
    pub name_span: Span,
    /// The whole statement.
    pub span: Span,
    pub parent: Option<usize>,
    /// For a `use` line: the path it names, as written.
    pub path: Option<String>,
}

/// A written use of a symbol's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occurrence {
    pub symbol: usize,
    pub span: Span,
}

/// A reference path as written: each segment's text and span, and the symbol that encloses it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path {
    pub segments: Vec<(String, Span)>,
    pub scope: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SymbolTable {
    pub symbols: Vec<Symbol>,
    pub occurrences: Vec<Occurrence>,
    pub paths: Vec<Path>,
}

/// Layout statements named after the net they carry, not declaring a name of their own.
const NET_NAMED: &[&str] = &["track", "arc", "via", "pour"];

fn is_circuit(dialect: Dialect) -> bool {
    matches!(dialect, Dialect::Circuit | Dialect::CircuitLayout)
}

impl SymbolTable {
    pub fn build(syntax: &SourceSyntax) -> Self {
        let mut table = SymbolTable::default();
        let source = syntax.source();
        let root = syntax.syntax();
        let mut net_named = Vec::new();
        collect(
            syntax.dialect,
            source,
            &root,
            None,
            &mut table,
            &mut net_named,
        );
        let mut paths = Vec::new();
        for node in root.descendants() {
            match node.kind() {
                NodeKind::Reference => {
                    let segments: Vec<(String, Span)> = node
                        .significant_tokens()
                        .filter(|token| token.kind() != FeaTokenKind::Dot)
                        .map(|token| (token.text().to_string(), token.span()))
                        .collect();
                    if !segments.is_empty() {
                        paths.push((segments, node.trimmed_span()));
                    }
                }
                NodeKind::StringLiteral if is_circuit(syntax.dialect) => {
                    if let Some(segments) = string_segments(source, node.trimmed_span()) {
                        paths.push((segments, node.trimmed_span()));
                    }
                }
                _ => {}
            }
        }
        for (name, span, at) in net_named {
            paths.push((vec![(name, span)], at));
        }
        for (segments, at) in paths {
            let path = Path {
                scope: table.scope_at(at.start),
                segments,
            };
            for (symbol, span) in table.resolve(syntax.dialect, &path) {
                table.occurrences.push(Occurrence { symbol, span });
            }
            table.paths.push(path);
        }
        table
            .occurrences
            .sort_by_key(|o| (o.span.start, o.span.end));
        table.occurrences.dedup();
        table
    }

    /// The innermost symbol whose statement holds `offset`.
    pub fn scope_at(&self, offset: usize) -> Option<usize> {
        self.symbols
            .iter()
            .enumerate()
            .filter(|(_, s)| s.span.start <= offset && offset < s.span.end)
            .min_by_key(|(_, s)| s.span.end - s.span.start)
            .map(|(index, _)| index)
    }

    fn ancestors(&self, mut scope: Option<usize>) -> Vec<usize> {
        let mut out = Vec::new();
        while let Some(index) = scope {
            out.push(index);
            scope = self.symbols[index].parent;
        }
        out
    }

    /// The symbol named `name` (of `kind` when given) seen from `scope`: one declared in the
    /// nearest enclosing block first, else anywhere in the file.
    pub fn lookup(&self, kind: Option<&str>, name: &str, scope: Option<usize>) -> Option<usize> {
        let ancestors = self.ancestors(scope);
        let candidates: Vec<usize> = (0..self.symbols.len())
            .filter(|&i| {
                let s = &self.symbols[i];
                s.name == name && kind.is_none_or(|k| s.kind == k)
            })
            .collect();
        candidates
            .iter()
            .copied()
            .min_by_key(|&i| {
                let parent = self.symbols[i].parent;
                match parent {
                    None => ancestors.len(),
                    Some(p) => ancestors.iter().position(|&a| a == p).unwrap_or(usize::MAX),
                }
            })
            .or_else(|| candidates.first().copied())
    }

    pub fn children(&self, parent: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.symbols.len()).filter(move |&i| self.symbols[i].parent == Some(parent))
    }

    pub fn has_kind(&self, kind: &str) -> bool {
        self.symbols.iter().any(|s| s.kind == kind)
    }

    /// The symbols each segment of `path` names, with the segment's span.
    pub fn resolve(&self, dialect: Dialect, path: &Path) -> Vec<(usize, Span)> {
        let segments = &path.segments;
        let mut out = Vec::new();
        let mut next = 0;
        let mut current = None;
        if segments.len() >= 2 && self.has_kind(&segments[0].0) {
            if let Some(symbol) = self.lookup(Some(&segments[0].0), &segments[1].0, path.scope) {
                out.push((symbol, segments[1].1));
                current = Some(symbol);
                next = 2;
            }
        }
        if current.is_none() && (segments.len() >= 2 || is_circuit(dialect)) {
            if let Some(symbol) = self.lookup(None, &segments[0].0, path.scope) {
                // A part's bare names are words (`mode = cut`, `through_all`); only a dotted
                // path into a sketch (`s1.l1`) or another block names a symbol there.
                let dotted_into = segments.len() >= 2 && self.children(symbol).next().is_some();
                if is_circuit(dialect) || dotted_into {
                    out.push((symbol, segments[0].1));
                    current = Some(symbol);
                    next = 1;
                }
            }
        }
        while let (Some(parent), Some((name, span))) = (current, segments.get(next)) {
            let child = self
                .children(parent)
                .find(|&c| self.symbols[c].name == *name);
            match child {
                Some(child) => {
                    out.push((child, *span));
                    current = Some(child);
                    next += 1;
                }
                None => break,
            }
        }
        out
    }

    /// The symbol a name or a use of it at `offset` stands for, and the span written there.
    pub fn at(&self, offset: usize) -> Option<(usize, Span)> {
        if let Some(o) = self
            .occurrences
            .iter()
            .find(|o| o.span.start <= offset && offset <= o.span.end && o.span.end > o.span.start)
        {
            return Some((o.symbol, o.span));
        }
        self.symbols
            .iter()
            .enumerate()
            .find(|(_, s)| s.name_span.start <= offset && offset <= s.name_span.end)
            .map(|(i, s)| (i, s.name_span))
    }

    /// Every span the symbol's name is written at: its declaration first, then each use.
    pub fn spans_of(&self, symbol: usize) -> Vec<Span> {
        let mut out = vec![self.symbols[symbol].name_span];
        out.extend(
            self.occurrences
                .iter()
                .filter(|o| o.symbol == symbol)
                .map(|o| o.span),
        );
        out
    }
}

/// A quoted name's span without its quotes, when the text has no escapes.
fn name_span(source: &str, node: &SyntaxNode) -> Span {
    let span = node.trimmed_span();
    let text = source.get(span.start..span.end).unwrap_or("");
    if text.len() >= 2 && text.starts_with('"') && text.ends_with('"') && !text.contains('\\') {
        Span {
            start: span.start + 1,
            end: span.end - 1,
        }
    } else {
        span
    }
}

/// A string's dotted segments (`"RS.1"`, `"net.+5V"`, `"+5V"`) and their spans, when it has
/// no escapes and no path separators.
fn string_segments(source: &str, span: Span) -> Option<Vec<(String, Span)>> {
    let text = source.get(span.start..span.end)?;
    let inner = text.strip_prefix('"')?.strip_suffix('"')?;
    if inner.is_empty() || inner.contains(['\\', '/', ' ']) {
        return None;
    }
    let mut out = Vec::new();
    let mut at = span.start + 1;
    for segment in inner.split('.') {
        if segment.is_empty() {
            return None;
        }
        out.push((
            segment.to_string(),
            Span {
                start: at,
                end: at + segment.len(),
            },
        ));
        at += segment.len() + 1;
    }
    Some(out)
}

fn push(table: &mut SymbolTable, symbol: Symbol) -> usize {
    table.symbols.push(symbol);
    table.symbols.len() - 1
}

fn collect(
    dialect: Dialect,
    source: &str,
    node: &SyntaxNode,
    parent: Option<usize>,
    table: &mut SymbolTable,
    net_named: &mut Vec<(String, Span, Span)>,
) {
    for child in node.children() {
        let mut scope = parent;
        match child.kind() {
            NodeKind::Declaration | NodeKind::BlockStatement => {
                let (keyword, name) = match child.kind() {
                    NodeKind::Declaration => {
                        let d = runmat_design_syntax::ast::Declaration::cast(child.clone())
                            .expect("a declaration node");
                        (d.keyword(), d.name())
                    }
                    _ => {
                        let b = runmat_design_syntax::ast::BlockStatement::cast(child.clone())
                            .expect("a block statement node");
                        (b.keyword(), b.name())
                    }
                };
                if let Some(name) = name {
                    scope = Some(push(
                        table,
                        Symbol {
                            kind: keyword,
                            name: name.text(),
                            name_span: name_span(source, name.syntax()),
                            span: child.trimmed_span(),
                            parent,
                            path: None,
                        },
                    ));
                }
            }
            NodeKind::ValueStatement => {
                let v = runmat_design_syntax::ast::ValueStatement::cast(child.clone())
                    .expect("a value statement node");
                if let Some(name) = v.name() {
                    let keyword = v.keyword();
                    if dialect == Dialect::CircuitLayout && NET_NAMED.contains(&keyword.as_str()) {
                        net_named.push((
                            name.text(),
                            name_span(source, name.syntax()),
                            child.trimmed_span(),
                        ));
                    } else {
                        scope = Some(push(
                            table,
                            Symbol {
                                kind: keyword,
                                name: name.text(),
                                name_span: name_span(source, name.syntax()),
                                span: child.trimmed_span(),
                                parent,
                                path: None,
                            },
                        ));
                    }
                }
            }
            NodeKind::Assignment => {
                let a = runmat_design_syntax::ast::Assignment::cast(child.clone())
                    .expect("an assignment node");
                let target = a.target();
                if let (Some(_), [name], Some(Expr::Call(call))) =
                    (parent, target.as_slice(), a.value())
                {
                    let kind = call
                        .callee()
                        .map(|callee| callee.syntax().text().trim().to_string())
                        .unwrap_or_default();
                    let span = a.target_span();
                    scope = Some(push(
                        table,
                        Symbol {
                            kind,
                            name: name.clone(),
                            name_span: span,
                            span: child.trimmed_span(),
                            parent,
                            path: None,
                        },
                    ));
                }
            }
            NodeKind::UseDeclaration => {
                let u = runmat_design_syntax::ast::UseDeclaration::cast(child.clone())
                    .expect("a use node");
                let path = match u.source() {
                    Some(runmat_design_syntax::ast::UseSource::Path(p)) => Some(p),
                    _ => None,
                };
                let named = u
                    .alias()
                    .map(|alias| (None, alias))
                    .or_else(|| u.symbol().map(|(kind, name)| (Some(kind), name)));
                if let Some((kind, name)) = named {
                    let kind = kind
                        .map(|k| k.text())
                        .or_else(|| u.symbol().map(|(k, _)| k.text()))
                        .unwrap_or_else(|| "use".into());
                    push(
                        table,
                        Symbol {
                            kind,
                            name: name.text(),
                            name_span: name_span(source, name.syntax()),
                            span: child.trimmed_span(),
                            parent,
                            path,
                        },
                    );
                }
            }
            _ => {}
        }
        collect(dialect, source, &child, scope, table, net_named);
    }
}
