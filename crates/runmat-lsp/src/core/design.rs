//! Language support for the design formats: `.part`, `.assembly`, `.drawing`, `.circuit` and
//! `.circuit.layout`, served on the same path as `.fea`.
//!
//! - Highlighting: lexer tokens take the classes of the dialect's generated TextMate grammar
//!   (`runmat_design_syntax::textmate_grammar`), the one grammar docs, the site and Desktop use;
//!   declared names carry the `declaration` modifier and uses of them read as variables.
//! - Diagnostics: the real compilers. A part regenerates on the reference kernel and its
//!   regeneration diagnostics land on the text they are about; assemblies and drawings report
//!   their front ends' syntax and model errors; circuits and layouts run
//!   `runmat_circuit::runtime::check` (reading, elaboration and ERC).
//! - Hover: the statement as written, with the compiler's resolved values (an input's value in
//!   SI units, a feature's status and references, a component's part, value and instances, a
//!   net's class and pins).
//! - Completion, definition, references, rename and document symbols from the file's symbol
//!   table ([`symbols`]); formatting with the shared design formatter.

use std::collections::BTreeMap;

use lsp_types::{
    CompletionItem, CompletionItemKind, Diagnostic, DiagnosticSeverity, DocumentSymbol, Hover,
    HoverContents, Location, MarkupContent, MarkupKind, NumberOrString, Position, Range,
    SemanticToken, SemanticTokens, SymbolKind, TextEdit, Url, WorkspaceEdit,
};
use runmat_design_syntax::{parse_source, Dialect, FeaFormatOptions, SourceSyntax};
use runmat_types::Span;
use serde_json::Value;

use super::fea::highlight::{self, class, TokenClass, FUNCTION, KEYWORD, NAMESPACE, VARIABLE};
use super::position::{offset_to_position, position_to_offset};

pub mod symbols;
#[cfg(test)]
mod tests;

use symbols::SymbolTable;

/// Workspace path → text: the files a design source reads (its circuit, lock, parts).
pub type DesignFiles = BTreeMap<String, String>;

/// Semantic-token modifier bit 0 (`declaration`, `semantic_tokens::legend`).
const DECLARATION: u32 = 1;

/// The design dialect of a document, by its file name; `.fea` is served by [`super::fea`].
pub fn design_dialect(uri: &Url) -> Option<Dialect> {
    let path = uri.path();
    Dialect::for_path(path).filter(|dialect| *dialect != Dialect::Fea)
}

#[cfg_attr(not(feature = "wasm"), allow(dead_code))] // the browser server's exports
pub fn is_design_uri(uri: &Url) -> bool {
    design_dialect(uri).is_some()
}

/// The files `path` needs beyond itself, as workspace paths: a layout's circuit and lock, a
/// circuit's used circuits, lock and layout, an assembly's parts, a drawing's models.
pub fn design_dependencies(dialect: Dialect, path: &str, text: &str) -> Vec<String> {
    match dialect {
        Dialect::Circuit | Dialect::CircuitLayout => {
            runmat_circuit::runtime::dependencies(path, text)
        }
        Dialect::Assembly | Dialect::Drawing => {
            let syntax = parse_source(dialect, text);
            let table = SymbolTable::build(&syntax);
            let mut out: Vec<String> = table
                .symbols
                .iter()
                .filter_map(|s| s.path.as_deref())
                .chain(string_paths(&syntax).iter().map(String::as_str))
                .filter(|p| p.ends_with(".part") || p.ends_with(".assembly"))
                .map(|p| runmat_circuit::runtime::resolve_path(path, p))
                .collect();
            out.sort();
            out.dedup();
            out
        }
        _ => Vec::new(),
    }
}

/// Every quoted string that names a design file (`file = "./clip.part"`).
fn string_paths(syntax: &SourceSyntax) -> Vec<String> {
    let source = syntax.source();
    syntax
        .syntax()
        .descendants()
        .into_iter()
        .filter(|node| node.kind() == runmat_design_syntax::tree::NodeKind::StringLiteral)
        .filter_map(|node| {
            let span = node.trimmed_span();
            let text = source.get(span.start..span.end)?;
            let inner = text.strip_prefix('"')?.strip_suffix('"')?;
            Dialect::for_path(inner).map(|_| inner.to_string())
        })
        .collect()
}

/// A compiler problem at a byte range of this file.
#[derive(Debug, Clone, PartialEq)]
struct Problem {
    span: Span,
    severity: DiagnosticSeverity,
    code: Option<String>,
    message: String,
}

#[derive(Clone)]
pub struct DesignDocumentAnalysis {
    dialect: Dialect,
    uri: Url,
    syntax: SourceSyntax,
    symbols: SymbolTable,
    problems: Vec<Problem>,
    /// The compiler's projection (`runmat_cad::index` or `runmat_circuit::index`), for hover.
    projection: Option<Value>,
}

impl DesignDocumentAnalysis {
    /// Reads, compiles and indexes one design document. `files` holds the texts of the files
    /// [`design_dependencies`] named (open editors first, then the workspace).
    pub fn analyze(uri: &Url, dialect: Dialect, text: &str, files: &DesignFiles) -> Self {
        let syntax = parse_source(dialect, text);
        let symbols = SymbolTable::build(&syntax);
        let path = uri.path().to_string();
        let mut problems: Vec<Problem> = syntax
            .diagnostics
            .iter()
            .map(|d| Problem {
                span: d.span,
                severity: DiagnosticSeverity::ERROR,
                code: Some("design.syntax".into()),
                message: d.message.clone(),
            })
            .collect();
        let mut projection = None;
        if syntax.diagnostics.is_empty() {
            let (more, compiled) = compile(dialect, &path, text, files, &symbols);
            problems.extend(more);
            projection = compiled;
        }
        DesignDocumentAnalysis {
            dialect,
            uri: uri.clone(),
            syntax,
            symbols,
            problems,
            projection,
        }
    }

    #[cfg_attr(not(feature = "wasm"), allow(dead_code))] // the browser server's exports
    pub fn dialect(&self) -> Dialect {
        self.dialect
    }

    fn source(&self) -> &str {
        self.syntax.source()
    }

    fn range(&self, span: Span) -> Range {
        let source = self.source();
        let start = span.start.min(source.len());
        let end = span.end.max(start).min(source.len());
        Range::new(
            offset_to_position(source, start),
            offset_to_position(source, end),
        )
    }

    pub fn status_message(&self) -> String {
        let errors = self
            .problems
            .iter()
            .filter(|p| p.severity == DiagnosticSeverity::ERROR)
            .count();
        let label = self.dialect.spec().label;
        if errors == 0 {
            format!("{label} compiles")
        } else {
            format!("{label} has {errors} error(s)")
        }
    }

    pub fn diagnostics(&self) -> Vec<Diagnostic> {
        self.problems
            .iter()
            .map(|p| Diagnostic {
                range: self.range(p.span),
                severity: Some(p.severity),
                code: p.code.clone().map(NumberOrString::String),
                source: Some(source_label(self.dialect).into()),
                message: p.message.clone(),
                ..Diagnostic::default()
            })
            .collect()
    }

    /// Tokens with the grammar's classes, before any compile: the first paint.
    #[cfg_attr(not(feature = "wasm"), allow(dead_code))] // the browser server's exports
    pub fn lexical_semantic_tokens(dialect: Dialect, text: &str) -> SemanticTokens {
        let syntax = parse_source(dialect, text);
        encode(text, classify(&syntax, None))
    }

    pub fn semantic_tokens(&self) -> SemanticTokens {
        encode(self.source(), classify(&self.syntax, Some(&self.symbols)))
    }

    pub fn formatting_edits(&self) -> Vec<TextEdit> {
        let source = self.source();
        let formatted =
            runmat_design_syntax::format_source(self.dialect, source, FeaFormatOptions::default());
        if !formatted.diagnostics.is_empty() || formatted.source == source {
            return Vec::new();
        }
        vec![TextEdit {
            range: Range::new(
                Position::new(0, 0),
                offset_to_position(source, source.len()),
            ),
            new_text: formatted.source,
        }]
    }

    /// Every declaration as `kind.name` (`feature.boss`, `net.VCC`, `footprint.U1`), for
    /// workspace symbol search across open design files.
    pub fn workspace_symbols(&self, query: Option<&str>) -> Vec<lsp_types::SymbolInformation> {
        let query = query.map(str::to_ascii_lowercase);
        let table = &self.symbols;
        table
            .symbols
            .iter()
            .filter_map(|s| {
                let name = format!("{}.{}", s.kind, s.name);
                if query
                    .as_deref()
                    .is_some_and(|q| !q.is_empty() && !name.to_ascii_lowercase().contains(q))
                {
                    return None;
                }
                let container = s.parent.map(|p| {
                    let p = &table.symbols[p];
                    format!("{}.{}", p.kind, p.name)
                });
                serde_json::from_value(serde_json::json!({
                    "name": name,
                    "kind": symbol_kind(&s.kind),
                    "location": { "uri": self.uri.clone(), "range": self.range(s.name_span) },
                    "containerName": container,
                }))
                .ok()
            })
            .collect()
    }

    pub fn document_symbols(&self) -> Vec<DocumentSymbol> {
        fn build(analysis: &DesignDocumentAnalysis, parent: Option<usize>) -> Vec<DocumentSymbol> {
            let table = &analysis.symbols;
            (0..table.symbols.len())
                .filter(|&i| table.symbols[i].parent == parent)
                .map(|i| {
                    let s = &table.symbols[i];
                    let children = build(analysis, Some(i));
                    #[allow(deprecated)]
                    DocumentSymbol {
                        name: s.name.clone(),
                        detail: Some(s.kind.clone()),
                        kind: symbol_kind(&s.kind),
                        tags: None,
                        deprecated: None,
                        range: analysis.range(s.span),
                        selection_range: analysis.range(s.name_span),
                        children: (!children.is_empty()).then_some(children),
                    }
                })
                .collect()
        }
        build(self, None)
    }

    fn symbol_at(&self, position: &Position) -> Option<(usize, Span)> {
        let offset = position_to_offset(self.source(), position);
        self.symbols.at(offset)
    }

    pub fn definition(&self, position: &Position) -> Option<Location> {
        let offset = position_to_offset(self.source(), position);
        if let Some(location) = self.use_target(offset) {
            return Some(location);
        }
        let (symbol, _) = self.symbols.at(offset)?;
        let s = &self.symbols.symbols[symbol];
        if let Some(path) = s.path.as_deref() {
            if let Some(location) = self.file_location(path) {
                return Some(location);
            }
        }
        Some(Location::new(self.uri.clone(), self.range(s.name_span)))
    }

    /// A `use … from "<path>"` string under the cursor: the file it names.
    fn use_target(&self, offset: usize) -> Option<Location> {
        let s = self.symbols.symbols.iter().find(|s| {
            s.path.is_some()
                && s.span.start <= offset
                && offset <= s.span.end
                && !(s.name_span.start <= offset && offset <= s.name_span.end)
        })?;
        let text = self.source().get(s.span.start..s.span.end)?;
        let quoted = format!("\"{}\"", s.path.as_deref()?);
        let at = text.find(&quoted)? + s.span.start;
        (at <= offset && offset <= at + quoted.len())
            .then(|| self.file_location(s.path.as_deref()?))
            .flatten()
    }

    fn file_location(&self, written: &str) -> Option<Location> {
        let mut target = runmat_circuit::runtime::resolve_path(self.uri.path(), written);
        if Dialect::for_path(&target).is_none() {
            // `use circuit io from "./io"` and other extension-free paths name the kind's file.
            return None;
        }
        if !target.starts_with('/') {
            target = format!("/{target}");
        }
        let mut uri = self.uri.clone();
        uri.set_path(&target);
        Some(Location::new(uri, Range::default()))
    }

    pub fn references(&self, position: &Position, include_declaration: bool) -> Vec<Location> {
        let Some((symbol, _)) = self.symbol_at(position) else {
            return Vec::new();
        };
        let spans = self.symbols.spans_of(symbol);
        spans
            .into_iter()
            .skip(usize::from(!include_declaration))
            .map(|span| Location::new(self.uri.clone(), self.range(span)))
            .collect()
    }

    /// The range and current text of a renamable name under the cursor.
    pub fn prepare_rename(&self, position: &Position) -> Option<(Range, String)> {
        let (symbol, span) = self.symbol_at(position)?;
        Some((self.range(span), self.symbols.symbols[symbol].name.clone()))
    }

    /// Renames the symbol under the cursor at its declaration and every use. A name that
    /// isn't a plain identifier is written quoted where it stands as a declared name.
    pub fn rename(
        &self,
        position: &Position,
        new_name: &str,
    ) -> Result<Option<WorkspaceEdit>, String> {
        let Some((symbol, _)) = self.symbol_at(position) else {
            return Ok(None);
        };
        if new_name.is_empty() || new_name.contains(['"', '\\', '\n', '.']) {
            return Err(format!("`{new_name}` isn't a valid name"));
        }
        let s = &self.symbols.symbols[symbol];
        if let Some(clash) = self.symbols.symbols.iter().find(|other| {
            other.name == new_name && other.kind == s.kind && other.parent == s.parent
        }) {
            return Err(format!("{} {} already exists", clash.kind, clash.name));
        }
        let identifier = is_identifier(new_name);
        let source = self.source();
        let edits: Vec<TextEdit> = self
            .symbols
            .spans_of(symbol)
            .into_iter()
            .map(|span| {
                let quoted = span.start > 0 && source.as_bytes()[span.start - 1] == b'"';
                let new_text = if quoted || identifier {
                    new_name.to_string()
                } else {
                    format!("\"{new_name}\"")
                };
                TextEdit {
                    range: self.range(span),
                    new_text,
                }
            })
            .collect();
        let mut changes = std::collections::HashMap::new();
        changes.insert(self.uri.clone(), edits);
        Ok(Some(WorkspaceEdit {
            changes: Some(changes),
            ..WorkspaceEdit::default()
        }))
    }

    pub fn hover(&self, position: &Position) -> Option<Hover> {
        let (symbol, span) = self.symbol_at(position)?;
        let s = &self.symbols.symbols[symbol];
        let statement = self.source().get(s.span.start..s.span.end).unwrap_or("");
        let head = statement
            .lines()
            .next()
            .unwrap_or("")
            .trim_end_matches('{')
            .trim();
        let head = if head.chars().count() > 160 {
            format!("{}…", head.chars().take(160).collect::<String>())
        } else {
            head.to_string()
        };
        let mut text = format!("```{}\n{}\n```", fence(self.dialect), head);
        let facts = self.facts(symbol);
        if !facts.is_empty() {
            text.push_str("\n\n");
            text.push_str(&facts.join("\n\n"));
        }
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: text,
            }),
            range: Some(self.range(span)),
        })
    }

    /// The compiler's resolved values for a symbol, as Markdown lines.
    fn facts(&self, symbol: usize) -> Vec<String> {
        let s = &self.symbols.symbols[symbol];
        let Some(projection) = &self.projection else {
            return Vec::new();
        };
        let mut out = Vec::new();
        match self.dialect {
            Dialect::Part => {
                let index = &projection["index"];
                let named = |list: &str, key: &str| {
                    index[list]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .find(|row| row[key].as_str() == Some(s.name.as_str()))
                        .cloned()
                };
                match s.kind.as_str() {
                    "input" => {
                        if let Some(row) = named("inputs", "name") {
                            if let Some(error) = row["error"].as_str() {
                                out.push(format!("**Value:** doesn't evaluate: {error}"));
                            } else if let Some(si) = row["si"].as_f64() {
                                let kind = row["kind"].as_str().unwrap_or("");
                                out.push(format!("**Value:** {} ({kind})", si_text(si, kind)));
                            }
                        }
                    }
                    "feature" => {
                        if let Some(row) = named("features", "id") {
                            let status = row["status"]["status"].as_str().unwrap_or("unknown");
                            let kind = row["kind"].as_str().unwrap_or("");
                            let mut line = format!("**{kind}**, {}", status.replace('_', " "));
                            if let Some(d) = row["status"]["diagnostic"].as_u64() {
                                if let Some(message) = index["diagnostics"]
                                    .as_array()
                                    .and_then(|ds| ds.iter().find(|x| x["id"].as_u64() == Some(d)))
                                    .and_then(|x| x["message"].as_str())
                                {
                                    line.push_str(&format!(": {message}"));
                                }
                            }
                            out.push(line);
                            let refs: Vec<String> = row["references"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .map(|r| {
                                    format!(
                                        "- `{}` → {} ({})",
                                        r["param"].as_str().unwrap_or(""),
                                        r["display"].as_str().unwrap_or(""),
                                        r["status"].as_str().unwrap_or("")
                                    )
                                })
                                .collect();
                            if !refs.is_empty() {
                                out.push(format!("**References**\n{}", refs.join("\n")));
                            }
                        }
                    }
                    "region" => {
                        if let Some(row) = named("regions", "name") {
                            let count = row["entities"].as_array().map_or(0, Vec::len);
                            out.push(format!(
                                "**{} region**, {count} {}{}",
                                row["kind"].as_str().unwrap_or(""),
                                if count == 1 { "entity" } else { "entities" },
                                if row["ok"].as_bool() == Some(false) {
                                    ", doesn't resolve"
                                } else {
                                    ""
                                }
                            ));
                        }
                    }
                    "publish" => {
                        if let Some(row) = named("publishes", "name") {
                            out.push(if row["ok"].as_bool() == Some(false) {
                                "**Publish**, doesn't resolve".into()
                            } else {
                                "**Publish**, resolves".into()
                            });
                        }
                    }
                    "output" | "part" => {
                        if let Some(row) = named("parts", "name") {
                            let mut line = "**Part**".to_string();
                            if let Some(pn) = row["part_number"].as_str() {
                                line.push_str(&format!(" {pn}"));
                            }
                            if let Some(m) = row["material"].as_str() {
                                line.push_str(&format!(", material {m}"));
                            }
                            out.push(line);
                        }
                    }
                    _ => {}
                }
            }
            Dialect::Circuit | Dialect::CircuitLayout => {
                let components = projection["components"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                match s.kind.as_str() {
                    "component" | "footprint" => {
                        let sheet = s.parent.map(|p| self.symbols.symbols[p].name.clone());
                        let rows: Vec<&Value> = components
                            .iter()
                            .filter(|row| {
                                if s.kind == "footprint" {
                                    return row["designator"].as_str() == Some(s.name.as_str());
                                }
                                let path = row["path"].as_str().unwrap_or("");
                                path.rsplit('/').next() == Some(s.name.as_str())
                                    && sheet
                                        .as_deref()
                                        .is_none_or(|sh| row["sheet"].as_str() == Some(sh))
                            })
                            .collect();
                        if let Some(first) = rows.first() {
                            let mut line = String::new();
                            if let Some(part) = first["part"].as_str() {
                                line.push_str(&format!("**{part}**"));
                            }
                            if let Some(value) = first["value"].as_str() {
                                line.push_str(&format!(", {value}"));
                            }
                            if let Some(d) = first["description"].as_str() {
                                line.push_str(&format!(". {d}"));
                            }
                            if !line.is_empty() {
                                out.push(line);
                            }
                            let params: Vec<String> = first["params"]
                                .as_object()
                                .into_iter()
                                .flatten()
                                .map(|(k, v)| format!("{k} = {}", v.as_str().unwrap_or("")))
                                .collect();
                            if !params.is_empty() {
                                out.push(params.join(", "));
                            }
                            let instances: Vec<String> = rows
                                .iter()
                                .map(|row| {
                                    format!(
                                        "{} ({})",
                                        row["designator"].as_str().unwrap_or(""),
                                        row["path"].as_str().unwrap_or("")
                                    )
                                })
                                .collect();
                            out.push(format!("**Instances:** {}", instances.join(", ")));
                        }
                    }
                    "net" => {
                        let nets = projection["nets"].as_array().cloned().unwrap_or_default();
                        let rows: Vec<&Value> = nets
                            .iter()
                            .filter(|row| {
                                let name = row["name"].as_str().unwrap_or("");
                                name == s.name || name.rsplit('/').next() == Some(s.name.as_str())
                            })
                            .collect();
                        for row in rows.iter().take(4) {
                            let pins: Vec<&str> = row["pins"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .filter_map(Value::as_str)
                                .collect();
                            let class = row["class"]
                                .as_str()
                                .map(|c| format!(", class {c}"))
                                .unwrap_or_default();
                            let shown: Vec<&str> = pins.iter().take(12).copied().collect();
                            out.push(format!(
                                "**{}**{class}: {} pin{} ({}{})",
                                row["name"].as_str().unwrap_or(""),
                                pins.len(),
                                if pins.len() == 1 { "" } else { "s" },
                                shown.join(", "),
                                if pins.len() > shown.len() {
                                    ", …"
                                } else {
                                    ""
                                }
                            ));
                        }
                        if rows.len() > 4 {
                            out.push(format!("and {} more instances", rows.len() - 4));
                        }
                    }
                    "net_class" => {
                        if let Some(row) = projection["net_classes"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .find(|row| row["name"].as_str() == Some(s.name.as_str()))
                        {
                            let params: Vec<String> = row["params"]
                                .as_object()
                                .into_iter()
                                .flatten()
                                .map(|(k, v)| format!("{k} = {}", v.as_str().unwrap_or("")))
                                .collect();
                            out.push(format!(
                                "**{} nets**{}",
                                row["nets"].as_u64().unwrap_or(0),
                                if params.is_empty() {
                                    String::new()
                                } else {
                                    format!(": {}", params.join(", "))
                                }
                            ));
                        }
                    }
                    "sheet" => {
                        let paths: Vec<&str> = projection["hierarchy"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter(|row| row["sheet"].as_str() == Some(s.name.as_str()))
                            .filter_map(|row| row["path"].as_str())
                            .collect();
                        if !paths.is_empty() {
                            out.push(format!("**Instances:** {}", paths.join(", ")));
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        out
    }

    pub fn completion_items(&self, position: &Position) -> Vec<CompletionItem> {
        let source = self.source();
        let offset = position_to_offset(source, position);
        let line_start = source[..offset].rfind('\n').map_or(0, |i| i + 1);
        let before = &source[line_start..offset];
        let scope = self.symbols.scope_at(offset);
        if let Some(path) = dotted_prefix(before) {
            return self.member_completions(&path, scope);
        }
        let vocabulary = highlight::dialect_vocabulary(self.dialect);
        let after_number = before
            .trim_end_matches(|c: char| c.is_alphanumeric() || c == '_')
            .trim_end()
            .ends_with(|c: char| c.is_ascii_digit());
        let mut items = Vec::new();
        if after_number {
            let mut units: Vec<&String> = vocabulary.units.iter().collect();
            units.sort();
            items.extend(units.into_iter().map(|u| CompletionItem {
                label: u.clone(),
                kind: Some(CompletionItemKind::UNIT),
                ..CompletionItem::default()
            }));
            return items;
        }
        if let Some(field) = assigned_field(before) {
            items.extend(self.value_completions(field));
        }
        let mut words: Vec<(&String, &TokenClass)> = vocabulary.words.iter().collect();
        words.sort_by_key(|(word, _)| *word);
        items.extend(words.into_iter().map(|(word, token)| CompletionItem {
            label: word.clone(),
            kind: Some(match token.token_type {
                KEYWORD => CompletionItemKind::KEYWORD,
                FUNCTION => CompletionItemKind::FUNCTION,
                NAMESPACE => CompletionItemKind::CLASS,
                _ => CompletionItemKind::ENUM_MEMBER,
            }),
            ..CompletionItem::default()
        }));
        let circuit = matches!(self.dialect, Dialect::Circuit | Dialect::CircuitLayout);
        for s in &self.symbols.symbols {
            let label = if circuit {
                s.name.clone()
            } else {
                format!("{}.{}", s.kind, s.name)
            };
            if !circuit
                && s.parent
                    .is_some_and(|p| self.symbols.symbols[p].parent.is_some())
            {
                continue;
            }
            items.push(CompletionItem {
                label,
                detail: Some(s.kind.clone()),
                kind: Some(completion_kind(&s.kind)),
                ..CompletionItem::default()
            });
        }
        items.sort_by(|a, b| a.label.cmp(&b.label));
        items.dedup_by(|a, b| a.label == b.label);
        items
    }

    /// Completions after `<path>.`: the members a reference can take there.
    fn member_completions(&self, path: &[String], scope: Option<usize>) -> Vec<CompletionItem> {
        let table = &self.symbols;
        let words = |list: &[&str], kind: CompletionItemKind| -> Vec<CompletionItem> {
            list.iter()
                .map(|w| CompletionItem {
                    label: (*w).to_string(),
                    kind: Some(kind),
                    ..CompletionItem::default()
                })
                .collect()
        };
        let named = |indices: Vec<usize>| -> Vec<CompletionItem> {
            let mut items: Vec<CompletionItem> = indices
                .into_iter()
                .map(|i| {
                    let s = &table.symbols[i];
                    CompletionItem {
                        label: s.name.clone(),
                        detail: Some(s.kind.clone()),
                        kind: Some(completion_kind(&s.kind)),
                        ..CompletionItem::default()
                    }
                })
                .collect();
            items.sort_by(|a, b| a.label.cmp(&b.label));
            items.dedup_by(|a, b| a.label == b.label);
            items
        };
        let head = path[0].as_str();
        if self.dialect == Dialect::Part || self.dialect == Dialect::Assembly {
            match path {
                [_] if head == "origin" => {
                    return words(
                        &["xy", "yz", "zx", "x", "y", "z", "point"],
                        CompletionItemKind::ENUM_MEMBER,
                    )
                }
                [f, _] if f == "feature" => {
                    return words(
                        runmat_cad::syntax::refs::TOPOLOGY,
                        CompletionItemKind::PROPERTY,
                    )
                }
                [f, _, topo] if f == "feature" && (topo == "face" || topo == "edge") => {
                    let mut roles: Vec<&str> = runmat_cad::syntax::refs::ROLES
                        .iter()
                        .map(|role| role.as_str())
                        .collect();
                    roles.sort_unstable();
                    roles.dedup();
                    return words(&roles, CompletionItemKind::ENUM_MEMBER);
                }
                _ => {}
            }
        }
        if path.len() == 1 && table.has_kind(head) {
            return named(
                (0..table.symbols.len())
                    .filter(|&i| table.symbols[i].kind == head)
                    .collect(),
            );
        }
        // A path through declared names: the last one's members.
        let probe = symbols::Path {
            segments: path.iter().map(|s| (s.clone(), Span::default())).collect(),
            scope,
        };
        let mut resolved = table.resolve(self.dialect, &probe);
        if resolved.is_empty() && path.len() == 1 {
            // `s1.` in a part: the sketch's entities (a bare name only resolves once dotted).
            resolved.extend(
                table
                    .lookup(None, head, scope)
                    .map(|s| (s, Span::default())),
            );
        }
        if let Some(&(last, _)) = resolved.last() {
            let consumed = resolved.len() + usize::from(table.has_kind(head) && path.len() >= 2);
            if consumed == path.len() {
                let children: Vec<usize> = table.children(last).collect();
                if !children.is_empty() {
                    return named(children);
                }
                if let Some(pins) = self.pins_of(last) {
                    return words(
                        &pins.iter().map(String::as_str).collect::<Vec<_>>(),
                        CompletionItemKind::FIELD,
                    );
                }
            }
        }
        Vec::new()
    }

    /// A component's pins, from the compiled netlist (`U.A.IN+` takes `A`, then `IN+`).
    fn pins_of(&self, symbol: usize) -> Option<Vec<String>> {
        let s = &self.symbols.symbols[symbol];
        if s.kind != "component" {
            return None;
        }
        let nets = self.projection.as_ref()?["nets"].as_array()?.clone();
        let components = self.projection.as_ref()?["components"].as_array()?.clone();
        let designators: Vec<String> = components
            .iter()
            .filter(|row| {
                row["path"]
                    .as_str()
                    .is_some_and(|p| p.rsplit('/').next() == Some(s.name.as_str()))
            })
            .filter_map(|row| row["designator"].as_str().map(str::to_string))
            .collect();
        let mut pins: Vec<String> = nets
            .iter()
            .flat_map(|row| row["pins"].as_array().cloned().unwrap_or_default())
            .filter_map(|pin| {
                let pin = pin.as_str()?;
                let (designator, name) = pin.split_once('.')?;
                designators
                    .iter()
                    .any(|d| d == designator)
                    .then(|| name.to_string())
            })
            .collect();
        pins.sort();
        pins.dedup();
        (!pins.is_empty()).then_some(pins)
    }

    /// Values for `<field> = `: parts for `part`, packages for `package`, nets for `class`'s
    /// kin, sheets for `sheet`.
    fn value_completions(&self, field: &str) -> Vec<CompletionItem> {
        let item = |label: String, detail: &str, kind: CompletionItemKind| CompletionItem {
            label,
            detail: Some(detail.to_string()),
            kind: Some(kind),
            ..CompletionItem::default()
        };
        let pool = || runmat_circuit::fixture::standard_pool();
        match (self.dialect, field) {
            (Dialect::Circuit, "part") => {
                let pool = pool();
                pool.parts
                    .values()
                    .map(|p| item(p.mpn.clone(), &p.description, CompletionItemKind::CONSTANT))
                    .collect()
            }
            (Dialect::CircuitLayout, "package") => {
                let pool = pool();
                let mut items: Vec<CompletionItem> = pool
                    .packages
                    .values()
                    .map(|p| item(p.name.clone(), &p.description, CompletionItemKind::CONSTANT))
                    .collect();
                items.extend(
                    self.symbols
                        .symbols
                        .iter()
                        .filter(|s| s.kind == "package")
                        .map(|s| item(s.name.clone(), "package", CompletionItemKind::CONSTANT)),
                );
                items
            }
            (_, "class") => self.kind_items("net_class"),
            (_, "sheet") => self.kind_items("sheet"),
            (_, "members") => self.kind_items("footprint"),
            (Dialect::CircuitLayout, "net") => self.kind_items("net"),
            _ => Vec::new(),
        }
    }

    fn kind_items(&self, kind: &str) -> Vec<CompletionItem> {
        self.symbols
            .symbols
            .iter()
            .filter(|s| s.kind == kind)
            .map(|s| CompletionItem {
                label: s.name.clone(),
                detail: Some(kind.to_string()),
                kind: Some(completion_kind(kind)),
                ..CompletionItem::default()
            })
            .collect()
    }
}

/// The dotted path before the cursor when it ends in `.`, or a member being typed after one
/// (`feature.body.fa` → `feature`, `body`).
fn dotted_prefix(before: &str) -> Option<Vec<String>> {
    let tail_start = before
        .char_indices()
        .rev()
        .find(|(_, c)| !(c.is_alphanumeric() || matches!(c, '_' | '.' | '+' | '-')))
        .map_or(0, |(i, c)| i + c.len_utf8());
    let tail = &before[tail_start..];
    let (path, _partial) = tail.rsplit_once('.')?;
    if path.is_empty() || path.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    let segments: Vec<String> = path.split('.').map(str::to_string).collect();
    (!segments.iter().any(String::is_empty)).then_some(segments)
}

/// The field a value is being written for (`part = ` → `part`).
fn assigned_field(before: &str) -> Option<&str> {
    let trimmed = before.trim_end_matches(|c: char| c.is_alphanumeric() || matches!(c, '_' | '-'));
    let trimmed = trimmed
        .trim_end()
        .trim_end_matches(['[', ',', ' '])
        .trim_end();
    let head = trimmed.strip_suffix('=')?.trim_end();
    let start = head
        .char_indices()
        .rev()
        .find(|(_, c)| !(c.is_alphanumeric() || *c == '_'))
        .map_or(0, |(i, c)| i + c.len_utf8());
    let field = &head[start..];
    (!field.is_empty()).then_some(field)
}

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn si_text(si: f64, kind: &str) -> String {
    let unit = match kind {
        "length" => "m",
        "angle" => "rad",
        "mass" => "kg",
        "area" => "m²",
        "volume" => "m³",
        "force" => "N",
        "pressure" => "Pa",
        "temperature" => "K",
        "time" => "s",
        _ => "",
    };
    let value = if si != 0.0 && (si.abs() < 1e-3 || si.abs() >= 1e6) {
        format!("{si:e}")
    } else {
        format!("{}", (si * 1e9).round() / 1e9)
    };
    if kind == "length" {
        format!("{} mm ({value} m)", (si * 1e3 * 1e6).round() / 1e6)
    } else if kind == "angle" {
        format!("{}° ({value} rad)", (si.to_degrees() * 1e6).round() / 1e6)
    } else if unit.is_empty() {
        value
    } else {
        format!("{value} {unit}")
    }
}

fn fence(dialect: Dialect) -> &'static str {
    match dialect {
        Dialect::Part => "part",
        Dialect::Assembly => "assembly",
        Dialect::Drawing => "drawing",
        Dialect::Circuit => "circuit",
        Dialect::CircuitLayout => "circuit-layout",
        Dialect::Fea => "fea",
    }
}

fn source_label(dialect: Dialect) -> &'static str {
    match dialect {
        Dialect::Circuit | Dialect::CircuitLayout => "runmat-circuit",
        _ => "runmat-cad",
    }
}

fn symbol_kind(kind: &str) -> SymbolKind {
    match kind {
        "part" | "assembly" | "drawing" | "circuit" | "layout" => SymbolKind::MODULE,
        "sheet" | "package" | "stackup" => SymbolKind::CLASS,
        "feature" | "feature_def" => SymbolKind::FUNCTION,
        "sketch" | "region" | "room" => SymbolKind::NAMESPACE,
        "input" | "let" | "property" | "param" => SymbolKind::VARIABLE,
        "net" | "bus" | "port" => SymbolKind::FIELD,
        "net_class" | "component_class" | "material" => SymbolKind::STRUCT,
        "component" | "footprint" | "instance" | "occurrence" => SymbolKind::OBJECT,
        "configuration" | "variant" => SymbolKind::ENUM,
        "publish" | "output" => SymbolKind::INTERFACE,
        _ => SymbolKind::CONSTANT,
    }
}

fn completion_kind(kind: &str) -> CompletionItemKind {
    match symbol_kind(kind) {
        SymbolKind::MODULE => CompletionItemKind::MODULE,
        SymbolKind::CLASS => CompletionItemKind::CLASS,
        SymbolKind::FUNCTION => CompletionItemKind::FUNCTION,
        SymbolKind::VARIABLE => CompletionItemKind::VARIABLE,
        SymbolKind::FIELD => CompletionItemKind::FIELD,
        SymbolKind::STRUCT => CompletionItemKind::STRUCT,
        SymbolKind::ENUM => CompletionItemKind::ENUM,
        SymbolKind::INTERFACE => CompletionItemKind::INTERFACE,
        _ => CompletionItemKind::REFERENCE,
    }
}

/// Every token's class: the grammar's, then declared names and their uses from the table.
fn classify(syntax: &SourceSyntax, table: Option<&SymbolTable>) -> Vec<(Span, TokenClass)> {
    let source = syntax.source();
    let vocabulary = highlight::dialect_vocabulary(syntax.dialect);
    let mut out: Vec<(Span, TokenClass)> =
        highlight::classify_with(vocabulary, source, &syntax.tokens.tokens)
            .into_iter()
            .map(|(token, token_class)| (token.span, token_class))
            .collect();
    let Some(table) = table else {
        return out;
    };
    let mut marks: BTreeMap<usize, (Span, TokenClass)> = BTreeMap::new();
    for s in &table.symbols {
        marks.insert(
            s.name_span.start,
            (s.name_span, class(VARIABLE, DECLARATION)),
        );
    }
    for o in &table.occurrences {
        marks
            .entry(o.span.start)
            .or_insert((o.span, class(VARIABLE, 0)));
    }
    // A mark replaces the token it starts (a quoted name's mark sits inside its string token).
    for entry in out.iter_mut() {
        if let Some((span, mark)) = marks.get(&entry.0.start) {
            if span.end == entry.0.end {
                entry.1 = *mark;
            }
        }
    }
    let mut extra: Vec<(Span, TokenClass)> = Vec::new();
    for (span, mark) in marks.values() {
        if !out
            .iter()
            .any(|(s, _)| s.start == span.start && s.end == span.end)
        {
            extra.push((*span, *mark));
        }
    }
    if !extra.is_empty() {
        // Split string tokens around names inside them so tokens never overlap.
        let mut split = Vec::with_capacity(out.len() + extra.len() * 2);
        for (span, token_class) in out {
            let inside: Vec<&(Span, TokenClass)> = extra
                .iter()
                .filter(|(e, _)| e.start >= span.start && e.end <= span.end)
                .collect();
            if inside.is_empty() {
                split.push((span, token_class));
                continue;
            }
            let mut at = span.start;
            for (e, mark) in inside {
                if e.start > at {
                    split.push((
                        Span {
                            start: at,
                            end: e.start,
                        },
                        token_class,
                    ));
                }
                split.push((*e, *mark));
                at = e.end;
            }
            if at < span.end {
                split.push((
                    Span {
                        start: at,
                        end: span.end,
                    },
                    token_class,
                ));
            }
        }
        out = split;
    }
    out
}

fn encode(source: &str, tokens: Vec<(Span, TokenClass)>) -> SemanticTokens {
    let mut data = Vec::with_capacity(tokens.len());
    let mut previous = Position::new(0, 0);
    let mut first = true;
    for (span, token_class) in tokens {
        // A token spanning lines (a block comment, a long string) is sent line by line.
        let text = source.get(span.start..span.end).unwrap_or("");
        let mut start = span.start;
        for piece in text.split_inclusive('\n') {
            let content = piece.trim_end_matches(['\n', '\r']);
            if !content.is_empty() {
                let position = offset_to_position(source, start);
                let delta_line = if first {
                    position.line
                } else {
                    position.line - previous.line
                };
                let delta_start = if first || delta_line > 0 {
                    position.character
                } else {
                    position.character - previous.character
                };
                data.push(SemanticToken {
                    delta_line,
                    delta_start,
                    length: content.encode_utf16().count() as u32,
                    token_type: token_class.token_type,
                    token_modifiers_bitset: token_class.modifiers,
                });
                previous = position;
                first = false;
            }
            start += piece.len();
        }
    }
    SemanticTokens {
        result_id: None,
        data,
    }
}

/// Runs the dialect's compiler: its problems in this file and its projection.
fn compile(
    dialect: Dialect,
    path: &str,
    text: &str,
    files: &DesignFiles,
    table: &SymbolTable,
) -> (Vec<Problem>, Option<Value>) {
    match dialect {
        Dialect::Part => compile_part(path, text, table),
        Dialect::Assembly => {
            let (mut problems, projection) = cad_front_end(
                runmat_cad::syntax::parse_assembly(path, text),
                path,
                files,
                text,
            );
            problems.extend(assembly_problems(path, text, files));
            (problems, projection)
        }
        Dialect::Drawing => cad_front_end(
            runmat_cad::syntax::parse_drawing(path, text),
            path,
            files,
            text,
        ),
        Dialect::Circuit | Dialect::CircuitLayout => {
            let out = runmat_circuit::runtime::check(path, text, files, &Default::default());
            let problems = out
                .diagnostics
                .iter()
                .map(|d| {
                    let in_file = d.path == path;
                    Problem {
                        span: if in_file {
                            Span {
                                start: byte_offset(text, d.start),
                                end: byte_offset(text, d.end),
                            }
                        } else {
                            Span::default()
                        },
                        severity: severity(&d.severity),
                        code: None,
                        message: if in_file {
                            d.message.clone()
                        } else {
                            format!("{}: {}", d.path, d.message)
                        },
                    }
                })
                .collect();
            (
                problems,
                out.index.and_then(|index| serde_json::to_value(index).ok()),
            )
        }
        Dialect::Fea => (Vec::new(), None),
    }
}

fn severity(text: &str) -> DiagnosticSeverity {
    match text {
        "warning" => DiagnosticSeverity::WARNING,
        "info" => DiagnosticSeverity::INFORMATION,
        _ => DiagnosticSeverity::ERROR,
    }
}

/// The byte offset of a UTF-16 offset.
fn byte_offset(text: &str, utf16: usize) -> usize {
    if text.is_ascii() {
        return utf16.min(text.len());
    }
    let mut units = 0;
    for (byte, ch) in text.char_indices() {
        if units >= utf16 {
            return byte;
        }
        units += ch.len_utf16();
    }
    text.len()
}

/// A part regenerated on the reference kernel; regeneration diagnostics at their sites.
fn compile_part(path: &str, text: &str, table: &SymbolTable) -> (Vec<Problem>, Option<Value>) {
    use runmat_cad::regen::{regenerate, FakeKernel, RegenCache, RegenContext, RegenRequest};
    let parsed = runmat_cad::syntax::parse_part(path, text);
    let mut problems = model_problems(&parsed.diagnostics);
    let Some(document) = parsed.document else {
        return (problems, None);
    };
    let kernel = FakeKernel::new();
    let ctx = RegenContext {
        kernel: &kernel,
        solver: &runmat_cad::sketch_solver::ExplicitSketchSolver,
        units: &runmat_cad::units::StubUnits,
        dynamic: &runmat_cad::value::NoDynamicValues,
        lock: None,
        catalog: None,
        policy: Default::default(),
        slots: None,
    };
    let result = regenerate(
        &document,
        &ctx,
        &RegenRequest::default(),
        &mut RegenCache::default(),
    );
    let mut report = result.diagnostics.clone();
    report.attach_spans(&parsed.spans);
    for d in &report.diagnostics {
        let span = d
            .span
            .as_ref()
            .map(|s| Span {
                start: s.start as usize,
                end: s.end as usize,
            })
            .or_else(|| site_span(&d.site, table))
            .unwrap_or_default();
        let cascade = matches!(d.cause, runmat_cad::diagnostics::Cause::Cascade { .. });
        problems.push(Problem {
            span,
            severity: match d.severity {
                runmat_cad::diagnostics::Severity::Error if cascade => DiagnosticSeverity::WARNING,
                runmat_cad::diagnostics::Severity::Error => DiagnosticSeverity::ERROR,
                runmat_cad::diagnostics::Severity::Warning => DiagnosticSeverity::WARNING,
                _ => DiagnosticSeverity::INFORMATION,
            },
            code: serde_json::to_value(d.code)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string)),
            message: d.message.clone(),
        });
    }
    let index = runmat_cad::index::build_index(&document, &result, None);
    let projection = serde_json::json!({ "document": &document, "index": &index });
    (problems, Some(projection))
}

/// The declaration a diagnostic's site names, when the span map has no entry for it.
fn site_span(site: &runmat_cad::diagnostics::Site, table: &SymbolTable) -> Option<Span> {
    let key = site.key();
    let base = key.split('#').next().unwrap_or(&key);
    let (kind, name) = base.split_once('.')?;
    table
        .symbols
        .iter()
        .find(|s| s.kind == kind && s.name == name)
        .map(|s| s.name_span)
}

fn model_problems(diagnostics: &[runmat_design_syntax::FeaSyntaxDiagnostic]) -> Vec<Problem> {
    diagnostics
        .iter()
        .map(|d| Problem {
            span: d.span,
            severity: DiagnosticSeverity::ERROR,
            code: Some("design.model".into()),
            message: d.message.clone(),
        })
        .collect()
}

/// An assembly's or drawing's front end: model errors, and each design file it names that
/// isn't open or in the workspace.
fn cad_front_end<T>(
    parsed: runmat_cad::syntax::Parsed<T>,
    path: &str,
    files: &DesignFiles,
    text: &str,
) -> (Vec<Problem>, Option<Value>) {
    let mut problems = model_problems(&parsed.diagnostics);
    let syntax = parse_source(Dialect::for_path(path).unwrap_or(Dialect::Assembly), text);
    let source = syntax.source();
    for node in syntax.syntax().descendants() {
        if node.kind() != runmat_design_syntax::tree::NodeKind::StringLiteral {
            continue;
        }
        let span = node.trimmed_span();
        let Some(inner) = source
            .get(span.start..span.end)
            .and_then(|t| t.strip_prefix('"'))
            .and_then(|t| t.strip_suffix('"'))
        else {
            continue;
        };
        if !(inner.ends_with(".part") || inner.ends_with(".assembly")) {
            continue;
        }
        let resolved = runmat_circuit::runtime::resolve_path(path, inner);
        if !files.contains_key(&resolved)
            && !files
                .keys()
                .any(|k| k.ends_with(&format!("/{}", inner.trim_start_matches("./"))))
        {
            problems.push(Problem {
                span,
                severity: DiagnosticSeverity::WARNING,
                code: Some("cad.file.missing".into()),
                message: format!("{inner} isn't open or in the workspace, so it draws as its box"),
            });
        }
    }
    (problems, None)
}

/// An assembly's resolve diagnostics (mates, patterns, occurrences, configurations), from the
/// design tools' check over the parts it uses. Front-end and missing-file problems are left to
/// [`cad_front_end`], which places them on the `use` line.
fn assembly_problems(path: &str, text: &str, files: &DesignFiles) -> Vec<Problem> {
    let mut all = files.clone();
    all.insert(path.to_owned(), text.to_owned());
    let Ok(out) = runmat_design_tools::check::check(
        &runmat_design_tools::check::CheckInput {
            path: path.to_owned(),
        },
        &all,
    ) else {
        return Vec::new();
    };
    out.diagnostics
        .into_iter()
        .filter(|d| d.code.is_some())
        .map(|d| Problem {
            span: d
                .range
                .as_ref()
                .filter(|r| r.path == path)
                .map(|r| Span {
                    start: byte_offset(text, r.start),
                    end: byte_offset(text, r.end),
                })
                .unwrap_or_default(),
            severity: severity(&d.severity),
            code: d.code,
            message: d.message,
        })
        .collect()
}
