//! Design documents from registries: the shared front-end engine.
//!
//! A design front end describes its model to this engine with a [`DocumentGrammar`]: the word
//! that heads the file's declaration, and which statement words fill which model collections.
//! The engine parses a source into the model's JSON form ([`Json`]) and prints a model's JSON
//! form back as source, so every dialect gets the same statement shapes:
//!
//! ```text
//! runmat 0.7
//! std 1
//!
//! use material aluminum from "./materials"
//!
//! part bracket {
//!     input width : length = 80 mm                       // <word> <id> [: <type>] = <value>
//!     feature body = extrude { profile = …, extent = … } // tagged: <tag> { fields }
//!     publish hardpoint = { publishes = region, region = mount }
//!     material = { body = …, material = … }              // an unnamed entry
//!     link verifies feature.body to "checks/mass.fea#check.mass_budget"
//!     closed = true                                      // any other model field
//! }
//! ```
//!
//! Any model field without a statement word is written `<field> = <value>`, so every model the
//! front end can serialise prints, and parses back to the same model. A field whose value is
//! the model's default is left out ([`DocumentModel::normalize`] decides). Each entry and each
//! value inside it records its span under a site key (`feature.round`, `feature.round#edges[1]`).

use crate::ast::{self, Expr, Headed, Item, Statement};
use crate::json::{
    is_bare_word, is_identifier, join_path, lower_arguments, lower_expr, lower_record,
    print_arguments, print_json, print_key, quote, Json, LowerError, PathSpans, ValueSyntax,
    INDENT,
};
use crate::{parse_source, Dialect, FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};
use runmat_types::Span;
use std::collections::BTreeMap;

/// How a front end's model maps to statements.
#[derive(Debug, Clone, Copy)]
pub struct DocumentGrammar {
    pub dialect: Dialect,
    /// The word heading the file's one declaration (`part`).
    pub declaration: &'static str,
    /// The model field holding the declaration's name; `None`: the name is the file stem.
    pub name_field: Option<&'static str>,
    /// The model field holding the file path, filled from the path given to the parser.
    pub path_field: Option<&'static str>,
    /// The model field holding `{ language, std }`, written as the version header.
    pub header_field: Option<&'static str>,
    pub collections: &'static [Collection],
    pub links: Option<LinkGrammar>,
}

/// Statements of one word filling one collection of the model.
#[derive(Debug, Clone, Copy)]
pub struct Collection {
    /// The statement word (`input`, `feature`, `publish`).
    pub word: &'static str,
    /// The model field holding the entries (`inputs`).
    pub field: &'static str,
    /// The entry field holding its name (`name`, `id`); `None`: unnamed entries, written
    /// `<word> = <value>` and keyed by position.
    pub id: Option<&'static str>,
    /// The entry field written as the type after `:`.
    pub type_field: Option<&'static str>,
    /// The entry field written alone after `=` when no other field is set.
    pub value_field: Option<&'static str>,
    /// Entries whose `field` is a tagged enum, written `<tag value> { fields }`.
    pub tagged: Option<Tagged>,
    /// The site key prefix (`feature` → `feature.<id>`, `material` → `material[0]`).
    pub site: &'static str,
    /// Names are unique. When false (alternatives sharing a name), a repeated name's site key
    /// takes its ordinal: `occurrence.engine~2`.
    pub unique: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Tagged {
    /// The entry field holding the tagged value (`kind`).
    pub field: &'static str,
    /// The tag key inside it (`feature`).
    pub tag: &'static str,
    /// Entry fields written in the same record as the tagged value's fields (`suppressed`).
    pub extras: &'static [&'static str],
}

/// `link <kind> <item kind>.<item> to "<target>"`.
#[derive(Debug, Clone, Copy)]
pub struct LinkGrammar {
    pub field: &'static str,
    pub kind: &'static str,
    /// The entry field holding the local item, a tagged `{ <tag>: <item kind>, <item kind>:
    /// <name> }`.
    pub from: &'static str,
    pub from_tag: &'static str,
    pub to: &'static str,
    pub site: &'static str,
}

/// A front end's view of its model, for printing.
pub trait DocumentModel {
    /// Deserialises and serialises `json` (the model's canonical form), or `None` when it is
    /// not a valid model.
    fn normalize(&self, json: &Json) -> Option<Json>;
}

/// A front end's statement forms beyond the engine's generic ones, reached through
/// [`ValueSyntax::entry_forms`]. Every method has a default that keeps the generic forms.
pub trait EntryForms {
    /// The key of a tagged entry's second tag, written as a dotted callee: `plane` for the
    /// feature tag `plane` gives `feature mid = plane.offset(base = origin.xy, …)`.
    fn subtag(&self, _word: &str, _tag: &str) -> Option<&'static str> {
        None
    }

    /// Turns a call's arguments into the tagged value's fields (`distance = 10 mm` →
    /// `extent = { extent = blind, distance = 10 mm }`). Also returns, for each argument that
    /// moved, its key and the model path it now fills (`("distance", "extent.distance")`).
    #[allow(clippy::type_complexity)]
    fn desugar(
        &self,
        _word: &str,
        _tag: &str,
        arguments: Vec<(String, Json)>,
    ) -> Result<(Vec<(String, Json)>, Vec<(String, String)>), String> {
        Ok((arguments, Vec::new()))
    }

    /// The inverse of [`EntryForms::desugar`]: the tagged value's fields (defaults left out)
    /// as call arguments, or `None` to write the fields as they are.
    fn sugar(
        &self,
        _word: &str,
        _tag: &str,
        _fields: &[(String, Json)],
    ) -> Option<Vec<(String, Json)>> {
        None
    }

    /// A block statement of the front end's own (`sketch s1 on origin.xy { … }`).
    fn lower_block(
        &self,
        _block: &ast::BlockStatement,
        _hook: &dyn ValueSyntax,
    ) -> Option<Result<BlockEntry, LowerError>> {
        None
    }

    /// A `use` statement inside the declaration (`use part bracket from "…" as left { … }`).
    fn lower_use(
        &self,
        _use: &ast::UseDeclaration,
        _hook: &dyn ValueSyntax,
    ) -> Option<Result<BlockEntry, LowerError>> {
        None
    }

    /// Prints an entry as a whole statement in the front end's own form, or `None` for the
    /// engine's forms. `fields` are the entry's fields with defaults left out, its name
    /// excluded.
    fn print_statement(
        &self,
        _word: &str,
        _name: &str,
        _fields: &[(String, Json)],
        _hook: &dyn ValueSyntax,
    ) -> Option<String> {
        None
    }
}

/// An entry a front end read from one of its own statements.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockEntry {
    /// The statement word of the collection it joins (`feature`).
    pub word: String,
    pub name: Option<String>,
    /// The entry's fields besides its name.
    pub fields: Vec<(String, Json)>,
    /// Model paths inside the entry (`kind.entities[0]`) and the spans that wrote them.
    pub spans: PathSpans,
    /// The values as written, for structured edits: key, value and span.
    pub written: Vec<(String, Json, Span)>,
    /// Where the statement's head ends (its block or its overrides start).
    pub head_end: usize,
    /// The span replaced when the written values change shape.
    pub value_span: Option<Span>,
}

/// A `use` line.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UseItem {
    /// `Some((kind, name))`: one symbol; `None`: a whole folder.
    pub symbol: Option<(String, String)>,
    pub from: String,
    pub alias: Option<String>,
    pub revision: Option<String>,
    pub span: Span,
}

impl UseItem {
    pub fn print(&self) -> String {
        let mut out = String::from("use ");
        if let Some((kind, name)) = &self.symbol {
            out.push_str(&format!("{kind} {name} from "));
        }
        out.push_str(&quote(&self.from));
        if let Some(revision) = &self.revision {
            out.push_str(&format!(" @ {}", quote(revision)));
        }
        if let Some(alias) = &self.alias {
            out.push_str(&format!(" as {alias}"));
        }
        out
    }
}

/// One statement the engine read, as written: what structured edits compare and replace.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedEntry {
    /// The site key (`feature.round`, `material[0]`, `closed`).
    pub site: String,
    /// The model field it fills (`features`, or the field itself for `<field> = <value>`).
    pub field: String,
    /// The statement without surrounding trivia.
    pub span: Span,
    /// The statement up to its value (`feature round =`), whitespace-normalised.
    pub head: String,
    pub head_span: Span,
    /// The value after `=`.
    pub value_span: Option<Span>,
    /// The values as written, by key: a record's entries in order, `(value field, value)` for
    /// the short form, `("", value)` for a field. `None`: compared as a whole statement.
    pub written: Option<Vec<(String, Json, Span)>>,
}

/// A parsed design document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSyntax {
    /// The model's JSON form, ready to deserialise.
    pub json: Json,
    pub uses: Vec<UseItem>,
    /// Site key → the span of the text that wrote it.
    pub spans: BTreeMap<String, Span>,
    /// JSON path (`features[2].kind.extent`) → span, for placing deserialisation errors.
    pub json_spans: Vec<(String, Span)>,
    pub diagnostics: Vec<FeaSyntaxDiagnostic>,
    /// The declaration's span (or the file start when there is none).
    pub declaration: Span,
    /// The declaration's block, `{` to `}`.
    pub block: Option<Span>,
    /// Every statement the engine read, in source order.
    pub entries: Vec<ParsedEntry>,
}

impl DocumentSyntax {
    /// The span of the deepest value written at or above a JSON path.
    pub fn span_of_json_path(&self, path: &str) -> Span {
        let mut best: Option<(usize, Span)> = None;
        for (candidate, span) in &self.json_spans {
            let covers = path == candidate
                || (path.starts_with(candidate.as_str())
                    && matches!(path.as_bytes().get(candidate.len()), Some(b'.' | b'[')));
            if covers && best.is_none_or(|(length, _)| candidate.len() > length) {
                best = Some((candidate.len(), *span));
            }
        }
        best.map_or(self.declaration, |(_, span)| span)
    }

    /// A diagnostic for a model that failed to deserialise at `path`.
    pub fn model_error(&self, path: &str, message: &str) -> FeaSyntaxDiagnostic {
        FeaSyntaxDiagnostic {
            kind: FeaSyntaxDiagnosticKind::InvalidExpression,
            span: self.span_of_json_path(path),
            message: if path.is_empty() {
                message.to_string()
            } else {
                format!("{path}: {message}")
            },
        }
    }
}

fn diagnostic(span: Span, message: impl Into<String>) -> FeaSyntaxDiagnostic {
    FeaSyntaxDiagnostic {
        kind: FeaSyntaxDiagnosticKind::InvalidExpression,
        span,
        message: message.into(),
    }
}

/// Values as written: `(key, value, span)` in source order.
type Written = Vec<(String, Json, Span)>;

/// Collapses runs of whitespace to one space.
fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The top-level values of a lowered record, with the spans of their expressions. `extras`
/// spans (entry-level fields written in a tagged record) are searched too.
fn written_values(
    entries: &Result<Vec<(String, Json)>, FeaSyntaxDiagnostic>,
    spans: &[(String, Span)],
    extras: &[(String, Span)],
) -> Vec<(String, Json, Span)> {
    let Ok(entries) = entries else {
        return Vec::new();
    };
    entries
        .iter()
        .map(|(key, value)| {
            let span = spans
                .iter()
                .chain(extras)
                .find(|(path, _)| path == key)
                .map(|(_, span)| *span)
                .unwrap_or_default();
            (key.clone(), value.clone(), span)
        })
        .collect()
}

fn lower_error(error: LowerError) -> FeaSyntaxDiagnostic {
    diagnostic(error.span, error.message)
}

struct Parse<'a> {
    source: &'a str,
    entries: Vec<ParsedEntry>,
    block: Option<Span>,
    grammar: &'a DocumentGrammar,
    hook: &'a dyn ValueSyntax,
    fields: Vec<(String, Json)>,
    collections: BTreeMap<&'static str, Vec<Json>>,
    spans: BTreeMap<String, Span>,
    json_spans: Vec<(String, Span)>,
    diagnostics: Vec<FeaSyntaxDiagnostic>,
}

/// Parses `source` (at `path`) with `grammar`.
pub fn parse_document(
    grammar: &DocumentGrammar,
    path: &str,
    source: &str,
    hook: &dyn ValueSyntax,
) -> DocumentSyntax {
    let syntax = parse_source(grammar.dialect, source);
    let file = syntax.file();
    let mut parse = Parse {
        source,
        entries: Vec::new(),
        block: None,
        grammar,
        hook,
        fields: Vec::new(),
        collections: BTreeMap::new(),
        spans: BTreeMap::new(),
        json_spans: Vec::new(),
        diagnostics: syntax.diagnostics.clone(),
    };
    if let Some(field) = grammar.path_field {
        parse
            .fields
            .push((field.to_string(), Json::String(path.to_string())));
    }
    if let Some(field) = grammar.header_field {
        let mut header = Vec::new();
        if let Some(language) = file.version(false) {
            header.push(("language".to_string(), Json::String(language)));
        }
        if let Some(std) = file.version(true) {
            header.push(("std".to_string(), Json::String(std)));
        }
        if !header.is_empty() {
            parse.fields.push((field.to_string(), Json::Object(header)));
        }
    }
    let mut uses = Vec::new();
    let mut declaration_span = Span { start: 0, end: 0 };
    let mut seen_declaration = false;
    for item in file.items() {
        match item {
            Item::Use(declaration) => match use_item(&declaration) {
                Ok(item) => uses.push(item),
                Err(error) => parse.diagnostics.push(error),
            },
            Item::Import(import) => parse.diagnostics.push(diagnostic(
                import.span(),
                format!(
                    "a {} reads other files with `use`, not `import`",
                    grammar.dialect.spec().label.to_lowercase()
                ),
            )),
            Item::Declaration(declaration) => {
                if declaration.keyword() != grammar.declaration {
                    // The parser has already reported a declaration word of another dialect.
                    continue;
                }
                if seen_declaration {
                    parse.diagnostics.push(diagnostic(
                        declaration.keyword_span(),
                        format!("a file holds one `{}` declaration", grammar.declaration),
                    ));
                    continue;
                }
                seen_declaration = true;
                declaration_span = declaration.span();
                parse.spans.insert("document".into(), declaration_span);
                parse.declaration(&declaration, path);
            }
            Item::Error(_) => {}
        }
    }
    if !seen_declaration {
        parse.diagnostics.push(diagnostic(
            Span { start: 0, end: 0 },
            format!(
                "expected a `{} <name> {{ … }}` declaration",
                grammar.declaration
            ),
        ));
    }
    let mut fields = parse.fields;
    for collection in grammar.collections {
        let entries = parse
            .collections
            .remove(collection.field)
            .unwrap_or_default();
        fields.push((collection.field.to_string(), Json::Array(entries)));
    }
    if let Some(links) = grammar.links {
        let entries = parse.collections.remove(links.field).unwrap_or_default();
        fields.push((links.field.to_string(), Json::Array(entries)));
    }
    DocumentSyntax {
        json: Json::Object(fields),
        uses,
        spans: parse.spans,
        json_spans: parse.json_spans,
        diagnostics: parse.diagnostics,
        declaration: declaration_span,
        block: parse.block,
        entries: parse.entries,
    }
}

fn use_item(declaration: &ast::UseDeclaration) -> Result<UseItem, FeaSyntaxDiagnostic> {
    let from = match declaration.source() {
        Some(ast::UseSource::Path(path)) => path,
        Some(ast::UseSource::Package(segments)) => segments.join("."),
        None => return Err(diagnostic(declaration.span(), "`use` needs a source")),
    };
    if declaration.overrides().is_some() {
        return Err(diagnostic(
            declaration.span(),
            "overrides on a file-level `use` are not part of this dialect",
        ));
    }
    Ok(UseItem {
        symbol: declaration
            .symbol()
            .map(|(kind, name)| (kind.text(), name.text())),
        from,
        alias: declaration.alias().map(|alias| alias.text()),
        revision: declaration.pin(),
        span: declaration.span(),
    })
}

impl Parse<'_> {
    fn declaration(&mut self, declaration: &ast::Declaration, path: &str) {
        let name = declaration
            .name()
            .map(|name| name.text())
            .unwrap_or_default();
        match self.grammar.name_field {
            Some(field) => self.fields.push((field.to_string(), Json::String(name))),
            None => {
                let stem = self.grammar.dialect.stem(path);
                if !path.is_empty() && name != stem {
                    let span = declaration.name().map_or(declaration.span(), |n| n.span());
                    self.diagnostics.push(diagnostic(
                        span,
                        format!(
                            "this {} is named `{name}`; its file is `{stem}`",
                            self.grammar.declaration
                        ),
                    ));
                }
            }
        }
        if let Some(clause) = declaration.clauses().first() {
            self.diagnostics
                .push(diagnostic(clause.span(), "a declaration takes no clauses"));
        }
        let Some(block) = declaration.block() else {
            return;
        };
        self.block = Some(block.span());
        for statement in block.statements() {
            self.statement(&statement);
        }
    }

    fn collection(&self, word: &str) -> Option<&'static Collection> {
        self.grammar.collections.iter().find(|c| c.word == word)
    }

    fn statement(&mut self, statement: &Statement) {
        let span = statement.span();
        match statement {
            Statement::Value(value) => {
                let word = value.keyword();
                let Some(collection) = self.collection(&word).filter(|c| c.id.is_some()) else {
                    self.diagnostics.push(diagnostic(
                        span,
                        format!("`{word}` does not start a statement in a {}", self.label()),
                    ));
                    return;
                };
                let name = value.name().map(|n| n.text()).unwrap_or_default();
                self.entry(
                    collection,
                    Some(name),
                    value.type_text(),
                    value.value(),
                    span,
                );
            }
            Statement::Publish(publish) => {
                self.named_form("publish", publish.name(), publish.value(), span)
            }
            Statement::Check(check) => match check.block() {
                Some(block) => self.check_block(check.name(), block, span),
                None => self.named_form("check", check.name(), check.value(), span),
            },
            Statement::Block(block) => {
                let lowered = self
                    .hook
                    .entry_forms()
                    .and_then(|forms| forms.lower_block(block, self.hook));
                match lowered {
                    Some(Ok(entry)) => self.block_entry(entry, span),
                    Some(Err(error)) => self.diagnostics.push(lower_error(error)),
                    None => self.diagnostics.push(diagnostic(
                        span,
                        format!(
                            "`{}` does not start a statement in a {}",
                            block.keyword(),
                            self.label()
                        ),
                    )),
                }
            }
            Statement::Use(declaration) => {
                let lowered = self
                    .hook
                    .entry_forms()
                    .and_then(|forms| forms.lower_use(declaration, self.hook));
                match lowered {
                    Some(Ok(entry)) => self.block_entry(entry, span),
                    Some(Err(error)) => self.diagnostics.push(lower_error(error)),
                    None => self.diagnostics.push(diagnostic(
                        span,
                        format!(
                            "`use` inside a {} declaration is not part of this dialect",
                            self.label()
                        ),
                    )),
                }
            }
            Statement::Assignment(assignment) => {
                let target = assignment.target();
                if assignment.relation() != ast::Relation::Equal || target.len() != 1 {
                    self.diagnostics.push(diagnostic(
                        assignment.target_span(),
                        format!("expected `<field> = <value>` in a {}", self.label()),
                    ));
                    return;
                }
                let word = target[0].clone();
                if let Some(collection) = self.collection(&word) {
                    if collection.id.is_some() {
                        self.diagnostics.push(diagnostic(
                            assignment.target_span(),
                            format!("`{word}` entries are written `{word} <name> = <value>`"),
                        ));
                        return;
                    }
                    self.entry(collection, None, None, assignment.value(), span);
                    return;
                }
                self.field(&word, assignment.target_span(), assignment.value(), span);
            }
            Statement::Link(link) => self.link(link, span),
            other => {
                let word = match other {
                    Statement::Command(command) => command.word(),
                    _ => other
                        .syntax()
                        .first_child_token()
                        .map(|t| t.text().to_string())
                        .unwrap_or_default(),
                };
                self.diagnostics.push(diagnostic(
                    span,
                    format!("`{word}` does not start a statement in a {}", self.label()),
                ));
            }
        }
    }

    fn label(&self) -> String {
        self.grammar.dialect.spec().label.to_lowercase()
    }

    fn named_form(&mut self, word: &str, name: Option<ast::Name>, value: Option<Expr>, span: Span) {
        match self.collection(word).filter(|c| c.id.is_some()) {
            Some(collection) => {
                let name = name.map(|n| n.text()).unwrap_or_default();
                self.entry(collection, Some(name), None, value, span);
            }
            None => self.diagnostics.push(diagnostic(
                span,
                format!("`{word}` does not start a statement in a {}", self.label()),
            )),
        }
    }

    fn lower(
        &mut self,
        expr: &Expr,
        site: &str,
        json_path: &str,
    ) -> Result<Json, FeaSyntaxDiagnostic> {
        let mut spans = Vec::new();
        let result = lower_expr(expr, self.hook, "", &mut spans);
        self.record_spans(site, json_path, spans);
        result.map_err(lower_error)
    }

    fn record_spans(&mut self, site: &str, json_path: &str, spans: Vec<(String, Span)>) {
        for (path, span) in spans {
            if !path.is_empty() {
                self.spans.entry(format!("{site}#{path}")).or_insert(span);
            }
            let full = if path.is_empty() {
                json_path.to_string()
            } else if path.starts_with('[') {
                format!("{json_path}{path}")
            } else {
                join_path(json_path, &path)
            };
            self.json_spans.push((full, span));
        }
    }

    /// The site key and JSON path the next entry of `collection` named `name` takes, or
    /// `None` (with a diagnostic) when the name is taken.
    fn place(
        &mut self,
        collection: &'static Collection,
        name: Option<&str>,
        span: Span,
    ) -> Option<(String, String)> {
        let entries = self.collections.entry(collection.field).or_default();
        let index = entries.len();
        let same_name = name.map_or(0, |name| {
            entries
                .iter()
                .filter(|entry| {
                    collection
                        .id
                        .and_then(|id| entry.get(id))
                        .and_then(Json::as_str)
                        == Some(name)
                })
                .count()
        });
        let site = match name {
            Some(name) if same_name > 0 && !collection.unique => {
                format!("{}.{name}~{}", collection.site, same_name + 1)
            }
            Some(name) => format!("{}.{name}", collection.site),
            None => format!("{}[{index}]", collection.site),
        };
        if let Some(name) = name.filter(|_| same_name > 0 && collection.unique) {
            self.diagnostics.push(diagnostic(
                span,
                format!("`{} {name}` is declared twice", collection.word),
            ));
            return None;
        }
        Some((site, format!("{}[{index}]", collection.field)))
    }

    /// Adds an entry: its name, its type and the fields its value gave.
    #[allow(clippy::too_many_arguments)]
    fn push_entry(
        &mut self,
        collection: &'static Collection,
        name: Option<String>,
        type_text: Option<String>,
        site: String,
        json_path: &str,
        span: Span,
        head_span: Span,
        value_span: Option<Span>,
        rest: Vec<(String, Json)>,
        written: Written,
    ) {
        let mut fields: Vec<(String, Json)> = Vec::new();
        if let (Some(id), Some(name)) = (collection.id, name) {
            fields.push((id.to_string(), Json::String(name)));
        }
        if let Some(type_text) = type_text {
            match collection.type_field {
                Some(field) => fields.push((field.to_string(), Json::String(type_text))),
                None => {
                    self.diagnostics.push(diagnostic(
                        span,
                        format!("`{}` entries take no type", collection.word),
                    ));
                    return;
                }
            }
        }
        for (key, item) in rest {
            if fields.iter().any(|(existing, _)| *existing == key) {
                self.diagnostics.push(diagnostic(
                    value_span.unwrap_or(span),
                    format!("`{key}` is given twice"),
                ));
                return;
            }
            fields.push((key, item));
        }
        self.spans.insert(site.clone(), span);
        self.json_spans.push((json_path.to_string(), span));
        self.entries.push(ParsedEntry {
            site,
            field: collection.field.to_string(),
            span,
            head: normalize(&self.source[head_span.start..head_span.end]),
            head_span,
            value_span,
            written: Some(written),
        });
        self.collections
            .entry(collection.field)
            .or_default()
            .push(Json::Object(fields));
    }

    fn entry(
        &mut self,
        collection: &'static Collection,
        name: Option<String>,
        type_text: Option<String>,
        value: Option<Expr>,
        span: Span,
    ) {
        let Some((site, json_path)) = self.place(collection, name.as_deref(), span) else {
            return;
        };
        let Some(value) = value else {
            self.diagnostics.push(diagnostic(
                span,
                format!("`{}` needs `= <value>`", collection.word),
            ));
            return;
        };
        match self.entry_value(collection, &value, &site, &json_path) {
            Ok((rest, written)) => {
                let head_span = Span {
                    start: span.start,
                    end: value.span().start,
                };
                self.push_entry(
                    collection,
                    name,
                    type_text,
                    site,
                    &json_path,
                    span,
                    head_span,
                    Some(value.span()),
                    rest,
                    written,
                );
            }
            Err(error) => self.diagnostics.push(error),
        }
    }

    /// An entry a front end read from its own statement form.
    fn block_entry(&mut self, block: BlockEntry, span: Span) {
        let Some(collection) = self.collection(&block.word) else {
            self.diagnostics.push(diagnostic(
                span,
                format!(
                    "`{}` does not start a statement in a {}",
                    block.word,
                    self.label()
                ),
            ));
            return;
        };
        let Some((site, json_path)) = self.place(collection, block.name.as_deref(), span) else {
            return;
        };
        let spans: Vec<(String, Span)> = block.spans;
        for (path, at) in &spans {
            if !path.is_empty() {
                let site_path = match collection.tagged {
                    Some(tagged) => path
                        .strip_prefix(tagged.field)
                        .and_then(|rest| rest.strip_prefix('.'))
                        .unwrap_or(path),
                    None => path,
                };
                self.spans
                    .entry(format!("{site}#{site_path}"))
                    .or_insert(*at);
            }
            self.json_spans.push((
                if path.is_empty() {
                    json_path.clone()
                } else {
                    join_path(&json_path, path)
                },
                *at,
            ));
        }
        let head_span = Span {
            start: span.start,
            end: block.head_end.clamp(span.start, span.end),
        };
        self.push_entry(
            collection,
            block.name,
            None,
            site,
            &json_path,
            span,
            head_span,
            block.value_span,
            block.fields,
            block.written,
        );
    }

    /// The entry's fields besides its name and type.
    fn entry_value(
        &mut self,
        collection: &'static Collection,
        value: &Expr,
        site: &str,
        json_path: &str,
    ) -> Result<(Vec<(String, Json)>, Written), FeaSyntaxDiagnostic> {
        if let (Some(tagged), Expr::Call(call)) = (collection.tagged, value) {
            return self.tagged_call(collection, tagged, call, site, json_path);
        }
        if let (Some(tagged), Expr::PrefixApplication(application)) = (collection.tagged, value) {
            let tag = match application.callee() {
                Some(Expr::Reference(reference)) if reference.segments().len() == 1 => {
                    reference.segments().remove(0)
                }
                _ => return Err(diagnostic(value.span(), "expected `<kind> { … }`")),
            };
            let Some(Expr::Record(record)) = application.argument() else {
                return Err(diagnostic(
                    value.span(),
                    format!("expected `{tag} {{ … }}`"),
                ));
            };
            let mut spans = Vec::new();
            let entries = lower_record(&record, self.hook, "", &mut spans).map_err(lower_error);
            let kind_path = join_path(json_path, tagged.field);
            let (extras, kind_spans): (Vec<_>, Vec<_>) =
                spans.into_iter().partition(|(path, _)| {
                    tagged.extras.iter().any(|extra| {
                        path == extra
                            || path.starts_with(&format!("{extra}."))
                            || path.starts_with(&format!("{extra}["))
                    })
                });
            for (path, span) in &kind_spans {
                if !path.is_empty() {
                    self.spans.entry(format!("{site}#{path}")).or_insert(*span);
                }
                self.json_spans.push((
                    if path.is_empty() {
                        kind_path.clone()
                    } else {
                        join_path(&kind_path, path)
                    },
                    *span,
                ));
            }
            let written = written_values(&entries, &kind_spans, &extras);
            self.record_spans(site, json_path, extras);
            let entries = entries?;
            let mut kind = vec![(tagged.tag.to_string(), Json::String(tag))];
            let mut out = Vec::new();
            for (key, item) in entries {
                if tagged.extras.contains(&key.as_str()) {
                    out.push((key, item));
                } else {
                    kind.push((key, item));
                }
            }
            out.insert(0, (tagged.field.to_string(), Json::Object(kind)));
            return Ok((out, written));
        }
        if let Expr::Record(record) = value {
            let explicit = match collection.value_field {
                Some(field) => record
                    .entries()
                    .iter()
                    .any(|entry| entry.name().is_some_and(|n| n.text() == field)),
                None => true,
            };
            if explicit {
                let mut spans = Vec::new();
                let entries = lower_record(record, self.hook, "", &mut spans).map_err(lower_error);
                let written = written_values(&entries, &spans, &[]);
                self.record_spans(site, json_path, spans);
                return entries.map(|entries| (entries, written));
            }
        }
        match collection.value_field {
            Some(field) => {
                let item = self.lower(value, site, &join_path(json_path, field))?;
                self.spans
                    .entry(format!("{site}#{field}"))
                    .or_insert(value.span());
                Ok((
                    vec![(field.to_string(), item.clone())],
                    vec![(field.to_string(), item, value.span())],
                ))
            }
            None => Err(diagnostic(
                value.span(),
                format!("expected `{} … = {{ … }}`", collection.word),
            )),
        }
    }

    /// `<tag>(<arguments>)` or `<tag>.<variant>(<arguments>)`: a tagged entry in the call
    /// form, through the front end's [`EntryForms::desugar`].
    fn tagged_call(
        &mut self,
        collection: &'static Collection,
        tagged: Tagged,
        call: &ast::Call,
        site: &str,
        json_path: &str,
    ) -> Result<(Vec<(String, Json)>, Written), FeaSyntaxDiagnostic> {
        let callee = match call.callee() {
            Some(Expr::Reference(reference)) => reference,
            _ => return Err(diagnostic(call.span(), "expected `<kind>(…)`")),
        };
        let callee_span = callee.span();
        let segments = callee.segments();
        let forms = self.hook.entry_forms();
        let (tag, variant) = match segments.as_slice() {
            [tag] => (tag.clone(), None),
            [tag, variant] => {
                let Some(key) = forms.and_then(|f| f.subtag(collection.word, tag)) else {
                    return Err(diagnostic(
                        callee_span,
                        format!("`{tag}` has no kinds written `{tag}.<kind>`"),
                    ));
                };
                (tag.clone(), Some((key, variant.clone())))
            }
            _ => return Err(diagnostic(callee_span, "expected `<kind>(…)`")),
        };
        let mut spans = Vec::new();
        let arguments =
            lower_arguments(call, self.hook, Some(&tag), "", &mut spans).map_err(lower_error)?;
        let mut written: Written = Vec::new();
        if let Some((key, variant)) = &variant {
            written.push((key.to_string(), Json::String(variant.clone()), callee_span));
        }
        for (key, value) in &arguments {
            let span = spans
                .iter()
                .find(|(path, _)| path == key)
                .map_or(call.span(), |(_, span)| *span);
            written.push((key.clone(), value.clone(), span));
        }
        let (fields, moved) = match forms {
            Some(forms) => forms
                .desugar(collection.word, &tag, arguments)
                .map_err(|message| diagnostic(call.span(), message))?,
            None => (arguments, Vec::new()),
        };
        // Spans follow the arguments to the model paths they fill.
        let spans: Vec<(String, Span)> = spans
            .into_iter()
            .map(|(path, span)| {
                for (key, target) in &moved {
                    if path == *key {
                        return (target.clone(), span);
                    }
                    if let Some(rest) = path.strip_prefix(key.as_str()) {
                        if rest.starts_with('.') || rest.starts_with('[') {
                            return (format!("{target}{rest}"), span);
                        }
                    }
                }
                (path, span)
            })
            .collect();
        let kind_path = join_path(json_path, tagged.field);
        let is_extra = |path: &str| {
            tagged.extras.iter().any(|extra| {
                path == *extra
                    || path.starts_with(&format!("{extra}."))
                    || path.starts_with(&format!("{extra}["))
            })
        };
        let mut kind = vec![(tagged.tag.to_string(), Json::String(tag))];
        if let Some((key, variant)) = variant {
            kind.push((key.to_string(), Json::String(variant)));
        }
        let mut out = Vec::new();
        for (key, item) in fields {
            if tagged.extras.contains(&key.as_str()) {
                out.push((key, item));
            } else if kind.iter().any(|(existing, _)| *existing == key) {
                return Err(diagnostic(call.span(), format!("`{key}` is given twice")));
            } else {
                kind.push((key, item));
            }
        }
        for (path, span) in spans {
            if !path.is_empty() {
                self.spans.entry(format!("{site}#{path}")).or_insert(span);
            }
            let full = if is_extra(&path) {
                join_path(json_path, &path)
            } else if path.is_empty() {
                kind_path.clone()
            } else {
                join_path(&kind_path, &path)
            };
            self.json_spans.push((full, span));
        }
        out.insert(0, (tagged.field.to_string(), Json::Object(kind)));
        Ok((out, written))
    }

    /// `check <name> { <field> = <value> … }`: the block form of `check <name> = { … }`.
    fn check_block(&mut self, name: Option<ast::Name>, block: ast::Block, span: Span) {
        let Some(collection) = self.collection("check").filter(|c| c.id.is_some()) else {
            self.diagnostics.push(diagnostic(
                span,
                format!("`check` does not start a statement in a {}", self.label()),
            ));
            return;
        };
        let name = name.map(|n| n.text()).unwrap_or_default();
        let Some((site, json_path)) = self.place(collection, Some(&name), span) else {
            return;
        };
        let mut rest: Vec<(String, Json)> = Vec::new();
        let mut written: Written = Vec::new();
        for statement in block.statements() {
            let Statement::Assignment(assignment) = &statement else {
                self.diagnostics.push(diagnostic(
                    statement.span(),
                    "a check's block holds `<field> = <value>` lines",
                ));
                return;
            };
            let target = assignment.target();
            let (true, [key], Some(value)) = (
                assignment.relation() == ast::Relation::Equal,
                target.as_slice(),
                assignment.value(),
            ) else {
                self.diagnostics.push(diagnostic(
                    assignment.target_span(),
                    "expected `<field> = <value>`",
                ));
                return;
            };
            if rest.iter().any(|(existing, _)| existing == key) {
                self.diagnostics.push(diagnostic(
                    assignment.target_span(),
                    format!("`{key}` is given twice"),
                ));
                return;
            }
            let mut spans = Vec::new();
            match crate::json::lower_field(key, &value, self.hook, None, key, &mut spans) {
                Ok(item) => {
                    self.record_spans(&site, &json_path, spans);
                    written.push((key.clone(), item.clone(), value.span()));
                    rest.push((key.clone(), item));
                }
                Err(error) => {
                    self.diagnostics.push(lower_error(error));
                    return;
                }
            }
        }
        let head_span = Span {
            start: span.start,
            end: block.span().start,
        };
        self.push_entry(
            collection,
            Some(name),
            None,
            site,
            &json_path,
            span,
            head_span,
            Some(block.span()),
            rest,
            written,
        );
    }

    fn field(&mut self, word: &str, target: Span, value: Option<Expr>, span: Span) {
        let reserved = [
            self.grammar.name_field,
            self.grammar.path_field,
            self.grammar.header_field,
        ]
        .into_iter()
        .flatten()
        .chain(self.grammar.collections.iter().map(|c| c.field))
        .chain(self.grammar.links.map(|l| l.field));
        if reserved.into_iter().any(|field| field == word) {
            self.diagnostics.push(diagnostic(
                target,
                format!("`{word}` is not set by assignment"),
            ));
            return;
        }
        if self.fields.iter().any(|(existing, _)| existing == word) {
            self.diagnostics
                .push(diagnostic(target, format!("`{word}` is set twice")));
            return;
        }
        let Some(value) = value else {
            return;
        };
        let site = word.to_string();
        self.spans.insert(site.clone(), span);
        match self.lower(&value, &site, word) {
            Ok(item) => {
                let head_span = Span {
                    start: span.start,
                    end: value.span().start,
                };
                self.entries.push(ParsedEntry {
                    site,
                    field: word.to_string(),
                    span,
                    head: normalize(&self.source[head_span.start..head_span.end]),
                    head_span,
                    value_span: Some(value.span()),
                    written: Some(vec![(String::new(), item.clone(), value.span())]),
                });
                self.fields.push((word.to_string(), item));
            }
            Err(error) => self.diagnostics.push(error),
        }
    }

    fn link(&mut self, link: &ast::Link, span: Span) {
        let Some(grammar) = self.grammar.links else {
            self.diagnostics.push(diagnostic(
                span,
                format!("a {} takes no links", self.label()),
            ));
            return;
        };
        let kind = link.kind().map(|k| k.text()).unwrap_or_default();
        let from = match link.from() {
            Some(Expr::Reference(reference)) if reference.segments().len() == 2 => {
                reference.segments()
            }
            other => {
                let at = other.map_or(span, |e| e.span());
                self.diagnostics.push(diagnostic(
                    at,
                    "a link starts from `<kind>.<name>` (`feature.body`)",
                ));
                return;
            }
        };
        let to = match link.to() {
            Some(Expr::String(string)) => string.value(),
            other => {
                let at = other.map_or(span, |e| e.span());
                self.diagnostics
                    .push(diagnostic(at, "a link goes to a quoted item path or URI"));
                return;
            }
        };
        let entries = self.collections.entry(grammar.field).or_default();
        let index = entries.len();
        self.entries.push(ParsedEntry {
            site: format!("{}[{index}]", grammar.site),
            field: grammar.field.to_string(),
            span,
            head: normalize(&self.source[span.start..span.end]),
            head_span: span,
            value_span: None,
            written: None,
        });
        self.spans
            .insert(format!("{}[{index}]", grammar.site), span);
        self.json_spans
            .push((format!("{}[{index}]", grammar.field), span));
        entries.push(Json::Object(vec![
            (grammar.kind.to_string(), Json::String(kind)),
            (
                grammar.from.to_string(),
                Json::Object(vec![
                    (grammar.from_tag.to_string(), Json::String(from[0].clone())),
                    (from[0].clone(), Json::String(from[1].clone())),
                ]),
            ),
            (grammar.to.to_string(), Json::String(to)),
        ]));
    }
}

// ------------------------------------------------------------------------------------ printing

/// Why a model could not be printed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrintError(pub String);

impl std::fmt::Display for PrintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PrintError {}

/// Prints a model (its JSON form) as source. `name` is the declaration's name when the
/// grammar keeps no name field (the file stem).
pub fn print_document(
    grammar: &DocumentGrammar,
    model: &dyn DocumentModel,
    json: &Json,
    uses: &[UseItem],
    name: &str,
    hook: &dyn ValueSyntax,
) -> Result<String, PrintError> {
    let Json::Object(fields) = json else {
        return Err(PrintError("a document is a JSON object".into()));
    };
    let canonical = model
        .normalize(json)
        .ok_or_else(|| PrintError("the model does not deserialise".into()))?
        .to_value();
    let mut out = String::new();
    if let Some(header) = grammar.header_field.and_then(|field| json.get(field)) {
        for (word, key) in [("runmat", "language"), ("std", "std")] {
            if let Some(version) = header.get(key).and_then(Json::as_str) {
                out.push_str(&format!("{word} {}\n", version_text(version)));
            }
        }
        out.push('\n');
    }
    for item in uses {
        out.push_str(&item.print());
        out.push('\n');
    }
    if !uses.is_empty() {
        out.push('\n');
    }
    let declaration_name = match grammar.name_field.and_then(|field| json.get(field)) {
        Some(Json::String(text)) => text.clone(),
        Some(_) => return Err(PrintError("the declaration name is not text".into())),
        None => name.to_string(),
    };
    if !is_bare_word(&declaration_name) {
        return Err(PrintError(format!(
            "`{declaration_name}` cannot be written as a name"
        )));
    }
    let mut body: Vec<String> = Vec::new();
    let mut groups_written = Vec::new();
    for collection in grammar.collections {
        let Some(Json::Array(entries)) = json.get(collection.field) else {
            continue;
        };
        let mut group = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            let elider = Elider::for_entry(
                grammar, collection, index, entry, json, &canonical, model, hook,
            );
            group.push(print_entry(collection, entry, &elider, hook)?);
        }
        if !group.is_empty() {
            groups_written.push(group);
        }
    }
    if let Some(links) = grammar.links {
        if let Some(Json::Array(entries)) = json.get(links.field) {
            let mut group = Vec::new();
            for entry in entries {
                group.push(print_link(&links, entry)?);
            }
            if !group.is_empty() {
                groups_written.push(group);
            }
        }
    }
    let mut extra = Vec::new();
    for (key, value) in fields {
        let handled = [grammar.name_field, grammar.path_field, grammar.header_field]
            .into_iter()
            .flatten()
            .chain(grammar.collections.iter().map(|c| c.field))
            .chain(grammar.links.map(|l| l.field))
            .any(|field| field == key);
        if handled || removable(json, &[Step::Key(key.clone())], &canonical, model) {
            continue;
        }
        if !is_bare_word(key) {
            return Err(PrintError(format!(
                "the field `{key}` cannot be written as a name"
            )));
        }
        extra.push(format!("{key} = {}", print_json(value, hook, 1)));
    }
    if !extra.is_empty() {
        groups_written.push(extra);
    }
    for (index, group) in groups_written.into_iter().enumerate() {
        if index > 0 {
            body.push(String::new());
        }
        body.extend(group);
    }
    out.push_str(&format!("{} {declaration_name} {{\n", grammar.declaration));
    for statement in body {
        if statement.is_empty() {
            out.push('\n');
            continue;
        }
        out.push_str(INDENT);
        out.push_str(&statement.replace('\n', &format!("\n{INDENT}")));
        out.push('\n');
    }
    out.push_str("}\n");
    Ok(out)
}

fn version_text(version: &str) -> String {
    let stream = crate::lexical::tokenize_design(version);
    let single_number = stream.diagnostics.is_empty()
        && stream.tokens.len() == 2
        && stream.tokens[0].kind == crate::lexical::FeaTokenKind::Number
        && stream.tokens[0].span.end == version.len();
    if single_number || is_bare_word(version) {
        version.to_string()
    } else {
        quote(version)
    }
}

#[derive(Debug, Clone)]
enum Step {
    Key(String),
    Index(usize),
}

fn without(json: &Json, path: &[Step]) -> Json {
    let mut copy = json.clone();
    let (last, parents) = path.split_last().expect("a non-empty path");
    let mut node = &mut copy;
    for step in parents {
        node = match (step, node) {
            (Step::Key(key), Json::Object(entries)) => {
                &mut entries
                    .iter_mut()
                    .find(|(k, _)| k == key)
                    .expect("the path exists")
                    .1
            }
            (Step::Index(index), Json::Array(items)) => &mut items[*index],
            _ => unreachable!("the path follows the value"),
        };
    }
    if let (Step::Key(key), Json::Object(entries)) = (last, node) {
        entries.retain(|(k, _)| k != key);
    }
    copy
}

/// Whether the field at `path` can be left out without changing the model.
fn removable(
    json: &Json,
    path: &[Step],
    canonical: &serde_json::Value,
    model: &dyn DocumentModel,
) -> bool {
    model
        .normalize(&without(json, path))
        .is_some_and(|normalized| normalized.to_value() == *canonical)
}

/// Leaves out an entry's fields, at any depth, that hold their default.
///
/// It tests each field on a copy of the document that holds only this entry, so the cost of
/// a test does not grow with the document. Values the dialect prints in its own forms
/// ([`ValueSyntax::print`]) are kept whole.
struct Elider<'a> {
    document: Json,
    canonical: serde_json::Value,
    /// The entry's path in `document`.
    path: Vec<Step>,
    model: &'a dyn DocumentModel,
    hook: &'a dyn ValueSyntax,
}

impl<'a> Elider<'a> {
    #[allow(clippy::too_many_arguments)]
    fn for_entry(
        grammar: &DocumentGrammar,
        collection: &Collection,
        index: usize,
        entry: &Json,
        json: &Json,
        canonical: &serde_json::Value,
        model: &'a dyn DocumentModel,
        hook: &'a dyn ValueSyntax,
    ) -> Self {
        let lists: Vec<&str> = grammar
            .collections
            .iter()
            .map(|c| c.field)
            .chain(grammar.links.map(|l| l.field))
            .collect();
        if let Json::Object(fields) = json {
            let reduced = Json::Object(
                fields
                    .iter()
                    .map(|(key, value)| {
                        if key == collection.field {
                            (key.clone(), Json::Array(vec![entry.clone()]))
                        } else if lists.contains(&key.as_str()) {
                            (key.clone(), Json::Array(Vec::new()))
                        } else {
                            (key.clone(), value.clone())
                        }
                    })
                    .collect(),
            );
            if let Some(normalized) = model.normalize(&reduced) {
                return Elider {
                    document: reduced,
                    canonical: normalized.to_value(),
                    path: vec![Step::Key(collection.field.to_string()), Step::Index(0)],
                    model,
                    hook,
                };
            }
        }
        Elider {
            document: json.clone(),
            canonical: canonical.clone(),
            path: vec![Step::Key(collection.field.to_string()), Step::Index(index)],
            model,
            hook,
        }
    }

    /// The entry's field `key` (a path below the entry) can be left out.
    fn removable(&self, below: &[Step]) -> bool {
        let mut path = self.path.clone();
        path.extend_from_slice(below);
        removable(&self.document, &path, &self.canonical, self.model)
    }

    /// `value` (at `below` in the entry) without the fields that hold their default.
    fn elide(&self, value: &Json, below: &[Step]) -> Json {
        if self.hook.print(value).is_some() {
            return value.clone();
        }
        match value {
            Json::Object(entries) => Json::Object(
                entries
                    .iter()
                    .filter_map(|(key, item)| {
                        let mut path = below.to_vec();
                        path.push(Step::Key(key.clone()));
                        if self.removable(&path) {
                            None
                        } else {
                            Some((key.clone(), self.elide(item, &path)))
                        }
                    })
                    .collect(),
            ),
            Json::Array(items) => Json::Array(
                items
                    .iter()
                    .enumerate()
                    .map(|(index, item)| {
                        let mut path = below.to_vec();
                        path.push(Step::Index(index));
                        self.elide(item, &path)
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }
}

fn print_entry(
    collection: &Collection,
    entry: &Json,
    elider: &Elider<'_>,
    hook: &dyn ValueSyntax,
) -> Result<String, PrintError> {
    let Json::Object(fields) = entry else {
        return Err(PrintError(format!(
            "a `{}` entry is not an object",
            collection.word
        )));
    };
    let mut head = collection.word.to_string();
    let mut rest: Vec<(String, Json)> = Vec::new();
    let mut type_text = None;
    let mut name = None;
    for (key, value) in fields {
        if Some(key.as_str()) == collection.id {
            let Json::String(text) = value else {
                return Err(PrintError(format!(
                    "a `{}` name is not text",
                    collection.word
                )));
            };
            name = Some(text.clone());
            head.push(' ');
            head.push_str(&if is_bare_word(text) {
                text.clone()
            } else {
                quote(text)
            });
            continue;
        }
        if Some(key.as_str()) == collection.type_field {
            if let Json::String(text) = value {
                if is_bare_word(text) {
                    type_text = Some(text.clone());
                    continue;
                }
            }
        }
        let path = [Step::Key(key.clone())];
        if !elider.removable(&path) {
            rest.push((key.clone(), elider.elide(value, &path)));
        }
    }
    if let Some(forms) = hook.entry_forms() {
        let mut all = rest.clone();
        if let (Some(field), Some(text)) = (collection.type_field, &type_text) {
            all.insert(0, (field.to_string(), Json::String(text.clone())));
        }
        if let Some(text) = forms.print_statement(
            collection.word,
            name.as_deref().unwrap_or_default(),
            &all,
            hook,
        ) {
            return Ok(text);
        }
    }
    if let Some(type_text) = type_text {
        head.push_str(&format!(" : {type_text}"));
    }
    if let Some(tagged) = collection.tagged {
        if let Some(text) = print_tagged(collection, tagged, &head, &rest, hook) {
            return Ok(text);
        }
    }
    if let Some(field) = collection.value_field {
        if let [(key, value)] = rest.as_slice() {
            if key == field && (!matches!(value, Json::Object(_)) || hook.print(value).is_some()) {
                return Ok(format!("{head} = {}", print_json(value, hook, 0)));
            }
        }
        // The record form names the value field, so it parses as the record form.
        if !rest.iter().any(|(key, _)| key == field) {
            if let Some((key, value)) = fields.iter().find(|(key, _)| key == field) {
                rest.insert(0, (key.clone(), value.clone()));
            }
        }
    }
    Ok(format!(
        "{head} = {}",
        crate::json::print_record(&rest, hook, 0)
    ))
}

/// A tagged entry in the call form (`feature body = extrude(…)`), or in the record form
/// (`feature body = extrude { … }`) when a field name cannot be an argument name.
fn print_tagged(
    collection: &Collection,
    tagged: Tagged,
    head: &str,
    rest: &[(String, Json)],
    hook: &dyn ValueSyntax,
) -> Option<String> {
    let position = rest.iter().position(|(key, _)| key == tagged.field)?;
    let Json::Object(kind) = &rest[position].1 else {
        return None;
    };
    let tag = kind
        .iter()
        .find(|(key, _)| key == tagged.tag)
        .and_then(|(_, v)| v.as_str())?;
    let collides = kind
        .iter()
        .any(|(key, _)| tagged.extras.contains(&key.as_str()));
    let others_are_extras = rest
        .iter()
        .all(|(key, _)| key == tagged.field || tagged.extras.contains(&key.as_str()));
    if collides || !others_are_extras || !is_bare_word(tag) {
        return None;
    }
    let mut record: Vec<(String, Json)> = kind
        .iter()
        .filter(|(key, _)| key != tagged.tag)
        .cloned()
        .collect();
    let extras: Vec<(String, Json)> = rest
        .iter()
        .filter(|(key, _)| key != tagged.field)
        .cloned()
        .collect();
    let forms = hook.entry_forms();
    let mut callee = tag.to_string();
    if let Some(key) = forms.and_then(|f| f.subtag(collection.word, tag)) {
        if let Some((first, Json::String(variant))) = record.first() {
            if first == key && is_identifier(variant) {
                callee = format!("{tag}.{variant}");
                record.remove(0);
            }
        }
    }
    let mut arguments = forms
        .and_then(|f| f.sugar(collection.word, tag, &record))
        .unwrap_or_else(|| record.clone());
    arguments.extend(extras.iter().cloned());
    if is_identifier(tag) && arguments.iter().all(|(key, _)| is_identifier(key)) {
        return Some(format!(
            "{head} = {callee}{}",
            print_arguments(&arguments, hook, Some(tag), 0)
        ));
    }
    if callee != tag {
        return None;
    }
    record.extend(extras);
    Some(format!(
        "{head} = {tag} {}",
        crate::json::print_record(&record, hook, 0)
    ))
}

fn print_link(grammar: &LinkGrammar, entry: &Json) -> Result<String, PrintError> {
    let kind = entry.get(grammar.kind).and_then(Json::as_str);
    let from = entry.get(grammar.from);
    let item_kind = from
        .and_then(|f| f.get(grammar.from_tag))
        .and_then(Json::as_str);
    let item = item_kind
        .and_then(|k| from.and_then(|f| f.get(k)))
        .and_then(Json::as_str);
    let to = entry.get(grammar.to).and_then(Json::as_str);
    match (kind, item_kind, item, to) {
        (Some(kind), Some(item_kind), Some(item), Some(to))
            if is_bare_word(kind) && crate::json::is_bare_key(item_kind) && is_bare_word(item) =>
        {
            Ok(format!("link {kind} {item_kind}.{item} to {}", quote(to)))
        }
        _ => Err(PrintError(
            "a link that cannot be written as `link <kind> <item> to \"…\"`".into(),
        )),
    }
}

/// The key a value at `json_path` gets in the printed record (`print_key`), for callers that
/// place edits.
pub fn key_text(key: &str) -> String {
    print_key(key)
}
