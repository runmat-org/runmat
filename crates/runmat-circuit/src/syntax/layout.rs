//! The `.circuit.layout` front end on the shared design grammar: the `sync { }` section.
//!
//! The section holds what the board last took from the schematic ([`BoardSync`]): the base of
//! the ECO's three-way compare, each footprint's link to its schematic component, and the
//! comparator settings the board was last updated with.
//!
//! ```text
//! runmat 0.7
//! std 1
//!
//! layout ctl {
//!     sync {
//!         version = 1
//!         compare = { ignore = [rooms], ignored_parameters = [datasheet], remove_unlinked_footprints = true, … }
//!         footprint R5 = "/ch[0]/R1"
//!         component "/ch[0]/R1" = { designator = R5, package = R0402, pads = { "1" = VIN, "2" = "ch[0]/OUT" } }
//!         net_class power = { trace_width = "0.5 mm" }
//!         diff_pair usb = { p = DP, n = DN }
//!         component_class "ch[0]" = [R5, C3]
//!     }
//! }
//! ```
//!
//! - `footprint <designator> = "<component path>"` links a footprint to the component it
//!   realizes. The links live only here: they are the identity the ECO matches by.
//! - `component "<path>" = { … }` is the synced designator, package and pad nets of one
//!   component; `net_class`, `diff_pair` and `component_class` are the synced classes.
//! - `compare` holds every comparator setting, defaults included, so a file reads the same
//!   whatever a later release's defaults are.
//!
//! **Versions.** `version` is the section's format version, [`SYNC_VERSION`]. The writer always
//! writes the current version. A reader takes the current version and migrates each older one
//! it knows; version 1 is the first, so there is nothing to migrate yet. A newer version is an
//! error, and the board then has no sync record rather than a misread one. A layout without the
//! section (a KiCad import, a board laid out before its first ECO) has an empty record: its
//! first compare is two-way, and footprints link to components by designator when
//! `relink_by_designator` is on. Applying that ECO writes the section.
//!
//! Reading leaves every other statement in the layout block to the board's own front end; the
//! writer ([`write_sync`]) replaces only the section's text, so the rest of the file stays
//! byte for byte. It refuses a source with errors, so it never overwrites a section it could
//! not read.

use crate::eco::{BoardSync, ComparatorOptions, Difference, SyncedComponent};
use crate::layout::Board;
use runmat_design_syntax::ast::{Expr, Headed, Item, Statement};
use runmat_design_syntax::json::{
    is_bare_word, lower_expr, print_json, quote, Json, PlainData, INDENT,
};
use runmat_design_syntax::{parse_source, Dialect, FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};
use runmat_types::Span;
use std::collections::{BTreeMap, BTreeSet};

/// The `.circuit.layout` statement words this front end reads: its registry.
pub mod words {
    pub const LAYOUT: &str = "layout";
    pub const SYNC: &str = "sync";
    pub const VERSION: &str = "version";
    pub const COMPARE: &str = "compare";
    pub const FOOTPRINT: &str = "footprint";
    pub const COMPONENT: &str = "component";
    pub const NET_CLASS: &str = "net_class";
    pub const DIFF_PAIR: &str = "diff_pair";
    pub const COMPONENT_CLASS: &str = "component_class";
    /// Every statement word, for completion and the highlighter.
    pub const ALL: &[&str] = &[
        LAYOUT,
        SYNC,
        VERSION,
        COMPARE,
        FOOTPRINT,
        COMPONENT,
        NET_CLASS,
        DIFF_PAIR,
        COMPONENT_CLASS,
    ];
}

/// The `compare = { … }` keys.
mod keys {
    pub const IGNORE: &str = "ignore";
    pub const IGNORED_PARAMETERS: &str = "ignored_parameters";
    pub const REMOVE_UNLINKED: &str = "remove_unlinked_footprints";
    pub const RELINK: &str = "relink_by_designator";
    pub const ROOMS_PER_CHANNEL: &str = "rooms_per_channel";
    pub const CLASSES_PER_CHANNEL: &str = "classes_per_channel";
    pub const DESIGNATOR: &str = "designator";
    pub const PACKAGE: &str = "package";
    pub const PADS: &str = "pads";
}

/// The format version this release writes and reads.
pub const SYNC_VERSION: u32 = 1;

/// The `sync { }` section's model: the board's sync record and its footprint links.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SyncSection {
    pub sync: BoardSync,
    /// Footprint designator → component path.
    pub links: BTreeMap<String, String>,
}

impl SyncSection {
    /// The section a board writes.
    pub fn of(board: &Board) -> SyncSection {
        SyncSection {
            sync: board.sync.clone(),
            links: board
                .footprints
                .values()
                .filter_map(|f| Some((f.reference.clone(), f.component.clone()?)))
                .collect(),
        }
    }

    /// Puts the section on a board: its sync record, and each link on the footprint with that
    /// designator. Returns the designators no footprint has (their links are dropped).
    pub fn install(&self, board: &mut Board) -> Vec<String> {
        board.sync = self.sync.clone();
        let mut missing = Vec::new();
        for (reference, path) in &self.links {
            match board
                .footprints
                .values_mut()
                .find(|f| &f.reference == reference)
            {
                Some(f) => f.component = Some(path.clone()),
                None => missing.push(reference.clone()),
            }
        }
        missing
    }
}

/// A parsed `.circuit.layout` file, as far as this front end reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedLayout {
    /// The `layout` declaration's name.
    pub name: Option<String>,
    /// The section, when the file has one and it reads without errors. `None` with no
    /// diagnostics: the file has no section (an empty sync record).
    pub sync: Option<SyncSection>,
    /// The `sync { … }` statement.
    pub section: Option<Span>,
    /// The layout declaration's block, `{` to `}`.
    pub block: Option<Span>,
    pub diagnostics: Vec<FeaSyntaxDiagnostic>,
}

fn diagnostic(span: Span, message: impl Into<String>) -> FeaSyntaxDiagnostic {
    FeaSyntaxDiagnostic {
        kind: FeaSyntaxDiagnosticKind::InvalidExpression,
        span,
        message: message.into(),
    }
}

type Read<T> = Result<T, FeaSyntaxDiagnostic>;

/// Parses a `.circuit.layout` source's declaration and `sync { }` section.
pub fn parse_layout(path: &str, source: &str) -> ParsedLayout {
    let syntax = parse_source(Dialect::CircuitLayout, source);
    let mut out = ParsedLayout {
        name: None,
        sync: None,
        section: None,
        block: None,
        diagnostics: syntax.diagnostics.clone(),
    };
    let mut seen = false;
    for item in syntax.file().items() {
        let Item::Declaration(declaration) = item else {
            continue;
        };
        if declaration.keyword() != words::LAYOUT {
            continue;
        }
        if seen {
            out.diagnostics.push(diagnostic(
                declaration.keyword_span(),
                "a file holds one `layout`",
            ));
            continue;
        }
        seen = true;
        let name = declaration.name().map(|n| n.text()).unwrap_or_default();
        let stem = Dialect::CircuitLayout.stem(path);
        if !path.is_empty() && name != stem {
            out.diagnostics.push(diagnostic(
                declaration.name().map_or(declaration.span(), |n| n.span()),
                format!("this layout is named `{name}`; its file is `{stem}`"),
            ));
        }
        out.name = Some(name);
        let Some(block) = declaration.block() else {
            continue;
        };
        out.block = Some(block.span());
        for statement in block.statements() {
            let Statement::Block(section) = &statement else {
                continue;
            };
            if section.keyword() != words::SYNC {
                continue;
            }
            if out.section.is_some() {
                out.diagnostics.push(diagnostic(
                    section.keyword_span(),
                    "a layout holds one `sync` section",
                ));
                continue;
            }
            out.section = Some(statement.span());
            if let Some(name) = section.name() {
                out.diagnostics
                    .push(diagnostic(name.span(), "the `sync` section takes no name"));
            }
            let statements = section.block().map(|b| b.statements()).unwrap_or_default();
            let mut reader = SectionReader::default();
            reader.read(&statements, statement.span());
            if reader.diagnostics.is_empty() {
                out.sync = Some(reader.section);
            }
            out.diagnostics.extend(reader.diagnostics);
        }
    }
    if !seen {
        out.diagnostics.push(diagnostic(
            Span::default(),
            "expected a `layout <name> { … }` declaration",
        ));
    }
    if !out.diagnostics.is_empty() {
        out.sync = None;
    }
    out
}

#[derive(Default)]
struct SectionReader {
    section: SyncSection,
    diagnostics: Vec<FeaSyntaxDiagnostic>,
}

impl SectionReader {
    fn note(&mut self, result: Read<()>) {
        if let Err(error) = result {
            self.diagnostics.push(error);
        }
    }

    fn read(&mut self, statements: &[Statement], span: Span) {
        let mut version: Option<(u32, Span)> = None;
        let mut compare_seen = false;
        // The version decides how everything else reads, so it is read first.
        for statement in statements {
            if let Statement::Assignment(a) = statement {
                if a.target() == [words::VERSION] {
                    let read = match a.value() {
                        Some(Expr::Number(n)) => n.text().parse::<u32>().ok(),
                        _ => None,
                    };
                    match read {
                        Some(v) if version.is_none() => version = Some((v, statement.span())),
                        Some(_) => self
                            .diagnostics
                            .push(diagnostic(statement.span(), "`version` is given twice")),
                        None => self.diagnostics.push(diagnostic(
                            statement.span(),
                            "`version` is a whole number: `version = 1`",
                        )),
                    }
                }
            }
        }
        match version {
            None => {
                self.diagnostics.push(diagnostic(
                    span,
                    format!("the sync section starts with `version = {SYNC_VERSION}`"),
                ));
                return;
            }
            Some((v, at)) if v > SYNC_VERSION => {
                self.diagnostics.push(diagnostic(
                    at,
                    format!(
                        "this sync section is version {v}, written by a newer RunMat; this \
                         release reads version {SYNC_VERSION}"
                    ),
                ));
                return;
            }
            Some((v, at)) if v < 1 => {
                self.diagnostics.push(diagnostic(
                    at,
                    format!("there is no sync section version {v}"),
                ));
                return;
            }
            // Version 1 is the first; older versions would migrate here.
            Some(_) => {}
        }
        for statement in statements {
            let at = statement.span();
            match statement {
                Statement::Assignment(a) if a.target() == [words::VERSION] => {}
                Statement::Assignment(a) if a.target() == [words::COMPARE] => {
                    if compare_seen {
                        self.diagnostics
                            .push(diagnostic(at, "`compare` is given twice"));
                        continue;
                    }
                    compare_seen = true;
                    let result = value(a.value(), at)
                        .and_then(|v| read_compare(&v, at))
                        .map(|o| self.section.sync.options = o);
                    self.note(result);
                }
                Statement::Value(v) => {
                    let word = v.keyword();
                    let name = v.name().map(|n| n.text()).unwrap_or_default();
                    let result =
                        value(v.value(), at).and_then(|value| self.entry(&word, name, &value, at));
                    self.note(result);
                }
                _ => self.diagnostics.push(diagnostic(
                    at,
                    "expected `version`, `compare`, `footprint`, `component`, `net_class`, \
                     `diff_pair` or `component_class`",
                )),
            }
        }
    }

    fn entry(&mut self, word: &str, name: String, value: &Json, at: Span) -> Read<()> {
        let sync = &mut self.section.sync;
        let taken = match word {
            words::FOOTPRINT => self
                .section
                .links
                .insert(name.clone(), string(value, at)?)
                .is_some(),
            words::COMPONENT => {
                let record = record(value, at)?;
                let mut component = SyncedComponent::default();
                for (key, v) in record {
                    match key.as_str() {
                        keys::DESIGNATOR => component.designator = string(v, at)?,
                        keys::PACKAGE => component.package = string(v, at)?,
                        keys::PADS => component.pads = string_map(v, at)?,
                        other => {
                            return Err(diagnostic(
                                at,
                                format!("a synced component has no `{other}`"),
                            ))
                        }
                    }
                }
                sync.components.insert(name.clone(), component).is_some()
            }
            words::NET_CLASS => sync
                .net_classes
                .insert(name.clone(), string_map(value, at)?)
                .is_some(),
            words::DIFF_PAIR => {
                let (mut p, mut n) = (None, None);
                for (key, v) in record(value, at)? {
                    match key.as_str() {
                        "p" => p = Some(string(v, at)?),
                        "n" => n = Some(string(v, at)?),
                        other => {
                            return Err(diagnostic(
                                at,
                                format!("a differential pair has no `{other}`"),
                            ))
                        }
                    }
                }
                let (Some(p), Some(n)) = (p, n) else {
                    return Err(diagnostic(at, "a differential pair is `{ p = …, n = … }`"));
                };
                sync.diff_pairs.insert(name.clone(), (p, n)).is_some()
            }
            words::COMPONENT_CLASS => sync
                .component_classes
                .insert(name.clone(), strings(value, at)?)
                .is_some(),
            other => {
                return Err(diagnostic(
                    at,
                    format!("`{other}` does not start a sync statement"),
                ))
            }
        };
        if taken {
            return Err(diagnostic(at, format!("{word} `{name}` is given twice")));
        }
        Ok(())
    }
}

fn value(expr: Option<Expr>, at: Span) -> Read<Json> {
    let expr = expr.ok_or_else(|| diagnostic(at, "expected `= <value>`"))?;
    lower_expr(&expr, &PlainData, "", &mut Vec::new())
        .map_err(|error| diagnostic(error.span, error.message))
}

fn string(value: &Json, at: Span) -> Read<String> {
    match value {
        Json::String(s) => Ok(s.clone()),
        _ => Err(diagnostic(at, "expected a name or quoted text")),
    }
}

fn boolean(value: &Json, at: Span) -> Read<bool> {
    match value {
        Json::Bool(b) => Ok(*b),
        _ => Err(diagnostic(at, "expected `true` or `false`")),
    }
}

fn strings(value: &Json, at: Span) -> Read<Vec<String>> {
    match value {
        Json::Array(items) => items.iter().map(|v| string(v, at)).collect(),
        _ => Err(diagnostic(at, "expected a list `[ … ]`")),
    }
}

fn record(value: &Json, at: Span) -> Read<&[(String, Json)]> {
    match value {
        Json::Object(entries) => Ok(entries),
        _ => Err(diagnostic(at, "expected a record `{ … }`")),
    }
}

fn string_map(value: &Json, at: Span) -> Read<BTreeMap<String, String>> {
    record(value, at)?
        .iter()
        .map(|(k, v)| Ok((k.clone(), string(v, at)?)))
        .collect()
}

fn read_compare(value: &Json, at: Span) -> Read<ComparatorOptions> {
    let mut o = ComparatorOptions::default();
    for (key, v) in record(value, at)? {
        match key.as_str() {
            keys::IGNORE => {
                for word in strings(v, at)? {
                    let d = Difference::parse(&word).ok_or_else(|| {
                        diagnostic(
                            at,
                            format!(
                                "`{word}` is not a difference type; expected one of {}",
                                Difference::ALL.map(Difference::as_str).join(", ")
                            ),
                        )
                    })?;
                    o.set_ignore(d, true);
                }
            }
            keys::IGNORED_PARAMETERS => {
                o.ignored_parameters = strings(v, at)?.into_iter().collect()
            }
            keys::REMOVE_UNLINKED => o.remove_unlinked_footprints = boolean(v, at)?,
            keys::RELINK => o.relink_by_designator = boolean(v, at)?,
            keys::ROOMS_PER_CHANNEL => o.rooms_per_channel = boolean(v, at)?,
            keys::CLASSES_PER_CHANNEL => o.classes_per_channel = boolean(v, at)?,
            other => {
                return Err(diagnostic(
                    at,
                    format!("the comparator has no setting `{other}`"),
                ))
            }
        }
    }
    Ok(o)
}

/* ------------------------------------------------------------------------------------------------
 * Printing
 * --------------------------------------------------------------------------------------------- */

fn s(text: &str) -> Json {
    Json::String(text.to_owned())
}

fn list<'a>(items: impl IntoIterator<Item = &'a String>) -> Json {
    Json::Array(items.into_iter().map(|t| s(t)).collect())
}

fn map(entries: &BTreeMap<String, String>) -> Json {
    Json::Object(entries.iter().map(|(k, v)| (k.clone(), s(v))).collect())
}

fn compare_json(o: &ComparatorOptions) -> Json {
    let ignored: Vec<String> = o.ignored().iter().map(|d| d.as_str().to_owned()).collect();
    let named: BTreeSet<&String> = o.ignored_parameters.iter().collect();
    Json::Object(vec![
        (keys::IGNORE.into(), list(&ignored)),
        (keys::IGNORED_PARAMETERS.into(), list(named)),
        (
            keys::REMOVE_UNLINKED.into(),
            Json::Bool(o.remove_unlinked_footprints),
        ),
        (keys::RELINK.into(), Json::Bool(o.relink_by_designator)),
        (
            keys::ROOMS_PER_CHANNEL.into(),
            Json::Bool(o.rooms_per_channel),
        ),
        (
            keys::CLASSES_PER_CHANNEL.into(),
            Json::Bool(o.classes_per_channel),
        ),
    ])
}

fn name(text: &str) -> String {
    if is_bare_word(text) {
        text.to_owned()
    } else {
        quote(text)
    }
}

/// The `sync { … }` statement, its first line unindented and the rest indented for nesting
/// `depth` (1 inside a layout block).
pub fn print_sync(section: &SyncSection, depth: usize) -> String {
    let inner = INDENT.repeat(depth + 1);
    let mut out = String::from("sync {\n");
    let mut line = |head: String, value: &Json| {
        out.push_str(&inner);
        out.push_str(&head);
        out.push_str(" = ");
        out.push_str(&print_json(value, &PlainData, depth + 1));
        out.push('\n');
    };
    line(words::VERSION.into(), &Json::Number(SYNC_VERSION.into()));
    line(words::COMPARE.into(), &compare_json(&section.sync.options));
    for (reference, path) in &section.links {
        line(
            format!("{} {}", words::FOOTPRINT, name(reference)),
            &s(path),
        );
    }
    let sync = &section.sync;
    for (path, c) in &sync.components {
        let record = Json::Object(vec![
            (keys::DESIGNATOR.into(), s(&c.designator)),
            (keys::PACKAGE.into(), s(&c.package)),
            (keys::PADS.into(), map(&c.pads)),
        ]);
        line(format!("{} {}", words::COMPONENT, name(path)), &record);
    }
    for (class, params) in &sync.net_classes {
        line(
            format!("{} {}", words::NET_CLASS, name(class)),
            &map(params),
        );
    }
    for (pair, (p, n)) in &sync.diff_pairs {
        let record = Json::Object(vec![("p".into(), s(p)), ("n".into(), s(n))]);
        line(format!("{} {}", words::DIFF_PAIR, name(pair)), &record);
    }
    for (class, members) in &sync.component_classes {
        line(
            format!("{} {}", words::COMPONENT_CLASS, name(class)),
            &list(members),
        );
    }
    out.push_str(&INDENT.repeat(depth));
    out.push('}');
    out
}

/// A new `.circuit.layout` file holding only the section.
pub fn print_layout(layout: &str, section: &SyncSection) -> String {
    format!(
        "runmat 0.7\nstd 1\n\n{} {} {{\n{INDENT}{}\n}}\n",
        words::LAYOUT,
        name(layout),
        print_sync(section, 1)
    )
}

/// `source` with its `sync { }` section replaced by `section`'s (or added at the end of the
/// layout block). Only the section's text changes. An empty source becomes a new layout
/// named for the file. A source with errors is not written, so a section from a newer
/// release is never replaced by an older one.
pub fn write_sync(
    path: &str,
    source: &str,
    section: &SyncSection,
) -> Result<String, Vec<FeaSyntaxDiagnostic>> {
    if source.trim().is_empty() {
        return Ok(print_layout(&Dialect::CircuitLayout.stem(path), section));
    }
    let parsed = parse_layout(path, source);
    // A section that does not read (a newer version among them) is not overwritten.
    if !parsed.diagnostics.is_empty() {
        return Err(parsed.diagnostics);
    }
    let Some(block) = parsed.block else {
        return Err(vec![diagnostic(
            Span::default(),
            "the layout declaration has no `{ … }` block",
        )]);
    };
    let text = print_sync(section, 1);
    let mut out = String::with_capacity(source.len() + text.len());
    match parsed.section {
        Some(at) => {
            out.push_str(&source[..at.start]);
            out.push_str(&text);
            out.push_str(&source[at.end..]);
        }
        None => {
            // Before the closing `}`, on its own line.
            let close = block.end - 1;
            let head = source[..close].trim_end_matches([' ', '\t']);
            out.push_str(head);
            if !head.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(INDENT);
            out.push_str(&text);
            out.push('\n');
            out.push_str(&source[close..]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SyncSection {
        let mut section = SyncSection::default();
        let o = &mut section.sync.options;
        o.set_ignore(Difference::Rooms, true);
        o.set_ignore(Difference::Values, true);
        o.ignored_parameters = ["datasheet".to_owned(), "stock code".to_owned()].into();
        o.relink_by_designator = false;
        section.links.insert("R5".into(), "/ch[0]/R1".into());
        section.links.insert("TP1".into(), "/main/TP 1".into());
        section.sync.components.insert(
            "/ch[0]/R1".into(),
            SyncedComponent {
                designator: "R5".into(),
                package: "R0402".into(),
                pads: [
                    ("1".to_owned(), "+3V3".to_owned()),
                    ("2".to_owned(), "ch[0]/OUT".to_owned()),
                ]
                .into(),
            },
        );
        section.sync.net_classes.insert(
            "power".into(),
            [("trace_width".to_owned(), "0.5 mm".to_owned())].into(),
        );
        section
            .sync
            .diff_pairs
            .insert("usb".into(), ("DP".into(), "DN".into()));
        section
            .sync
            .component_classes
            .insert("ch[0]".into(), vec!["R5".into(), "true".into()]);
        section
    }

    #[test]
    fn the_section_round_trips() {
        for section in [SyncSection::default(), sample()] {
            let text = print_layout("ctl", &section);
            let parsed = parse_layout("boards/ctl.circuit.layout", &text);
            assert!(
                parsed.diagnostics.is_empty(),
                "{text}\n{:#?}",
                parsed.diagnostics
            );
            assert_eq!(parsed.sync.as_ref(), Some(&section), "{text}");
            assert_eq!(parsed.name.as_deref(), Some("ctl"));
            // Printing is a fixed point.
            assert_eq!(print_layout("ctl", parsed.sync.as_ref().unwrap()), text);
        }
    }

    #[test]
    fn versions_and_errors() {
        let file = |body: &str| format!("layout ctl {{\n    sync {{\n{body}\n    }}\n}}\n");
        let newer = parse_layout("ctl.circuit.layout", &file("        version = 2"));
        assert!(newer.sync.is_none());
        assert!(newer.diagnostics[0].message.contains("newer RunMat"));
        let missing = parse_layout(
            "ctl.circuit.layout",
            &file("        footprint R1 = \"/R1\""),
        );
        assert!(missing.diagnostics[0].message.contains("version = 1"));
        let unknown = parse_layout(
            "ctl.circuit.layout",
            &file("        version = 1\n        compare = { ignore = [layers] }"),
        );
        assert!(unknown.diagnostics[0]
            .message
            .contains("not a difference type"));
        let twice = parse_layout(
            "ctl.circuit.layout",
            &file(
                "        version = 1\n        footprint R1 = \"/a\"\n        footprint R1 = \"/b\"",
            ),
        );
        assert!(twice.diagnostics[0].message.contains("given twice"));
        // No section: an empty record, no error.
        let none = parse_layout("ctl.circuit.layout", "layout ctl {\n}\n");
        assert!(none.diagnostics.is_empty() && none.sync.is_none() && none.section.is_none());
    }

    #[test]
    fn writing_replaces_only_the_section() {
        let before = "runmat 0.7\nstd 1\n\n// The controller board.\nlayout ctl {\n    outline = \"keep me\"\n}\n";
        let added = write_sync("ctl.circuit.layout", before, &sample()).unwrap();
        assert!(added.starts_with("runmat 0.7\nstd 1\n\n// The controller board.\nlayout ctl {\n    outline = \"keep me\"\n    sync {\n"));
        assert!(added.ends_with("    }\n}\n"));
        let parsed = parse_layout("ctl.circuit.layout", &added);
        assert_eq!(parsed.sync, Some(sample()));
        // Replacing keeps the text around the section.
        let mut next = sample();
        next.sync.options = ComparatorOptions::default();
        let replaced = write_sync("ctl.circuit.layout", &added, &next).unwrap();
        let at = parsed.section.unwrap();
        assert_eq!(&replaced[..at.start], &added[..at.start]);
        assert!(replaced.ends_with(&added[at.end..]));
        assert_eq!(
            parse_layout("ctl.circuit.layout", &replaced).sync,
            Some(next)
        );
        // A section from a newer release is never overwritten.
        let newer = "layout ctl {\n    sync {\n        version = 9\n    }\n}\n";
        assert!(write_sync("ctl.circuit.layout", newer, &sample()).is_err());
        assert_eq!(
            write_sync("ctl.circuit.layout", "", &sample()).unwrap(),
            print_layout("ctl", &sample())
        );
    }
}
