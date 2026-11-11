//! Structured edits on `.part`, `.assembly` and `.drawing` sources.
//!
//! An edit changes the model; [`diff_source`] then finds the smallest text change that makes
//! the source describe the new model:
//!
//! - an entry whose written values are unchanged keeps its text, comments and layout;
//! - a changed value is replaced in place (`radius = 2 mm` → `radius = 3 mm`), and only a
//!   changed set of keys reprints the entry's value;
//! - a removed entry takes its own lines and comments with it;
//! - an added entry goes next to its neighbours in the new order, at their indentation;
//! - a reordered entry moves with its comments.
//!
//! The result is checked: the edited source must parse back to the new model. If it would
//! not, the whole file is reprinted instead and [`SourceChange::reprinted`] says so.
//!
//! The catalog operations ([`add_feature`], [`set_feature_param`], [`remove_feature`],
//! [`reorder_feature`], [`rename_feature`], [`set_value`], [`add_publish`], …) are model edits
//! run through [`diff_source`].

use super::{grammar_parse_in, print_with};
use crate::ids::{ConfigurationName, FeatureId, FilePath, InputName};
use crate::model::{Feature, PartDocument, PublishEntry};
use crate::modules::ModuleScope;
use crate::refs::lock::{ItemPath, LockError, LockSection, UuidMinter};
use crate::scope::UseDecl;
use crate::value::ValueExpr;
use runmat_design_syntax::document::{DocumentGrammar, DocumentSyntax, ParsedEntry, PrintError};
use runmat_design_syntax::edit::{
    apply_edits, block_statements, indentation_at, insert_statement, own_lines, remove_statement,
    EditError, TextEdit,
};
use runmat_design_syntax::json::Json;
use runmat_design_syntax::tree::SyntaxNode;
use runmat_design_syntax::{parse_source, FeaSyntaxDiagnostic};
use runmat_types::Span;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The text change for a model edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChange {
    pub edits: Vec<TextEdit>,
    /// The source after the edits.
    pub source: String,
    /// The minimal edits did not reproduce the model, so the file was reprinted whole.
    pub reprinted: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EditFailure {
    /// The source has errors; edits need a valid file.
    Invalid(Vec<FeaSyntaxDiagnostic>),
    /// The new model cannot be written as source.
    Print(PrintError),
    Apply(EditError),
    /// The edit names something the model does not have, or would make it invalid.
    Model(String),
    Lock(LockError),
}

impl std::fmt::Display for EditFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditFailure::Invalid(diagnostics) => write!(
                f,
                "the source has {} error(s); the first: {}",
                diagnostics.len(),
                diagnostics.first().map_or("", |d| d.message.as_str())
            ),
            EditFailure::Print(error) => write!(f, "{error}"),
            EditFailure::Apply(error) => write!(f, "{error}"),
            EditFailure::Model(message) => f.write_str(message),
            EditFailure::Lock(error) => write!(f, "{error:?}"),
        }
    }
}

impl std::error::Error for EditFailure {}

/// `(old site, new site)`: an entry renamed by the edit, compared as one entry.
pub type Rename = (String, String);

struct Located<'a> {
    entry: &'a ParsedEntry,
    position: usize,
}

fn by_site(syntax: &DocumentSyntax) -> BTreeMap<&str, Located<'_>> {
    syntax
        .entries
        .iter()
        .enumerate()
        .map(|(position, entry)| (entry.site.as_str(), Located { entry, position }))
        .collect()
}

/// Re-indents `text` written at `from` indentation (its continuation lines) to `to`.
fn reindent(text: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            out.push('\n');
            let body = line.strip_prefix(from).unwrap_or(line.trim_start());
            if !body.is_empty() {
                out.push_str(to);
            }
            out.push_str(body);
        } else {
            out.push_str(line);
        }
    }
    out
}

fn statement_node(root: &SyntaxNode, span: Span) -> Option<SyntaxNode> {
    root.descendants().into_iter().find(|node| {
        // An occurrence (`use part … as b { … }`) is a use declaration inside the
        // assembly's block: an entry like any statement.
        (node.kind().is_statement()
            || node.kind() == runmat_design_syntax::tree::NodeKind::UseDeclaration)
            && node.trimmed_span() == span
    })
}

fn declaration_block(root: &SyntaxNode, span: Span) -> Option<SyntaxNode> {
    root.descendants().into_iter().find(|node| {
        node.kind() == runmat_design_syntax::tree::NodeKind::Block && node.span() == span
    })
}

/// Longest increasing subsequence of `values` (indices into it).
fn stable_indices(values: &[usize]) -> BTreeSet<usize> {
    let mut tails: Vec<usize> = Vec::new();
    let mut previous = vec![usize::MAX; values.len()];
    for (index, value) in values.iter().enumerate() {
        let position = tails.partition_point(|tail| values[*tail] < *value);
        if position > 0 {
            previous[index] = tails[position - 1];
        }
        if position == tails.len() {
            tails.push(index);
        } else {
            tails[position] = index;
        }
    }
    let mut out = BTreeSet::new();
    let mut cursor = tails.last().copied();
    while let Some(index) = cursor {
        out.insert(index);
        cursor = (previous[index] != usize::MAX).then(|| previous[index]);
    }
    out
}

/// The minimal edits turning `source` into a source for `new`. `renames` pairs an entry's old
/// site with its new one.
pub fn diff_source<T: Serialize + DeserializeOwned + PartialEq>(
    grammar: &DocumentGrammar,
    path: &str,
    source: &str,
    new: &T,
    uses: &[UseDecl],
    renames: &[Rename],
) -> Result<SourceChange, EditFailure> {
    let scope = ModuleScope::from_files([(path, source)]);
    diff_source_in(grammar, path, source, new, uses, renames, &scope)
}

/// [`diff_source`] for a file whose designs resolve through the folders of `scope`.
pub fn diff_source_in<T: Serialize + DeserializeOwned + PartialEq>(
    grammar: &DocumentGrammar,
    path: &str,
    source: &str,
    new: &T,
    uses: &[UseDecl],
    renames: &[Rename],
    scope: &ModuleScope,
) -> Result<SourceChange, EditFailure> {
    let old = grammar_parse_in::<T>(grammar, path, source, scope);
    if !old.is_valid() {
        return Err(EditFailure::Invalid(old.diagnostics));
    }
    let printed = print_with(grammar, new, uses).map_err(EditFailure::Print)?;
    let fresh = grammar_parse_in::<T>(grammar, path, &printed, scope);
    if fresh.document.as_ref() != Some(new) {
        return Err(EditFailure::Print(PrintError(
            "the new model does not read back from its own text".into(),
        )));
    }
    let (old_syntax, new_syntax) = (&old.syntax, &fresh.syntax);
    let tree = parse_source(grammar.dialect, source).syntax();
    let block = old_syntax
        .block
        .and_then(|span| declaration_block(&tree, span))
        .ok_or_else(|| EditFailure::Model("the file has no declaration block".into()))?;
    let statements = block_statements(&block);
    let index_of = |span: Span| {
        statements
            .iter()
            .position(|node| node.trimmed_span() == span)
    };

    let renamed: BTreeMap<&str, &str> = renames
        .iter()
        .map(|(from, to)| (to.as_str(), from.as_str()))
        .collect();
    let old_sites = by_site(old_syntax);
    let mut matched: BTreeSet<&str> = BTreeSet::new();
    let mut edits: Vec<TextEdit> = Vec::new();
    // Line-by-line edits inside block entries (a sketch's entities), and the whole-value
    // edits tried when those do not read back.
    let (mut fine, mut coarse): (Vec<TextEdit>, Vec<TextEdit>) = (Vec::new(), Vec::new());
    let new_indent = indentation_at(
        &printed,
        new_syntax.entries.first().map_or(0, |e| e.span.start),
    )
    .to_string();

    // Pairs in new order, per field, for the reorder pass.
    let mut pairs: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
    for (new_position, entry) in new_syntax.entries.iter().enumerate() {
        let site = renamed
            .get(entry.site.as_str())
            .copied()
            .unwrap_or(entry.site.as_str());
        if let Some(located) = old_sites.get(site) {
            matched.insert(site);
            pairs
                .entry(entry.field.as_str())
                .or_default()
                .push((new_position, located.position));
        }
    }
    let mut moved: BTreeSet<usize> = BTreeSet::new();
    for list in pairs.values() {
        let old_positions: Vec<usize> = list.iter().map(|(_, old)| *old).collect();
        let stable = stable_indices(&old_positions);
        for (index, (new_position, _)) in list.iter().enumerate() {
            if !stable.contains(&index) {
                moved.insert(*new_position);
            }
        }
    }

    let text_of = |entry: &ParsedEntry| printed[entry.span.start..entry.span.end].to_string();
    for (new_position, entry) in new_syntax.entries.iter().enumerate() {
        let site = renamed
            .get(entry.site.as_str())
            .copied()
            .unwrap_or(entry.site.as_str());
        let Some(located) = old_sites.get(site) else {
            continue;
        };
        let previous = located.entry;
        if moved.contains(&new_position) {
            let node = statement_node(&tree, previous.span)
                .ok_or_else(|| EditFailure::Model(format!("`{site}` has no statement")))?;
            let lines = own_lines(source, &node);
            let unchanged = previous.head == entry.head && same_written(previous, entry);
            let text = match (unchanged, lines) {
                (true, Some(lines)) => source[lines.start..lines.end]
                    .trim_end_matches('\n')
                    .to_string(),
                (true, None) => source[previous.span.start..previous.span.end].to_string(),
                (false, _) => text_of(entry),
            };
            edits.push(remove_statement(source, &node));
            let target = insertion_index(
                new_syntax,
                new_position,
                &old_sites,
                &renamed,
                &moved,
                &index_of,
                statements.len(),
            );
            let indent = indentation_at(source, previous.span.start).to_string();
            let text = if unchanged {
                // Old text: strip the statement's own indentation from its first line.
                text.strip_prefix(indent.as_str())
                    .unwrap_or(&text)
                    .to_string()
            } else {
                reindent(&text, &new_indent, &indent)
            };
            edits.push(insert_statement(
                source,
                &block,
                target,
                &dedent_first(&text, &indent),
            ));
            continue;
        }
        let plain = entry_edits(source, &printed, previous, entry, &new_indent);
        match line_edits(source, &tree, &printed, previous, entry, &new_indent) {
            Some(lines) => {
                fine.extend(lines);
                coarse.extend(plain);
            }
            None => edits.extend(plain),
        }
    }
    // Removed entries.
    for entry in &old_syntax.entries {
        if matched.contains(entry.site.as_str()) {
            continue;
        }
        let node = statement_node(&tree, entry.span)
            .ok_or_else(|| EditFailure::Model(format!("`{}` has no statement", entry.site)))?;
        edits.push(remove_statement(source, &node));
    }
    // Added entries.
    for (new_position, entry) in new_syntax.entries.iter().enumerate() {
        let site = renamed
            .get(entry.site.as_str())
            .copied()
            .unwrap_or(entry.site.as_str());
        if old_sites.contains_key(site) {
            continue;
        }
        let target = insertion_index(
            new_syntax,
            new_position,
            &old_sites,
            &renamed,
            &moved,
            &index_of,
            statements.len(),
        );
        let indent = statements
            .first()
            .map(|node| indentation_at(source, node.trimmed_span().start).to_string())
            .unwrap_or_else(|| new_indent.clone());
        let text = reindent(&text_of(entry), &new_indent, &indent);
        edits.push(insert_statement(
            source,
            &block,
            target,
            &dedent_first(&text, &indent),
        ));
    }
    let old_document = old
        .document
        .as_ref()
        .ok_or_else(|| EditFailure::Model("the source has no model".into()))?;
    let old_printed = print_with(grammar, old_document, uses).map_err(EditFailure::Print)?;
    edits.extend(header_edits(source, &old_printed, &printed));

    let mut attempts = vec![[edits.clone(), fine].concat()];
    if !coarse.is_empty() {
        attempts.push([edits, coarse].concat());
    }
    let tries = attempts.len();
    let mut failures = Vec::new();
    for edits in attempts {
        let edited = match apply_edits(source, &edits) {
            Ok(edited) => edited,
            Err(error) => {
                failures.push(error);
                continue;
            }
        };
        let check = grammar_parse_in::<T>(grammar, path, &edited, scope);
        if check.document.as_ref() == Some(new) && check.diagnostics.is_empty() {
            return Ok(SourceChange {
                edits,
                source: edited,
                reprinted: false,
            });
        }
    }
    if failures.len() == tries {
        return Err(EditFailure::Apply(failures.remove(0)));
    }
    Ok(SourceChange {
        edits: vec![TextEdit::replace(
            Span {
                start: 0,
                end: source.len(),
            },
            printed.clone(),
        )],
        source: printed,
        reprinted: true,
    })
}

/// Continuation lines of `text` are already at `indent`; `insert_statement` adds `indent` to
/// them again, so take it off here.
fn dedent_first(text: &str, indent: &str) -> String {
    reindent(text, indent, "")
}

fn same_written(old: &ParsedEntry, new: &ParsedEntry) -> bool {
    match (&old.written, &new.written) {
        (Some(a), Some(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(x, y)| x.0 == y.0 && x.1.to_value() == y.1.to_value())
        }
        (None, None) => true,
        _ => false,
    }
}

/// Where an added or moved entry goes: before the next entry in the new order that stays in
/// place, else at the end of the block.
fn insertion_index(
    new_syntax: &DocumentSyntax,
    new_position: usize,
    old_sites: &BTreeMap<&str, Located<'_>>,
    renamed: &BTreeMap<&str, &str>,
    moved: &BTreeSet<usize>,
    index_of: &dyn Fn(Span) -> Option<usize>,
    end: usize,
) -> usize {
    for (position, later) in new_syntax.entries.iter().enumerate().skip(new_position + 1) {
        if moved.contains(&position) {
            continue;
        }
        let site = renamed
            .get(later.site.as_str())
            .copied()
            .unwrap_or(later.site.as_str());
        if let Some(located) = old_sites.get(site) {
            if let Some(index) = index_of(located.entry.span) {
                return index;
            }
        }
    }
    end
}

fn entry_edits(
    source: &str,
    printed: &str,
    old: &ParsedEntry,
    new: &ParsedEntry,
    new_indent: &str,
) -> Vec<TextEdit> {
    let indent = indentation_at(source, old.span.start).to_string();
    let take = |span: Span| reindent(&printed[span.start..span.end], new_indent, &indent);
    let mut edits = Vec::new();
    if old.head != new.head {
        if old.value_span.is_none() {
            return vec![TextEdit::replace(old.span, take(new.span))];
        }
        edits.push(TextEdit::replace(old.head_span, take(new.head_span)));
    }
    match (&old.written, &new.written, old.value_span, new.value_span) {
        (Some(a), Some(b), _, _)
            if a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.0 == y.0) =>
        {
            let head_changed = old.head != new.head && old.value_span.is_some();
            for (x, y) in a.iter().zip(b) {
                // A value written in the head (an occurrence's source) is in the head's edit.
                if head_changed && x.2.start < old.head_span.end {
                    continue;
                }
                if x.1.to_value() != y.1.to_value() {
                    edits.push(TextEdit::replace(x.2, take(y.2)));
                }
            }
        }
        (_, _, Some(old_value), Some(new_value)) => {
            if !same_written(old, new) {
                edits.push(TextEdit::replace(old_value, take(new_value)));
            }
        }
        _ => {}
    }
    edits
}

/// Whether the written value at `span` is a line of its own in `text`: nothing but
/// indentation before it, and nothing but a `//` comment after it.
fn is_line(text: &str, span: Span) -> bool {
    let start = text[..span.start].rfind('\n').map_or(0, |i| i + 1);
    let end = text[span.end..]
        .find('\n')
        .map_or(text.len(), |i| span.end + i);
    let after = text[span.end..end].trim();
    text[start..span.start].trim().is_empty() && (after.is_empty() || after.starts_with("//"))
}

fn inside(span: Span, outer: Option<Span>) -> bool {
    outer.is_some_and(|outer| outer.start < span.start && span.end < outer.end)
}

/// The edits for a block entry whose lines were added or removed (a sketch's entities and
/// constraints): kept lines stay with their comments, changed lines are replaced, removed
/// lines go with their own comments, and added lines go next to their neighbours. `None`
/// when the entry is not written as lines of a block, or nothing was added or removed.
fn line_edits(
    source: &str,
    tree: &SyntaxNode,
    printed: &str,
    old: &ParsedEntry,
    new: &ParsedEntry,
    new_indent: &str,
) -> Option<Vec<TextEdit>> {
    let (a, b) = (old.written.as_ref()?, new.written.as_ref()?);
    if a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.0 == y.0) {
        return None;
    }
    let old_keys: BTreeMap<&str, usize> = a
        .iter()
        .enumerate()
        .map(|(i, x)| (x.0.as_str(), i))
        .collect();
    let new_keys: BTreeMap<&str, usize> = b
        .iter()
        .enumerate()
        .map(|(i, y)| (y.0.as_str(), i))
        .collect();
    if old_keys.len() != a.len() || new_keys.len() != b.len() {
        return None;
    }
    // Pairs in new order; the longest run in old order stays where it is.
    let pairs: Vec<(usize, usize)> = b
        .iter()
        .enumerate()
        .filter_map(|(j, y)| old_keys.get(y.0.as_str()).map(|i| (*i, j)))
        .collect();
    let stable: BTreeSet<usize> =
        stable_indices(&pairs.iter().map(|(i, _)| *i).collect::<Vec<_>>())
            .into_iter()
            .map(|k| pairs[k].1)
            .collect();
    let kept_old: BTreeMap<usize, usize> = pairs
        .iter()
        .filter(|(_, j)| stable.contains(j))
        .map(|(i, j)| (*j, *i))
        .collect();
    let kept: BTreeSet<usize> = kept_old.values().copied().collect();
    if kept.is_empty() {
        return None;
    }
    // Every line that moves is a line of the block, in both texts.
    for (i, x) in a.iter().enumerate() {
        let line = inside(x.2, old.value_span) && is_line(source, x.2);
        if !kept.contains(&i) && !line {
            return None;
        }
    }
    for (j, y) in b.iter().enumerate() {
        let line = inside(y.2, new.value_span) && is_line(printed, y.2);
        if !kept_old.contains_key(&j) && !line {
            return None;
        }
    }
    let indent = indentation_at(source, old.span.start).to_string();
    let take = |span: Span| reindent(&printed[span.start..span.end], new_indent, &indent);
    let mut edits = Vec::new();
    if old.head != new.head {
        edits.push(TextEdit::replace(old.head_span, take(new.head_span)));
    }
    for (j, i) in &kept_old {
        if a[*i].1.to_value() != b[*j].1.to_value() {
            edits.push(TextEdit::replace(a[*i].2, take(b[*j].2)));
        }
    }
    for (i, x) in a.iter().enumerate() {
        if !kept.contains(&i) {
            let node = statement_node(tree, x.2)?;
            edits.push(remove_statement(source, &node));
        }
    }
    // Added lines, in runs: each run goes on the lines after the kept line before it, or
    // before the first kept line.
    let mut j = 0;
    while j < b.len() {
        if kept_old.contains_key(&j) {
            j += 1;
            continue;
        }
        let run_start = j;
        while j < b.len() && !kept_old.contains_key(&j) {
            j += 1;
        }
        // Anchors are kept lines of the block (not values in the head, such as the plane).
        let anchor = |i: &usize| inside(a[*i].2, old.value_span) && is_line(source, a[*i].2);
        let before = kept_old
            .range(..run_start)
            .rev()
            .map(|(_, i)| *i)
            .find(&anchor);
        let (at, neighbour) = match before {
            Some(i) => {
                let span = a[i].2;
                let end = source[span.end..]
                    .find('\n')
                    .map_or(source.len(), |k| span.end + k + 1);
                (end, span)
            }
            None => {
                let i = kept_old.range(j..).map(|(_, i)| *i).find(&anchor)?;
                let span = a[i].2;
                let node = statement_node(tree, span)?;
                (own_lines(source, &node)?.start, span)
            }
        };
        if at == source.len() && !source.ends_with('\n') {
            return None;
        }
        let line_indent = indentation_at(source, neighbour.start).to_string();
        let mut text = String::new();
        for y in &b[run_start..j] {
            text.push_str(&line_indent);
            text.push_str(&take(y.2));
            text.push('\n');
        }
        edits.push(TextEdit::insert(at, text));
    }
    Some(edits)
}

fn is_header_line(line: &str) -> bool {
    line.starts_with("runmat ") || line.starts_with("std ")
}

/// Blank lines and `//` comments, which may come before the version header.
fn is_leading_trivia(line: &str) -> bool {
    line.is_empty() || line.starts_with("//")
}

/// The `runmat` and `std` lines at the top of a source (after any leading comments).
fn version_lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .skip_while(|line| is_leading_trivia(line))
        .take_while(|line| line.is_empty() || is_header_line(line))
        .filter(|line| !line.is_empty())
        .collect()
}

/// Rewrites the version header when the model's header changed: the whole header block is
/// replaced (or added at the top, or removed), so the result has exactly the new header lines
/// followed by one blank line. A header below leading comments is found there and edited in
/// place; the comments stay above it.
fn header_edits(source: &str, old_printed: &str, new_printed: &str) -> Vec<TextEdit> {
    let wanted = version_lines(new_printed);
    if version_lines(old_printed) == wanted {
        return Vec::new();
    }
    // The header block: its first line through the blank lines after its last line.
    let mut offset = 0;
    let mut start = None;
    let mut end = 0;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim();
        match start {
            None if is_header_line(trimmed) => {
                start = Some(offset);
                end = offset + line.len();
            }
            None if is_leading_trivia(trimmed) => {}
            None => break,
            Some(_) if is_header_line(trimmed) || trimmed.is_empty() => end = offset + line.len(),
            Some(_) => break,
        }
        offset += line.len();
    }
    let text = if wanted.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", wanted.join("\n"))
    };
    match start {
        Some(start) => vec![TextEdit::replace(Span { start, end }, text)],
        None => vec![TextEdit::insert(0, text)],
    }
}

// ------------------------------------------------------------------------------ the catalog

fn edit_part_with(
    source: &str,
    path: &str,
    renames: &[Rename],
    change: impl FnOnce(&mut PartDocument) -> Result<(), EditFailure>,
) -> Result<SourceChange, EditFailure> {
    let parsed = super::parse_part(path, source);
    let (Some(mut document), true) = (parsed.document.clone(), parsed.diagnostics.is_empty())
    else {
        return Err(EditFailure::Invalid(parsed.diagnostics));
    };
    change(&mut document)?;
    diff_source(&super::PART, path, source, &document, &parsed.uses, renames)
}

/// Any edit to a part's model, as the smallest text change.
pub fn edit_part(
    source: &str,
    path: &str,
    change: impl FnOnce(&mut PartDocument) -> Result<(), EditFailure>,
) -> Result<SourceChange, EditFailure> {
    edit_part_with(source, path, &[], change)
}

fn missing(what: &str, name: &str) -> EditFailure {
    EditFailure::Model(format!("the part has no {what} `{name}`"))
}

/// Adds a feature after `after` (or at the end of the history).
pub fn add_feature(
    source: &str,
    path: &str,
    feature: Feature,
    after: Option<&str>,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        if document.feature(&feature.id).is_some() {
            return Err(EditFailure::Model(format!(
                "the part already has a feature `{}`",
                feature.id
            )));
        }
        let at = match after {
            Some(name) => document
                .features
                .iter()
                .position(|f| f.id.0 == name)
                .map(|i| i + 1)
                .ok_or_else(|| missing("feature", name))?,
            None => document.features.len(),
        };
        document.features.insert(at, feature);
        Ok(())
    })
}

pub fn remove_feature(source: &str, path: &str, id: &str) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        let at = document
            .features
            .iter()
            .position(|f| f.id.0 == id)
            .ok_or_else(|| missing("feature", id))?;
        document.features.remove(at);
        Ok(())
    })
}

/// Moves a feature to just after `after` (or to the front of the history).
pub fn reorder_feature(
    source: &str,
    path: &str,
    id: &str,
    after: Option<&str>,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        if document.reorder(id, after) {
            Ok(())
        } else {
            Err(missing(
                "feature",
                after
                    .filter(|a| document.feature(&FeatureId::new(*a)).is_none())
                    .unwrap_or(id),
            ))
        }
    })
}

pub fn suppress_feature(
    source: &str,
    path: &str,
    id: &str,
    suppressed: bool,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        let feature = document
            .feature_mut(id)
            .ok_or_else(|| missing("feature", id))?;
        feature.suppressed = suppressed;
        Ok(())
    })
}

/// Sets a feature parameter by its path in the feature's parameters (`radius`, `edges[1]`,
/// `extent.distance`); `value` is the parameter's model JSON (a [`ValueExpr`] serialises to
/// one).
pub fn set_feature_param(
    source: &str,
    path: &str,
    id: &str,
    param: &str,
    value: serde_json::Value,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        let feature = document
            .feature_mut(id)
            .ok_or_else(|| missing("feature", id))?;
        let mut kind =
            serde_json::to_value(&feature.kind).map_err(|e| EditFailure::Model(e.to_string()))?;
        let slot = json_slot(&mut kind, param).ok_or_else(|| {
            EditFailure::Model(format!("feature `{id}` has no parameter `{param}`"))
        })?;
        *slot = value;
        feature.kind = serde_json::from_value(kind)
            .map_err(|e| EditFailure::Model(format!("`{param}` of feature `{id}`: {e}")))?;
        Ok(())
    })
}

/// The value at `a.b[2].c`, created for a missing last key of an object.
fn json_slot<'a>(
    value: &'a mut serde_json::Value,
    path: &str,
) -> Option<&'a mut serde_json::Value> {
    let mut steps = Vec::new();
    for part in path.split('.') {
        let (key, rest) = part.split_once('[').map_or((part, ""), |(k, r)| (k, r));
        if !key.is_empty() {
            steps.push(Err(key.to_string()));
        }
        for index in rest.split('[') {
            if let Some(number) = index.strip_suffix(']') {
                steps.push(Ok(number.parse::<usize>().ok()?));
            }
        }
    }
    let mut node = value;
    let count = steps.len();
    for (position, step) in steps.into_iter().enumerate() {
        node = match step {
            Ok(index) => node.as_array_mut()?.get_mut(index)?,
            Err(key) => {
                let object = node.as_object_mut()?;
                if position + 1 == count && !object.contains_key(&key) {
                    object.insert(key.clone(), serde_json::Value::Null);
                }
                object.get_mut(&key)?
            }
        };
    }
    Some(node)
}

/// Renames a feature. Its statement takes the new name and the lock records the old one as an
/// alias, so references written against the old name keep resolving (the lock is the one place
/// renames live).
pub fn rename_feature(
    source: &str,
    path: &str,
    from: &str,
    to: &str,
    lock: &mut LockSection,
    minter: &mut dyn UuidMinter,
) -> Result<SourceChange, EditFailure> {
    let renames = vec![(format!("feature.{from}"), format!("feature.{to}"))];
    let change = edit_part_with(source, path, &renames, |document| {
        if document.feature(&FeatureId::new(to)).is_some() {
            return Err(EditFailure::Model(format!(
                "the part already has a feature `{to}`"
            )));
        }
        let feature = document
            .feature_mut(from)
            .ok_or_else(|| missing("feature", from))?;
        feature.id = FeatureId::new(to);
        Ok(())
    })?;
    let file = FilePath::new(path);
    let old = ItemPath::feature(&file, &FeatureId::new(from));
    lock.ensure(&old, minter);
    lock.rename(&old, &ItemPath::feature(&file, &FeatureId::new(to)))
        .map_err(EditFailure::Lock)?;
    Ok(change)
}

/// Which configurations a value edit applies to (the Desktop's `ConfigurationScope`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum ConfigurationScope {
    /// The value itself: the input's default, every configuration's override cleared.
    All,
    This {
        configuration: String,
    },
    Specify {
        configurations: Vec<String>,
    },
}

/// A value field's edit, as the Desktop's value editor sends it (`ValueEdit`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueEdit {
    pub value: ValueExpr,
    /// The text it was typed as (kept for the editor; the source gets the canonical text).
    #[serde(default)]
    pub text: String,
    pub scope: ConfigurationScope,
}

/// Sets an input's value in the given scope: its default (`all`), or an override in the named
/// configurations (`this`, `specify`).
pub fn set_value(
    source: &str,
    path: &str,
    input: &str,
    edit: &ValueEdit,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        let name = InputName::new(input);
        let declared = document
            .inputs
            .iter_mut()
            .find(|i| i.name == name)
            .ok_or_else(|| missing("input", input))?;
        let targets: Vec<String> = match &edit.scope {
            ConfigurationScope::All => {
                declared.default = edit.value.clone();
                for configuration in &mut document.configurations {
                    configuration.overrides.remove(&name);
                }
                return Ok(());
            }
            ConfigurationScope::This { configuration } => vec![configuration.clone()],
            ConfigurationScope::Specify { configurations } => configurations.clone(),
        };
        for target in targets {
            let configuration = document
                .configurations
                .iter_mut()
                .find(|c| c.name == ConfigurationName::new(&target))
                .ok_or_else(|| missing("configuration", &target))?;
            configuration
                .overrides
                .insert(name.clone(), edit.value.clone());
        }
        Ok(())
    })
}

pub fn add_publish(
    source: &str,
    path: &str,
    entry: PublishEntry,
) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        if document.publish(&entry.name).is_some() {
            return Err(EditFailure::Model(format!(
                "the part already publishes `{}`",
                entry.name
            )));
        }
        document.publishes.push(entry);
        Ok(())
    })
}

pub fn remove_publish(source: &str, path: &str, name: &str) -> Result<SourceChange, EditFailure> {
    edit_part(source, path, |document| {
        let at = document
            .publishes
            .iter()
            .position(|p| p.name.0 == name)
            .ok_or_else(|| missing("publish", name))?;
        document.publishes.remove(at);
        Ok(())
    })
}

/// The written values of a part's entry, for hosts that show what a statement holds.
pub fn written_values(syntax: &DocumentSyntax, site: &str) -> Option<Vec<(String, Json)>> {
    let entry = syntax.entries.iter().find(|entry| entry.site == site)?;
    Some(
        entry
            .written
            .as_ref()?
            .iter()
            .map(|(k, v, _)| (k.clone(), v.clone()))
            .collect(),
    )
}
