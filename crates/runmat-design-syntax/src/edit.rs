//! Minimal text edits over the lossless tree.
//!
//! A structured edit names a node and what changes about it; these helpers turn that into the
//! smallest byte-range replacement. Text outside the edited ranges, comments and blank lines
//! included, is never rewritten. Statement helpers work on whole lines: removing or moving a
//! statement takes its trailing comment and the comment lines directly above it along, and an
//! inserted statement takes the indentation of its neighbours.

use crate::tree::{NodeKind, SyntaxNode};
use runmat_types::Span;

/// Replace the bytes of `span` with `text` (an empty span inserts, empty text deletes).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TextEdit {
    pub span: Span,
    pub text: String,
}

impl TextEdit {
    pub fn replace(span: Span, text: impl Into<String>) -> Self {
        Self {
            span,
            text: text.into(),
        }
    }

    pub fn insert(at: usize, text: impl Into<String>) -> Self {
        Self::replace(Span { start: at, end: at }, text)
    }

    pub fn delete(span: Span) -> Self {
        Self::replace(span, "")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditError {
    /// Two edits touch the same bytes.
    Overlap { first: Span, second: Span },
    /// An edit reaches past the end of the source or splits a character.
    OutOfRange { span: Span },
}

impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditError::Overlap { first, second } => write!(
                f,
                "edits overlap at {}..{} and {}..{}",
                first.start, first.end, second.start, second.end
            ),
            EditError::OutOfRange { span } => {
                write!(f, "edit {}..{} is outside the source", span.start, span.end)
            }
        }
    }
}

impl std::error::Error for EditError {}

/// Applies non-overlapping edits. Insertions at the same offset apply in the order given.
pub fn apply_edits(source: &str, edits: &[TextEdit]) -> Result<String, EditError> {
    let mut ordered: Vec<(usize, &TextEdit)> = edits.iter().enumerate().collect();
    ordered.sort_by_key(|(order, edit)| (edit.span.start, edit.span.end, *order));
    for window in ordered.windows(2) {
        let (first, second) = (&window[0].1.span, &window[1].1.span);
        if second.start < first.end {
            return Err(EditError::Overlap {
                first: *first,
                second: *second,
            });
        }
    }
    let mut out = String::with_capacity(source.len());
    let mut cursor = 0;
    for (_, edit) in ordered {
        let span = edit.span;
        if span.end > source.len()
            || span.start > span.end
            || !source.is_char_boundary(span.start)
            || !source.is_char_boundary(span.end)
        {
            return Err(EditError::OutOfRange { span });
        }
        out.push_str(&source[cursor..span.start]);
        out.push_str(&edit.text);
        cursor = span.end;
    }
    out.push_str(&source[cursor..]);
    Ok(out)
}

/// Replaces a node's text (without its surrounding trivia).
pub fn replace_node(node: &SyntaxNode, text: impl Into<String>) -> TextEdit {
    TextEdit::replace(node.trimmed_span(), text)
}

fn line_start(source: &str, offset: usize) -> usize {
    source[..offset].rfind('\n').map_or(0, |index| index + 1)
}

/// The end of the line holding `offset`, after its newline (or the end of the source).
fn line_end(source: &str, offset: usize) -> usize {
    source[offset..]
        .find('\n')
        .map_or(source.len(), |index| offset + index + 1)
}

/// The whitespace that starts the line holding `offset`.
pub fn indentation_at(source: &str, offset: usize) -> &str {
    let start = line_start(source, offset);
    let rest = &source[start..];
    let width = rest.len() - rest.trim_start_matches([' ', '\t']).len();
    &rest[..width]
}

/// The lines a statement occupies alone, with its trailing comment and the comment lines
/// directly above it (no blank line between); `None` when it shares a line with other code.
pub fn own_lines(source: &str, statement: &SyntaxNode) -> Option<Span> {
    let span = statement.trimmed_span();
    let start = line_start(source, span.start);
    let end_of_line = line_end(source, span.end);
    let after = source[span.end..end_of_line].trim();
    if !source[start..span.start].trim().is_empty()
        || !(after.is_empty() || after.starts_with("//"))
    {
        return None;
    }
    let mut first = start;
    while first > 0 {
        let previous = line_start(source, first - 1);
        if source[previous..first].trim().starts_with("//") {
            first = previous;
        } else {
            break;
        }
    }
    Some(Span {
        start: first,
        end: end_of_line,
    })
}

/// The bytes removing a statement takes: [`own_lines`], or just its text when it shares a
/// line.
pub fn statement_lines(source: &str, statement: &SyntaxNode) -> Span {
    own_lines(source, statement).unwrap_or_else(|| statement.trimmed_span())
}

/// Removes a statement and the lines it alone occupies.
pub fn remove_statement(source: &str, statement: &SyntaxNode) -> TextEdit {
    TextEdit::delete(statement_lines(source, statement))
}

/// The statements directly in a block, in order.
pub fn block_statements(block: &SyntaxNode) -> Vec<SyntaxNode> {
    block
        .children()
        .filter(|child| child.kind().is_statement() || child.kind() == NodeKind::Error)
        .collect()
}

/// Indents every line of `text` after the first by `indent`.
fn reindent(text: &str, indent: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            out.push('\n');
            if !line.is_empty() {
                out.push_str(indent);
            }
        }
        out.push_str(line);
    }
    out
}

/// Inserts `text` (one statement, its own continuation lines indented relative to its first
/// line) into `block` before the statement at `index`, or at the end when `index` is past the
/// last. The statement takes its neighbours' indentation, or the block's plus one unit.
pub fn insert_statement(source: &str, block: &SyntaxNode, index: usize, text: &str) -> TextEdit {
    let statements = block_statements(block);
    let span = block.trimmed_span();
    let open = span.start;
    let close = span.end.saturating_sub(1);
    let indent = match statements.first() {
        Some(first) => indentation_at(source, first.trimmed_span().start).to_string(),
        None => format!("{}{}", indentation_at(source, open), crate::json::INDENT),
    };
    let body = reindent(text, &indent);
    if let Some(next) = statements.get(index) {
        return match own_lines(source, next) {
            Some(lines) => TextEdit::insert(lines.start, format!("{indent}{body}\n")),
            None => TextEdit::insert(next.trimmed_span().start, format!("{body}\n{indent}")),
        };
    }
    let closing_indent = indentation_at(source, close).to_string();
    let close_alone = source[line_start(source, close)..close].trim().is_empty();
    if close_alone
        && statements
            .last()
            .is_some_and(|last| last.trimmed_span().end < line_start(source, close))
        || close_alone && statements.is_empty() && source[open + 1..close].contains('\n')
    {
        let at = line_start(source, close);
        return TextEdit::insert(at, format!("{indent}{body}\n"));
    }
    // The block closes on a line with code (`part a {}` or `{ x = 1 }`): open it up.
    let inner = source[open + 1..close].trim();
    let mut replacement = String::from("{\n");
    if !inner.is_empty() {
        replacement.push_str(&indent);
        replacement.push_str(inner);
        replacement.push('\n');
    }
    replacement.push_str(&indent);
    replacement.push_str(&body);
    replacement.push('\n');
    replacement.push_str(&closing_indent);
    replacement.push('}');
    TextEdit::replace(
        Span {
            start: open,
            end: close + 1,
        },
        replacement,
    )
}

/// Moves a statement to before the statement at `index` of `block` (or to its end), keeping
/// its comments. Returns no edits when it would not move.
pub fn move_statement(
    source: &str,
    statement: &SyntaxNode,
    block: &SyntaxNode,
    index: usize,
) -> Vec<TextEdit> {
    let statements = block_statements(block);
    let current = statements.iter().position(|node| node == statement);
    if current.is_some_and(|position| position == index || position + 1 == index) {
        return Vec::new();
    }
    let Some(lines) = own_lines(source, statement) else {
        // A statement sharing its line: move just its text.
        let span = statement.trimmed_span();
        let text = source[span.start..span.end].to_string();
        return vec![
            TextEdit::delete(span),
            insert_statement(source, block, index, &text),
        ];
    };
    let insert_at = match statements.get(index) {
        Some(next) => statement_lines(source, next).start,
        None => line_start(source, block.trimmed_span().end.saturating_sub(1)),
    };
    let text = source[lines.start..lines.end].to_string();
    vec![TextEdit::delete(lines), TextEdit::insert(insert_at, text)]
}

/// Renames a name token (`Name` node) in place.
pub fn rename(name: &SyntaxNode, new_name: &str) -> TextEdit {
    replace_node(name, new_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_source, Dialect};

    fn block(source: &str) -> SyntaxNode {
        let syntax = parse_source(Dialect::Part, source);
        assert!(syntax.diagnostics.is_empty(), "{:#?}", syntax.diagnostics);
        syntax
            .syntax()
            .descendants()
            .into_iter()
            .find(|node| node.kind() == NodeKind::Block)
            .unwrap()
    }

    const SOURCE: &str = "part p {\n    // width of the plate\n    input w : length = 8 mm   // keep\n\n    let h = 2 mm\n    let k = 3 mm\n}\n";

    #[test]
    fn removing_a_statement_takes_its_comments_and_keeps_the_rest() {
        let block = block(SOURCE);
        let statements = block_statements(&block);
        let out = apply_edits(SOURCE, &[remove_statement(SOURCE, &statements[0])]).unwrap();
        assert_eq!(out, "part p {\n\n    let h = 2 mm\n    let k = 3 mm\n}\n");
    }

    #[test]
    fn inserting_takes_the_neighbours_indentation() {
        let block = block(SOURCE);
        let end = insert_statement(SOURCE, &block, 9, "let z = {\n    a = 1\n}");
        let out = apply_edits(SOURCE, &[end]).unwrap();
        assert!(
            out.ends_with("    let k = 3 mm\n    let z = {\n        a = 1\n    }\n}\n"),
            "{out}"
        );
        let middle = insert_statement(SOURCE, &block, 2, "let m = 1 mm");
        let out = apply_edits(SOURCE, &[middle]).unwrap();
        assert!(
            out.contains("    let h = 2 mm\n    let m = 1 mm\n    let k = 3 mm\n"),
            "{out}"
        );
        let first = insert_statement(SOURCE, &block, 0, "let a = 1 mm");
        let out = apply_edits(SOURCE, &[first]).unwrap();
        assert!(
            out.starts_with("part p {\n    let a = 1 mm\n    // width"),
            "{out}"
        );
    }

    #[test]
    fn inserting_into_an_empty_block_opens_it() {
        let source = "part p {}\n";
        let out = apply_edits(
            source,
            &[insert_statement(source, &block(source), 0, "let a = 1")],
        )
        .unwrap();
        assert_eq!(out, "part p {\n    let a = 1\n}\n");
        let source = "part p {\n}\n";
        let out = apply_edits(
            source,
            &[insert_statement(source, &block(source), 0, "let a = 1")],
        )
        .unwrap();
        assert_eq!(out, "part p {\n    let a = 1\n}\n");
    }

    #[test]
    fn moving_a_statement_keeps_its_comment() {
        let block = block(SOURCE);
        let statements = block_statements(&block);
        let edits = move_statement(SOURCE, &statements[0], &block, 3);
        let out = apply_edits(SOURCE, &edits).unwrap();
        assert_eq!(
            out,
            "part p {\n\n    let h = 2 mm\n    let k = 3 mm\n    // width of the plate\n    input w : length = 8 mm   // keep\n}\n"
        );
        assert!(move_statement(SOURCE, &statements[1], &block, 2).is_empty());
    }

    #[test]
    fn overlapping_edits_are_refused() {
        let edits = [
            TextEdit::replace(Span { start: 0, end: 4 }, "x"),
            TextEdit::replace(Span { start: 2, end: 6 }, "y"),
        ];
        assert!(matches!(
            apply_edits(SOURCE, &edits),
            Err(EditError::Overlap { .. })
        ));
    }
}
