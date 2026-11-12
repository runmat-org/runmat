//! Authored text in rewritten layout statements.
//!
//! The printer writes lengths as bare millimetres and angles as bare degrees. When an edit
//! rewrites a statement, [`keep_written`] puts back every value the engineer wrote that still
//! reads the same: `width = 12 mil` stays `12 mil` when only the track's layer changed, and
//! `at = [0.3 in, 20]` keeps its `0.3 in` when only the y moved.
//!
//! [`write_literal`] writes one value as the user typed it (`0.3 in`, `12 mil`), and
//! [`edit_board_written`] runs a board edit that way: Properties keeps a typed literal in
//! another unit as written.
//!
//! Callers check that the edited file reads back as the edited board, so a merge that would
//! change the board is never kept.

use super::read::head;
use super::*;
use crate::syntax::board_edit::{edit_parsed, EditFailure, SourceChange};
use runmat_design_syntax::ast::{Item, Statement};
use runmat_design_syntax::edit::apply_edits;

/// One step from a statement's value to a value inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step<'a> {
    /// A record field (`width`).
    Key(&'a str),
    /// A list element (`at`'s x is `Index(0)`).
    Index(usize),
}

/// `text` read as one layout statement: its value and the offset of `text` in the wrapper the
/// reader saw.
fn statement_value(text: &str) -> Option<(Val, usize)> {
    const OPEN: &str = "layout probe {\n    ";
    let wrapped = format!("{OPEN}{text}\n}}\n");
    let syntax = parse_source(Dialect::CircuitLayout, &wrapped);
    if !syntax.diagnostics.is_empty() {
        return None;
    }
    let statement: Statement = syntax.file().items().into_iter().find_map(|item| match item {
        Item::Declaration(d) => d.block().and_then(|b| b.statements().into_iter().next()),
        _ => None,
    })?;
    let value = head(&statement).ok()??.value?;
    Some((value, OPEN.len()))
}

/// Whether the written value `old` reads as the printed value `new`.
fn reads_the_same(old: &Val, new: &Val) -> bool {
    match (&old.v, &new.v) {
        (V::Num(a), V::Num(b)) => a == b || a.parse::<f64>().ok() == b.parse::<f64>().ok(),
        (V::Qty(..), V::Num(_)) => {
            if let (Ok(a), Ok(b)) = (len(old), len(new)) {
                return a == b;
            }
            matches!((angle(old), angle(new)), (Ok(a), Ok(b)) if (a - b).abs() <= 1e-9 * b.abs().max(1.0))
        }
        (V::Qty(a, u), V::Qty(b, w)) => a == b && u == w,
        _ => false,
    }
}

/// The replacements (span in the new text → written text) that keep `old`'s values in `new`.
fn kept(old: (&Val, &str, usize), new: (&Val, &str, usize), out: &mut Vec<(Span, String)>) {
    let ((o, old_text, old_at), (n, new_text, new_at)) = (old, new);
    match (&o.v, &n.v) {
        (V::Rec(olds), V::Rec(news)) => {
            for (key, nv) in news {
                if let Some((_, ov)) = olds.iter().find(|(k, _)| k == key) {
                    kept((ov, old_text, old_at), (nv, new_text, new_at), out);
                }
            }
        }
        (V::List(olds), V::List(news)) if olds.len() == news.len() => {
            for (ov, nv) in olds.iter().zip(news) {
                kept((ov, old_text, old_at), (nv, new_text, new_at), out);
            }
        }
        (V::Num(_) | V::Qty(..), V::Num(_) | V::Qty(..)) if reads_the_same(o, n) => {
            let from = trim_span(old_text, shift(o.span, old_at));
            let to = trim_span(new_text, shift(n.span, new_at));
            if let Some(written) = old_text.get(from.start..from.end) {
                out.push((to, written.to_owned()));
            }
        }
        _ => {}
    }
}

fn shift(span: Span, by: usize) -> Span {
    Span {
        start: span.start.saturating_sub(by),
        end: span.end.saturating_sub(by),
    }
}

/// `text` with each span replaced; spans don't overlap.
fn replaced(text: &str, mut spans: Vec<(Span, String)>) -> Option<String> {
    spans.sort_by_key(|(s, _)| s.start);
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for (span, with) in spans {
        if span.start < at || span.end > text.len() {
            return None;
        }
        out.push_str(text.get(at..span.start)?);
        out.push_str(&with);
        at = span.end;
    }
    out.push_str(text.get(at..)?);
    Some(out)
}

/// `printed`, the printer's text for a statement whose source text was `written`, with every
/// value that reads the same as the written one kept as written. `printed` unchanged when
/// either text doesn't read as one statement.
pub fn keep_written(written: &str, printed: &str) -> String {
    let (Some((old, old_at)), Some((new, new_at))) =
        (statement_value(written), statement_value(printed))
    else {
        return printed.to_owned();
    };
    let mut spans = Vec::new();
    // Spans are in the wrappers' coordinates; `kept` shifts them back into each text.
    kept((&old, written, old_at), (&new, printed, new_at), &mut spans);
    spans.retain(|(span, with)| printed.get(span.start..span.end) != Some(with.as_str()));
    if spans.is_empty() {
        return printed.to_owned();
    }
    replaced(printed, spans).unwrap_or_else(|| printed.to_owned())
}

/// The statement `text` with the value at `at` written as `literal`, or `None` when the
/// statement has no value there.
pub fn write_literal(text: &str, at: &[Step], literal: &str) -> Option<String> {
    let (value, offset) = statement_value(text)?;
    let mut v = &value;
    for step in at {
        v = match (step, &v.v) {
            (Step::Key(key), V::Rec(entries)) => &entries.iter().find(|(k, _)| k == key)?.1,
            (Step::Index(i), V::List(items)) => items.get(*i)?,
            _ => return None,
        };
    }
    if !matches!(v.v, V::Num(_) | V::Qty(..)) {
        return None;
    }
    let span = trim_span(text, shift(v.span, offset));
    replaced(text, vec![(span, literal.to_owned())])
}

/// A value written as the user typed it: the item, where in its statement, and the text.
#[derive(Clone, Copy, Debug)]
pub struct Written<'a> {
    pub item: ItemId,
    pub at: &'a [Step<'a>],
    pub text: &'a str,
}

/// [`edit_parsed`](crate::syntax::board_edit::edit_parsed) on `source`, then `written`'s value
/// written as typed in the item's rewritten statement. The typed text is kept only when the
/// file still reads as the edited board; otherwise the printed value stands.
pub fn edit_board_written<T>(
    path: &str,
    source: &str,
    parsed: &ParsedBoard,
    written: Option<Written<'_>>,
    edit: impl FnOnce(&mut Board) -> Result<T, String>,
) -> Result<(SourceChange, T), EditFailure> {
    let (change, out) = edit_parsed(path, source, parsed, edit)?;
    let Some(written) = written.filter(|_| !change.reprinted) else {
        return Ok((change, out));
    };
    let Some(span) = parsed
        .statements
        .iter()
        .find(|s| s.item == Some(written.item))
        .map(|s| s.span)
    else {
        return Ok((change, out));
    };
    let mut edits = change.edits.clone();
    let Some(edit) = edits.iter_mut().find(|e| e.span == span) else {
        return Ok((change, out));
    };
    let Some(text) = write_literal(&edit.text, written.at, written.text) else {
        return Ok((change, out));
    };
    edit.text = text;
    let Ok(typed) = apply_edits(source, &edits) else {
        return Ok((change, out));
    };
    let board = |text: &str| parse_board(path, text).board.map(|b| canonical(&b));
    if board(&typed).is_some() && board(&typed) == board(&change.source) {
        return Ok((
            SourceChange {
                edits,
                source: typed,
                reprinted: false,
            },
            out,
        ));
    }
    Ok((change, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rewritten_statement_keeps_the_values_that_did_not_change() {
        let written = "track GND = { layer = F.Cu, width = 12 mil, from = [0.1 in, 20], to = [30, 20] }";
        let printed = "track GND = { layer = B.Cu, width = 0.3048, from = [2.54, 20], to = [31, 20] }";
        assert_eq!(
            keep_written(written, printed),
            "track GND = { layer = B.Cu, width = 12 mil, from = [0.1 in, 20], to = [31, 20] }"
        );
    }

    #[test]
    fn a_changed_value_takes_the_printed_text() {
        let written = "track GND = { layer = F.Cu, width = 12 mil, from = [0, 0], to = [1, 0] }";
        let printed = "track GND = { layer = F.Cu, width = 0.5, from = [0, 0], to = [1, 0] }";
        assert_eq!(keep_written(written, printed), printed);
    }

    #[test]
    fn angles_in_another_unit_are_kept() {
        let written = "footprint R1 = { package = R0402, at = [1, 2], rotation = 90 deg }";
        let printed = "footprint R1 = { package = R0402, at = [1, 3], rotation = 90 }";
        assert!(keep_written(written, printed).ends_with("rotation = 90 deg }"));
    }

    #[test]
    fn a_literal_is_written_where_the_step_points() {
        let text = "via GND = { at = [5, 5], size = 0.6, drill = 0.3048 }";
        assert_eq!(
            write_literal(text, &[Step::Key("drill")], "12 mil").as_deref(),
            Some("via GND = { at = [5, 5], size = 0.6, drill = 12 mil }")
        );
        assert_eq!(
            write_literal(text, &[Step::Key("at"), Step::Index(1)], "0.2 in").as_deref(),
            Some("via GND = { at = [5, 0.2 in], size = 0.6, drill = 0.3048 }")
        );
        assert_eq!(write_literal(text, &[Step::Key("width")], "1 mm"), None);
    }
}
