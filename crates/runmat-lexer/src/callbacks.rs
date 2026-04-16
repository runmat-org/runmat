use crate::scan::find_line_terminator;
use crate::token::Token;
use logos::{Filter, Lexer};

pub(crate) fn double_quoted_string_emit(lexer: &mut Lexer<Token>) -> Filter<()> {
    // Always emit and mark as value
    lexer.extras.last_was_value = true;
    Filter::Emit(())
}

pub(crate) fn ellipsis_emit_and_skip_to_eol(lex: &mut Lexer<Token>) -> Filter<()> {
    // After an ellipsis, ignore the remainder of the physical line (including comments)
    let rest = lex.remainder();
    if let Some((idx, len)) = find_line_terminator(rest) {
        lex.bump(idx + len); // consume through the newline so no standalone newline token is emitted
    } else {
        lex.bump(rest.len());
    }
    lex.extras.last_was_value = true; // e.g., '1 + ...\n 2' the ellipsis does not reset value-ness
    Filter::Emit(())
}

pub(crate) fn newline_skip(lex: &mut Lexer<Token>) -> Filter<()> {
    lex.extras.line_start = true;
    lex.extras.last_was_value = false;
    Filter::Emit(())
}

pub(crate) fn section_marker(lex: &mut Lexer<Token>) -> Filter<()> {
    // Only emit a Section token when at start of line; otherwise, treat as a comment and skip
    if lex.extras.line_start {
        lex.extras.line_start = true;
        lex.extras.last_was_value = false;
        if let Some((_, len)) = find_line_terminator(lex.remainder()) {
            lex.bump(len);
        }
        Filter::Emit(())
    } else {
        // Skip to end of line (already consumed by regex except for the newline char)
        Filter::Skip
    }
}

pub(crate) fn block_comment_skip(lex: &mut Lexer<Token>) -> Filter<()> {
    // We matched '%{'. Skip until the first '%}' or end of input.
    let rest = lex.remainder();
    if let Some(end) = rest.find("%}") {
        lex.bump(end + 2); // consume up to and including '%}'
    } else {
        lex.bump(rest.len()); // consume to end if no terminator
    }
    if let Some((_, len)) = find_line_terminator(lex.remainder()) {
        lex.bump(len);
        lex.extras.line_start = true;
        lex.extras.last_was_value = false;
    }
    Filter::Skip
}

pub(crate) fn line_comment_start(lex: &mut Lexer<Token>) -> Filter<()> {
    // We just consumed a single '%'. Skip to the end of the line.
    let rest = lex.remainder();
    if let Some(pos) = rest.find('\n') {
        lex.bump(pos);
    } else {
        lex.bump(rest.len());
    }
    Filter::Skip
}
