mod assignment;
mod classdef;
mod command;
mod cursor;
mod expr;
mod stmt;

use runmat_lexer::Token;

use crate::{ParserOptions, Program, Stmt, SyntaxError};

#[derive(Clone)]
struct TokenInfo {
    token: Token,
    lexeme: String,
    position: usize,
    end: usize,
}

struct Parser {
    tokens: Vec<TokenInfo>,
    pos: usize,
    input: String,
    options: ParserOptions,
    in_matrix_expr: bool,
    current_classdef_name: Option<String>,
}

pub fn parse(input: &str) -> Result<Program, SyntaxError> {
    parse_with_options(input, ParserOptions::default())
}

pub fn parse_with_options(input: &str, options: ParserOptions) -> Result<Program, SyntaxError> {
    use runmat_lexer::tokenize_detailed;

    let toks = tokenize_detailed(input);
    let mut tokens = Vec::new();
    let mut skip_newlines = false;

    for t in toks {
        if matches!(t.token, Token::Error) {
            return Err(invalid_token_error(input, &t.lexeme, t.start));
        }
        // Skip layout-only tokens from lexing.
        if matches!(
            t.token,
            Token::Ellipsis | Token::Section | Token::LineComment | Token::BlockComment
        ) {
            // After ellipsis, also drop any immediately following Newline tokens.
            // The lexer callback already consumed the first \n after `...`; any
            // additional blank lines should be treated as part of the continuation.
            skip_newlines = matches!(t.token, Token::Ellipsis);
            continue;
        }
        if skip_newlines && matches!(t.token, Token::Newline) {
            continue;
        }
        skip_newlines = false;
        tokens.push(TokenInfo {
            token: t.token,
            lexeme: t.lexeme,
            position: t.start,
            end: t.end,
        });
    }

    let mut parser = Parser {
        tokens,
        pos: 0,
        input: input.to_string(),
        options,
        in_matrix_expr: false,
        current_classdef_name: None,
    };
    parser.parse_program()
}

/// Build the error for a character the lexer cannot tokenize.
///
/// The message names the line and column and quotes the character once, so it
/// reads on its own without a source excerpt. A hint classifies the common
/// paste mistakes: Python code, curly quotes, and non-ASCII identifier letters.
fn invalid_token_error(input: &str, lexeme: &str, start: usize) -> SyntaxError {
    let bad_char = lexeme.chars().next().unwrap_or('?');
    let (line, column) = line_and_column(input, start);
    let mut message = format!("Invalid character '{bad_char}' at line {line}, column {column}.");
    if let Some(hint) = invalid_character_hint(input, bad_char) {
        message.push(' ');
        message.push_str(&hint);
    }
    SyntaxError {
        message,
        position: start,
        found_token: None,
        expected: None,
    }
}

fn invalid_character_hint(input: &str, bad_char: char) -> Option<String> {
    match bad_char {
        '#' => {
            let mut hint = String::from("Use '%' for comments in MATLAB.");
            if input.contains("import ") || input.contains("def ") || input.contains("plt.") {
                hint.push_str(" This code looks like Python.");
            }
            Some(hint)
        }
        '\u{2018}' | '\u{2019}' => {
            Some("Replace the curly quote with a straight quote (').".to_string())
        }
        '\u{201C}' | '\u{201D}' => {
            Some("Replace the curly quote with a straight quote (\").".to_string())
        }
        other if !other.is_ascii() && other.is_alphabetic() => Some(format!(
            "'{other}' is not a valid character in an identifier."
        )),
        _ => None,
    }
}

fn line_and_column(input: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut column = 1usize;
    for (index, character) in input.char_indices() {
        if index >= offset {
            break;
        }
        if character == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}

impl Parser {
    fn parse_program(&mut self) -> Result<Program, SyntaxError> {
        let mut body = Vec::new();
        while self.pos < self.tokens.len() {
            if self.consume(&Token::Semicolon)
                || self.consume(&Token::Comma)
                || self.consume(&Token::Newline)
            {
                continue;
            }
            body.push(self.parse_stmt_with_semicolon()?);
        }
        Ok(Program { body })
    }

    fn finalize_stmt(&self, stmt: Stmt, is_semicolon_terminated: bool) -> Stmt {
        match stmt {
            Stmt::ExprStmt(expr, _, span) => Stmt::ExprStmt(expr, is_semicolon_terminated, span),
            Stmt::Assign(name, expr, _, span) => {
                Stmt::Assign(name, expr, is_semicolon_terminated, span)
            }
            Stmt::MultiAssign(names, expr, _, span) => {
                Stmt::MultiAssign(names, expr, is_semicolon_terminated, span)
            }
            Stmt::AssignLValue(lv, expr, _, span) => {
                Stmt::AssignLValue(lv, expr, is_semicolon_terminated, span)
            }
            other => other,
        }
    }
}
