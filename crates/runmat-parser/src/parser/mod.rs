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
}

pub fn parse(input: &str) -> Result<Program, SyntaxError> {
    parse_with_options(input, ParserOptions::default())
}

pub fn parse_with_options(input: &str, options: ParserOptions) -> Result<Program, SyntaxError> {
    use runmat_lexer::tokenize_detailed;

    let toks = tokenize_detailed(input);
    let mut tokens = Vec::new();

    for t in toks {
        if matches!(t.token, Token::Error) {
            return Err(SyntaxError {
                message: format!("Invalid token: '{}'", t.lexeme),
                position: t.start,
                found_token: Some(t.lexeme),
                expected: None,
            });
        }
        // Skip layout-only tokens from lexing.
        if matches!(t.token, Token::Ellipsis | Token::Section) {
            continue;
        }
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
    };
    parser.parse_program()
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
