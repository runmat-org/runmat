use runmat_lexer::Token;

use crate::{BinOp, Expr, Span, SyntaxError, UnOp};

use super::{Parser, TokenInfo};

impl Parser {
    pub(super) fn skip_newlines(&mut self) {
        while self.consume(&Token::Newline) {}
    }

    pub(super) fn tokens_adjacent(&self, left: usize, right: usize) -> bool {
        match (self.tokens.get(left), self.tokens.get(right)) {
            (Some(a), Some(b)) => a.end == b.position,
            _ => false,
        }
    }

    pub(super) fn span_from(&self, start: usize, end: usize) -> Span {
        Span { start, end }
    }

    pub(super) fn span_between(&self, start: Span, end: Span) -> Span {
        Span {
            start: start.start,
            end: end.end,
        }
    }

    pub(super) fn last_token_end(&self) -> usize {
        self.tokens
            .get(self.pos.saturating_sub(1))
            .map(|t| t.end)
            .unwrap_or(self.input.len())
    }

    pub(super) fn make_binary(&self, left: Expr, op: BinOp, right: Expr) -> Expr {
        let span = self.span_between(left.span(), right.span());
        Expr::Binary(Box::new(left), op, Box::new(right), span)
    }

    pub(super) fn make_unary(&self, op: UnOp, operand: Expr, op_start: usize) -> Expr {
        let span = self.span_from(op_start, operand.span().end);
        Expr::Unary(op, Box::new(operand), span)
    }

    pub(super) fn error(&self, message: &str) -> SyntaxError {
        SyntaxError {
            message: message.to_string(),
            position: self.current_position(),
            found_token: self.peek().map(|t| t.lexeme.clone()),
            expected: None,
        }
    }

    pub(super) fn error_with_expected(&self, message: &str, expected: &str) -> SyntaxError {
        SyntaxError {
            message: message.to_string(),
            position: self.current_position(),
            found_token: self.peek().map(|t| t.lexeme.clone()),
            expected: Some(expected.to_string()),
        }
    }

    pub(super) fn peek(&self) -> Option<&TokenInfo> {
        self.tokens.get(self.pos)
    }

    pub(super) fn current_position(&self) -> usize {
        self.peek()
            .map(|t| t.position)
            .unwrap_or_else(|| self.input.len())
    }

    pub(super) fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|t| &t.token)
    }

    pub(super) fn peek_token_at(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset).map(|t| &t.token)
    }

    pub(super) fn next(&mut self) -> Option<TokenInfo> {
        if self.pos < self.tokens.len() {
            let info = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(info)
        } else {
            None
        }
    }

    pub(super) fn consume(&mut self, t: &Token) -> bool {
        if self.peek_token() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}
