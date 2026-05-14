use runmat_lexer::Token;

use crate::{Stmt, SyntaxError};

use super::{Parser, TokenInfo};

impl Parser {
    pub(super) fn parse_stmt_with_semicolon(&mut self) -> Result<Stmt, SyntaxError> {
        let stmt = self.parse_stmt()?;
        let is_semicolon_terminated = self.consume(&Token::Semicolon);
        Ok(self.finalize_stmt(stmt, is_semicolon_terminated))
    }

    fn parse_stmt(&mut self) -> Result<Stmt, SyntaxError> {
        match self.peek_token() {
            Some(Token::If) => self.parse_if().map_err(|e| e.into()),
            Some(Token::For) => self.parse_for().map_err(|e| e.into()),
            Some(Token::While) => self.parse_while().map_err(|e| e.into()),
            Some(Token::Switch) => self.parse_switch().map_err(|e| e.into()),
            Some(Token::Try) => self.parse_try_catch().map_err(|e| e.into()),
            Some(Token::Import) => self.parse_import().map_err(|e| e.into()),
            Some(Token::ClassDef) => self.parse_classdef().map_err(|e| e.into()),
            Some(Token::Global) => self.parse_global().map_err(|e| e.into()),
            Some(Token::Persistent) => self.parse_persistent().map_err(|e| e.into()),
            Some(Token::Break) => {
                let token = &self.tokens[self.pos];
                self.pos += 1;
                Ok(Stmt::Break(self.span_from(token.position, token.end)))
            }
            Some(Token::Continue) => {
                let token = &self.tokens[self.pos];
                self.pos += 1;
                Ok(Stmt::Continue(self.span_from(token.position, token.end)))
            }
            Some(Token::Return) => {
                let token = &self.tokens[self.pos];
                self.pos += 1;
                Ok(Stmt::Return(self.span_from(token.position, token.end)))
            }
            Some(Token::Function) => self.parse_function().map_err(|e| e.into()),
            Some(Token::LBracket) => {
                if matches!(self.peek_token_at(1), Some(Token::Ident | Token::Tilde)) {
                    match self.try_parse_multi_assign() {
                        Ok(stmt) => Ok(stmt),
                        Err(msg) => Err(self.error(&msg)),
                    }
                } else {
                    let expr = self.parse_expr()?;
                    let span = expr.span();
                    Ok(Stmt::ExprStmt(expr, false, span))
                }
            }
            _ => {
                if self.peek_token() == Some(&Token::Ident)
                    && self.peek_token_at(1) == Some(&Token::Assign)
                {
                    let name_token = self
                        .next()
                        .ok_or_else(|| self.error("expected identifier"))?;
                    if !self.consume(&Token::Assign) {
                        return Err(self.error_with_expected("expected assignment operator", "'='"));
                    }
                    let expr = self.parse_expr()?;
                    let span = self.span_from(name_token.position, expr.span().end);
                    Ok(Stmt::Assign(name_token.lexeme, expr, false, span))
                } else if self.is_simple_assignment_ahead() {
                    let name = self.expect_ident_syntax()?;
                    let start = self.tokens[self.pos.saturating_sub(1)].position;
                    if !self.consume(&Token::Assign) {
                        return Err(self.error_with_expected("expected assignment operator", "'='"));
                    }
                    let expr = self.parse_expr()?;
                    let span = self.span_from(start, expr.span().end);
                    Ok(Stmt::Assign(name, expr, false, span))
                } else if self.peek_token() == Some(&Token::Ident) {
                    if let Some(lv) = self.try_parse_lvalue_assign()? {
                        return Ok(lv);
                    }
                    if self.can_start_command_form() {
                        self.parse_command_stmt()
                    } else {
                        if matches!(self.peek_token_at(1), Some(Token::Ident))
                            && matches!(
                                self.peek_token_at(2),
                                Some(
                                    Token::LParen
                                        | Token::Dot
                                        | Token::LBracket
                                        | Token::LBrace
                                        | Token::Transpose
                                )
                            )
                        {
                            return Err(self.error(
                                "Unexpected adjacency: interpret as function call? Use parentheses (e.g., foo(b(1))).",
                            ));
                        }
                        let expr = self.parse_expr()?;
                        let span = expr.span();
                        Ok(Stmt::ExprStmt(expr, false, span))
                    }
                } else {
                    let expr = self.parse_expr()?;
                    let span = expr.span();
                    Ok(Stmt::ExprStmt(expr, false, span))
                }
            }
        }
    }

    fn parse_if(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::If);
        let cond = self.parse_expr()?;
        let then_body =
            self.parse_block(|t| matches!(t, Token::Else | Token::ElseIf | Token::End))?;
        let mut elseif_blocks = Vec::new();
        while self.consume(&Token::ElseIf) {
            let c = self.parse_expr()?;
            let body =
                self.parse_block(|t| matches!(t, Token::Else | Token::ElseIf | Token::End))?;
            elseif_blocks.push((c, body));
        }
        let else_body = if self.consume(&Token::Else) {
            Some(self.parse_block(|t| matches!(t, Token::End))?)
        } else {
            None
        };
        if !self.consume(&Token::End) {
            return Err("expected 'end'".into());
        }
        let end = self.last_token_end();
        Ok(Stmt::If {
            cond,
            then_body,
            elseif_blocks,
            else_body,
            span: self.span_from(start, end),
        })
    }

    fn parse_while(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::While);
        let cond = self.parse_expr()?;
        let body = self.parse_block(|t| matches!(t, Token::End))?;
        if !self.consume(&Token::End) {
            return Err("expected 'end'".into());
        }
        let end = self.last_token_end();
        Ok(Stmt::While {
            cond,
            body,
            span: self.span_from(start, end),
        })
    }

    fn parse_for(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::For);
        let var = self.expect_ident()?;
        if !self.consume(&Token::Assign) {
            return Err("expected '='".into());
        }
        let expr = self.parse_expr()?;
        let body = self.parse_block(|t| matches!(t, Token::End))?;
        if !self.consume(&Token::End) {
            return Err("expected 'end'".into());
        }
        let end = self.last_token_end();
        Ok(Stmt::For {
            var,
            expr,
            body,
            span: self.span_from(start, end),
        })
    }

    fn parse_function(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::Function);
        let mut outputs = Vec::new();
        if self.consume(&Token::LBracket) {
            outputs.push(self.expect_ident_or_tilde()?);
            while self.consume(&Token::Comma) {
                outputs.push(self.expect_ident_or_tilde()?);
            }
            if !self.consume(&Token::RBracket) {
                return Err("expected ']'".into());
            }
            if !self.consume(&Token::Assign) {
                return Err("expected '='".into());
            }
        } else if self.peek_token() == Some(&Token::Ident)
            && self.peek_token_at(1) == Some(&Token::Assign)
        {
            outputs.push(self.next().unwrap().lexeme);
            self.consume(&Token::Assign);
        }
        let name = self.expect_ident()?;
        if !self.consume(&Token::LParen) {
            return Err("expected '('".into());
        }
        let mut params = Vec::new();
        if !self.consume(&Token::RParen) {
            params.push(self.expect_ident()?);
            while self.consume(&Token::Comma) {
                params.push(self.expect_ident()?);
            }
            if !self.consume(&Token::RParen) {
                return Err("expected ')'".into());
            }
        }

        if let Some(idx) = params.iter().position(|p| p == "varargin") {
            if idx != params.len() - 1 {
                return Err("'varargin' must be the last input parameter".into());
            }
            if params.iter().filter(|p| p.as_str() == "varargin").count() > 1 {
                return Err("'varargin' cannot appear more than once".into());
            }
        }
        if let Some(idx) = outputs.iter().position(|o| o == "varargout") {
            if idx != outputs.len() - 1 {
                return Err("'varargout' must be the last output parameter".into());
            }
            if outputs.iter().filter(|o| o.as_str() == "varargout").count() > 1 {
                return Err("'varargout' cannot appear more than once".into());
            }
        }

        // Optional function-level arguments block.
        if self.peek_token() == Some(&Token::Arguments) {
            self.pos += 1;
            loop {
                if self.consume(&Token::End) {
                    break;
                }
                if self.consume(&Token::Semicolon) || self.consume(&Token::Comma) {
                    continue;
                }
                if matches!(self.peek_token(), Some(Token::Ident)) {
                    let _ = self.expect_ident()?;
                    continue;
                }
                if self.peek_token().is_none() {
                    break;
                }
                break;
            }
        }

        let body = self.parse_block(|t| matches!(t, Token::End))?;
        if !self.consume(&Token::End) {
            return Err("expected 'end'".into());
        }
        let end = self.last_token_end();
        Ok(Stmt::Function {
            name,
            params,
            outputs,
            body,
            span: self.span_from(start, end),
        })
    }

    pub(super) fn parse_block<F>(&mut self, term: F) -> Result<Vec<Stmt>, String>
    where
        F: Fn(&Token) -> bool,
    {
        let mut body = Vec::new();
        while let Some(tok) = self.peek_token() {
            if term(tok) {
                break;
            }
            if self.consume(&Token::Semicolon)
                || self.consume(&Token::Comma)
                || self.consume(&Token::Newline)
            {
                continue;
            }
            let stmt = if self.peek_token() == Some(&Token::LBracket) {
                self.try_parse_multi_assign()?
            } else {
                self.parse_stmt().map_err(|e| e.message)?
            };
            let is_semicolon_terminated = self.consume(&Token::Semicolon);
            body.push(self.finalize_stmt(stmt, is_semicolon_terminated));
        }
        Ok(body)
    }

    fn parse_switch(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::Switch);
        let control = self.parse_expr()?;
        let mut cases = Vec::new();
        let mut otherwise: Option<Vec<Stmt>> = None;
        loop {
            if self.consume(&Token::Newline) || self.consume(&Token::Semicolon) {
                continue;
            }
            if self.consume(&Token::Case) {
                let val = self.parse_expr()?;
                let body =
                    self.parse_block(|t| matches!(t, Token::Case | Token::Otherwise | Token::End))?;
                cases.push((val, body));
            } else if self.consume(&Token::Otherwise) {
                let body = self.parse_block(|t| matches!(t, Token::End))?;
                otherwise = Some(body);
            } else if self.consume(&Token::Comma) {
                continue;
            } else {
                break;
            }
        }
        if !self.consume(&Token::End) {
            return Err("expected 'end' for switch".into());
        }
        let end = self.last_token_end();
        Ok(Stmt::Switch {
            expr: control,
            cases,
            otherwise,
            span: self.span_from(start, end),
        })
    }

    fn parse_try_catch(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::Try);
        let try_body = self.parse_block(|t| matches!(t, Token::Catch | Token::End))?;
        if !self.consume(&Token::Catch) {
            return Err("expected 'catch' after try".into());
        }
        let catch_var = if self.peek_token() == Some(&Token::Ident) {
            Some(self.expect_ident()?)
        } else {
            None
        };
        let catch_body = self.parse_block(|t| matches!(t, Token::End))?;
        if !self.consume(&Token::End) {
            return Err("expected 'end' after catch".into());
        }
        let end = self.last_token_end();
        Ok(Stmt::TryCatch {
            try_body,
            catch_var,
            catch_body,
            span: self.span_from(start, end),
        })
    }

    fn parse_import(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::Import);
        let mut path = Vec::new();
        path.push(self.expect_ident()?);
        let mut wildcard = false;
        loop {
            if self.consume(&Token::DotStar) {
                wildcard = true;
                break;
            }
            if self.consume(&Token::Dot) {
                if self.consume(&Token::Star) {
                    wildcard = true;
                    break;
                } else {
                    path.push(self.expect_ident()?);
                    continue;
                }
            }
            break;
        }
        let end = self.last_token_end();
        Ok(Stmt::Import {
            path,
            wildcard,
            span: self.span_from(start, end),
        })
    }

    fn parse_global(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::Global);
        let mut names = Vec::new();
        names.push(self.expect_ident()?);
        loop {
            if self.consume(&Token::Comma) {
                names.push(self.expect_ident()?);
                continue;
            }
            if self.peek_token() == Some(&Token::Ident) {
                names.push(self.expect_ident()?);
                continue;
            }
            break;
        }
        let end = self.last_token_end();
        Ok(Stmt::Global(names, self.span_from(start, end)))
    }

    fn parse_persistent(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::Persistent);
        let mut names = Vec::new();
        names.push(self.expect_ident()?);
        loop {
            if self.consume(&Token::Comma) {
                names.push(self.expect_ident()?);
                continue;
            }
            if self.peek_token() == Some(&Token::Ident) {
                names.push(self.expect_ident()?);
                continue;
            }
            break;
        }
        let end = self.last_token_end();
        Ok(Stmt::Persistent(names, self.span_from(start, end)))
    }

    pub(super) fn expect_ident(&mut self) -> Result<String, String> {
        match self.next() {
            Some(TokenInfo {
                token: Token::Ident,
                lexeme,
                ..
            }) => Ok(lexeme),
            _ => Err("expected identifier".into()),
        }
    }

    pub(super) fn expect_ident_syntax(&mut self) -> Result<String, SyntaxError> {
        let token = self.peek().cloned();
        self.expect_ident().map_err(|message| SyntaxError {
            message,
            position: token
                .as_ref()
                .map(|token| token.position)
                .unwrap_or_else(|| self.input.len()),
            found_token: token.map(|token| token.lexeme),
            expected: Some("identifier".into()),
        })
    }

    pub(super) fn expect_ident_or_tilde(&mut self) -> Result<String, String> {
        match self.next() {
            Some(TokenInfo {
                token: Token::Ident,
                lexeme,
                ..
            }) => Ok(lexeme),
            Some(TokenInfo {
                token: Token::Tilde,
                ..
            }) => Ok("~".to_string()),
            _ => Err("expected identifier or '~'".into()),
        }
    }
}
