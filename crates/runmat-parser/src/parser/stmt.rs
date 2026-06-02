use runmat_lexer::Token;

use crate::ast::{
    FunctionArgDim, FunctionArgSizeSpec, FunctionArgValidationDecl, FunctionArgValidatorDecl,
};
use crate::{Expr, Stmt, SyntaxError};

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
            Some(Token::Function | Token::Isolated | Token::Async) => {
                self.parse_function().map_err(|e| e.into())
            }
            Some(Token::LBracket) => {
                if self.looks_like_multi_assign_lhs() {
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
                    if self.looks_like_super_constructor_stmt() {
                        return self.parse_super_constructor_stmt();
                    }
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

    fn looks_like_super_constructor_stmt(&self) -> bool {
        if !matches!(
            (
                self.peek_token(),
                self.peek_token_at(1),
                self.peek_token_at(2)
            ),
            (Some(Token::Ident), Some(Token::At), Some(Token::Ident))
        ) {
            return false;
        }
        let mut idx = 3usize;
        while matches!(
            (self.peek_token_at(idx), self.peek_token_at(idx + 1)),
            (Some(Token::Dot), Some(Token::Ident))
        ) {
            idx += 2;
        }
        matches!(self.peek_token_at(idx), Some(Token::LParen))
    }

    fn parse_super_constructor_stmt(&mut self) -> Result<Stmt, SyntaxError> {
        let target = self
            .next()
            .ok_or_else(|| self.error("expected constructor target"))?;
        let target_name = target.lexeme.clone();
        let start = target.position;
        if !self.consume(&Token::At) {
            return Err(
                self.error_with_expected("expected '@' for superclass constructor syntax", "'@'")
            );
        }
        let mut super_parts = Vec::new();
        super_parts.push(self.expect_ident_syntax()?);
        while self.consume(&Token::Dot) {
            super_parts.push(self.expect_ident_syntax()?);
        }
        let super_name = super_parts.join(".");
        if !self.consume(&Token::LParen) {
            return Err(
                self.error_with_expected("expected '(' after superclass constructor name", "'('")
            );
        }
        let mut args = Vec::new();
        if !self.consume(&Token::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if self.consume(&Token::Comma) {
                    continue;
                }
                if self.consume(&Token::RParen) {
                    break;
                }
                return Err(
                    self.error_with_expected("expected ',' or ')' in argument list", "',' or ')'")
                );
            }
        }
        let end = self.last_token_end();
        let span = self.span_from(start, end);
        let class_name = self.current_classdef_name.clone().ok_or_else(|| {
            self.error("superclass constructor syntax is only valid inside classdef methods")
        })?;
        let call = Expr::SuperConstructorCall {
            current_class: class_name,
            super_class: super_name,
            args,
            span,
        };
        Ok(Stmt::Assign(target_name, call, false, span))
    }

    fn looks_like_multi_assign_lhs(&self) -> bool {
        if self.peek_token() != Some(&Token::LBracket) {
            return false;
        }
        let mut i = self.pos;
        let mut paren_depth = 0usize;
        let mut brace_depth = 0usize;
        let mut bracket_depth = 0usize;
        while let Some(info) = self.tokens.get(i) {
            match info.token {
                Token::LParen => paren_depth += 1,
                Token::RParen => {
                    if paren_depth == 0 {
                        return false;
                    }
                    paren_depth -= 1;
                }
                Token::LBrace => brace_depth += 1,
                Token::RBrace => {
                    if brace_depth == 0 {
                        return false;
                    }
                    brace_depth -= 1;
                }
                Token::LBracket => bracket_depth += 1,
                Token::RBracket => {
                    if bracket_depth == 0 {
                        return false;
                    }
                    bracket_depth -= 1;
                    if bracket_depth == 0 {
                        return matches!(
                            self.tokens.get(i + 1).map(|next| &next.token),
                            Some(Token::Assign)
                        );
                    }
                }
                _ => {}
            }
            i += 1;
        }
        false
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
        let mut isolated = false;
        let mut is_async = false;
        loop {
            if self.consume(&Token::Isolated) {
                if isolated {
                    return Err("duplicate 'isolated' function modifier".into());
                }
                isolated = true;
            } else if self.consume(&Token::Async) {
                if is_async {
                    return Err("duplicate 'async' function modifier".into());
                }
                is_async = true;
            } else {
                break;
            }
        }
        if (isolated || is_async) && self.peek_token() != Some(&Token::Function) {
            return Err("expected 'function' after function modifier".into());
        }
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
        let mut name = self.expect_ident()?;
        if self.current_classdef_name.is_some() && self.consume(&Token::Dot) {
            let member = self.expect_ident()?;
            name = format!("{name}.{member}");
        }
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
        // Allow leading separators/newlines between the signature and `arguments`.
        while self.consume(&Token::Semicolon)
            || self.consume(&Token::Comma)
            || self.consume(&Token::Newline)
        {}
        let mut argument_validations = Vec::new();
        if self.peek_token() == Some(&Token::Arguments) {
            self.pos += 1;
            // Parse simple MATLAB function arguments-block declarations.
            while let Some(token) = self.peek_token() {
                if matches!(token, Token::End) {
                    self.pos += 1;
                    break;
                }
                if self.consume(&Token::Semicolon)
                    || self.consume(&Token::Comma)
                    || self.consume(&Token::Newline)
                {
                    continue;
                }
                argument_validations.push(self.parse_function_argument_validation_decl()?);
            }
            while self.consume(&Token::Semicolon)
                || self.consume(&Token::Comma)
                || self.consume(&Token::Newline)
            {}
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
            argument_validations,
            body,
            isolated,
            is_async,
            span: self.span_from(start, end),
        })
    }

    fn parse_function_argument_validation_decl(
        &mut self,
    ) -> Result<FunctionArgValidationDecl, String> {
        let name = self.expect_ident()?;
        let mut size = None;
        if self.consume(&Token::LParen) {
            let rows = self.parse_function_argument_dim()?;
            if !self.consume(&Token::Comma) {
                return Err("expected ',' in arguments size spec".into());
            }
            let cols = self.parse_function_argument_dim()?;
            if !self.consume(&Token::RParen) {
                return Err("expected ')' after arguments size spec".into());
            }
            size = Some(FunctionArgSizeSpec { rows, cols });
        }

        let mut class_name = None;
        let mut validators = Vec::new();
        if matches!(self.peek_token(), Some(Token::Ident)) {
            let candidate = self.parse_qualified_name_in_stmt()?;
            if self.peek_token() == Some(&Token::LParen) {
                validators.push(self.parse_function_argument_validator_decl_with_name(candidate)?);
            } else {
                class_name = Some(candidate);
            }
        }
        loop {
            if self.consume(&Token::LBrace) {
                loop {
                    let validator = self.parse_function_argument_validator_decl()?;
                    validators.push(validator);
                    if self.consume(&Token::Comma) {
                        continue;
                    }
                    if !self.consume(&Token::RBrace) {
                        return Err("expected '}' after arguments validators".into());
                    }
                    break;
                }
                continue;
            }
            if matches!(self.peek_token(), Some(Token::Ident)) {
                validators.push(self.parse_function_argument_validator_decl()?);
                continue;
            }
            break;
        }

        let default_value = if self.consume(&Token::Assign) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        // Record unsupported trailing tokens on the same logical line.
        let mut has_unsupported_trailing = false;
        while let Some(token) = self.peek_token() {
            if matches!(
                token,
                Token::Semicolon | Token::Comma | Token::Newline | Token::End
            ) {
                break;
            }
            has_unsupported_trailing = true;
            self.pos += 1;
        }

        Ok(FunctionArgValidationDecl {
            name,
            size,
            class_name,
            validators,
            default_value,
            has_unsupported_trailing,
        })
    }

    fn parse_function_argument_dim(&mut self) -> Result<FunctionArgDim, String> {
        if self.consume(&Token::Colon) {
            return Ok(FunctionArgDim::Any);
        }
        match self.next() {
            Some(token) if matches!(token.token, Token::Integer | Token::Float) => {
                let parsed = token
                    .lexeme
                    .parse::<usize>()
                    .map_err(|_| "arguments size spec must use non-negative integer dimensions")?;
                Ok(FunctionArgDim::Exact(parsed))
            }
            _ => Err("expected numeric dimension or ':' in arguments size spec".into()),
        }
    }

    fn parse_qualified_name_in_stmt(&mut self) -> Result<String, String> {
        let mut parts = Vec::new();
        parts.push(self.expect_ident()?);
        while self.consume(&Token::Dot) {
            parts.push(self.expect_ident()?);
        }
        Ok(parts.join("."))
    }

    fn parse_function_argument_validator_decl(
        &mut self,
    ) -> Result<FunctionArgValidatorDecl, String> {
        let name = self.parse_qualified_name_in_stmt()?;
        self.parse_function_argument_validator_decl_with_name(name)
    }

    fn parse_function_argument_validator_decl_with_name(
        &mut self,
        name: String,
    ) -> Result<FunctionArgValidatorDecl, String> {
        let mut args = Vec::new();
        if self.consume(&Token::LParen) && !self.consume(&Token::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if self.consume(&Token::Comma) {
                    continue;
                }
                if !self.consume(&Token::RParen) {
                    return Err("expected ')' after arguments validator arguments".into());
                }
                break;
            }
        }
        Ok(FunctionArgValidatorDecl { name, args })
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
