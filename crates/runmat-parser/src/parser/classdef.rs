use runmat_lexer::Token;

use crate::{ast::ClassPropertyDecl, Attr, ClassMember, ClassNamedDecl, Stmt};

use super::Parser;

impl Parser {
    pub(super) fn parse_classdef(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::ClassDef);
        let attributes = self.parse_optional_attr_list();
        let name = self.parse_qualified_name()?;
        let mut super_class = None;
        if self.consume(&Token::Less) {
            super_class = Some(self.parse_qualified_name()?);
        }
        let mut members: Vec<ClassMember> = Vec::new();
        let previous_class_context = self.current_classdef_name.clone();
        self.current_classdef_name = Some(name.clone());
        loop {
            if self.consume(&Token::Semicolon)
                || self.consume(&Token::Comma)
                || self.consume(&Token::Newline)
            {
                continue;
            }
            match self.peek_token() {
                Some(Token::Properties) => {
                    let block_start = self.tokens[self.pos].position;
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let props = self.parse_properties_names_block()?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after properties".into());
                    }
                    let block_end = self.last_token_end();
                    members.push(ClassMember::Properties {
                        attributes: attrs,
                        names: props,
                        span: self.span_from(block_start, block_end),
                    });
                }
                Some(Token::Methods) => {
                    let block_start = self.tokens[self.pos].position;
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let body = self.parse_block(|t| matches!(t, Token::End))?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after methods".into());
                    }
                    let block_end = self.last_token_end();
                    members.push(ClassMember::Methods {
                        attributes: attrs,
                        body,
                        span: self.span_from(block_start, block_end),
                    });
                }
                Some(Token::Events) => {
                    let block_start = self.tokens[self.pos].position;
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let names = self.parse_name_block()?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after events".into());
                    }
                    let block_end = self.last_token_end();
                    members.push(ClassMember::Events {
                        attributes: attrs,
                        names,
                        span: self.span_from(block_start, block_end),
                    });
                }
                Some(Token::Enumeration) => {
                    let block_start = self.tokens[self.pos].position;
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let names = self.parse_name_block()?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after enumeration".into());
                    }
                    let block_end = self.last_token_end();
                    members.push(ClassMember::Enumeration {
                        attributes: attrs,
                        names,
                        span: self.span_from(block_start, block_end),
                    });
                }
                Some(Token::Arguments) => {
                    let block_start = self.tokens[self.pos].position;
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let names = self.parse_name_block()?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after arguments".into());
                    }
                    let block_end = self.last_token_end();
                    members.push(ClassMember::Arguments {
                        attributes: attrs,
                        names,
                        span: self.span_from(block_start, block_end),
                    });
                }
                Some(Token::End) => {
                    self.pos += 1;
                    break;
                }
                _ => break,
            }
        }
        let end = self.last_token_end();
        self.current_classdef_name = previous_class_context;
        Ok(Stmt::ClassDef {
            attributes,
            name,
            super_class,
            members,
            span: self.span_from(start, end),
        })
    }

    fn parse_name_block(&mut self) -> Result<Vec<ClassNamedDecl>, String> {
        let mut names = Vec::new();
        while let Some(tok) = self.peek_token() {
            if matches!(tok, Token::End) {
                break;
            }
            if self.consume(&Token::Semicolon)
                || self.consume(&Token::Comma)
                || self.consume(&Token::Newline)
            {
                continue;
            }
            if let Some(Token::Ident) = self.peek_token() {
                let start = self.tokens[self.pos].position;
                let name = self.expect_ident()?;
                names.push(ClassNamedDecl {
                    name,
                    span: self.span_from(start, self.last_token_end()),
                });
            } else {
                break;
            }
        }
        Ok(names)
    }

    fn parse_properties_names_block(&mut self) -> Result<Vec<ClassPropertyDecl>, String> {
        // Accept identifiers with optional default assignment: name, name = expr.
        let mut names = Vec::new();
        while let Some(tok) = self.peek_token() {
            if matches!(tok, Token::End) {
                break;
            }
            if self.consume(&Token::Semicolon)
                || self.consume(&Token::Comma)
                || self.consume(&Token::Newline)
            {
                continue;
            }
            if let Some(Token::Ident) = self.peek_token() {
                let start = self.tokens[self.pos].position;
                let name = self.expect_ident()?;
                let mut default = None;
                if self.consume(&Token::Assign) {
                    default = Some(self.parse_expr().map_err(|e| e.message)?);
                }
                names.push(ClassPropertyDecl {
                    name,
                    default,
                    span: self.span_from(start, self.last_token_end()),
                });
            } else {
                break;
            }
        }
        Ok(names)
    }

    pub(super) fn parse_optional_attr_list(&mut self) -> Vec<Attr> {
        // Minimal parsing of attribute lists: (Attr, Attr=Value, ...)
        let mut attrs: Vec<Attr> = Vec::new();
        if !self.consume(&Token::LParen) {
            return attrs;
        }
        loop {
            if self.consume(&Token::RParen) {
                break;
            }
            match self.peek_token() {
                Some(Token::Ident) => {
                    let start = self.tokens[self.pos].position;
                    let name = self.expect_ident().unwrap_or_else(|_| "".to_string());
                    let mut value: Option<String> = None;
                    if self.consume(&Token::Assign) {
                        value = self.parse_attribute_value();
                    }
                    attrs.push(Attr {
                        name,
                        value,
                        span: self.span_from(start, self.last_token_end()),
                    });
                    let _ = self.consume(&Token::Comma);
                }
                Some(Token::Comma) => {
                    self.pos += 1;
                }
                Some(Token::RParen) => {
                    self.pos += 1;
                    break;
                }
                Some(_) => {
                    self.pos += 1;
                }
                None => {
                    break;
                }
            }
        }
        attrs
    }

    fn parse_attribute_value(&mut self) -> Option<String> {
        let start = self.tokens.get(self.pos)?.position;
        let mut paren_depth = 0usize;
        let mut bracket_depth = 0usize;
        let mut brace_depth = 0usize;
        let mut end = start;
        while let Some(token) = self.tokens.get(self.pos) {
            let is_delimiter = paren_depth == 0
                && bracket_depth == 0
                && brace_depth == 0
                && matches!(token.token, Token::Comma | Token::RParen);
            if is_delimiter {
                break;
            }
            match token.token {
                Token::LParen => paren_depth += 1,
                Token::RParen => paren_depth = paren_depth.saturating_sub(1),
                Token::LBracket => bracket_depth += 1,
                Token::RBracket => bracket_depth = bracket_depth.saturating_sub(1),
                Token::LBrace => brace_depth += 1,
                Token::RBrace => brace_depth = brace_depth.saturating_sub(1),
                _ => {}
            }
            end = token.end;
            self.pos += 1;
        }
        (end > start).then(|| self.input[start..end].trim().to_owned())
    }

    fn parse_qualified_name(&mut self) -> Result<String, String> {
        let mut parts = Vec::new();
        parts.push(self.expect_ident()?);
        while self.consume(&Token::Dot) {
            parts.push(self.expect_ident()?);
        }
        Ok(parts.join("."))
    }
}
