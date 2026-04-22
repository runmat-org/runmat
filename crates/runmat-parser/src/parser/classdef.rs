use runmat_lexer::Token;

use crate::{Attr, ClassMember, Stmt};

use super::Parser;

impl Parser {
    pub(super) fn parse_classdef(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::ClassDef);
        let name = self.parse_qualified_name()?;
        let mut super_class = None;
        if self.consume(&Token::Less) {
            super_class = Some(self.parse_qualified_name()?);
        }
        let mut members: Vec<ClassMember> = Vec::new();
        loop {
            if self.consume(&Token::Semicolon)
                || self.consume(&Token::Comma)
                || self.consume(&Token::Newline)
            {
                continue;
            }
            match self.peek_token() {
                Some(Token::Properties) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let props = self.parse_properties_names_block()?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after properties".into());
                    }
                    members.push(ClassMember::Properties {
                        attributes: attrs,
                        names: props,
                    });
                }
                Some(Token::Methods) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let body = self.parse_block(|t| matches!(t, Token::End))?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after methods".into());
                    }
                    members.push(ClassMember::Methods {
                        attributes: attrs,
                        body,
                    });
                }
                Some(Token::Events) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let names = self.parse_name_block()?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after events".into());
                    }
                    members.push(ClassMember::Events {
                        attributes: attrs,
                        names,
                    });
                }
                Some(Token::Enumeration) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let names = self.parse_name_block()?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after enumeration".into());
                    }
                    members.push(ClassMember::Enumeration {
                        attributes: attrs,
                        names,
                    });
                }
                Some(Token::Arguments) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let names = self.parse_name_block()?;
                    if !self.consume(&Token::End) {
                        return Err("expected 'end' after arguments".into());
                    }
                    members.push(ClassMember::Arguments {
                        attributes: attrs,
                        names,
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
        Ok(Stmt::ClassDef {
            name,
            super_class,
            members,
            span: self.span_from(start, end),
        })
    }

    fn parse_name_block(&mut self) -> Result<Vec<String>, String> {
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
                names.push(self.expect_ident()?);
            } else {
                break;
            }
        }
        Ok(names)
    }

    fn parse_properties_names_block(&mut self) -> Result<Vec<String>, String> {
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
                names.push(self.expect_ident()?);
                // Parse and discard default initializers to preserve current permissive syntax behavior.
                if self.consume(&Token::Assign) {
                    let _ = self.parse_expr().map_err(|e| e.message)?;
                }
            } else {
                break;
            }
        }
        Ok(names)
    }

    fn parse_optional_attr_list(&mut self) -> Vec<Attr> {
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
                    let name = self.expect_ident().unwrap_or_else(|_| "".to_string());
                    let mut value: Option<String> = None;
                    if self.consume(&Token::Assign) {
                        if let Some(tok) = self.next() {
                            value = Some(tok.lexeme);
                        }
                    }
                    attrs.push(Attr { name, value });
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

    fn parse_qualified_name(&mut self) -> Result<String, String> {
        let mut parts = Vec::new();
        parts.push(self.expect_ident()?);
        while self.consume(&Token::Dot) {
            parts.push(self.expect_ident()?);
        }
        Ok(parts.join("."))
    }
}
