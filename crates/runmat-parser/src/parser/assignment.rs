use runmat_lexer::Token;

use crate::{Expr, LValue, MultiAssignTarget, Stmt, SyntaxError};

use super::Parser;

impl Parser {
    fn parse_lvalue_base_with_postfix(
        &mut self,
        base_token: super::TokenInfo,
    ) -> Result<Expr, SyntaxError> {
        let base_span = self.span_from(base_token.position, base_token.end);
        let mut base = Expr::Ident(base_token.lexeme, base_span);
        loop {
            if self.consume(&Token::LParen) {
                let args = self.parse_call_arguments_until(
                    Token::RParen,
                    "expected ')' after indices",
                    ")",
                )?;
                let end = self.last_token_end();
                let span = self.span_from(base.span().start, end);
                base = Expr::Index(Box::new(base), args, span);
            } else if self.consume(&Token::LBracket) {
                let mut idxs = Vec::new();
                idxs.push(self.parse_expr()?);
                while self.consume(&Token::Comma) {
                    idxs.push(self.parse_expr()?);
                }
                if !self.consume(&Token::RBracket) {
                    return Err(self.error_with_expected("expected ']'", "]"));
                }
                let end = self.last_token_end();
                let span = self.span_from(base.span().start, end);
                base = Expr::Index(Box::new(base), idxs, span);
            } else if self.consume(&Token::LBrace) {
                let mut idxs = Vec::new();
                idxs.push(self.parse_expr()?);
                while self.consume(&Token::Comma) {
                    idxs.push(self.parse_expr()?);
                }
                if !self.consume(&Token::RBrace) {
                    return Err(self.error_with_expected("expected '}'", "}"));
                }
                let end = self.last_token_end();
                let span = self.span_from(base.span().start, end);
                base = Expr::IndexCell(Box::new(base), idxs, span);
            } else if self.peek_token() == Some(&Token::Dot) {
                if self.peek_token_at(1) == Some(&Token::Transpose) {
                    break;
                }
                if self.peek_token_at(1) == Some(&Token::Plus)
                    || self.peek_token_at(1) == Some(&Token::Minus)
                {
                    break;
                }
                self.pos += 1; // consume '.'
                if self.consume(&Token::LParen) {
                    let name_expr = self.parse_expr()?;
                    if !self.consume(&Token::RParen) {
                        return Err(self.error_with_expected(
                            "expected ')' after dynamic field expression",
                            ")",
                        ));
                    }
                    let end = self.last_token_end();
                    let span = self.span_from(base.span().start, end);
                    base = Expr::MemberDynamic(Box::new(base), Box::new(name_expr), span);
                } else {
                    let name = self.expect_ident()?;
                    let end = self.last_token_end();
                    let span = self.span_from(base.span().start, end);
                    base = Expr::Member(Box::new(base), name, span);
                }
            } else {
                break;
            }
        }
        Ok(base)
    }

    fn expr_to_lvalue(&self, expr: Expr) -> Result<LValue, SyntaxError> {
        match expr {
            Expr::Member(b, name, _) => Ok(LValue::Member(b, name)),
            Expr::MemberDynamic(b, n, _) => Ok(LValue::MemberDynamic(b, n)),
            Expr::Index(b, idxs, _) => Ok(LValue::Index(b, idxs)),
            Expr::IndexCell(b, idxs, _) => Ok(LValue::IndexCell(b, idxs)),
            Expr::Ident(v, _) => Ok(LValue::Var(v)),
            _ => Err(self.error("invalid assignment target in multi-assignment")),
        }
    }

    pub(super) fn is_simple_assignment_ahead(&self) -> bool {
        // Heuristic: at statement start, if we see Ident ... '=' before a terminator, treat as assignment.
        self.peek_token() == Some(&Token::Ident) && self.peek_token_at(1) == Some(&Token::Assign)
    }

    pub(super) fn try_parse_lvalue_assign(&mut self) -> Result<Option<Stmt>, SyntaxError> {
        let save = self.pos;
        let lvalue = if self.peek_token() == Some(&Token::Ident) {
            let base_token = self.next().unwrap();
            self.parse_lvalue_base_with_postfix(base_token)?
        } else {
            self.pos = save;
            return Ok(None);
        };

        if !self.consume(&Token::Assign) {
            self.pos = save;
            return Ok(None);
        }

        let rhs = self.parse_expr()?;
        let stmt_span = self.span_between(lvalue.span(), rhs.span());
        let stmt = match lvalue {
            Expr::Member(b, name, _) => {
                Stmt::AssignLValue(LValue::Member(b, name), rhs, false, stmt_span)
            }
            Expr::MemberDynamic(b, n, _) => {
                Stmt::AssignLValue(LValue::MemberDynamic(b, n), rhs, false, stmt_span)
            }
            Expr::Index(b, idxs, _) => {
                Stmt::AssignLValue(LValue::Index(b, idxs), rhs, false, stmt_span)
            }
            Expr::IndexCell(b, idxs, _) => {
                Stmt::AssignLValue(LValue::IndexCell(b, idxs), rhs, false, stmt_span)
            }
            Expr::Ident(v, _) => Stmt::Assign(v, rhs, false, stmt_span),
            _ => {
                self.pos = save;
                return Ok(None);
            }
        };
        Ok(Some(stmt))
    }

    pub(super) fn try_parse_multi_assign(&mut self) -> Result<Stmt, String> {
        if !self.consume(&Token::LBracket) {
            return Err("not a multi-assign".into());
        }
        let start = self.tokens[self.pos.saturating_sub(1)].position;
        let mut targets = Vec::new();
        targets.push(self.parse_multi_assign_target()?);
        while self.consume(&Token::Comma) {
            targets.push(self.parse_multi_assign_target()?);
        }
        if !self.consume(&Token::RBracket) {
            return Err("expected ']'".into());
        }
        if !self.consume(&Token::Assign) {
            return Err("expected '='".into());
        }
        let rhs = self.parse_expr().map_err(|e| e.message)?;
        let span = self.span_from(start, rhs.span().end);
        Ok(Stmt::MultiAssign(targets, rhs, false, span))
    }

    fn parse_multi_assign_target(&mut self) -> Result<MultiAssignTarget, String> {
        if self.consume(&Token::Tilde) {
            return Ok(MultiAssignTarget::Discard);
        }
        if self.peek_token() != Some(&Token::Ident) {
            return Err("expected multi-assignment target".into());
        }
        let base_token = self.next().unwrap();
        let expr = self
            .parse_lvalue_base_with_postfix(base_token)
            .map_err(|e| e.message)?;
        let lvalue = self.expr_to_lvalue(expr).map_err(|e| e.message)?;
        Ok(MultiAssignTarget::LValue(lvalue))
    }
}
