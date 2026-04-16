use runmat_lexer::Token;

use crate::{BinOp, Expr, Span, SyntaxError, UnOp};

use super::{Parser, TokenInfo};

impl Parser {
    pub(super) fn parse_expr(&mut self) -> Result<Expr, SyntaxError> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<Expr, SyntaxError> {
        let mut node = self.parse_logical_and()?;
        while self.consume(&Token::OrOr) {
            let rhs = self.parse_logical_and()?;
            node = self.make_binary(node, BinOp::OrOr, rhs);
        }
        Ok(node)
    }

    fn parse_logical_and(&mut self) -> Result<Expr, SyntaxError> {
        let mut node = self.parse_bitwise_or()?;
        while self.consume(&Token::AndAnd) {
            let rhs = self.parse_bitwise_or()?;
            node = self.make_binary(node, BinOp::AndAnd, rhs);
        }
        Ok(node)
    }

    fn parse_bitwise_or(&mut self) -> Result<Expr, SyntaxError> {
        let mut node = self.parse_bitwise_and()?;
        while self.consume(&Token::Or) {
            let rhs = self.parse_bitwise_and()?;
            node = self.make_binary(node, BinOp::BitOr, rhs);
        }
        Ok(node)
    }

    fn parse_bitwise_and(&mut self) -> Result<Expr, SyntaxError> {
        let mut node = self.parse_range()?;
        while self.consume(&Token::And) {
            let rhs = self.parse_range()?;
            node = self.make_binary(node, BinOp::BitAnd, rhs);
        }
        Ok(node)
    }

    fn parse_range(&mut self) -> Result<Expr, SyntaxError> {
        let mut node = self.parse_comparison()?;
        if self.consume(&Token::Colon) {
            let mid = self.parse_comparison()?;
            if self.consume(&Token::Colon) {
                let end = self.parse_comparison()?;
                let span = self.span_between(node.span(), end.span());
                node = Expr::Range(Box::new(node), Some(Box::new(mid)), Box::new(end), span);
            } else {
                let span = self.span_between(node.span(), mid.span());
                node = Expr::Range(Box::new(node), None, Box::new(mid), span);
            }
        }
        Ok(node)
    }

    fn parse_comparison(&mut self) -> Result<Expr, SyntaxError> {
        let mut node = self.parse_add_sub()?;
        loop {
            let op = match self.peek_token() {
                Some(Token::Equal) => BinOp::Equal,
                Some(Token::NotEqual) => BinOp::NotEqual,
                Some(Token::Less) => BinOp::Less,
                Some(Token::LessEqual) => BinOp::LessEqual,
                Some(Token::Greater) => BinOp::Greater,
                Some(Token::GreaterEqual) => BinOp::GreaterEqual,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_add_sub()?;
            node = self.make_binary(node, op, rhs);
        }
        Ok(node)
    }

    fn parse_add_sub(&mut self) -> Result<Expr, String> {
        let mut node = self.parse_mul_div()?;
        loop {
            if self.in_matrix_expr
                && matches!(self.peek_token(), Some(Token::Plus | Token::Minus))
                && self.pos > 0
                && !self.tokens_adjacent(self.pos - 1, self.pos)
                && self.tokens_adjacent(self.pos, self.pos + 1)
            {
                let rhs_index = self.pos + 1;
                let rhs_is_imag_literal = self
                    .tokens
                    .get(rhs_index)
                    .map(|info| matches!(info.token, Token::Integer | Token::Float))
                    .unwrap_or(false)
                    && self
                        .tokens
                        .get(rhs_index + 1)
                        .map(|info| {
                            matches!(info.token, Token::Ident)
                                && (info.lexeme.eq_ignore_ascii_case("i")
                                    || info.lexeme.eq_ignore_ascii_case("j"))
                                && self.tokens_adjacent(rhs_index, rhs_index + 1)
                        })
                        .unwrap_or(false);
                if !rhs_is_imag_literal {
                    break;
                }
            }
            let op = if self.consume(&Token::Plus) {
                Some(BinOp::Add)
            } else if self.consume(&Token::Minus) {
                Some(BinOp::Sub)
            } else if self.peek_token() == Some(&Token::Dot)
                && (self.peek_token_at(1) == Some(&Token::Plus)
                    || self.peek_token_at(1) == Some(&Token::Minus))
            {
                self.pos += 2;
                if self.tokens[self.pos - 1].token == Token::Plus {
                    Some(BinOp::Add)
                } else {
                    Some(BinOp::Sub)
                }
            } else {
                None
            };
            let Some(op) = op else { break };
            let rhs = self.parse_mul_div()?;
            node = self.make_binary(node, op, rhs);
        }
        Ok(node)
    }

    fn parse_mul_div(&mut self) -> Result<Expr, String> {
        let mut node = self.parse_unary()?;
        loop {
            if self.peek_token() == Some(&Token::Ident) && self.pos > 0 {
                let prev = &self.tokens[self.pos - 1];
                let curr = &self.tokens[self.pos];
                let is_adjacent = self.tokens_adjacent(self.pos - 1, self.pos);
                let is_imag =
                    curr.lexeme.eq_ignore_ascii_case("i") || curr.lexeme.eq_ignore_ascii_case("j");
                if is_adjacent && is_imag && matches!(prev.token, Token::Integer | Token::Float) {
                    let token = self.next().unwrap();
                    let rhs = Expr::Ident(token.lexeme, self.span_from(token.position, token.end));
                    node = self.make_binary(node, BinOp::Mul, rhs);
                    continue;
                }
            }
            let op = match self.peek_token() {
                Some(Token::Star) => BinOp::Mul,
                Some(Token::DotStar) => BinOp::ElemMul,
                Some(Token::Slash) => BinOp::RightDiv,
                Some(Token::DotSlash) => BinOp::ElemDiv,
                Some(Token::Backslash) => BinOp::LeftDiv,
                Some(Token::DotBackslash) => BinOp::ElemLeftDiv,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_unary()?;
            node = self.make_binary(node, op, rhs);
        }
        Ok(node)
    }

    fn parse_pow(&mut self) -> Result<Expr, String> {
        let node = self.parse_postfix()?;
        if let Some(token) = self.peek_token() {
            let op = match token {
                Token::Caret => BinOp::Pow,
                Token::DotCaret => BinOp::ElemPow,
                _ => return Ok(node),
            };
            self.pos += 1;
            let rhs = self.parse_pow()?;
            Ok(self.make_binary(node, op, rhs))
        } else {
            Ok(node)
        }
    }

    fn parse_postfix_with_base(&mut self, mut expr: Expr) -> Result<Expr, String> {
        loop {
            if self.consume(&Token::LParen) {
                let start = expr.span().start;
                let mut args = Vec::new();
                if !self.consume(&Token::RParen) {
                    args.push(self.parse_expr()?);
                    while self.consume(&Token::Comma) {
                        args.push(self.parse_expr()?);
                    }
                    if !self.consume(&Token::RParen) {
                        return Err("expected ')' after arguments".into());
                    }
                }
                let end = self.last_token_end();
                let span = self.span_from(start, end);
                // Binder-based disambiguation:
                // If the callee is an identifier, defer call vs. index to HIR binding.
                // Parse as a function call now; HIR will rewrite to Index if a variable shadows the function.
                if let Expr::Ident(ref name, _) = expr {
                    expr = Expr::FuncCall(name.clone(), args, span);
                } else {
                    expr = Expr::Index(Box::new(expr), args, span);
                }
            } else if self.consume(&Token::LBracket) {
                let start = expr.span().start;
                let mut indices = Vec::new();
                indices.push(self.parse_expr()?);
                while self.consume(&Token::Comma) {
                    indices.push(self.parse_expr()?);
                }
                if !self.consume(&Token::RBracket) {
                    return Err("expected ']'".into());
                }
                let end = self.last_token_end();
                let span = self.span_from(start, end);
                expr = Expr::Index(Box::new(expr), indices, span);
            } else if self.consume(&Token::LBrace) {
                let start = expr.span().start;
                let mut indices = Vec::new();
                indices.push(self.parse_expr()?);
                while self.consume(&Token::Comma) {
                    indices.push(self.parse_expr()?);
                }
                if !self.consume(&Token::RBrace) {
                    return Err("expected '}'".into());
                }
                let end = self.last_token_end();
                let span = self.span_from(start, end);
                expr = Expr::IndexCell(Box::new(expr), indices, span);
            } else if self.peek_token() == Some(&Token::Dot) {
                if self.peek_token_at(1) == Some(&Token::Transpose) {
                    self.pos += 2;
                    let end = self.last_token_end();
                    let span = self.span_from(expr.span().start, end);
                    expr = Expr::Unary(UnOp::NonConjugateTranspose, Box::new(expr), span);
                    continue;
                }
                if self.peek_token_at(1) == Some(&Token::Plus)
                    || self.peek_token_at(1) == Some(&Token::Minus)
                {
                    break;
                }
                self.pos += 1;
                let name_token = match self.next() {
                    Some(TokenInfo {
                        token: Token::Ident,
                        lexeme,
                        position,
                        end,
                    }) => (lexeme, position, end),
                    _ => return Err("expected member name after '.'".into()),
                };
                if self.consume(&Token::LParen) {
                    let mut args = Vec::new();
                    if !self.consume(&Token::RParen) {
                        args.push(self.parse_expr()?);
                        while self.consume(&Token::Comma) {
                            args.push(self.parse_expr()?);
                        }
                        if !self.consume(&Token::RParen) {
                            return Err("expected ')' after method arguments".into());
                        }
                    }
                    let end = self.last_token_end();
                    let span = self.span_from(expr.span().start, end);
                    if matches!(expr, Expr::MetaClass(_, _)) {
                        expr = Expr::MethodCall(Box::new(expr), name_token.0, args, span);
                    } else {
                        expr = Expr::DottedInvoke(Box::new(expr), name_token.0, args, span);
                    }
                } else {
                    let span = self.span_from(expr.span().start, name_token.2);
                    expr = Expr::Member(Box::new(expr), name_token.0, span);
                }
            } else if self.consume(&Token::Transpose) {
                let end = self.last_token_end();
                let span = self.span_from(expr.span().start, end);
                expr = Expr::Unary(UnOp::Transpose, Box::new(expr), span);
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let expr = self.parse_primary()?;
        self.parse_postfix_with_base(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if self.peek_token() == Some(&Token::Plus) {
            let start = self.tokens[self.pos].position;
            self.pos += 1;
            let expr = self.parse_unary()?;
            Ok(self.make_unary(UnOp::Plus, expr, start))
        } else if self.peek_token() == Some(&Token::Minus) {
            let start = self.tokens[self.pos].position;
            self.pos += 1;
            let expr = self.parse_unary()?;
            Ok(self.make_unary(UnOp::Minus, expr, start))
        } else if self.peek_token() == Some(&Token::Tilde) {
            let start = self.tokens[self.pos].position;
            self.pos += 1;
            let expr = self.parse_unary()?;
            Ok(self.make_unary(UnOp::Not, expr, start))
        } else if self.peek_token() == Some(&Token::Question) {
            let start = self.tokens[self.pos].position;
            self.pos += 1;
            // Meta-class query with controlled qualified name consumption to allow postfix chaining.
            // Consume packages (lowercase-leading) and exactly one Class segment (uppercase-leading), then stop.
            let mut parts: Vec<String> = Vec::new();
            let first = self.expect_ident()?;
            let class_consumed = first
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);
            parts.push(first);
            while self.peek_token() == Some(&Token::Dot)
                && matches!(self.peek_token_at(1), Some(Token::Ident))
            {
                let next_lex = if let Some(ti) = self.tokens.get(self.pos + 1) {
                    ti.lexeme.clone()
                } else {
                    String::new()
                };
                let is_upper = next_lex
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false);
                if class_consumed {
                    break;
                }
                self.pos += 1;
                let seg = self.expect_ident()?;
                parts.push(seg);
                if is_upper {
                    break;
                }
            }
            let end = self.last_token_end();
            let span = self.span_from(start, end);
            let base = Expr::MetaClass(parts.join("."), span);
            self.parse_postfix_with_base(base)
        } else {
            self.parse_pow()
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(info) => match info.token {
                Token::Integer | Token::Float => {
                    let span = self.span_from(info.position, info.end);
                    Ok(Expr::Number(info.lexeme, span))
                }
                Token::Str => {
                    let span = self.span_from(info.position, info.end);
                    Ok(Expr::String(info.lexeme, span))
                }
                Token::True => {
                    let span = self.span_from(info.position, info.end);
                    Ok(Expr::Ident("true".into(), span))
                }
                Token::False => {
                    let span = self.span_from(info.position, info.end);
                    Ok(Expr::Ident("false".into(), span))
                }
                Token::Ident => {
                    let span = self.span_from(info.position, info.end);
                    Ok(Expr::Ident(info.lexeme, span))
                }
                // Treat 'end' as EndKeyword in expression contexts; in command-form we allow
                // 'end' to be consumed as an identifier via command-args path.
                Token::End => {
                    let span = self.span_from(info.position, info.end);
                    Ok(Expr::EndKeyword(span))
                }
                Token::At => {
                    let start = info.position;
                    if self.consume(&Token::LParen) {
                        let mut params = Vec::new();
                        if !self.consume(&Token::RParen) {
                            params.push(self.expect_ident()?);
                            while self.consume(&Token::Comma) {
                                params.push(self.expect_ident()?);
                            }
                            if !self.consume(&Token::RParen) {
                                return Err(
                                    "expected ')' after anonymous function parameters".into()
                                );
                            }
                        }
                        let body = self.parse_expr().map_err(|e| e.message)?;
                        let span = self.span_from(start, body.span().end);
                        Ok(Expr::AnonFunc {
                            params,
                            body: Box::new(body),
                            span,
                        })
                    } else {
                        let name = self.expect_ident()?;
                        let end = self.last_token_end();
                        let span = self.span_from(start, end);
                        Ok(Expr::FuncHandle(name, span))
                    }
                }
                Token::LParen => {
                    let start = info.position;
                    let expr = self.parse_expr()?;
                    if !self.consume(&Token::RParen) {
                        return Err("expected ')' to close parentheses".into());
                    }
                    let end = self.last_token_end();
                    let span = self.span_from(start, end);
                    Ok(expr.with_span(span))
                }
                Token::LBracket => {
                    let start = info.position;
                    let matrix = self.parse_matrix()?;
                    if !self.consume(&Token::RBracket) {
                        return Err("expected ']' to close matrix literal".into());
                    }
                    let end = self.last_token_end();
                    let span = self.span_from(start, end);
                    Ok(matrix.with_span(span))
                }
                Token::LBrace => {
                    let start = info.position;
                    let cell = self.parse_cell()?;
                    if !self.consume(&Token::RBrace) {
                        return Err("expected '}' to close cell literal".into());
                    }
                    let end = self.last_token_end();
                    let span = self.span_from(start, end);
                    Ok(cell.with_span(span))
                }
                Token::Colon => {
                    let span = self.span_from(info.position, info.end);
                    Ok(Expr::Colon(span))
                }
                _ => Err(format!("unexpected token: {:?}", info.token)),
            },
            None => Err("unexpected end of input".into()),
        }
    }

    fn parse_matrix(&mut self) -> Result<Expr, String> {
        self.skip_newlines();
        let mut rows = Vec::new();
        if self.peek_token() == Some(&Token::RBracket) {
            return Ok(Expr::Tensor(rows, Span::default()));
        }
        loop {
            self.skip_newlines();
            if self.peek_token() == Some(&Token::RBracket) {
                break;
            }
            let mut row = Vec::new();
            row.push(self.parse_matrix_expr()?);
            loop {
                if self.consume(&Token::Newline) {
                    continue;
                }
                if self.consume(&Token::Comma) {
                    row.push(self.parse_matrix_expr()?);
                    continue;
                }
                if matches!(
                    self.peek_token(),
                    Some(Token::Semicolon) | Some(Token::RBracket)
                ) {
                    break;
                }
                match self.peek_token() {
                    Some(
                        Token::Ident
                        | Token::Integer
                        | Token::Float
                        | Token::Str
                        | Token::LParen
                        | Token::LBracket
                        | Token::LBrace
                        | Token::At
                        | Token::Plus
                        | Token::Minus
                        | Token::Colon
                        | Token::True
                        | Token::False,
                    ) => {
                        row.push(self.parse_matrix_expr()?);
                    }
                    _ => {
                        break;
                    }
                }
            }
            rows.push(row);
            if self.consume(&Token::Semicolon) {
                self.skip_newlines();
                continue;
            } else {
                break;
            }
        }
        self.skip_newlines();
        Ok(Expr::Tensor(rows, Span::default()))
    }

    fn parse_matrix_expr(&mut self) -> Result<Expr, String> {
        let prior = self.in_matrix_expr;
        self.in_matrix_expr = true;
        let expr = self.parse_expr().map_err(|e| e.message);
        self.in_matrix_expr = prior;
        expr
    }

    fn parse_cell(&mut self) -> Result<Expr, String> {
        let mut rows = Vec::new();
        self.skip_newlines();
        if self.peek_token() == Some(&Token::RBrace) {
            return Ok(Expr::Cell(rows, Span::default()));
        }
        loop {
            self.skip_newlines();
            if self.peek_token() == Some(&Token::RBrace) {
                break;
            }
            let mut row = Vec::new();
            row.push(self.parse_expr()?);
            while self.consume(&Token::Comma) {
                row.push(self.parse_expr()?);
            }
            rows.push(row);
            if self.consume(&Token::Semicolon) {
                self.skip_newlines();
                continue;
            } else {
                break;
            }
        }
        self.skip_newlines();
        Ok(Expr::Cell(rows, Span::default()))
    }
}
