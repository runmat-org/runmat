use runmat_lexer::Token;

use crate::{BinOp, Expr, Span, SyntaxError, UnOp};

use super::{Parser, TokenInfo};

impl Parser {
    pub(super) fn parse_call_argument(&mut self) -> Result<Expr, SyntaxError> {
        if self.peek_token().is_some_and(Token::can_be_identifier)
            && self.peek_token_at(1) == Some(&Token::Assign)
        {
            let name = self.next().expect("peeked identifier");
            self.pos += 1; // consume '='
            let value = self.parse_nested_expr()?;
            let span = self.span_from(name.position, value.span().end);
            return Ok(Expr::NameValueArg(name.lexeme, Box::new(value), span));
        }
        self.parse_nested_expr()
    }

    pub(super) fn parse_call_arguments_until(
        &mut self,
        end_token: Token,
        expected_message: &str,
        expected: &str,
    ) -> Result<Vec<Expr>, SyntaxError> {
        let mut args = Vec::new();
        if self.consume(&end_token) {
            return Ok(args);
        }
        args.push(self.parse_call_argument()?);
        while self.consume(&Token::Comma) {
            args.push(self.parse_call_argument()?);
        }
        if !self.consume(&end_token) {
            return Err(self.error_with_expected(expected_message, expected));
        }
        Ok(args)
    }

    pub(super) fn parse_expr(&mut self) -> Result<Expr, SyntaxError> {
        self.parse_logical_or()
    }

    fn parse_nested_expr(&mut self) -> Result<Expr, SyntaxError> {
        let prior = self.at_matrix_element_boundary;
        self.at_matrix_element_boundary = false;
        let expr = self.parse_expr();
        self.at_matrix_element_boundary = prior;
        expr
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

    fn parse_add_sub(&mut self) -> Result<Expr, SyntaxError> {
        let mut node = self.parse_mul_div()?;
        loop {
            if self.at_matrix_element_boundary
                && matches!(self.peek_token(), Some(Token::Plus | Token::Minus))
                && self.pos > 0
                && !self.tokens_adjacent(self.pos - 1, self.pos)
                && self.tokens_adjacent(self.pos, self.pos + 1)
            {
                // A spaced sign attached to its operand begins another column,
                // including imaginary operands. Nested expressions do not have
                // matrix-element separator semantics.
                break;
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

    fn parse_mul_div(&mut self) -> Result<Expr, SyntaxError> {
        let mut node = self.parse_unary()?;
        loop {
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

    fn parse_pow(&mut self) -> Result<Expr, SyntaxError> {
        let node = self.parse_postfix()?;
        if let Some(token) = self.peek_token() {
            let op = match token {
                Token::Caret => BinOp::Pow,
                Token::DotCaret => BinOp::ElemPow,
                _ => return Ok(node),
            };
            self.pos += 1;
            // Prefix signs are valid on an exponent without parentheses.
            // Unary parsing delegates unsigned operands back to parse_pow,
            // preserving right association and the precedence of -2^2.
            let rhs = self.parse_unary()?;
            Ok(self.make_binary(node, op, rhs))
        } else {
            Ok(node)
        }
    }

    fn parse_postfix_with_base(&mut self, mut expr: Expr) -> Result<Expr, SyntaxError> {
        loop {
            // At the outer level of a matrix element, a space before an
            // opening delimiter starts the next column, rather than indexing
            // or calling the prior column.
            // Calls and grouping inside the element parse with this boundary off.
            if self.at_matrix_element_boundary
                && matches!(
                    self.peek_token(),
                    Some(Token::LParen | Token::LBracket | Token::LBrace)
                )
                && self.pos > 0
                && !self.tokens_adjacent(self.pos - 1, self.pos)
            {
                break;
            }
            if self.consume(&Token::LParen) {
                let start = expr.span().start;
                let args = self.parse_call_arguments_until(
                    Token::RParen,
                    "expected ')' after arguments",
                    "')'",
                )?;
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
                indices.push(self.parse_nested_expr()?);
                while self.consume(&Token::Comma) {
                    indices.push(self.parse_nested_expr()?);
                }
                if !self.consume(&Token::RBracket) {
                    return Err(self.error_with_expected("expected ']'", "']'"));
                }
                let end = self.last_token_end();
                let span = self.span_from(start, end);
                expr = Expr::Index(Box::new(expr), indices, span);
            } else if self.consume(&Token::LBrace) {
                if let Some(aggregate_expr) = self.try_parse_brace_aggregate_literal(&expr)? {
                    expr = aggregate_expr;
                    continue;
                }
                let start = expr.span().start;
                let mut indices = Vec::new();
                indices.push(self.parse_nested_expr()?);
                while self.consume(&Token::Comma) {
                    indices.push(self.parse_nested_expr()?);
                }
                if !self.consume(&Token::RBrace) {
                    return Err(self.error_with_expected("expected '}'", "'}'"));
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
                if self.consume(&Token::LParen) {
                    let name_expr = self.parse_nested_expr()?;
                    if !self.consume(&Token::RParen) {
                        return Err(self.error_with_expected(
                            "expected ')' after dynamic member expression",
                            "')'",
                        ));
                    }
                    let end = self.last_token_end();
                    let span = self.span_from(expr.span().start, end);
                    expr = Expr::MemberDynamic(Box::new(expr), Box::new(name_expr), span);
                    continue;
                }
                let name_token = match self.peek().cloned() {
                    Some(TokenInfo {
                        token,
                        lexeme,
                        position,
                        end,
                    }) if token.can_be_identifier() => {
                        self.pos += 1;
                        (lexeme, position, end)
                    }
                    Some(token) => {
                        return Err(SyntaxError {
                            message: "expected member name after '.'".to_string(),
                            position: token.position,
                            found_token: Some(token.lexeme),
                            expected: Some("identifier".to_string()),
                        });
                    }
                    _ => {
                        return Err(self
                            .error_with_expected("expected member name after '.'", "identifier"));
                    }
                };
                if self.consume(&Token::LParen) {
                    let args = self.parse_call_arguments_until(
                        Token::RParen,
                        "expected ')' after method arguments",
                        "')'",
                    )?;
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

    fn parse_postfix(&mut self) -> Result<Expr, SyntaxError> {
        let expr = self.parse_primary()?;
        self.parse_postfix_with_base(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, SyntaxError> {
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
            let first = self.expect_ident_syntax()?;
            let class_consumed = first
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);
            parts.push(first);
            while self.peek_token() == Some(&Token::Dot)
                && self.peek_token_at(1).is_some_and(Token::can_be_identifier)
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
                let seg = self.expect_ident_syntax()?;
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

    fn parse_primary(&mut self) -> Result<Expr, SyntaxError> {
        match self.next() {
            Some(info) => match info.token {
                Token::Integer | Token::Float => {
                    // The suffix is syntax of the numeric atom, not a variable
                    // named i/j. Preserve the numeric text for later f64 decoding.
                    if self.peek_token() == Some(&Token::Ident)
                        && self.tokens_adjacent(self.pos - 1, self.pos)
                        && self.tokens[self.pos].lexeme.len() == 1
                        && matches!(
                            self.tokens[self.pos].lexeme.as_bytes()[0],
                            b'i' | b'I' | b'j' | b'J'
                        )
                    {
                        let suffix = self.next().expect("peeked imaginary suffix");
                        Ok(Expr::ImaginaryNumber(
                            format!("{}{}", info.lexeme, suffix.lexeme),
                            self.span_from(info.position, suffix.end),
                        ))
                    } else {
                        let span = self.span_from(info.position, info.end);
                        Ok(Expr::Number(info.lexeme, span))
                    }
                }
                Token::RadixInteger => {
                    let span = self.span_from(info.position, info.end);
                    let literal =
                        crate::IntegerLiteral::parse(&info.lexeme).map_err(|message| {
                            SyntaxError {
                                message,
                                position: info.position,
                                found_token: Some(info.lexeme.clone()),
                                expected: Some(
                                    "a valid hexadecimal or binary integer literal".into(),
                                ),
                            }
                        })?;
                    Ok(Expr::IntegerLiteral(literal, span))
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
                token if token.can_be_identifier() => {
                    let span = self.span_from(info.position, info.end);
                    if self.peek_token() == Some(&Token::At) {
                        let method_name = info.lexeme;
                        self.pos += 1; // consume '@'
                        let mut super_parts = Vec::new();
                        super_parts.push(self.expect_ident_syntax()?);
                        while self.consume(&Token::Dot) {
                            super_parts.push(self.expect_ident_syntax()?);
                        }
                        let super_name = super_parts.join(".");
                        if !self.consume(&Token::LParen) {
                            return Err(self.error_with_expected(
                                "expected '(' after superclass method name",
                                "'('",
                            ));
                        }
                        let args = self.parse_call_arguments_until(
                            Token::RParen,
                            "expected ',' or ')' in argument list",
                            "',' or ')'",
                        )?;
                        let end = self.last_token_end();
                        let span = self.span_from(info.position, end);
                        let class_name = self.current_classdef_name.clone().ok_or_else(|| {
                            self.error(
                                "superclass method syntax is only valid inside classdef methods",
                            )
                        })?;
                        Ok(Expr::SuperMethodCall {
                            current_class: class_name,
                            super_class: super_name,
                            method: method_name,
                            args,
                            span,
                        })
                    } else {
                        Ok(Expr::Ident(info.lexeme, span))
                    }
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
                            params.push(self.expect_ident_syntax()?);
                            while self.consume(&Token::Comma) {
                                params.push(self.expect_ident_syntax()?);
                            }
                            if !self.consume(&Token::RParen) {
                                return Err(self.error_with_expected(
                                    "expected ')' after anonymous function parameters",
                                    "')'",
                                ));
                            }
                        }
                        let body = self.parse_nested_expr()?;
                        let span = self.span_from(start, body.span().end);
                        Ok(Expr::AnonFunc {
                            params,
                            body: Box::new(body),
                            span,
                        })
                    } else {
                        let mut name = self.expect_ident_syntax()?;
                        while self.peek_token() == Some(&Token::Dot)
                            && self.peek_token_at(1).is_some_and(Token::can_be_identifier)
                        {
                            self.consume(&Token::Dot);
                            name.push('.');
                            name.push_str(&self.expect_ident_syntax()?);
                        }
                        let end = self.last_token_end();
                        let span = self.span_from(start, end);
                        Ok(Expr::FuncHandle(name, span))
                    }
                }
                Token::LParen => {
                    let start = info.position;
                    let expr = self.parse_nested_expr()?;
                    if !self.consume(&Token::RParen) {
                        return Err(
                            self.error_with_expected("expected ')' to close parentheses", "')'")
                        );
                    }
                    let end = self.last_token_end();
                    let span = self.span_from(start, end);
                    Ok(expr.with_span(span))
                }
                Token::LBracket => {
                    let start = info.position;
                    let matrix = self.parse_matrix()?;
                    if !self.consume(&Token::RBracket) {
                        return Err(
                            self.error_with_expected("expected ']' to close matrix literal", "']'")
                        );
                    }
                    let end = self.last_token_end();
                    let span = self.span_from(start, end);
                    Ok(matrix.with_span(span))
                }
                Token::LBrace => {
                    let start = info.position;
                    let cell = self.parse_cell()?;
                    if !self.consume(&Token::RBrace) {
                        return Err(
                            self.error_with_expected("expected '}' to close cell literal", "'}'")
                        );
                    }
                    let end = self.last_token_end();
                    let span = self.span_from(start, end);
                    Ok(cell.with_span(span))
                }
                Token::Colon => {
                    let span = self.span_from(info.position, info.end);
                    Ok(Expr::Colon(span))
                }
                _ => Err(SyntaxError {
                    message: format!("unexpected token: {:?}", info.token),
                    position: info.position,
                    found_token: Some(info.lexeme),
                    expected: None,
                }),
            },
            None => Err(self.error("unexpected end of input")),
        }
    }

    fn parse_matrix(&mut self) -> Result<Expr, SyntaxError> {
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
                    Some(token)
                        if token.can_be_identifier()
                            || matches!(
                                token,
                                Token::Integer
                                    | Token::RadixInteger
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
                                    | Token::False
                            ) =>
                    {
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

    fn parse_matrix_expr(&mut self) -> Result<Expr, SyntaxError> {
        let prior_boundary = self.at_matrix_element_boundary;
        self.at_matrix_element_boundary = true;
        let expr = self.parse_expr();
        self.at_matrix_element_boundary = prior_boundary;
        expr
    }

    fn parse_cell(&mut self) -> Result<Expr, SyntaxError> {
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
            row.push(self.parse_nested_expr()?);
            while self.consume(&Token::Comma) {
                row.push(self.parse_nested_expr()?);
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

    fn try_parse_brace_aggregate_literal(
        &mut self,
        base: &Expr,
    ) -> Result<Option<Expr>, SyntaxError> {
        enum AggregateBase {
            Struct,
            Object(String),
        }

        let aggregate_base = match base {
            Expr::Ident(name, _) if name == "struct" => AggregateBase::Struct,
            Expr::MetaClass(class_name, _) => AggregateBase::Object(class_name.clone()),
            _ => return Ok(None),
        };

        let checkpoint = self.pos;
        let mut fields: Vec<(String, Expr)> = Vec::new();

        self.skip_newlines();
        if self.consume(&Token::RBrace) {
            let span = self.span_from(base.span().start, self.last_token_end());
            return Ok(Some(match aggregate_base {
                AggregateBase::Struct => Expr::StructLiteral(fields, span),
                AggregateBase::Object(class_name) => Expr::ObjectLiteral(class_name, fields, span),
            }));
        }

        loop {
            self.skip_newlines();
            let Some(token) = self.next() else {
                self.pos = checkpoint;
                return Ok(None);
            };
            let field_name = if token.token.can_be_identifier() {
                token.lexeme
            } else {
                self.pos = checkpoint;
                return Ok(None);
            };
            if !self.consume(&Token::Assign) {
                self.pos = checkpoint;
                return Ok(None);
            }
            let value = match self.parse_nested_expr() {
                Ok(value) => value,
                Err(_) => {
                    self.pos = checkpoint;
                    return Ok(None);
                }
            };
            fields.push((field_name, value));
            if self.consume(&Token::Comma) || self.consume(&Token::Semicolon) {
                continue;
            }
            break;
        }

        self.skip_newlines();
        if !self.consume(&Token::RBrace) {
            self.pos = checkpoint;
            return Ok(None);
        }

        let span = self.span_from(base.span().start, self.last_token_end());
        Ok(Some(match aggregate_base {
            AggregateBase::Struct => Expr::StructLiteral(fields, span),
            AggregateBase::Object(class_name) => Expr::ObjectLiteral(class_name, fields, span),
        }))
    }
}
