// use logos::Logos; // Not needed since we use rustmat_lexer::tokenize
use rustmat_lexer::Token;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Expr {
    Number(String),
    String(String),
    Ident(String),
    Unary(UnOp, Box<Expr>),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Matrix(Vec<Vec<Expr>>),
    Index(Box<Expr>, Vec<Expr>),
    Range(Box<Expr>, Option<Box<Expr>>, Box<Expr>),
    Colon,
    FuncCall(String, Vec<Expr>),
}

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    LeftDiv,
    Colon,
    // Element-wise operations
    ElemMul,     // .*
    ElemDiv,     // ./
    ElemPow,     // .^
    ElemLeftDiv, // .\
    // Comparison operations
    Equal,        // ==
    NotEqual,     // ~=
    Less,         // <
    LessEqual,    // <=
    Greater,      // >
    GreaterEqual, // >=
}

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub enum UnOp {
    Plus,
    Minus,
    Transpose,
}

#[derive(Debug, PartialEq)]
pub enum Stmt {
    ExprStmt(Expr),
    Assign(String, Expr),
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        elseif_blocks: Vec<(Expr, Vec<Stmt>)>,
        else_body: Option<Vec<Stmt>>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    For {
        var: String,
        expr: Expr,
        body: Vec<Stmt>,
    },
    Break,
    Continue,
    Return,
    Function {
        name: String,
        params: Vec<String>,
        outputs: Vec<String>,
        body: Vec<Stmt>,
    },
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub body: Vec<Stmt>,
}

#[derive(Clone)]
struct TokenInfo {
    token: Token,
    lexeme: String,
    position: usize,
}

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
    pub found_token: Option<String>,
    pub expected: Option<String>,
}

pub fn parse(input: &str) -> Result<Program, ParseError> {
    use rustmat_lexer::tokenize_detailed;

    let toks = tokenize_detailed(input);
    let mut tokens = Vec::new();

    for t in toks {
        if matches!(t.token, Token::Error) {
            return Err(ParseError {
                message: format!("Invalid token: '{}'", t.lexeme),
                position: t.start,
                found_token: Some(t.lexeme),
                expected: None,
            });
        }
        tokens.push(TokenInfo {
            token: t.token,
            lexeme: t.lexeme,
            position: t.start,
        });
    }

    let mut parser = Parser {
        tokens,
        pos: 0,
        input: input.to_string(),
    };
    parser.parse_program()
}

// For backward compatibility
pub fn parse_simple(input: &str) -> Result<Program, String> {
    parse(input).map_err(|e| format!("{}", e))
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Parse error at position {}: {}",
            self.position, self.message
        )?;
        if let Some(found) = &self.found_token {
            write!(f, " (found: '{}')", found)?;
        }
        if let Some(expected) = &self.expected {
            write!(f, " (expected: {})", expected)?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

impl From<String> for ParseError {
    fn from(message: String) -> Self {
        ParseError {
            message,
            position: 0,
            found_token: None,
            expected: None,
        }
    }
}

impl From<ParseError> for String {
    fn from(error: ParseError) -> Self {
        format!("{}", error)
    }
}

struct Parser {
    tokens: Vec<TokenInfo>,
    pos: usize,
    input: String,
}

impl Parser {
    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut body = Vec::new();
        while self.pos < self.tokens.len() {
            if self.consume(&Token::Semicolon) {
                continue;
            }
            body.push(self.parse_stmt()?);
            self.consume(&Token::Semicolon);
        }
        Ok(Program { body })
    }

    fn error(&self, message: &str) -> ParseError {
        let (position, found_token) = if let Some(token_info) = self.tokens.get(self.pos) {
            (token_info.position, Some(token_info.lexeme.clone()))
        } else {
            (self.input.len(), None)
        };

        ParseError {
            message: message.to_string(),
            position,
            found_token,
            expected: None,
        }
    }

    fn error_with_expected(&self, message: &str, expected: &str) -> ParseError {
        let (position, found_token) = if let Some(token_info) = self.tokens.get(self.pos) {
            (token_info.position, Some(token_info.lexeme.clone()))
        } else {
            (self.input.len(), None)
        };

        ParseError {
            message: message.to_string(),
            position,
            found_token,
            expected: Some(expected.to_string()),
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek_token() {
            Some(Token::If) => self.parse_if().map_err(|e| e.into()),
            Some(Token::For) => self.parse_for().map_err(|e| e.into()),
            Some(Token::While) => self.parse_while().map_err(|e| e.into()),
            Some(Token::Break) => {
                self.pos += 1;
                Ok(Stmt::Break)
            }
            Some(Token::Continue) => {
                self.pos += 1;
                Ok(Stmt::Continue)
            }
            Some(Token::Return) => {
                self.pos += 1;
                Ok(Stmt::Return)
            }
            Some(Token::Function) => self.parse_function().map_err(|e| e.into()),
            _ => {
                if self.peek_token() == Some(&Token::Ident)
                    && self.peek_token_at(1) == Some(&Token::Assign)
                {
                    let name = self
                        .next()
                        .ok_or_else(|| self.error("expected identifier"))?
                        .lexeme;
                    if !self.consume(&Token::Assign) {
                        return Err(self.error_with_expected("expected assignment operator", "'='"));
                    }
                    let expr = self.parse_expr()?;
                    Ok(Stmt::Assign(name, expr))
                } else {
                    let expr = self.parse_expr()?;
                    Ok(Stmt::ExprStmt(expr))
                }
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_range()
    }

    fn parse_range(&mut self) -> Result<Expr, ParseError> {
        let mut node = self.parse_comparison()?;
        if self.consume(&Token::Colon) {
            let mid = self.parse_comparison()?;
            if self.consume(&Token::Colon) {
                let end = self.parse_comparison()?;
                node = Expr::Range(Box::new(node), Some(Box::new(mid)), Box::new(end));
            } else {
                node = Expr::Range(Box::new(node), None, Box::new(mid));
            }
        }
        Ok(node)
    }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
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
            self.pos += 1; // consume op
            let rhs = self.parse_add_sub()?;
            node = Expr::Binary(Box::new(node), op, Box::new(rhs));
        }
        Ok(node)
    }

    fn parse_add_sub(&mut self) -> Result<Expr, String> {
        let mut node = self.parse_mul_div()?;
        loop {
            let op = match self.peek_token() {
                Some(Token::Plus) => BinOp::Add,
                Some(Token::Minus) => BinOp::Sub,
                _ => break,
            };
            self.pos += 1; // consume op
            let rhs = self.parse_mul_div()?;
            node = Expr::Binary(Box::new(node), op, Box::new(rhs));
        }
        Ok(node)
    }

    fn parse_mul_div(&mut self) -> Result<Expr, String> {
        let mut node = self.parse_unary()?;
        loop {
            let op = match self.peek_token() {
                Some(Token::Star) => BinOp::Mul,
                Some(Token::DotStar) => BinOp::ElemMul,
                Some(Token::Slash) => BinOp::Div,
                Some(Token::DotSlash) => BinOp::ElemDiv,
                Some(Token::Backslash) => BinOp::LeftDiv,
                Some(Token::DotBackslash) => BinOp::ElemLeftDiv,
                _ => break,
            };
            self.pos += 1; // consume op
            let rhs = self.parse_unary()?;
            node = Expr::Binary(Box::new(node), op, Box::new(rhs));
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
            self.pos += 1; // consume
            let rhs = self.parse_pow()?; // right associative
            Ok(Expr::Binary(Box::new(node), op, Box::new(rhs)))
        } else {
            Ok(node)
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.consume(&Token::LParen) {
                // Parse as function call initially - semantic analysis will resolve ambiguity
                let func_name = match expr {
                    Expr::Ident(name) => name,
                    _ => return Err("expected function name before '('".into()),
                };
                let mut args = Vec::new();
                if !self.consume(&Token::RParen) {
                    args.push(self.parse_expr()?);
                    while self.consume(&Token::Comma) {
                        args.push(self.parse_expr()?);
                    }
                    if !self.consume(&Token::RParen) {
                        return Err("expected ')' after function arguments".into());
                    }
                }
                expr = Expr::FuncCall(func_name, args);
            } else if self.consume(&Token::LBracket) {
                // Array indexing
                let mut indices = Vec::new();
                indices.push(self.parse_expr()?);
                while self.consume(&Token::Comma) {
                    indices.push(self.parse_expr()?);
                }
                if !self.consume(&Token::RBracket) {
                    return Err("expected ']'".into());
                }
                expr = Expr::Index(Box::new(expr), indices);
            } else if self.consume(&Token::Transpose) {
                // Matrix transpose (postfix operator)
                expr = Expr::Unary(UnOp::Transpose, Box::new(expr));
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if self.consume(&Token::Plus) {
            Ok(Expr::Unary(UnOp::Plus, Box::new(self.parse_unary()?)))
        } else if self.consume(&Token::Minus) {
            Ok(Expr::Unary(UnOp::Minus, Box::new(self.parse_unary()?)))
        } else {
            self.parse_pow()
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(info) => match info.token {
                Token::Integer | Token::Float => Ok(Expr::Number(info.lexeme)),
                Token::Str => Ok(Expr::String(info.lexeme)),
                Token::Ident => Ok(Expr::Ident(info.lexeme)),
                Token::LParen => {
                    let expr = self.parse_expr()?;
                    if !self.consume(&Token::RParen) {
                        return Err("expected ')' to close parentheses".into());
                    }
                    Ok(expr)
                }
                Token::LBracket => {
                    let matrix = self.parse_matrix()?;
                    if !self.consume(&Token::RBracket) {
                        return Err("expected ']' to close matrix literal".into());
                    }
                    Ok(matrix)
                }
                Token::Colon => Ok(Expr::Colon),
                _ => {
                    // Provide detailed error message about what token was unexpected
                    let token_desc = match info.token {
                        Token::Semicolon => "semicolon ';' (statement separator)",
                        Token::Comma => "comma ',' (list separator)",
                        Token::RParen => {
                            "closing parenthesis ')' (no matching opening parenthesis)"
                        }
                        Token::RBracket => "closing bracket ']' (no matching opening bracket)",
                        Token::If => "keyword 'if' (expected in statement context)",
                        Token::For => "keyword 'for' (expected in statement context)",
                        Token::While => "keyword 'while' (expected in statement context)",
                        Token::Function => "keyword 'function' (expected in statement context)",
                        Token::End => "keyword 'end' (no matching control structure)",
                        Token::Equal => "equality operator '==' (expected in comparison context)",
                        Token::Assign => "assignment operator '=' (expected in assignment context)",
                        Token::Error => "invalid character or symbol",
                        _ => {
                            return Err(format!(
                                "unexpected token '{}' in expression context",
                                info.lexeme
                            ))
                        }
                    };
                    Err(format!("unexpected {} in expression context", token_desc))
                }
            },
            None => Err("unexpected end of input, expected expression".into()),
        }
    }

    fn parse_matrix(&mut self) -> Result<Expr, String> {
        let mut rows = Vec::new();
        if self.peek_token() == Some(&Token::RBracket) {
            return Ok(Expr::Matrix(rows));
        }
        loop {
            let mut row = Vec::new();
            row.push(self.parse_expr()?);
            while self.consume(&Token::Comma) {
                row.push(self.parse_expr()?);
            }
            rows.push(row);
            if self.consume(&Token::Semicolon) {
                continue;
            } else {
                break;
            }
        }
        Ok(Expr::Matrix(rows))
    }

    fn parse_if(&mut self) -> Result<Stmt, String> {
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
        Ok(Stmt::If {
            cond,
            then_body,
            elseif_blocks,
            else_body,
        })
    }

    fn parse_while(&mut self) -> Result<Stmt, String> {
        self.consume(&Token::While);
        let cond = self.parse_expr()?;
        let body = self.parse_block(|t| matches!(t, Token::End))?;
        if !self.consume(&Token::End) {
            return Err("expected 'end'".into());
        }
        Ok(Stmt::While { cond, body })
    }

    fn parse_for(&mut self) -> Result<Stmt, String> {
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
        Ok(Stmt::For { var, expr, body })
    }

    fn parse_function(&mut self) -> Result<Stmt, String> {
        self.consume(&Token::Function);
        let mut outputs = Vec::new();
        if self.consume(&Token::LBracket) {
            outputs.push(self.expect_ident()?);
            while self.consume(&Token::Comma) {
                outputs.push(self.expect_ident()?);
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
        let body = self.parse_block(|t| matches!(t, Token::End))?;
        if !self.consume(&Token::End) {
            return Err("expected 'end'".into());
        }
        Ok(Stmt::Function {
            name,
            params,
            outputs,
            body,
        })
    }

    fn parse_block<F>(&mut self, term: F) -> Result<Vec<Stmt>, String>
    where
        F: Fn(&Token) -> bool,
    {
        let mut body = Vec::new();
        while let Some(tok) = self.peek_token() {
            if term(tok) {
                break;
            }
            if self.consume(&Token::Semicolon) {
                continue;
            }
            body.push(self.parse_stmt()?);
            self.consume(&Token::Semicolon);
        }
        Ok(body)
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.next() {
            Some(TokenInfo {
                token: Token::Ident,
                lexeme,
                ..
            }) => Ok(lexeme),
            _ => Err("expected identifier".into()),
        }
    }

    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|t| &t.token)
    }

    fn peek_token_at(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset).map(|t| &t.token)
    }

    fn next(&mut self) -> Option<TokenInfo> {
        if self.pos < self.tokens.len() {
            let info = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(info)
        } else {
            None
        }
    }

    fn consume(&mut self, t: &Token) -> bool {
        if self.peek_token() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}
