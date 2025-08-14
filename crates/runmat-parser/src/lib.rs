use runmat_lexer::Token;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Expr {
    Number(String),
    String(String),
    Ident(String),
    EndKeyword, // 'end' used in indexing contexts
    Unary(UnOp, Box<Expr>),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Tensor(Vec<Vec<Expr>>),
    Cell(Vec<Vec<Expr>>),
    Index(Box<Expr>, Vec<Expr>),
    IndexCell(Box<Expr>, Vec<Expr>),
    Range(Box<Expr>, Option<Box<Expr>>, Box<Expr>),
    Colon,
    FuncCall(String, Vec<Expr>),
    Member(Box<Expr>, String),
    // Dynamic field: s.(expr)
    MemberDynamic(Box<Expr>, Box<Expr>),
    MethodCall(Box<Expr>, String, Vec<Expr>),
    AnonFunc { params: Vec<String>, body: Box<Expr> },
    FuncHandle(String),
    MetaClass(String),
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
    // Logical operations
    AndAnd, // && (short-circuit)
    OrOr,   // || (short-circuit)
    BitAnd, // &
    BitOr,  // |
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
    NonConjugateTranspose,
    Not, // ~
}

#[derive(Debug, PartialEq)]
pub enum Stmt {
    ExprStmt(Expr, bool), // Expression and whether it's semicolon-terminated (suppressed)
    Assign(String, Expr, bool), // Variable, Expression, and whether it's semicolon-terminated (suppressed)
    MultiAssign(Vec<String>, Expr, bool),
    AssignLValue(LValue, Expr, bool),
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
    Switch {
        expr: Expr,
        cases: Vec<(Expr, Vec<Stmt>)>,
        otherwise: Option<Vec<Stmt>>,
    },
    TryCatch {
        try_body: Vec<Stmt>,
        catch_var: Option<String>,
        catch_body: Vec<Stmt>,
    },
    Global(Vec<String>),
    Persistent(Vec<String>),
    Break,
    Continue,
    Return,
    Function {
        name: String,
        params: Vec<String>,
        outputs: Vec<String>,
        body: Vec<Stmt>,
    },
    Import { path: Vec<String>, wildcard: bool },
    ClassDef {
        name: String,
        super_class: Option<String>,
        members: Vec<ClassMember>,
    },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum LValue {
    Var(String),
    Member(Box<Expr>, String),
    MemberDynamic(Box<Expr>, Box<Expr>),
    Index(Box<Expr>, Vec<Expr>),
    IndexCell(Box<Expr>, Vec<Expr>),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Attr {
    pub name: String,
    pub value: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum ClassMember {
    Properties { attributes: Vec<Attr>, names: Vec<String> },
    Methods { attributes: Vec<Attr>, body: Vec<Stmt> },
    Events { attributes: Vec<Attr>, names: Vec<String> },
    Enumeration { attributes: Vec<Attr>, names: Vec<String> },
    Arguments { attributes: Vec<Attr>, names: Vec<String> },
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
    use runmat_lexer::tokenize_detailed;

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
        // Skip layout-only tokens from lexing
        if matches!(t.token, Token::Ellipsis | Token::Section) {
            continue;
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
    parse(input).map_err(|e| format!("{e}"))
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Parse error at position {}: {}",
            self.position, self.message
        )?;
        if let Some(found) = &self.found_token {
            write!(f, " (found: '{found}')")?;
        }
        if let Some(expected) = &self.expected {
            write!(f, " (expected: {expected})")?;
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
        format!("{error}")
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
            if self.consume(&Token::Semicolon) || self.consume(&Token::Comma) {
                continue;
            }
            body.push(self.parse_stmt_with_semicolon()?);
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

    fn parse_stmt_with_semicolon(&mut self) -> Result<Stmt, ParseError> {
        let stmt = self.parse_stmt()?;
        let is_semicolon_terminated = self.consume(&Token::Semicolon);

        // Expression statements: semicolon indicates output suppression
        // Top-level assignments: set true only if a following top-level statement exists
        // (i.e., not the final trailing semicolon at EOF).
        match stmt {
            Stmt::ExprStmt(expr, _) => Ok(Stmt::ExprStmt(expr, is_semicolon_terminated)),
            Stmt::Assign(name, expr, _) => {
                let has_more_toplevel_tokens = self.pos < self.tokens.len();
                Ok(Stmt::Assign(
                    name,
                    expr,
                    is_semicolon_terminated && has_more_toplevel_tokens,
                ))
            }
            other => Ok(other),
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek_token() {
            Some(Token::If) => self.parse_if().map_err(|e| e.into()),
            Some(Token::For) => self.parse_for().map_err(|e| e.into()),
            Some(Token::While) => self.parse_while().map_err(|e| e.into()),
            Some(Token::Switch) => self.parse_switch().map_err(|e| e.into()),
            Some(Token::Try) => self.parse_try_catch().map_err(|e| e.into()),
            Some(Token::Import) => self.parse_import().map_err(|e| e.into()),
            Some(Token::Global) => self.parse_global().map_err(|e| e.into()),
            Some(Token::Persistent) => self.parse_persistent().map_err(|e| e.into()),
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
            Some(Token::ClassDef) => self.parse_classdef().map_err(|e| e.into()),
            // Multi-assign like [a,b] = f()
            Some(Token::LBracket) => {
                let save = self.pos;
                match self.try_parse_multi_assign() {
                    Ok(stmt) => Ok(stmt),
                    Err(_) => {
                        self.pos = save;
                        let expr = self.parse_expr()?;
                        Ok(Stmt::ExprStmt(expr, false))
                    }
                }
            }
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
                    Ok(Stmt::Assign(name, expr, false)) // Will be updated by parse_stmt_with_semicolon
                } else if self.peek_token() == Some(&Token::Ident)
                    && !matches!(
                        self.peek_token_at(1),
                        Some(
                            Token::Assign
                                | Token::LParen
                                | Token::Dot
                                | Token::LBracket
                                | Token::LBrace
                                | Token::Plus
                                | Token::Minus
                                | Token::Star
                                | Token::Transpose
                        )
                    )
                    && !matches!(
                        self.peek_token_at(1),
                        Some(Token::AndAnd | Token::OrOr | Token::And | Token::Or)
                    )
                {
                    // Command/function duality: name arg1 arg2 ...
                    let name = self.next().unwrap().lexeme;
                    let args = self.parse_command_args();
                    Ok(Stmt::ExprStmt(Expr::FuncCall(name, args), false))
                } else if let Some(lv) = self.try_parse_lvalue_assign()? {
                    Ok(lv)
                } else {
                    let expr = self.parse_expr()?;
                    Ok(Stmt::ExprStmt(expr, false)) // Will be updated by parse_stmt_with_semicolon
                }
            }
        }
    }

    fn parse_command_args(&mut self) -> Vec<Expr> {
        let mut args = Vec::new();
        loop {
            match self.peek_token() {
                Some(Token::Ident) => {
                    let ident = self.next().unwrap().lexeme;
                    args.push(Expr::Ident(ident));
                }
                Some(Token::Integer) | Some(Token::Float) => {
                    let num = self.next().unwrap().lexeme;
                    args.push(Expr::Number(num));
                }
                Some(Token::Str) => {
                    let s = self.next().unwrap().lexeme;
                    args.push(Expr::String(s));
                }
                _ => break,
            }
        }
        args
    }

    fn try_parse_lvalue_assign(&mut self) -> Result<Option<Stmt>, ParseError> {
        let save = self.pos;
        // Parse potential LValue: Member/Index/IndexCell
        let lvalue = if self.peek_token() == Some(&Token::Ident) {
            // Start with primary
            let base_ident = self.next().unwrap().lexeme;
            let mut base = Expr::Ident(base_ident);
            loop {
                if self.consume(&Token::LParen) {
                    let mut args = Vec::new();
                    if !self.consume(&Token::RParen) {
                        args.push(self.parse_expr()?);
                        while self.consume(&Token::Comma) {
                            args.push(self.parse_expr()?);
                        }
                        if !self.consume(&Token::RParen) {
                            return Err(self.error_with_expected("expected ')' after indices", ")"));
                        }
                    }
                    base = Expr::Index(Box::new(base), args);
                } else if self.consume(&Token::LBracket) {
                    let mut idxs = Vec::new();
                    idxs.push(self.parse_expr()?);
                    while self.consume(&Token::Comma) { idxs.push(self.parse_expr()?); }
                    if !self.consume(&Token::RBracket) {
                        return Err(self.error_with_expected("expected ']'", "]"));
                    }
                    base = Expr::Index(Box::new(base), idxs);
                } else if self.consume(&Token::LBrace) {
                    let mut idxs = Vec::new();
                    idxs.push(self.parse_expr()?);
                    while self.consume(&Token::Comma) { idxs.push(self.parse_expr()?); }
                    if !self.consume(&Token::RBrace) {
                        return Err(self.error_with_expected("expected '}'", "}"));
                    }
                    base = Expr::IndexCell(Box::new(base), idxs);
                } else if self.peek_token() == Some(&Token::Dot) {
                    // If this is .', it's a non-conjugate transpose, not a member
                    if self.peek_token_at(1) == Some(&Token::Transpose) {
                        break;
                    }
                    // If this is .+ or .-, it's an additive operator, not a member
                    if self.peek_token_at(1) == Some(&Token::Plus)
                        || self.peek_token_at(1) == Some(&Token::Minus)
                    {
                        break;
                    }
                    // Otherwise, member access
                    self.pos += 1; // consume '.'
                    // Support dynamic field: .(expr) or static .ident
                    if self.consume(&Token::LParen) {
                        let name_expr = self.parse_expr()?;
                        if !self.consume(&Token::RParen) { return Err(self.error_with_expected("expected ')' after dynamic field expression", ")")); }
                        base = Expr::MemberDynamic(Box::new(base), Box::new(name_expr));
                    } else {
                        let name = self.expect_ident()?;
                        base = Expr::Member(Box::new(base), name);
                    }
                } else {
                    break;
                }
            }
            base
        } else {
            self.pos = save; return Ok(None);
        };
        if !self.consume(&Token::Assign) { self.pos = save; return Ok(None); }
        let rhs = self.parse_expr()?;
        let stmt = match lvalue {
            Expr::Member(b, name) => Stmt::AssignLValue(LValue::Member(b, name), rhs, false),
            Expr::MemberDynamic(b, n) => Stmt::AssignLValue(LValue::MemberDynamic(b, n), rhs, false),
            Expr::Index(b, idxs) => Stmt::AssignLValue(LValue::Index(b, idxs), rhs, false),
            Expr::IndexCell(b, idxs) => Stmt::AssignLValue(LValue::IndexCell(b, idxs), rhs, false),
            Expr::Ident(v) => Stmt::Assign(v, rhs, false),
            _ => { self.pos = save; return Ok(None); }
        };
        Ok(Some(stmt))
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<Expr, ParseError> {
        let mut node = self.parse_logical_and()?;
        while self.consume(&Token::OrOr) {
            let rhs = self.parse_logical_and()?;
            node = Expr::Binary(Box::new(node), BinOp::OrOr, Box::new(rhs));
        }
        Ok(node)
    }

    fn parse_logical_and(&mut self) -> Result<Expr, ParseError> {
        let mut node = self.parse_bitwise_or()?;
        while self.consume(&Token::AndAnd) {
            let rhs = self.parse_bitwise_or()?;
            node = Expr::Binary(Box::new(node), BinOp::AndAnd, Box::new(rhs));
        }
        Ok(node)
    }

    fn parse_bitwise_or(&mut self) -> Result<Expr, ParseError> {
        let mut node = self.parse_bitwise_and()?;
        while self.consume(&Token::Or) {
            let rhs = self.parse_bitwise_and()?;
            node = Expr::Binary(Box::new(node), BinOp::BitOr, Box::new(rhs));
        }
        Ok(node)
    }

    fn parse_bitwise_and(&mut self) -> Result<Expr, ParseError> {
        let mut node = self.parse_range()?;
        while self.consume(&Token::And) {
            let rhs = self.parse_range()?;
            node = Expr::Binary(Box::new(node), BinOp::BitAnd, Box::new(rhs));
        }
        Ok(node)
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
            let op = if self.consume(&Token::Plus) {
                Some(BinOp::Add)
            } else if self.consume(&Token::Minus) {
                Some(BinOp::Sub)
            } else if self.peek_token() == Some(&Token::Dot)
                && (self.peek_token_at(1) == Some(&Token::Plus)
                    || self.peek_token_at(1) == Some(&Token::Minus))
            {
                // '.+' or '.-' tokenized as Dot then Plus/Minus; treat like Add/Sub
                // consume two tokens
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
                // Heuristic: If arguments indicate indexing (colon/end/range), treat as indexing; else if
                // the callee is an identifier, treat as function call; otherwise as indexing.
                let has_index_hint = args.iter().any(|a| self.expr_suggests_indexing(a));
                if has_index_hint {
                    expr = Expr::Index(Box::new(expr), args);
                } else {
                    match expr {
                        Expr::Ident(ref name) => expr = Expr::FuncCall(name.clone(), args),
                        _ => expr = Expr::Index(Box::new(expr), args),
                    }
                }
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
            } else if self.consume(&Token::LBrace) {
                // Cell content indexing
                let mut indices = Vec::new();
                indices.push(self.parse_expr()?);
                while self.consume(&Token::Comma) {
                    indices.push(self.parse_expr()?);
                }
                if !self.consume(&Token::RBrace) {
                    return Err("expected '}'".into());
                }
                expr = Expr::IndexCell(Box::new(expr), indices);
            } else if self.peek_token() == Some(&Token::Dot) {
                // Could be .', .+ , .- or member access
                if self.peek_token_at(1) == Some(&Token::Transpose) {
                    self.pos += 2; // '.' and '''
                    expr = Expr::Unary(UnOp::NonConjugateTranspose, Box::new(expr));
                    continue;
                }
                if self.peek_token_at(1) == Some(&Token::Plus)
                    || self.peek_token_at(1) == Some(&Token::Minus)
                {
                    // '.+' or '.-' belong to additive level; stop postfix loop
                    break;
                }
                // Otherwise, member access
                self.pos += 1; // consume '.'
                let name = match self.next() {
                    Some(TokenInfo { token: Token::Ident, lexeme, .. }) => lexeme,
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
                    expr = Expr::MethodCall(Box::new(expr), name, args);
                } else {
                    expr = Expr::Member(Box::new(expr), name);
                }
            } else if self.consume(&Token::Transpose) {
                // Matrix transpose (postfix operator)
                expr = Expr::Unary(UnOp::Transpose, Box::new(expr));
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn expr_suggests_indexing(&self, e: &Expr) -> bool {
        match e {
            Expr::Colon | Expr::EndKeyword | Expr::Range(_, _, _) => true,
            Expr::Binary(_, op, _) => matches!(
                op,
                BinOp::Colon
                    | BinOp::Equal
                    | BinOp::NotEqual
                    | BinOp::Less
                    | BinOp::LessEqual
                    | BinOp::Greater
                    | BinOp::GreaterEqual
                    | BinOp::AndAnd
                    | BinOp::OrOr
                    | BinOp::BitAnd
                    | BinOp::BitOr
            ),
            _ => false,
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if self.consume(&Token::Plus) {
            Ok(Expr::Unary(UnOp::Plus, Box::new(self.parse_unary()?)))
        } else if self.consume(&Token::Minus) {
            Ok(Expr::Unary(UnOp::Minus, Box::new(self.parse_unary()?)))
        } else if self.consume(&Token::Tilde) {
            Ok(Expr::Unary(UnOp::Not, Box::new(self.parse_unary()?)))
        } else if self.consume(&Token::Question) {
            // Meta-class query: ?Qualified.Name
            let name = self.parse_qualified_name()?;
            Ok(Expr::MetaClass(name))
        } else {
            self.parse_pow()
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(info) => match info.token {
                Token::Integer | Token::Float => Ok(Expr::Number(info.lexeme)),
                Token::Str => Ok(Expr::String(info.lexeme)),
                Token::True => Ok(Expr::Ident("true".into())),
                Token::False => Ok(Expr::Ident("false".into())),
                Token::Ident => Ok(Expr::Ident(info.lexeme)),
                Token::End => Ok(Expr::EndKeyword),
                Token::At => {
                    // Anonymous function or function handle
                    if self.consume(&Token::LParen) {
                        let mut params = Vec::new();
                        if !self.consume(&Token::RParen) {
                            params.push(self.expect_ident()?);
                            while self.consume(&Token::Comma) {
                                params.push(self.expect_ident()?);
                            }
                            if !self.consume(&Token::RParen) {
                                return Err("expected ')' after anonymous function parameters".into());
                            }
                        }
                        let body = self.parse_expr().map_err(|e| e.message)?;
                        Ok(Expr::AnonFunc { params, body: Box::new(body) })
                    } else {
                        // function handle @name
                        let name = self.expect_ident()?;
                        Ok(Expr::FuncHandle(name))
                    }
                }
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
                Token::LBrace => {
                    let cell = self.parse_cell()?;
                    if !self.consume(&Token::RBrace) {
                        return Err("expected '}' to close cell literal".into());
                    }
                    Ok(cell)
                }
                Token::Colon => Ok(Expr::Colon),
                Token::ClassDef => {
                    // Allow classdef at statement position only; if encountered in expression, surface a clearer error
                    Err("unexpected token 'classdef' in expression context".into())
                }
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
                    Err(format!("unexpected {token_desc} in expression context"))
                }
            },
            None => Err("unexpected end of input, expected expression".into()),
        }
    }

    fn parse_matrix(&mut self) -> Result<Expr, String> {
        let mut rows = Vec::new();
        if self.peek_token() == Some(&Token::RBracket) {
            return Ok(Expr::Tensor(rows));
        }
        loop {
            let mut row = Vec::new();
            // First element in the row
            row.push(self.parse_expr()?);
            // Accept either comma-separated or whitespace-separated elements until ';' or ']'
            loop {
                if self.consume(&Token::Comma) {
                    row.push(self.parse_expr()?);
                    continue;
                }
                // If next token ends the row/matrix, stop
                if matches!(self.peek_token(), Some(Token::Semicolon) | Some(Token::RBracket)) {
                    break;
                }
                // Otherwise, treat whitespace as a separator and parse the next element
                // Only proceed if the next token can start an expression
                match self.peek_token() {
                    Some(Token::Ident
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
                        | Token::False
                    ) => {
                        row.push(self.parse_expr()?);
                    }
                    _ => { break; }
                }
            }
            rows.push(row);
            if self.consume(&Token::Semicolon) {
                continue;
            } else {
                break;
            }
        }
        Ok(Expr::Tensor(rows))
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

        // Enforce MATLAB varargs placement constraints at parse time
        // varargin: at most once, must be last in params if present
        if let Some(idx) = params.iter().position(|p| p == "varargin") {
            if idx != params.len() - 1 {
                return Err("'varargin' must be the last input parameter".into());
            }
            if params.iter().filter(|p| p.as_str() == "varargin").count() > 1 {
                return Err("'varargin' cannot appear more than once".into());
            }
        }
        // varargout: at most once, must be last in outputs if present
        if let Some(idx) = outputs.iter().position(|o| o == "varargout") {
            if idx != outputs.len() - 1 {
                return Err("'varargout' must be the last output parameter".into());
            }
            if outputs.iter().filter(|o| o.as_str() == "varargout").count() > 1 {
                return Err("'varargout' cannot appear more than once".into());
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
            if self.consume(&Token::Semicolon) || self.consume(&Token::Comma) {
                continue;
            }
            let stmt = self.parse_stmt().map_err(|e| e.message)?;
            let is_semicolon_terminated = self.consume(&Token::Semicolon);

            // Only expression statements are display-suppressed by semicolon.
            let final_stmt = match stmt {
                Stmt::ExprStmt(expr, _) => Stmt::ExprStmt(expr, is_semicolon_terminated),
                Stmt::Assign(name, expr, _) => Stmt::Assign(name, expr, false),
                other => other,
            };
            body.push(final_stmt);
        }
        Ok(body)
    }

    fn parse_cell(&mut self) -> Result<Expr, String> {
        let mut rows = Vec::new();
        if self.peek_token() == Some(&Token::RBrace) {
            return Ok(Expr::Cell(rows));
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
        Ok(Expr::Cell(rows))
    }

    fn parse_switch(&mut self) -> Result<Stmt, String> {
        self.consume(&Token::Switch);
        let control = self.parse_expr()?;
        let mut cases = Vec::new();
        let mut otherwise: Option<Vec<Stmt>> = None;
        loop {
            if self.consume(&Token::Case) {
                let val = self.parse_expr()?;
                let body = self.parse_block(|t| matches!(t, Token::Case | Token::Otherwise | Token::End))?;
                cases.push((val, body));
            } else if self.consume(&Token::Otherwise) {
                let body = self.parse_block(|t| matches!(t, Token::End))?;
                otherwise = Some(body);
            } else {
                break;
            }
        }
        if !self.consume(&Token::End) {
            return Err("expected 'end' for switch".into());
        }
        Ok(Stmt::Switch { expr: control, cases, otherwise })
    }

    fn parse_try_catch(&mut self) -> Result<Stmt, String> {
        self.consume(&Token::Try);
        let try_body = self.parse_block(|t| matches!(t, Token::Catch | Token::End))?;
        let (catch_var, catch_body) = if self.consume(&Token::Catch) {
            let maybe_ident = if let Some(Token::Ident) = self.peek_token() {
                Some(self.expect_ident()?)
            } else {
                None
            };
            let body = self.parse_block(|t| matches!(t, Token::End))?;
            (maybe_ident, body)
        } else {
            (None, Vec::new())
        };
        if !self.consume(&Token::End) {
            return Err("expected 'end' after try/catch".into());
        }
        Ok(Stmt::TryCatch { try_body, catch_var, catch_body })
    }

    fn parse_import(&mut self) -> Result<Stmt, String> {
        self.consume(&Token::Import);
        // import pkg.sub.Class or import pkg.*
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
        Ok(Stmt::Import { path, wildcard })
    }

    fn parse_classdef(&mut self) -> Result<Stmt, String> {
        self.consume(&Token::ClassDef);
        let name = self.parse_qualified_name()?;
        let mut super_class = None;
        if self.consume(&Token::Less) {
            super_class = Some(self.parse_qualified_name()?);
        }
        let mut members: Vec<ClassMember> = Vec::new();
        loop {
            // Skip layout separators between member blocks
            if self.consume(&Token::Semicolon) || self.consume(&Token::Comma) {
                continue;
            }
            match self.peek_token() {
                Some(Token::Properties) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let props = self.parse_properties_names_block()?;
                    if !self.consume(&Token::End) { return Err("expected 'end' after properties".into()); }
                    members.push(ClassMember::Properties { attributes: attrs, names: props });
                }
                Some(Token::Methods) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let body = self.parse_block(|t| matches!(t, Token::End))?;
                    if !self.consume(&Token::End) { return Err("expected 'end' after methods".into()); }
                    members.push(ClassMember::Methods { attributes: attrs, body });
                }
                Some(Token::Events) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let names = self.parse_name_block()?;
                    if !self.consume(&Token::End) { return Err("expected 'end' after events".into()); }
                    members.push(ClassMember::Events { attributes: attrs, names });
                }
                Some(Token::Enumeration) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let names = self.parse_name_block()?;
                    if !self.consume(&Token::End) { return Err("expected 'end' after enumeration".into()); }
                    members.push(ClassMember::Enumeration { attributes: attrs, names });
                }
                Some(Token::Arguments) => {
                    self.pos += 1;
                    let attrs = self.parse_optional_attr_list();
                    let names = self.parse_name_block()?;
                    if !self.consume(&Token::End) { return Err("expected 'end' after arguments".into()); }
                    members.push(ClassMember::Arguments { attributes: attrs, names });
                }
                Some(Token::End) => { self.pos += 1; break; }
                _ => break,
            }
        }
        Ok(Stmt::ClassDef { name, super_class, members })
    }

    fn parse_name_block(&mut self) -> Result<Vec<String>, String> {
        let mut names = Vec::new();
        while let Some(tok) = self.peek_token() {
            if matches!(tok, Token::End) { break; }
            if self.consume(&Token::Semicolon) || self.consume(&Token::Comma) { continue; }
            if let Some(Token::Ident) = self.peek_token() {
                names.push(self.expect_ident()?);
            } else {
                break;
            }
        }
        Ok(names)
    }

    fn parse_properties_names_block(&mut self) -> Result<Vec<String>, String> {
        // Accept identifiers with optional default assignment: name, name = expr
        let mut names = Vec::new();
        while let Some(tok) = self.peek_token() {
            if matches!(tok, Token::End) { break; }
            if self.consume(&Token::Semicolon) || self.consume(&Token::Comma) { continue; }
            if let Some(Token::Ident) = self.peek_token() {
                names.push(self.expect_ident()?);
                // Optional default initializer: skip over `= expr` syntactically
                if self.consume(&Token::Assign) {
                    // Parse and discard expression to keep the grammar permissive; initializer is HIR/semantics concern
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
        if !self.consume(&Token::LParen) { return attrs; }
        loop {
            if self.consume(&Token::RParen) { break; }
            match self.peek_token() {
                Some(Token::Ident) => {
                    let name = self.expect_ident().unwrap_or_else(|_| "".to_string());
                    let mut value: Option<String> = None;
                    if self.consume(&Token::Assign) {
                        // Value could be ident, string or number; capture raw lexeme
                        if let Some(tok) = self.next() { value = Some(tok.lexeme); }
                    }
                    attrs.push(Attr { name, value });
                    let _ = self.consume(&Token::Comma);
                }
                Some(Token::Comma) => { self.pos += 1; }
                Some(Token::RParen) => { self.pos += 1; break; }
                Some(_) => { self.pos += 1; }
                None => { break; }
            }
        }
        attrs
    }

    fn parse_global(&mut self) -> Result<Stmt, String> {
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
        Ok(Stmt::Global(names))
    }

    fn parse_persistent(&mut self) -> Result<Stmt, String> {
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
        Ok(Stmt::Persistent(names))
    }

    fn try_parse_multi_assign(&mut self) -> Result<Stmt, String> {
        if !self.consume(&Token::LBracket) { return Err("not a multi-assign".into()); }
        let mut names = Vec::new();
        names.push(self.expect_ident_or_tilde()?);
        while self.consume(&Token::Comma) {
            names.push(self.expect_ident_or_tilde()?);
        }
        if !self.consume(&Token::RBracket) { return Err("expected ']'".into()); }
        if !self.consume(&Token::Assign) { return Err("expected '='".into()); }
        let rhs = self.parse_expr().map_err(|e| e.message)?;
        Ok(Stmt::MultiAssign(names, rhs, false))
    }

    fn parse_qualified_name(&mut self) -> Result<String, String> {
        let mut parts = Vec::new();
        parts.push(self.expect_ident()?);
        while self.consume(&Token::Dot) {
            parts.push(self.expect_ident()?);
        }
        Ok(parts.join("."))
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

    fn expect_ident_or_tilde(&mut self) -> Result<String, String> {
        match self.next() {
            Some(TokenInfo { token: Token::Ident, lexeme, .. }) => Ok(lexeme),
            Some(TokenInfo { token: Token::Tilde, .. }) => Ok("~".to_string()),
            _ => Err("expected identifier or '~'".into()),
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
