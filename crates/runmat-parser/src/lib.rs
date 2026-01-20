use runmat_lexer::Token;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompatMode {
    #[default]
    Matlab,
    Strict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParserOptions {
    #[serde(default)]
    pub compat_mode: CompatMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Default for ParserOptions {
    fn default() -> Self {
        Self {
            compat_mode: CompatMode::Matlab,
        }
    }
}

impl ParserOptions {
    pub fn new(compat_mode: CompatMode) -> Self {
        Self { compat_mode }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Expr {
    Number(String, Span),
    String(String, Span),
    Ident(String, Span),
    EndKeyword(Span), // 'end' used in indexing contexts
    Unary(UnOp, Box<Expr>, Span),
    Binary(Box<Expr>, BinOp, Box<Expr>, Span),
    Tensor(Vec<Vec<Expr>>, Span),
    Cell(Vec<Vec<Expr>>, Span),
    Index(Box<Expr>, Vec<Expr>, Span),
    IndexCell(Box<Expr>, Vec<Expr>, Span),
    Range(Box<Expr>, Option<Box<Expr>>, Box<Expr>, Span),
    Colon(Span),
    FuncCall(String, Vec<Expr>, Span),
    Member(Box<Expr>, String, Span),
    // Dynamic field: s.(expr)
    MemberDynamic(Box<Expr>, Box<Expr>, Span),
    MethodCall(Box<Expr>, String, Vec<Expr>, Span),
    AnonFunc {
        params: Vec<String>,
        body: Box<Expr>,
        span: Span,
    },
    FuncHandle(String, Span),
    MetaClass(String, Span),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Number(_, span)
            | Expr::String(_, span)
            | Expr::Ident(_, span)
            | Expr::EndKeyword(span)
            | Expr::Unary(_, _, span)
            | Expr::Binary(_, _, _, span)
            | Expr::Tensor(_, span)
            | Expr::Cell(_, span)
            | Expr::Index(_, _, span)
            | Expr::IndexCell(_, _, span)
            | Expr::Range(_, _, _, span)
            | Expr::Colon(span)
            | Expr::FuncCall(_, _, span)
            | Expr::Member(_, _, span)
            | Expr::MemberDynamic(_, _, span)
            | Expr::MethodCall(_, _, _, span)
            | Expr::FuncHandle(_, span)
            | Expr::MetaClass(_, span) => *span,
            Expr::AnonFunc { span, .. } => *span,
        }
    }

    pub fn with_span(self, span: Span) -> Expr {
        match self {
            Expr::Number(value, _) => Expr::Number(value, span),
            Expr::String(value, _) => Expr::String(value, span),
            Expr::Ident(value, _) => Expr::Ident(value, span),
            Expr::EndKeyword(_) => Expr::EndKeyword(span),
            Expr::Unary(op, expr, _) => Expr::Unary(op, expr, span),
            Expr::Binary(lhs, op, rhs, _) => Expr::Binary(lhs, op, rhs, span),
            Expr::Tensor(rows, _) => Expr::Tensor(rows, span),
            Expr::Cell(rows, _) => Expr::Cell(rows, span),
            Expr::Index(base, indices, _) => Expr::Index(base, indices, span),
            Expr::IndexCell(base, indices, _) => Expr::IndexCell(base, indices, span),
            Expr::Range(start, step, end, _) => Expr::Range(start, step, end, span),
            Expr::Colon(_) => Expr::Colon(span),
            Expr::FuncCall(name, args, _) => Expr::FuncCall(name, args, span),
            Expr::Member(base, name, _) => Expr::Member(base, name, span),
            Expr::MemberDynamic(base, name, _) => Expr::MemberDynamic(base, name, span),
            Expr::MethodCall(base, name, args, _) => Expr::MethodCall(base, name, args, span),
            Expr::AnonFunc { params, body, .. } => Expr::AnonFunc { params, body, span },
            Expr::FuncHandle(name, _) => Expr::FuncHandle(name, span),
            Expr::MetaClass(name, _) => Expr::MetaClass(name, span),
        }
    }
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
    ExprStmt(Expr, bool, Span), // Expression and whether it's semicolon-terminated (suppressed)
    Assign(String, Expr, bool, Span), // Variable, Expression, and whether it's semicolon-terminated (suppressed)
    MultiAssign(Vec<String>, Expr, bool, Span),
    AssignLValue(LValue, Expr, bool, Span),
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        elseif_blocks: Vec<(Expr, Vec<Stmt>)>,
        else_body: Option<Vec<Stmt>>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    For {
        var: String,
        expr: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    Switch {
        expr: Expr,
        cases: Vec<(Expr, Vec<Stmt>)>,
        otherwise: Option<Vec<Stmt>>,
        span: Span,
    },
    TryCatch {
        try_body: Vec<Stmt>,
        catch_var: Option<String>,
        catch_body: Vec<Stmt>,
        span: Span,
    },
    Global(Vec<String>, Span),
    Persistent(Vec<String>, Span),
    Break(Span),
    Continue(Span),
    Return(Span),
    Function {
        name: String,
        params: Vec<String>,
        outputs: Vec<String>,
        body: Vec<Stmt>,
        span: Span,
    },
    Import {
        path: Vec<String>,
        wildcard: bool,
        span: Span,
    },
    ClassDef {
        name: String,
        super_class: Option<String>,
        members: Vec<ClassMember>,
        span: Span,
    },
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::ExprStmt(_, _, span)
            | Stmt::Assign(_, _, _, span)
            | Stmt::MultiAssign(_, _, _, span)
            | Stmt::AssignLValue(_, _, _, span)
            | Stmt::Global(_, span)
            | Stmt::Persistent(_, span)
            | Stmt::Break(span)
            | Stmt::Continue(span)
            | Stmt::Return(span) => *span,
            Stmt::If { span, .. }
            | Stmt::While { span, .. }
            | Stmt::For { span, .. }
            | Stmt::Switch { span, .. }
            | Stmt::TryCatch { span, .. }
            | Stmt::Function { span, .. }
            | Stmt::Import { span, .. }
            | Stmt::ClassDef { span, .. } => *span,
        }
    }
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
    Properties {
        attributes: Vec<Attr>,
        names: Vec<String>,
    },
    Methods {
        attributes: Vec<Attr>,
        body: Vec<Stmt>,
    },
    Events {
        attributes: Vec<Attr>,
        names: Vec<String>,
    },
    Enumeration {
        attributes: Vec<Attr>,
        names: Vec<String>,
    },
    Arguments {
        attributes: Vec<Attr>,
        names: Vec<String>,
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
    end: usize,
}

#[derive(Debug)]
pub struct SyntaxError {
    pub message: String,
    pub position: usize,
    pub found_token: Option<String>,
    pub expected: Option<String>,
}

pub fn parse(input: &str) -> Result<Program, SyntaxError> {
    parse_with_options(input, ParserOptions::default())
}

pub fn parse_with_options(input: &str, options: ParserOptions) -> Result<Program, SyntaxError> {
    use runmat_lexer::tokenize_detailed;

    let toks = tokenize_detailed(input);
    let mut tokens = Vec::new();

    for t in toks {
        if matches!(t.token, Token::Error) {
            return Err(SyntaxError {
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
            end: t.end,
        });
    }

    let mut parser = Parser {
        tokens,
        pos: 0,
        input: input.to_string(),
        options,
    };
    parser.parse_program()
}

impl std::fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Syntax error at position {}: {}",
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

impl std::error::Error for SyntaxError {}

impl From<String> for SyntaxError {
    fn from(value: String) -> Self {
        SyntaxError {
            message: value,
            position: 0,
            found_token: None,
            expected: None,
        }
    }
}

impl From<SyntaxError> for String {
    fn from(error: SyntaxError) -> Self {
        error.to_string()
    }
}

struct Parser {
    tokens: Vec<TokenInfo>,
    pos: usize,
    input: String,
    options: ParserOptions,
}

#[derive(Clone, Copy)]
struct CommandVerb {
    name: &'static str,
    arg_kind: CommandArgKind,
}

#[derive(Clone, Copy)]
enum CommandArgKind {
    Keyword {
        allowed: &'static [&'static str],
        optional: bool,
    },
    Any,
}

const COMMAND_VERBS: &[CommandVerb] = &[
    CommandVerb {
        name: "hold",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["on", "off", "all", "reset"],
            optional: false,
        },
    },
    CommandVerb {
        name: "grid",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["on", "off"],
            optional: false,
        },
    },
    CommandVerb {
        name: "box",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["on", "off"],
            optional: false,
        },
    },
    CommandVerb {
        name: "axis",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["auto", "manual", "tight", "equal", "ij", "xy"],
            optional: false,
        },
    },
    CommandVerb {
        name: "shading",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["flat", "interp", "faceted"],
            optional: false,
        },
    },
    CommandVerb {
        name: "colormap",
        arg_kind: CommandArgKind::Keyword {
            allowed: &[
                "parula", "jet", "hsv", "hot", "cool", "spring", "summer", "autumn", "winter",
                "gray", "bone", "copper", "pink",
            ],
            optional: false,
        },
    },
    CommandVerb {
        name: "colorbar",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["on", "off"],
            optional: true,
        },
    },
    CommandVerb {
        name: "figure",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "subplot",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "clf",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "cla",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "close",
        arg_kind: CommandArgKind::Any,
    },
];

impl Parser {
    fn skip_newlines(&mut self) {
        while self.consume(&Token::Newline) {}
    }

    fn tokens_adjacent(&self, left: usize, right: usize) -> bool {
        match (self.tokens.get(left), self.tokens.get(right)) {
            (Some(a), Some(b)) => a.end == b.position,
            _ => false,
        }
    }

    fn span_from(&self, start: usize, end: usize) -> Span {
        Span { start, end }
    }

    fn span_between(&self, start: Span, end: Span) -> Span {
        Span {
            start: start.start,
            end: end.end,
        }
    }

    fn last_token_end(&self) -> usize {
        self.tokens
            .get(self.pos.saturating_sub(1))
            .map(|t| t.end)
            .unwrap_or(self.input.len())
    }

    fn make_binary(&self, left: Expr, op: BinOp, right: Expr) -> Expr {
        let span = self.span_between(left.span(), right.span());
        Expr::Binary(Box::new(left), op, Box::new(right), span)
    }

    fn make_unary(&self, op: UnOp, operand: Expr, op_start: usize) -> Expr {
        let span = self.span_from(op_start, operand.span().end);
        Expr::Unary(op, Box::new(operand), span)
    }

    fn is_simple_assignment_ahead(&self) -> bool {
        // Heuristic: at statement start, if we see Ident ... '=' before a terminator, treat as assignment
        self.peek_token() == Some(&Token::Ident) && self.peek_token_at(1) == Some(&Token::Assign)
    }
    fn parse_program(&mut self) -> Result<Program, SyntaxError> {
        let mut body = Vec::new();
        while self.pos < self.tokens.len() {
            if self.consume(&Token::Semicolon)
                || self.consume(&Token::Comma)
                || self.consume(&Token::Newline)
            {
                continue;
            }
            body.push(self.parse_stmt_with_semicolon()?);
        }
        Ok(Program { body })
    }

    fn error(&self, message: &str) -> SyntaxError {
        SyntaxError {
            message: message.to_string(),
            position: self.current_position(),
            found_token: self.peek().map(|t| t.lexeme.clone()),
            expected: None,
        }
    }

    fn error_with_expected(&self, message: &str, expected: &str) -> SyntaxError {
        SyntaxError {
            message: message.to_string(),
            position: self.current_position(),
            found_token: self.peek().map(|t| t.lexeme.clone()),
            expected: Some(expected.to_string()),
        }
    }

    fn parse_stmt_with_semicolon(&mut self) -> Result<Stmt, SyntaxError> {
        let stmt = self.parse_stmt()?;
        let is_semicolon_terminated = self.consume(&Token::Semicolon);

        // Expression statements: semicolon indicates output suppression.
        // Assignments/lvalues are now suppressed whenever a semicolon is present, even at EOF.
        match stmt {
            Stmt::ExprStmt(expr, _, span) => {
                Ok(Stmt::ExprStmt(expr, is_semicolon_terminated, span))
            }
            Stmt::Assign(name, expr, _, span) => {
                Ok(Stmt::Assign(name, expr, is_semicolon_terminated, span))
            }
            Stmt::MultiAssign(names, expr, _, span) => Ok(Stmt::MultiAssign(
                names,
                expr,
                is_semicolon_terminated,
                span,
            )),
            Stmt::AssignLValue(lv, expr, _, span) => {
                Ok(Stmt::AssignLValue(lv, expr, is_semicolon_terminated, span))
            }
            other => Ok(other),
        }
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
            // Multi-assign like [a,b] = f()
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
                    // Fallback: treat as simple assignment if '=' appears before terminator
                    let name = self.expect_ident().map_err(|e| self.error(&e))?;
                    let start = self.tokens[self.pos.saturating_sub(1)].position;
                    if !self.consume(&Token::Assign) {
                        return Err(self.error_with_expected("expected assignment operator", "'='"));
                    }
                    let expr = self.parse_expr()?;
                    let span = self.span_from(start, expr.span().end);
                    Ok(Stmt::Assign(name, expr, false, span))
                } else if self.peek_token() == Some(&Token::Ident) {
                    // First, try complex lvalue assignment starting from an identifier: A(1)=x, A{1}=x, s.f=x, s.(n)=x
                    if let Some(lv) = self.try_parse_lvalue_assign()? {
                        return Ok(lv);
                    }
                    // Command-form at statement start if it looks like a sequence of simple arguments
                    // and is not immediately followed by indexing/member syntax.
                    if self.can_start_command_form() {
                        if self.options.compat_mode == CompatMode::Strict {
                            return Err(self.error(
                                "Command syntax is disabled in strict compatibility mode; call functions with parentheses.",
                            ));
                        }
                        let name_token = self.next().unwrap();
                        let mut args = self.parse_command_args();
                        if let Some(command) = self.lookup_command(&name_token.lexeme) {
                            self.normalize_command_args(command, &mut args[..])?;
                        }
                        let end = self.last_token_end();
                        let span = self.span_from(name_token.position, end);
                        Ok(Stmt::ExprStmt(
                            Expr::FuncCall(name_token.lexeme, args, span),
                            false,
                            span,
                        ))
                    } else {
                        // If we see Ident <space> Ident immediately followed by postfix opener,
                        // this is an ambiguous adjacency (e.g., "foo b(1)"). Emit a targeted error.
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

    fn can_start_command_form(&self) -> bool {
        // At entry, peek_token() is Some(Ident) for callee
        let Some(current) = self.tokens.get(self.pos) else {
            return false;
        };
        let verb = current.lexeme.as_str();
        let command = self.lookup_command(verb);
        let zero_arg_allowed = matches!(
            command,
            Some(CommandVerb {
                arg_kind: CommandArgKind::Any,
                ..
            })
        ) || matches!(
            command,
            Some(CommandVerb {
                arg_kind: CommandArgKind::Keyword { optional: true, .. },
                ..
            })
        );

        let mut i = 1;
        let mut saw_arg = false;
        while matches!(
            self.peek_token_at(i),
            Some(Token::Newline | Token::Ellipsis)
        ) {
            i += 1;
        }
        // At least one simple arg must follow
        if !matches!(
            self.peek_token_at(i),
            Some(Token::Ident | Token::Integer | Token::Float | Token::Str | Token::End)
        ) {
            if !zero_arg_allowed {
                return false;
            }
        } else {
            saw_arg = true;
        }
        // Consume all contiguous simple args
        loop {
            match self.peek_token_at(i) {
                Some(Token::Ident | Token::Integer | Token::Float | Token::Str | Token::End) => {
                    saw_arg = true;
                    i += 1;
                }
                Some(Token::Newline | Token::Ellipsis) => {
                    i += 1;
                }
                _ => break,
            }
        }
        if !saw_arg && !zero_arg_allowed {
            return false;
        }
        // If the next token begins indexing/member or other expression syntax, do not use command-form
        match self.peek_token_at(i) {
            Some(Token::LParen)
            | Some(Token::Dot)
            | Some(Token::LBracket)
            | Some(Token::LBrace)
            | Some(Token::Transpose) => false,
            // If next token is assignment, also not a command-form (would be ambiguous)
            Some(Token::Assign) => false,
            // End of statement is okay for command-form
            None | Some(Token::Semicolon) | Some(Token::Comma) | Some(Token::Newline) => true,
            // Otherwise conservatively allow
            _ => true,
        }
    }

    fn parse_command_args(&mut self) -> Vec<Expr> {
        let mut args = Vec::new();
        loop {
            if self.consume(&Token::Newline) {
                continue;
            }
            if self.consume(&Token::Ellipsis) {
                continue;
            }
            match self.peek_token() {
                Some(Token::Ident) => {
                    let token = self.next().unwrap();
                    let span = self.span_from(token.position, token.end);
                    args.push(Expr::Ident(token.lexeme, span));
                }
                // In command-form, accept 'end' as a literal identifier token for compatibility
                Some(Token::End) => {
                    let token = &self.tokens[self.pos];
                    self.pos += 1;
                    let span = self.span_from(token.position, token.end);
                    args.push(Expr::Ident("end".to_string(), span));
                }
                Some(Token::Integer) | Some(Token::Float) => {
                    let token = self.next().unwrap();
                    let span = self.span_from(token.position, token.end);
                    args.push(Expr::Number(token.lexeme, span));
                }
                Some(Token::Str) => {
                    let token = self.next().unwrap();
                    let span = self.span_from(token.position, token.end);
                    args.push(Expr::String(token.lexeme, span));
                }
                // Stop on tokens that would start normal expression syntax
                Some(Token::Slash)
                | Some(Token::Star)
                | Some(Token::Backslash)
                | Some(Token::Plus)
                | Some(Token::Minus)
                | Some(Token::LParen)
                | Some(Token::Dot)
                | Some(Token::LBracket)
                | Some(Token::LBrace)
                | Some(Token::Transpose) => break,
                _ => break,
            }
        }
        args
    }

    fn lookup_command(&self, name: &str) -> Option<&'static CommandVerb> {
        COMMAND_VERBS
            .iter()
            .find(|cmd| cmd.name.eq_ignore_ascii_case(name))
    }

    fn normalize_command_args(
        &self,
        command: &CommandVerb,
        args: &mut [Expr],
    ) -> Result<(), SyntaxError> {
        match command.arg_kind {
            CommandArgKind::Keyword { allowed, optional } => {
                if args.is_empty() {
                    if optional {
                        return Ok(());
                    }
                    return Err(self.error(&format!(
                        "'{}' command syntax requires an argument",
                        command.name
                    )));
                }
                if args.len() > 1 {
                    return Err(self.error(&format!(
                        "'{}' command syntax accepts only one argument",
                        command.name
                    )));
                }
                let keyword = extract_keyword(&args[0]).ok_or_else(|| {
                    self.error(&format!(
                        "'{}' command syntax expects a keyword argument",
                        command.name
                    ))
                })?;
                if allowed
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(&keyword))
                {
                    let span = args[0].span();
                    args[0] = Expr::String(format!("\"{}\"", keyword), span);
                } else {
                    return Err(self.error(&format!(
                        "'{}' command syntax does not support '{}'",
                        command.name, keyword
                    )));
                }
            }
            CommandArgKind::Any => {
                // Accept general expressions; no normalization needed.
            }
        }
        Ok(())
    }

    fn try_parse_lvalue_assign(&mut self) -> Result<Option<Stmt>, SyntaxError> {
        let save = self.pos;
        // Parse potential LValue: Member/Index/IndexCell
        let lvalue = if self.peek_token() == Some(&Token::Ident) {
            // Start with primary
            let base_token = self.next().unwrap();
            let base_span = self.span_from(base_token.position, base_token.end);
            let mut base = Expr::Ident(base_token.lexeme, base_span);
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
            base
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

    fn parse_expr(&mut self) -> Result<Expr, SyntaxError> {
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
            self.pos += 1; // consume op
            let rhs = self.parse_add_sub()?;
            node = self.make_binary(node, op, rhs);
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
                Some(Token::Slash) => BinOp::Div,
                Some(Token::DotSlash) => BinOp::ElemDiv,
                Some(Token::Backslash) => BinOp::LeftDiv,
                Some(Token::DotBackslash) => BinOp::ElemLeftDiv,
                _ => break,
            };
            self.pos += 1; // consume op
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
            self.pos += 1; // consume
            let rhs = self.parse_pow()?; // right associative
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
                    // For non-ident bases (e.g., X(1), (A+B)(1)), this is indexing.
                    expr = Expr::Index(Box::new(expr), args, span);
                }
            } else if self.consume(&Token::LBracket) {
                // Array indexing
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
                // Cell content indexing
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
                // Could be .', .+ , .- or member access
                if self.peek_token_at(1) == Some(&Token::Transpose) {
                    self.pos += 2; // '.' and '''
                    let end = self.last_token_end();
                    let span = self.span_from(expr.span().start, end);
                    expr = Expr::Unary(UnOp::NonConjugateTranspose, Box::new(expr), span);
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
                    expr = Expr::MethodCall(Box::new(expr), name_token.0, args, span);
                } else {
                    let span = self.span_from(expr.span().start, name_token.2);
                    expr = Expr::Member(Box::new(expr), name_token.0, span);
                }
            } else if self.consume(&Token::Transpose) {
                // Matrix transpose (postfix operator)
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
            // Meta-class query with controlled qualified name consumption to allow postfix chaining
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
                // Lookahead at the next identifier lexeme
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
                // Consume dot and ident
                self.pos += 1; // consume '.'
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
                // Treat 'end' as EndKeyword in expression contexts; in command-form we allow 'end' to be consumed as an identifier via command-args path.
                Token::End => {
                    let span = self.span_from(info.position, info.end);
                    Ok(Expr::EndKeyword(span))
                }
                Token::At => {
                    let start = info.position;
                    // Anonymous function or function handle
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
                        // function handle @name
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
            // First element in the row
            row.push(self.parse_expr()?);
            // Accept either comma-separated or whitespace-separated elements until ';' or ']'
            loop {
                if self.consume(&Token::Newline) {
                    continue;
                }
                if self.consume(&Token::Comma) {
                    row.push(self.parse_expr()?);
                    continue;
                }
                // If next token ends the row/matrix, stop
                if matches!(
                    self.peek_token(),
                    Some(Token::Semicolon) | Some(Token::RBracket)
                ) {
                    break;
                }
                // Otherwise, treat whitespace as a separator and parse the next element
                // Only proceed if the next token can start an expression
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
                        row.push(self.parse_expr()?);
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

        // Enforce varargs placement constraints at parse time
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

        // Optional function-level arguments block
        // arguments ... end  (we accept a sequence of identifiers, validation semantics handled in HIR/runtime)
        if self.peek_token() == Some(&Token::Arguments) {
            self.pos += 1; // consume 'arguments'
                           // Accept a flat list of identifiers optionally separated by commas/semicolons
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
                // Tolerate newlines/whitespace-only between entries
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

    fn parse_block<F>(&mut self, term: F) -> Result<Vec<Stmt>, String>
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
            // Fast-path: handle multi-assign LHS at statement start inside blocks reliably
            let stmt = if self.peek_token() == Some(&Token::LBracket) {
                self.try_parse_multi_assign()?
            } else {
                self.parse_stmt().map_err(|e| e.message)?
            };
            let is_semicolon_terminated = self.consume(&Token::Semicolon);

            // Only expression statements are display-suppressed by semicolon.
            let final_stmt = match stmt {
                Stmt::ExprStmt(expr, _, span) => {
                    Stmt::ExprStmt(expr, is_semicolon_terminated, span)
                }
                Stmt::Assign(name, expr, _, span) => Stmt::Assign(name, expr, false, span),
                Stmt::MultiAssign(names, expr, _, span) => {
                    Stmt::MultiAssign(names, expr, false, span)
                }
                Stmt::AssignLValue(lv, expr, _, span) => Stmt::AssignLValue(lv, expr, false, span),
                other => other,
            };
            body.push(final_stmt);
        }
        Ok(body)
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
        let end = self.last_token_end();
        Ok(Stmt::Import {
            path,
            wildcard,
            span: self.span_from(start, end),
        })
    }

    fn parse_classdef(&mut self) -> Result<Stmt, String> {
        let start = self.tokens[self.pos].position;
        self.consume(&Token::ClassDef);
        let name = self.parse_qualified_name()?;
        let mut super_class = None;
        if self.consume(&Token::Less) {
            super_class = Some(self.parse_qualified_name()?);
        }
        let mut members: Vec<ClassMember> = Vec::new();
        loop {
            // Skip layout separators between member blocks
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
        // Accept identifiers with optional default assignment: name, name = expr
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
                        // Value could be ident, string or number; capture raw lexeme
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

    fn try_parse_multi_assign(&mut self) -> Result<Stmt, String> {
        if !self.consume(&Token::LBracket) {
            return Err("not a multi-assign".into());
        }
        let start = self.tokens[self.pos.saturating_sub(1)].position;
        let mut names = Vec::new();
        names.push(self.expect_ident_or_tilde()?);
        while self.consume(&Token::Comma) {
            names.push(self.expect_ident_or_tilde()?);
        }
        if !self.consume(&Token::RBracket) {
            return Err("expected ']'".into());
        }
        if !self.consume(&Token::Assign) {
            return Err("expected '='".into());
        }
        let rhs = self.parse_expr().map_err(|e| e.message)?;
        let span = self.span_from(start, rhs.span().end);
        Ok(Stmt::MultiAssign(names, rhs, false, span))
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

    fn peek(&self) -> Option<&TokenInfo> {
        self.tokens.get(self.pos)
    }

    fn current_position(&self) -> usize {
        self.peek()
            .map(|t| t.position)
            .unwrap_or_else(|| self.input.len())
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

fn extract_keyword(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(s, _) => Some(s.clone()),
        Expr::String(s, _) => Some(s.trim_matches(&['"', '\''][..]).to_string()),
        _ => None,
    }
}
