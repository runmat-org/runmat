use logos::Logos;
use rustmat_lexer::Token;

#[derive(Debug, PartialEq)]
pub enum Expr {
    Number(String),
    Ident(String),
    Unary(UnOp, Box<Expr>),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Matrix(Vec<Vec<Expr>>),
    Index(Box<Expr>, Vec<Expr>),
    Range(Box<Expr>, Option<Box<Expr>>, Box<Expr>),
    Colon,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    LeftDiv,
    Colon,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum UnOp {
    Plus,
    Minus,
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
}

pub fn parse(input: &str) -> Result<Program, String> {
    let mut lexer = Token::lexer(input);
    let mut tokens = Vec::new();
    while let Some(tok) = lexer.next() {
        let token = tok.unwrap_or(Token::Error);
        tokens.push(TokenInfo {
            token,
            lexeme: lexer.slice().to_string(),
        });
    }
    let mut parser = Parser { tokens, pos: 0 };
    parser.parse_program()
}

struct Parser {
    tokens: Vec<TokenInfo>,
    pos: usize,
}

impl Parser {
    fn parse_program(&mut self) -> Result<Program, String> {
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

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        match self.peek_token() {
            Some(Token::If) => self.parse_if(),
            Some(Token::For) => self.parse_for(),
            Some(Token::While) => self.parse_while(),
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
            Some(Token::Function) => self.parse_function(),
            _ => {
                if self.peek_token() == Some(&Token::Ident)
                    && self.peek_token_at(1) == Some(&Token::Assign)
                {
                    let name = self.next().unwrap().lexeme;
                    self.consume(&Token::Assign);
                    let expr = self.parse_expr()?;
                    Ok(Stmt::Assign(name, expr))
                } else {
                    let expr = self.parse_expr()?;
                    Ok(Stmt::ExprStmt(expr))
                }
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_range()
    }

    fn parse_range(&mut self) -> Result<Expr, String> {
        let mut node = self.parse_add_sub()?;
        if self.consume(&Token::Colon) {
            let mid = self.parse_add_sub()?;
            if self.consume(&Token::Colon) {
                let end = self.parse_add_sub()?;
                node = Expr::Range(Box::new(node), Some(Box::new(mid)), Box::new(end));
            } else {
                node = Expr::Range(Box::new(node), None, Box::new(mid));
            }
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
                Some(Token::Star) | Some(Token::DotStar) => BinOp::Mul,
                Some(Token::Slash) | Some(Token::DotSlash) => BinOp::Div,
                Some(Token::Backslash) | Some(Token::DotBackslash) => BinOp::LeftDiv,
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
        if matches!(self.peek_token(), Some(Token::Caret | Token::DotCaret)) {
            self.pos += 1; // consume
            let rhs = self.parse_pow()?; // right associative
            Ok(Expr::Binary(Box::new(node), BinOp::Pow, Box::new(rhs)))
        } else {
            Ok(node)
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut node = self.parse_primary()?;
        loop {
            if self.consume(&Token::LParen) {
                let mut args = Vec::new();
                if !self.consume(&Token::RParen) {
                    args.push(self.parse_expr()?);
                    while self.consume(&Token::Comma) {
                        args.push(self.parse_expr()?);
                    }
                    if !self.consume(&Token::RParen) {
                        return Err("expected ')'".into());
                    }
                }
                node = Expr::Index(Box::new(node), args);
            } else {
                break;
            }
        }
        Ok(node)
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
                Token::Ident => Ok(Expr::Ident(info.lexeme)),
                Token::LParen => {
                    let expr = self.parse_expr()?;
                    if !self.consume(&Token::RParen) {
                        return Err("expected ')'".into());
                    }
                    Ok(expr)
                }
                Token::LBracket => {
                    let matrix = self.parse_matrix()?;
                    if !self.consume(&Token::RBracket) {
                        return Err("expected ']'".into());
                    }
                    Ok(matrix)
                }
                Token::Colon => Ok(Expr::Colon),
                _ => Err("unexpected token".into()),
            },
            None => Err("unexpected end of input".into()),
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
