use logos::Logos;
use rustmat_lexer::Token;

#[derive(Debug, PartialEq)]
pub enum Expr {
    Number(String),
    Ident(String),
    Binary(Box<Expr>, BinOp, Box<Expr>),
}

#[derive(Debug, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, PartialEq)]
pub enum Stmt {
    ExprStmt(Expr),
    Assign(String, Expr),
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
            body.push(self.parse_stmt()?);
            if self.consume(&Token::Semicolon) {
                // continue
            } else {
                break;
            }
        }
        Ok(Program { body })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        if self.peek_token() == Some(&Token::Ident) && self.peek_token_at(1) == Some(&Token::Assign)
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

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_add_sub()
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
        let mut node = self.parse_primary()?;
        loop {
            let op = match self.peek_token() {
                Some(Token::Star) => BinOp::Mul,
                Some(Token::Slash) => BinOp::Div,
                _ => break,
            };
            self.pos += 1; // consume op
            let rhs = self.parse_primary()?;
            node = Expr::Binary(Box::new(node), op, Box::new(rhs));
        }
        Ok(node)
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
                _ => Err("unexpected token".into()),
            },
            None => Err("unexpected end of input".into()),
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
