use crate::callbacks::{
    block_comment_skip, double_quoted_string_emit, ellipsis_emit_and_skip_to_eol,
    line_comment_start, newline_skip, section_marker,
};
use crate::extras::LexerExtras;
use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
// Skip spaces, tabs and carriage returns, but NOT newlines; we need newlines to detect '%%' at line start
#[logos(skip r"[ \t\r]+")]
#[logos(extras = LexerExtras)]
pub enum Token {
    // Keywords
    #[token("function")]
    Function,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("elseif")]
    ElseIf,
    #[token("for")]
    For,
    #[token("while")]
    While,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,
    #[token("end")]
    End,

    // Object-oriented and function syntax keywords
    #[token("classdef")]
    ClassDef,
    #[token("properties")]
    Properties,
    #[token("methods")]
    Methods,
    #[token("events")]
    Events,
    #[token("enumeration")]
    Enumeration,
    #[token("arguments")]
    Arguments,

    // Importing packages/classes
    #[token("import")]
    Import,

    // Additional keywords (recognized by lexer; parser may treat as identifiers for now)
    #[token("switch")]
    Switch,
    #[token("case")]
    Case,
    #[token("otherwise")]
    Otherwise,
    #[token("try")]
    Try,
    #[token("catch")]
    Catch,
    #[token("global")]
    Global,
    #[token("persistent")]
    Persistent,
    #[token("true", |lex| { lex.extras.last_was_value = true; })]
    True,
    #[token("false", |lex| { lex.extras.last_was_value = true; })]
    False,

    // Identifiers and literals
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| { lex.extras.last_was_value = true; })]
    Ident,
    // Float with optional underscores as digit separators (strip later)
    #[regex(r"\d(?:_?\d)*\.(?:\d(?:_?\d)*)?(?:[eE][+-]?\d(?:_?\d)*)?", |lex| {
        lex.extras.last_was_value = true;
    })]
    #[regex(r"\d(?:_?\d)*[eE][+-]?\d(?:_?\d)*", |lex| {
        lex.extras.last_was_value = true;
    })]
    Float,
    // Integer with optional underscores as digit separators (strip later)
    #[regex(r"\d(?:_?\d)*", |lex| {
        lex.extras.last_was_value = true;
    })]
    Integer,
    // Apostrophe is handled contextually in tokenize_detailed: either Transpose or a single-quoted string
    #[token("'")]
    Apostrophe,
    // Double-quoted string scalar (treated as Str at lexer level). Always emit.
    #[regex(r#""([^"\n\r]|"")*""#, double_quoted_string_emit, priority = 1)]
    Str,
    #[token("...", ellipsis_emit_and_skip_to_eol)]
    Ellipsis,
    // Section marker: must be at start of line (after optional whitespace). We match until EOL and emit a single token.
    #[regex(r"%%[^\n]*", section_marker, priority = 3)]
    Section,
    #[token(".*")]
    DotStar,
    #[token("./")]
    DotSlash,
    #[token(".\\")]
    DotBackslash,
    #[token(".^")]
    DotCaret,
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("==")]
    Equal,
    #[token("~=")]
    NotEqual,
    #[token("<=")]
    LessEqual,
    #[token(">=")]
    GreaterEqual,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("\\")]
    Backslash,
    #[token("^")]
    Caret,
    #[token("&")]
    And,
    #[token("|")]
    Or,
    #[token("~")]
    Tilde,
    #[token("@")]
    At,
    // Meta-class (type) query operator: ?ClassName
    #[token("?")]
    Question,
    #[token("<")]
    Less,
    #[token(">")]
    Greater,
    #[token("=", |lex| { lex.extras.last_was_value = false; })]
    Assign,
    #[token(".")]
    Dot,
    // Semicolon ends a statement; next token should not be treated as a value.
    // This helps disambiguate that a following apostrophe starts a string, not a transpose.
    #[token(";", |lex| { lex.extras.last_was_value = false; })]
    Semicolon,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("(", |lex| { lex.extras.last_was_value = false; })]
    LParen,
    #[token(")", |lex| { lex.extras.last_was_value = true; })]
    RParen,
    #[token("[", |lex| { lex.extras.last_was_value = false; })]
    LBracket,
    #[token("]", |lex| { lex.extras.last_was_value = true; })]
    RBracket,
    #[token("{", |lex| { lex.extras.last_was_value = false; })]
    LBrace,
    #[token("}", |lex| { lex.extras.last_was_value = true; })]
    RBrace,

    // Newlines are skipped but set line_start for '%%' detection
    #[regex(r"\n+", newline_skip)]
    Newline,

    // Block comments: '%{' ... '%}' (non-nesting). Skipped entirely.
    #[regex(r"%\{", block_comment_skip, priority = 2)]
    BlockComment,

    // Line comments: single '%' handled here; '%%' and '%{' are matched by other rules first
    #[token("%", line_comment_start, priority = 0)]
    LineComment,

    Error,
    // Synthetic tokens (not produced by Logos directly)
    Transpose,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    pub token: Token,
    pub lexeme: String,
    pub start: usize,
    pub end: usize,
}
