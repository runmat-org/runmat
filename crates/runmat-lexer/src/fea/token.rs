use super::FeaTriviaKind;
use runmat_types::Span;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeaKeyword {
    Import,
    From,
    Geometry,
    Mesh,
    Material,
    Model,
    Study,
    Sweep,
    Execution,
    Function,
    Region,
    Input,
    Construct,
    Feature,
    Derive,
    Generated,
    Assign,
    To,
    Constraint,
    Load,
    LoadCase,
    Analysis,
    Output,
    Retain,
    Include,
    On,
    Select,
    Expect,
    Result,
    Case,
    Cases,
    Axis,
    Transition,
    Initial,
    Interface,
    Section,
    Layup,
    Orientation,
    Signal,
    Excitation,
}

impl FeaKeyword {
    pub fn from_source(value: &str) -> Option<Self> {
        Some(match value {
            "import" => Self::Import,
            "from" => Self::From,
            "geometry" => Self::Geometry,
            "mesh" => Self::Mesh,
            "material" => Self::Material,
            "model" => Self::Model,
            "study" => Self::Study,
            "sweep" => Self::Sweep,
            "execution" => Self::Execution,
            "function" => Self::Function,
            "region" => Self::Region,
            "input" => Self::Input,
            "construct" => Self::Construct,
            "feature" => Self::Feature,
            "derive" => Self::Derive,
            "generated" => Self::Generated,
            "assign" => Self::Assign,
            "to" => Self::To,
            "constraint" => Self::Constraint,
            "load" => Self::Load,
            "load_case" => Self::LoadCase,
            "analysis" => Self::Analysis,
            "output" => Self::Output,
            "retain" => Self::Retain,
            "include" => Self::Include,
            "on" => Self::On,
            "select" => Self::Select,
            "expect" => Self::Expect,
            "result" => Self::Result,
            "case" => Self::Case,
            "cases" => Self::Cases,
            "axis" => Self::Axis,
            "transition" => Self::Transition,
            "initial" => Self::Initial,
            "interface" => Self::Interface,
            "section" => Self::Section,
            "layup" => Self::Layup,
            "orientation" => Self::Orientation,
            "signal" => Self::Signal,
            "excitation" => Self::Excitation,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeaTokenKind {
    Keyword(FeaKeyword),
    Identifier,
    Number,
    String,
    Trivia(FeaTriviaKind),
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    LeftParen,
    RightParen,
    Colon,
    Comma,
    Dot,
    Range,
    Equal,
    EqualEqual,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Semicolon,
    Invalid,
    EndOfFile,
}

impl FeaTokenKind {
    pub fn is_trivia(self) -> bool {
        matches!(self, Self::Trivia(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaToken {
    pub kind: FeaTokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaTokenStream {
    pub source: String,
    pub tokens: Vec<FeaToken>,
    pub diagnostics: Vec<super::FeaLexDiagnostic>,
}

impl FeaTokenStream {
    pub fn lexeme(&self, token: FeaToken) -> &str {
        &self.source[token.span.start..token.span.end]
    }

    pub fn significant_tokens(&self) -> impl Iterator<Item = FeaToken> + '_ {
        self.tokens
            .iter()
            .copied()
            .filter(|token| !token.kind.is_trivia())
    }
}
