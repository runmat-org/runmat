use super::FeaTriviaKind;
use runmat_types::Span;
use serde::{Deserialize, Serialize};

// One lexical declaration owns the enum, recognition, spelling and inventory.
// Contextual authoring names remain identifiers and belong to the language owner.
macro_rules! fea_keywords {
    ($($variant:ident => $source:literal),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum FeaKeyword { $($variant),+ }
        impl FeaKeyword {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
            pub const fn as_source(self) -> &'static str {
                match self { $(Self::$variant => $source),+ }
            }
            pub fn from_source(value: &str) -> Option<Self> {
                match value { $($source => Some(Self::$variant),)+ _ => None }
            }
        }
    };
}

fea_keywords! {
    Import => "import", From => "from", Over => "over", Geometry => "geometry",
    Mesh => "mesh", Material => "material", Model => "model", Study => "study",
    Sweep => "sweep", Execution => "execution", Function => "function", Region => "region",
    Input => "input", Let => "let", Construct => "construct", Feature => "feature",
    Derive => "derive", Generated => "generated", Assign => "assign", To => "to",
    Constraint => "constraint", Load => "load", LoadCase => "load_case", Analysis => "analysis",
    Output => "output", Retain => "retain", Include => "include", On => "on",
    Select => "select", Expect => "expect", Result => "result", Case => "case",
    Cases => "cases", Axis => "axis", Transition => "transition", Initial => "initial",
    Interface => "interface", Connector => "connector", Section => "section", Layup => "layup",
    Orientation => "orientation", Signal => "signal", Table => "table", Excitation => "excitation",
    ResponseSpectrum => "response_spectrum", RandomExcitation => "random_excitation",
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
    /// `±`: a symmetric tolerance (design dialects).
    PlusMinus,
    /// `@`: a pin on a cross-partition `use` or published value.
    At,
    /// `%`: the percent unit.
    Percent,
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
