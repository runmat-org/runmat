mod declaration;
mod expression;
mod module;
mod statement;

pub use declaration::*;
pub use expression::*;
pub use module::*;
pub use statement::*;

use runmat_types::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeaTokenRange {
    pub start: usize,
    pub end: usize,
    pub span: Span,
}

impl FeaTokenRange {
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaName {
    pub text: String,
    pub span: Span,
}
