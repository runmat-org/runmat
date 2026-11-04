pub mod lexical;

mod diagnostic;
mod format;
mod parser;
pub mod syntax;
pub mod tree;

pub use diagnostic::*;
pub use format::*;
pub use parser::*;
pub use syntax::*;

#[cfg(test)]
mod tests;

/// The `.fea` TextMate grammar, generated from the compiler's closed tables by
/// `scripts/docs/fea-grammar.mjs --write` (a docs test keeps this copy current). Docs, site code
/// cards and the language server all color `.fea` from this one grammar; its `vocabulary` key lists
/// the same word classes for hosts that classify lexer tokens.
pub const FEA_TEXTMATE_GRAMMAR: &str = include_str!("../grammar/fea.tmLanguage.json");
