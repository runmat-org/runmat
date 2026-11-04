//! The lossless syntax tree: every byte of a source file, trivia included, in a green/red tree.
//!
//! - [`GreenNode`] / [`GreenToken`]: immutable, shareable, position-free (`Send + Sync`).
//! - [`SyntaxNode`] / [`SyntaxToken`]: positioned cursors with parents, built on demand.
//!
//! The parser records the tree as it parses ([`TreeRecorder`]); the `.fea` AST and the
//! design-dialect AST are typed views over the same constructs, each AST node at the span of a
//! tree node. Printing the tree's text reproduces the source exactly.

mod build;
mod green;
mod kind;
mod red;

pub(crate) use build::TreeRecorder;
pub use green::{GreenElement, GreenNode, GreenToken};
pub use kind::{NodeKind, TokenKind};
pub use red::{SyntaxElement, SyntaxNode, SyntaxToken};
