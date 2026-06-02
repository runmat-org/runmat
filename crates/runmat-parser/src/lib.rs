mod ast;
mod error;
mod options;
mod parser;
mod span;

pub use ast::{
    Attr, BinOp, ClassMember, Expr, FunctionArgDim, FunctionArgSizeSpec, FunctionArgValidationDecl,
    FunctionArgValidatorDecl, FunctionArgumentsBlockKind, LValue, MultiAssignTarget, Program, Stmt,
    UnOp,
};
pub use error::SyntaxError;
pub use options::{CompatMode, ParserOptions};
pub use parser::{parse, parse_with_options};
pub use span::Span;
