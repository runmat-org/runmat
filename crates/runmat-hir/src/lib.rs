pub use runmat_builtins::Type;

mod diagnostic;
mod error;
mod hir;
mod ids;
pub(crate) mod inference;
pub(crate) mod lowering;
mod lowering_context;
pub mod remapping;
mod span;
mod validation;

pub use diagnostic::{HirDiagnostic, HirDiagnosticSeverity};
pub use error::{set_error_namespace, SemanticError};
pub use hir::{
    HirClassMember, HirExpr, HirExprKind, HirLValue, HirProgram, HirStmt, LoweringResult,
};
pub use ids::{SourceId, VarId};
pub use inference::expr::infer_expr_type_with_env;
pub use inference::function_outputs::infer_function_output_types;
pub use inference::function_vars::infer_function_variable_types;
pub use inference::globals::infer_global_variable_types;
pub use inference::shared::eval_const_num;
pub use lowering::lower;
pub use lowering_context::LoweringContext;
pub use span::{merge_span, Span};
pub use validation::classdefs::validate_classdefs;
pub use validation::imports::{
    collect_imports, normalize_imports, validate_imports, NormalizedImport,
};
