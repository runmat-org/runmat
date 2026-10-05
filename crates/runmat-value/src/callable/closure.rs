use crate::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    pub function_name: String,
    pub bound_function: Option<usize>,
    /// Compiler/source-owned classification; display spelling is never authority.
    pub source_role: runmat_types::FunctionSourceRole,
    /// Compiler-registered origin retained with the callable across sessions.
    pub source_file: Option<String>,
    pub captures: Vec<Value>,
}
