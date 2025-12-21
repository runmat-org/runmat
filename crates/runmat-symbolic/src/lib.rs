//! RunMat Symbolic Mathematics Engine
//!
//! This crate provides symbolic computation capabilities for RunMat,
//! enabling MATLAB-compatible symbolic math operations.
//!
//! # Architecture
//!
//! The symbolic engine uses a tree-based expression representation with:
//! - Compact coefficient representation (rational + float fallback)
//! - Symbol table for variable interning
//! - Staged normalization pipeline
//! - Bytecode compilation for numeric evaluation
//!
//! # Design Principles
//!
//! 1. **RunMat-native**: Integrates with RunMat's IR and JIT
//! 2. **MATLAB-compatible**: Matches MATLAB Symbolic Toolbox behavior
//! 3. **Performance**: Structural interning, caching, normalization
//! 4. **Extensible**: Pluggable simplification rules

mod coeff;
mod compiler;
mod expr;
mod normalize;
mod symbol;

pub use coeff::Coefficient;
pub use compiler::{compile, compile_with_vars, BytecodeCompiler, BytecodeOp, CompiledExpr};
pub use expr::{SymExpr, SymExprKind};
pub use normalize::{NormPass, NormProof, StagedNormalizer};
pub use symbol::{Symbol, SymbolAttrs, SymbolId, SymbolTable};

/// Error type for symbolic operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum SymbolicError {
    #[error("undefined symbol: {0}")]
    UndefinedSymbol(String),

    #[error("invalid operation: {0}")]
    InvalidOperation(String),

    #[error("division by zero")]
    DivisionByZero,

    #[error("numeric overflow")]
    NumericOverflow,
}

pub type Result<T> = std::result::Result<T, SymbolicError>;
