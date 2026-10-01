pub mod codec;
pub mod contract;
pub mod declaration;
pub mod dimension;
pub mod fact;
pub mod geometry;
pub mod identity;
pub mod integer_literal;
pub mod rules;
pub mod real_unary;
pub mod reduction;
pub mod serde_f64;
pub mod source;
pub mod symbolic;
pub mod version;

pub use contract::*;
pub use declaration::{
    ClassArrayPayload, ClassArrayShape, ClassDeclaration, ClassKind, EnumerationDeclaration,
    EventDeclaration, ExternalClassDeclaration, ExternalMethodDeclaration,
    ExternalPropertyDeclaration, FunctionArgDefaultValue, FunctionArgDim,
    FunctionArgRangeInclusivity, FunctionArgSizeSpec, FunctionArgValidationLiteral,
    FunctionArgValidator, InheritanceDeclaration, MemberAccess, MethodAttributes,
    MethodDeclaration, PropertyAttributes, PropertyDeclaration, SemanticAttribute,
};
pub use dimension::{decode_shape_dimension, dimension_fits_platform, DimensionValueError};
pub use fact::*;
pub use identity::*;
pub use integer_literal::{IntegerClass, IntegerLiteral, IntegerLiteralClass};
pub use real_unary::{RealUnaryClassification, RealUnaryPredicate};
pub use reduction::{ReductionNaN, SumOutputClass};
mod floating_remainder;
pub use floating_remainder::FloatingRemainderKind;
pub use rules::*;
pub use source::{merge_span, Span};
pub use symbolic::{
    parse_symbolic_declaration, symbolic_declaration_tokens, SymbolicDeclaration,
    SymbolicDeclarationError,
};
pub use version::{SchemaIdentity, RUNMAT_TYPES_SCHEMA};
