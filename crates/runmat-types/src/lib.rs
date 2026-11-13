pub mod codec;
pub mod convolution;
pub mod contract;
pub mod declaration;
pub mod dimension;
mod exception;
pub use exception::ExceptionProperty;
pub mod fact;
pub mod geometry;
pub mod identity;
pub mod integer_literal;
pub mod operation;
pub mod physical;
mod matrix_work;
pub use matrix_work::MatrixWorkGeometry;
pub mod rules;
pub mod scalar_program;
pub mod real_unary;
pub mod reduction;
pub mod serde_f32;
pub mod serde_f64;
pub mod source;
pub mod version;
pub mod wire_integer;

pub use contract::*;
pub use convolution::ConvolutionMode;
pub use declaration::{
    CallSyntax, CapturedBinding, ClassArrayPayload, ClassArrayShape, ClassDeclaration, ClassKind,
    EnumerationDeclaration, EnvironmentEffect, EventAccess, EventAttributes, EventDeclaration,
    ExternalClassDeclaration, ExternalMethodDeclaration, ExternalPropertyDeclaration, FunctionAbi,
    FunctionArgDefaultValue, FunctionArgDim, FunctionArgRangeInclusivity, FunctionArgSizeSpec,
    FunctionArgValidationLiteral, FunctionArgValidator, FunctionArgumentValidation, FunctionKind,
    FunctionModifiers, InheritanceDeclaration, MemberAccess, MethodAttributes, MethodDeclaration,
    PropertyAttributes, PropertyDeclaration, SemanticAttribute, StringLiteral, WorkspaceEffect,
};
pub use dimension::{decode_shape_dimension, dimension_fits_platform, DimensionValueError};
pub use fact::*;
pub use identity::*;
pub use integer_literal::{
    integer_magnitude_is_exact_binary64, IntegerClass, IntegerLiteral, IntegerLiteralClass,
};
pub use operation::*;
pub use real_unary::{RealUnaryClassification, RealUnaryPredicate};
pub use reduction::{
    FusionReductionContract, FusionReductionCount, FusionReductionEmpty, FusionReductionOperation,
    FusionReductionTies, FusionReductionWorkPrior, ReductionNaN, SumInvocation, SumOutputClass,
};
mod floating_remainder;
pub use floating_remainder::FloatingRemainderKind;
pub use rules::*;
pub use scalar_program::{ScalarOperation, ScalarUnaryMath};
pub use source::{merge_span, Span};
pub use version::{SchemaIdentity, RUNMAT_TYPES_SCHEMA};

pub mod path;
pub use path::{NormalizedRelativePath, RelativePathError};
