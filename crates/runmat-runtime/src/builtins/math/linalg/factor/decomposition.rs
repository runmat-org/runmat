//! MATLAB-compatible `decomposition` objects for reusable linear solves.
use runmat_types::MemberAccess;

use std::collections::HashMap;

use num_complex::Complex64;
use runmat_builtins::*;
use runmat_builtins::{
    DECOMPOSITION_ERROR_INTERNAL as ERROR_INTERNAL,
    DECOMPOSITION_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexElement, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray,
    NumericDType, ObjectInstance, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::builtins::math::linalg::ops::matrix_arithmetic::solve::{
    self as shared_solve, SolveOrientation,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "decomposition";
const CLASS_IDENTITY: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("decomposition");

const MATRIX_FIELD: &str = "__matrix";
const TYPE_FIELD: &str = "Type";
const MATRIX_SIZE_FIELD: &str = "MatrixSize";
const CHECK_CONDITION_FIELD: &str = "CheckCondition";
const DATATYPE_FIELD: &str = "Datatype";
const IS_CONJUGATE_TRANSPOSED_FIELD: &str = "IsConjugateTransposed";
const IS_REAL_FIELD: &str = "IsReal";
const IS_SPARSE_FIELD: &str = "IsSparse";
const SCALE_FACTOR_FIELD: &str = "ScaleFactor";
const RANK_TOLERANCE_FIELD: &str = "RankTolerance";

static DECOMPOSITION_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CLASS_IDENTITY);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecompositionType {
    Auto,
    Lu,
    Qr,
    Chol,
    Triangular,
    Diagonal,
}

#[derive(Clone, Debug)]
struct DecompositionSpec {
    matrix: Value,
    decomp_type: DecompositionType,
    check_condition: bool,
    rank_tolerance: Option<f64>,
    is_conjugate_transposed: bool,
    is_sparse: bool,
    scale: (f64, f64),
}

fn validate_output_count() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(invalid("decomposition: expected at most one output"));
    }
    Ok(())
}

fn decomposition_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    decomposition_error(&ERROR_INVALID_INPUT, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    decomposition_error(&ERROR_INTERNAL, message)
}

fn ensure_decomposition_class_registered() {
    DECOMPOSITION_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in [
            MATRIX_SIZE_FIELD,
            TYPE_FIELD,
            CHECK_CONDITION_FIELD,
            DATATYPE_FIELD,
            IS_CONJUGATE_TRANSPOSED_FIELD,
            IS_REAL_FIELD,
            IS_SPARSE_FIELD,
            SCALE_FACTOR_FIELD,
        ] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            );
        }
        properties.insert(
            MATRIX_FIELD.into(),
            crate::class_registry::RuntimeProperty {
                name: MATRIX_FIELD.into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Private,
                set_access: MemberAccess::Private,
                default_value: None,
            },
        );

        let mut methods = HashMap::new();
        for method_name in [
            runmat_types::StaticMethodName::new("subsref"),
            runmat_types::StaticMethodName::new("uplus"),
            runmat_types::StaticMethodName::new("uminus"),
            runmat_types::StaticMethodName::new("ctranspose"),
            runmat_types::StaticMethodName::new("times"),
            runmat_types::StaticMethodName::new("mtimes"),
            runmat_types::StaticMethodName::new("rdivide"),
            runmat_types::StaticMethodName::new("mrdivide"),
            runmat_types::StaticMethodName::new("mldivide"),
        ] {
            methods.insert(
                method_name.into(),
                crate::class_registry::RuntimeMethod {
                    name: method_name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("{CLASS_IDENTITY}.{method_name}"),
                    implicit_class_argument: None,
                },
            );
        }

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: CLASS_IDENTITY.into(),
            parent: None,
            properties,
            methods,
        });
    });
}

#[runtime_builtin(
    name = "decomposition",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    validate_output_count()?;
    if args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECOMPOSITION_GPU_INPUT_EXTENSION,
            NAME,
        )?;
    }
    if has_integer_check_condition(&args) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECOMPOSITION_INTEGER_CHECK_CONDITION_EXTENSION,
            NAME,
        )?;
    }
    if has_integer_named_option(&args, "RankTolerance") {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECOMPOSITION_INTEGER_RANK_TOLERANCE_EXTENSION,
            NAME,
        )?;
    }
    if args.first().is_some_and(is_nonfloating_coefficient_matrix) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECOMPOSITION_NONFLOATING_INPUT_EXTENSION,
            NAME,
        )?;
    }
    ensure_decomposition_class_registered();
    let spec = parse_constructor(args).await?;
    Ok(Value::Object(spec_to_object(&spec)?))
}

fn has_integer_check_condition(args: &[Value]) -> bool {
    has_integer_named_option(args, "CheckCondition")
}

fn has_integer_named_option(args: &[Value], option: &str) -> bool {
    args.windows(2).any(|pair| {
        text_from_value(&pair[0]).is_some_and(|name| name.eq_ignore_ascii_case(option))
            && is_integer_value(&pair[1])
    })
}

fn is_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_nonfloating_coefficient_matrix(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => !matches!(
            tensor.numeric_dtype(),
            NumericDType::F64 | NumericDType::F32
        ),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
        }
        _ => false,
    }
}

#[runtime_builtin(
    name = "decomposition.subsref",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_subsref(obj: Value, subscript: Value) -> BuiltinResult<Value> {
    validate_output_count()?;
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(obj, path, decomposition_read_step, None).await
}

async fn decomposition_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<Value> {
    if step.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(invalid(
            "decomposition objects only support property indexing",
        ));
    }
    let field = text_from_value(&step.selector_value()?)
        .ok_or_else(|| invalid("decomposition property name must be a text scalar"))?;
    let spec = object_to_spec(&obj)?;
    ensure_nonfloating_object_enabled(&spec)?;
    let object = expect_object(&obj)?;
    public_property_value(object, &field)
}

#[runtime_builtin(
    name = "decomposition.uplus",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_uplus(d_a: Value) -> BuiltinResult<Value> {
    validate_output_count()?;
    let spec = object_to_spec(&d_a)?;
    ensure_nonfloating_object_enabled(&spec)?;
    Ok(Value::Object(spec_to_object(&spec)?))
}

#[runtime_builtin(
    name = "decomposition.uminus",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_uminus(d_a: Value) -> BuiltinResult<Value> {
    validate_output_count()?;
    let mut spec = object_to_spec(&d_a)?;
    ensure_nonfloating_object_enabled(&spec)?;
    spec.scale = complex_mul(spec.scale, (-1.0, 0.0));
    Ok(Value::Object(spec_to_object(&spec)?))
}

#[runtime_builtin(
    name = "decomposition.ctranspose",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_ctranspose(d_a: Value) -> BuiltinResult<Value> {
    validate_output_count()?;
    let mut spec = object_to_spec(&d_a)?;
    ensure_nonfloating_object_enabled(&spec)?;
    spec.is_conjugate_transposed = !spec.is_conjugate_transposed;
    spec.scale = complex_conj(spec.scale);
    Ok(Value::Object(spec_to_object(&spec)?))
}

#[runtime_builtin(
    name = "decomposition.times",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_times(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    validate_output_count()?;
    scale_product(lhs, rhs).await
}

#[runtime_builtin(
    name = "decomposition.mtimes",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_mtimes(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    validate_output_count()?;
    scale_product(lhs, rhs).await
}

#[runtime_builtin(
    name = "decomposition.rdivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_rdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    validate_output_count()?;
    divide_or_solve(lhs, rhs).await
}

#[runtime_builtin(
    name = "decomposition.mrdivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_mrdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    validate_output_count()?;
    divide_or_solve(lhs, rhs).await
}

#[runtime_builtin(
    name = "decomposition.mldivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_mldivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    validate_output_count()?;
    left_solve(lhs, rhs).await
}

#[path = "decomposition/constructor.rs"]
mod constructor;
use constructor::*;

#[path = "decomposition/structure.rs"]
mod structure;
use structure::*;

#[path = "decomposition/object.rs"]
mod object;
use object::*;

#[path = "decomposition/solve.rs"]
mod solve;
use solve::*;

#[path = "decomposition/transform.rs"]
mod transform;
use transform::*;

#[path = "decomposition/scalar.rs"]
mod scalar;
use scalar::*;

#[cfg(test)]
#[path = "decomposition/tests/mod.rs"]
mod tests;

#[path = "decomposition/triangle.rs"]
mod triangle;
use triangle::*;
