//! MATLAB-compatible `decomposition` objects for reusable linear solves.
use runmat_types::MemberAccess;

use std::collections::HashMap;

use num_complex::Complex64;
use runmat_builtins::*;
use runmat_builtins::{
    DecompositionKind as DecompositionType, DecompositionOption,
    DecompositionTriangle as TriangularFlag,
};
use runmat_builtins::{
    DECOMPOSITION_ERROR_INTERNAL as ERROR_INTERNAL,
    DECOMPOSITION_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::IntegerComplexStorage;
use runmat_value::{
    ComplexElement, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType,
    ObjectInstance, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "decomposition";
pub(crate) const DECOMPOSITION_CLASS_IDENTITY: runmat_types::StaticClassIdentity =
    runmat_types::standard::DECOMPOSITION;
const CLASS_IDENTITY: runmat_types::StaticClassIdentity = DECOMPOSITION_CLASS_IDENTITY;

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
const TRIANGLE_FIELD: &str = "__triangle";
const SOURCE_REAL_FIELD: &str = "__source_is_real";

static DECOMPOSITION_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CLASS_IDENTITY);

#[derive(Clone, Debug)]
struct DecompositionSpec {
    matrix: Value,
    decomp_type: DecompositionType,
    check_condition: bool,
    rank_tolerance: Option<f64>,
    is_conjugate_transposed: bool,
    is_sparse: bool,
    scale: (f64, f64),
    factor_data: Option<runmat_value::NumericFactorData>,
    triangle: Option<TriangularFlag>,
    source_is_real: Option<bool>,
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
        for field in [
            MATRIX_FIELD,
            TRIANGLE_FIELD,
            SOURCE_REAL_FIELD,
            RANK_TOLERANCE_FIELD,
        ] {
            properties.insert(
                field.into(),
                crate::class_registry::RuntimeProperty {
                    name: field.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Private,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            );
        }

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
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            );
        }

        crate::class_registry::register_builtin_class(
            CLASS_IDENTITY,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: CLASS_IDENTITY.into(),
                parent: None,
                properties,
                methods,
            },
        );
    });
}

#[runtime_builtin(
    name = "decomposition",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_builtin_context(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
    if has_integer_check_condition(args.view())? {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECOMPOSITION_INTEGER_CHECK_CONDITION_EXTENSION,
            NAME,
        )?;
    }
    if has_integer_named_option(args.view(), "RankTolerance")? {
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
    let runtime = native::selected_runtime(runtime)?;
    async {
        let spec = parse_constructor(args, Some(&runtime)).await?;
        native::publish_value(&spec, &runtime)
    }
    .await
    .map_err(native::terminal)
}

#[cfg(test)]
async fn decomposition_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    decomposition_builtin_context(None, args).await
}

fn has_integer_check_condition(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<bool> {
    has_integer_named_option(args, "CheckCondition")
}

fn has_integer_named_option(
    args: crate::call::FunctionArgumentView<'_>,
    option: &str,
) -> BuiltinResult<bool> {
    for index in 0..args.len().saturating_sub(1) {
        if text_from_value(args.get(index).expect("bounded option index"))?
            .is_some_and(|name| name.eq_ignore_ascii_case(option))
            && is_integer_value(args.get(index + 1).expect("bounded option value index"))
        {
            return Ok(true);
        }
    }
    Ok(false)
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
        Value::SparseTensor(tensor) => {
            tensor.is_logical()
                || !matches!(
                    tensor.numeric_dtype(),
                    Some(NumericDType::F32 | NumericDType::F64)
                )
        }
        _ => false,
    }
}

#[runtime_builtin(
    name = "decomposition.subsref",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_subsref_context(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    validate_output_count()?;
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let runtime = native::selected_runtime(runtime)?;
    crate::object::protocol::execute_owned_subsref_in_context(
        &runtime,
        obj,
        path,
        decomposition_read_step,
        None,
    )
    .await
    .map_err(native::terminal)
}

async fn decomposition_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<crate::execution::RootedValue> {
    if step.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(invalid(
            "decomposition objects only support property indexing",
        ));
    }
    let field = text_from_value(&step.selector_value()?)?
        .ok_or_else(|| invalid("decomposition property name must be a text scalar"))?;
    let spec = object_to_spec(&obj)?;
    ensure_nonfloating_object_enabled(&spec)?;
    let object = expect_object(&obj)?;
    let value = public_property_value(object, &field)?;
    crate::execution::RootedValue::new(value).map_err(|error| error.into_runtime_error())
}

#[runtime_builtin(
    name = "decomposition.uplus",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_uplus_context(
    runtime: Option<crate::context::RuntimeContext>,
    d_a: Value,
) -> BuiltinResult<Value> {
    validate_output_count()?;
    let spec = object_to_spec(&d_a)?;
    ensure_nonfloating_object_enabled(&spec)?;
    let runtime = native::selected_runtime(runtime)?;
    native::publish_value(&spec, &runtime).map_err(native::terminal)
}

#[runtime_builtin(
    name = "decomposition.uminus",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_uminus_context(
    runtime: Option<crate::context::RuntimeContext>,
    d_a: Value,
) -> BuiltinResult<Value> {
    validate_output_count()?;
    let mut spec = object_to_spec(&d_a)?;
    ensure_nonfloating_object_enabled(&spec)?;
    spec.scale = complex_mul(spec.scale, (-1.0, 0.0));
    let runtime = native::selected_runtime(runtime)?;
    native::publish_value(&spec, &runtime).map_err(native::terminal)
}

#[runtime_builtin(
    name = "decomposition.ctranspose",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_ctranspose_context(
    runtime: Option<crate::context::RuntimeContext>,
    d_a: Value,
) -> BuiltinResult<Value> {
    validate_output_count()?;
    let mut spec = object_to_spec(&d_a)?;
    ensure_nonfloating_object_enabled(&spec)?;
    spec.is_conjugate_transposed = !spec.is_conjugate_transposed;
    spec.scale = complex_conj(spec.scale);
    let runtime = native::selected_runtime(runtime)?;
    native::publish_value(&spec, &runtime).map_err(native::terminal)
}

#[runtime_builtin(
    name = "decomposition.times",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_times_context(
    runtime: Option<crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    validate_output_count()?;
    let runtime = native::selected_runtime(runtime)?;
    scale_product(lhs, rhs, &runtime)
        .await
        .map_err(native::terminal)
}

#[runtime_builtin(
    name = "decomposition.mtimes",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_mtimes_context(
    runtime: Option<crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    validate_output_count()?;
    let runtime = native::selected_runtime(runtime)?;
    scale_product(lhs, rhs, &runtime)
        .await
        .map_err(native::terminal)
}

#[runtime_builtin(
    name = "decomposition.rdivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_rdivide_context(
    runtime: Option<crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    validate_output_count()?;
    let runtime = native::selected_runtime(runtime)?;
    divide_or_solve(lhs, rhs, &runtime)
        .await
        .map_err(native::terminal)
}

#[runtime_builtin(
    name = "decomposition.mrdivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_mrdivide_context(
    runtime: Option<crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    validate_output_count()?;
    let runtime = native::selected_runtime(runtime)?;
    divide_or_solve(lhs, rhs, &runtime)
        .await
        .map_err(native::terminal)
}

#[runtime_builtin(
    name = "decomposition.mldivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::decomposition",
    context = true
)]
async fn decomposition_mldivide_context(
    runtime: Option<crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    validate_output_count()?;
    let runtime = native::selected_runtime(runtime)?;
    left_solve(lhs, rhs, &runtime)
        .await
        .map_err(native::terminal)
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

#[path = "decomposition/persistence.rs"]
mod persistence;
pub(crate) use persistence::{
    decomposition_persistence_kind, validate_factor_object_binding, DecompositionPersistenceKind,
};

#[path = "decomposition/initialize.rs"]
mod initialize;
#[path = "decomposition/native.rs"]
mod native;
#[path = "decomposition/publication.rs"]
mod publication;
pub(crate) use initialize::restore_decomposition_factor_object;

#[path = "decomposition/solve.rs"]
mod solve;
use solve::*;

#[path = "decomposition/retained_solve.rs"]
mod retained_solve;

#[path = "decomposition/transform.rs"]
mod transform;
use transform::*;

#[path = "decomposition/scalar.rs"]
mod scalar;
use scalar::*;

#[cfg(test)]
#[path = "decomposition/tests/mod.rs"]
mod tests;

// Existing direct unit fixtures enter the same selected bodies under their
// explicit ordinary policy; production registrations only use *_context.
#[cfg(test)]
macro_rules! ordinary_unary_fixture {
    ($name:ident,$selected:ident) => {
        async fn $name(value: Value) -> BuiltinResult<Value> {
            $selected(None, value).await
        }
    };
}
#[cfg(test)]
macro_rules! ordinary_binary_fixture {
    ($name:ident,$selected:ident) => {
        async fn $name(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
            $selected(None, lhs, rhs).await
        }
    };
}
#[cfg(test)]
ordinary_unary_fixture!(decomposition_uplus, decomposition_uplus_context);
#[cfg(test)]
ordinary_unary_fixture!(decomposition_uminus, decomposition_uminus_context);
#[cfg(test)]
ordinary_unary_fixture!(decomposition_ctranspose, decomposition_ctranspose_context);
#[cfg(test)]
ordinary_binary_fixture!(decomposition_times, decomposition_times_context);
#[cfg(test)]
ordinary_binary_fixture!(decomposition_mtimes, decomposition_mtimes_context);
#[cfg(test)]
ordinary_binary_fixture!(decomposition_rdivide, decomposition_rdivide_context);
#[cfg(test)]
ordinary_binary_fixture!(decomposition_mrdivide, decomposition_mrdivide_context);
#[cfg(test)]
ordinary_binary_fixture!(decomposition_mldivide, decomposition_mldivide_context);
#[cfg(test)]
async fn decomposition_subsref(
    obj: Value,
    subscript: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    decomposition_subsref_context(None, obj, subscript).await
}

#[path = "decomposition/triangle.rs"]
#[cfg(test)]
mod triangle;
#[cfg(test)]
use triangle::*;
