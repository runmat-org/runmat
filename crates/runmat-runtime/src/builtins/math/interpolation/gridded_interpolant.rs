//! MATLAB-compatible `griddedInterpolant` builtin.
//!
//! This implements RunMat's gridded interpolant object surface for real dense
//! data. The object stores compact grid vectors and evaluates through `subsref`
//! so MATLAB syntax such as `F(xq)` and `F(xq,yq)` works through the normal
//! object indexing path.

mod admission;
mod constructor;
mod execution;
mod methods;
mod object;
mod query;
#[cfg(test)]
mod tests;

use admission::*;
use constructor::*;
use execution::*;
use methods::*;
use object::*;
use query::*;

use runmat_types::MemberAccess;
use std::collections::HashMap;

use runmat_builtins::{
    BuiltinErrorDescriptor, GRIDDED_INTEGER_GRID_EXTENSION, GRIDDED_INTEGER_QUERY_EXTENSION,
    GRIDDED_INTEGER_VALUES_EXTENSION, GRIDDED_INTERPOLANT_CLASS_IDENTITY,
    GRIDDED_INTERPOLANT_ERROR_INTERNAL as ERROR_INTERNAL,
    GRIDDED_INTERPOLANT_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
    GRIDDED_INTERPOLANT_ERROR_UNSUPPORTED_METHOD as ERROR_UNSUPPORTED_METHOD,
};
#[cfg(test)]
use runmat_builtins::{GRIDDED_INTERPOLANT_EXTENSIONS, GRIDDED_INTERPOLANT_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, IntValue, IntegerStorage, NumericDType, ObjectInstance, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult, RuntimeError, OBJECT_SUBSREF_METHOD};

use super::pp::{
    build_pchip_pp, build_spline_pp, eval_pp_scalar, Extrapolation, NumericSeries,
    PiecewisePolynomial,
};

const BUILTIN_NAME: &str = "griddedInterpolant";
const CLASS_IDENTITY: runmat_types::StaticClassIdentity = GRIDDED_INTERPOLANT_CLASS_IDENTITY;
const BUILTIN_SUBSREF: &str = "griddedInterpolant.subsref";
const GRID_VECTORS: &str = "GridVectors";
const VALUES: &str = "Values";
const METHOD: &str = "Method";
const EXTRAPOLATION_METHOD: &str = "ExtrapolationMethod";

static GRIDDED_INTERPOLANT_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CLASS_IDENTITY);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterpMethod {
    Linear,
    Nearest,
    Next,
    Previous,
    Pchip,
    Spline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtrapolationMethod {
    Linear,
    Nearest,
    Next,
    Previous,
    Pchip,
    Spline,
    None,
}

#[derive(Clone)]
struct InterpolantSpec {
    grid_vectors: Vec<Vec<f64>>,
    values: Tensor,
    method: InterpMethod,
    extrapolation: ExtrapolationMethod,
}

#[derive(Clone)]
struct Bracket {
    lo: usize,
    hi: usize,
    t: f64,
}

#[derive(Clone)]
struct QueryPlan {
    coords: QueryCoords,
    shape: Vec<usize>,
}

#[derive(Clone)]
enum QueryCoords {
    Explicit(Vec<Vec<f64>>),
    FullGrid {
        vectors: Vec<Vec<f64>>,
        total: usize,
    },
}

impl QueryPlan {
    fn len(&self) -> usize {
        match &self.coords {
            QueryCoords::Explicit(points) => points.len(),
            QueryCoords::FullGrid { total, .. } => *total,
        }
    }

    fn for_each_point(&self, mut f: impl FnMut(&[f64])) {
        match &self.coords {
            QueryCoords::Explicit(points) => {
                for point in points {
                    f(point);
                }
            }
            QueryCoords::FullGrid { vectors, total } => {
                for linear in 0..*total {
                    let subs = unravel_column_major(linear, &self.shape);
                    let point = subs
                        .iter()
                        .enumerate()
                        .map(|(dim, &idx)| vectors[dim][idx])
                        .collect::<Vec<_>>();
                    f(&point);
                }
            }
        }
    }
}

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::interpolation::gridded_interpolant"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("gridded-interpolation"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "griddedInterpolant currently stores host tensors and evaluates through the runtime object indexing path.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::gridded_interpolant"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Interpolant object construction and evaluation are runtime sinks.",
};

fn gridded_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    gridded_error(&ERROR_INVALID_INPUT, detail)
}

fn unsupported(detail: impl AsRef<str>) -> RuntimeError {
    gridded_error(&ERROR_UNSUPPORTED_METHOD, detail)
}

fn internal(detail: impl AsRef<str>) -> RuntimeError {
    gridded_error(&ERROR_INTERNAL, detail)
}

fn ensure_gridded_interpolant_class_registered() {
    GRIDDED_INTERPOLANT_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in [GRID_VECTORS, VALUES, METHOD, EXTRAPOLATION_METHOD] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Public,
                    default_value: None,
                },
            );
        }
        let mut methods = HashMap::new();
        methods.insert(
            OBJECT_SUBSREF_METHOD.into(),
            crate::class_registry::RuntimeMethod {
                name: OBJECT_SUBSREF_METHOD.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: BUILTIN_SUBSREF.into(),
                implicit_class_argument: None,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: CLASS_IDENTITY.into(),
            parent: None,
            properties,
            methods,
        });
    });
}

#[runtime_builtin(
    name = "griddedInterpolant",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::interpolation::gridded_interpolant"
)]
async fn gridded_interpolant_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_constructor_integer_extensions(&args)?;
    let args = gather_values(args).await?;
    ensure_gridded_interpolant_class_registered();
    let spec = parse_constructor(args)?;
    Ok(Value::Object(spec_to_object(spec)?))
}

#[runtime_builtin(
    name = "griddedInterpolant.subsref",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::interpolation::gridded_interpolant"
)]
async fn gridded_interpolant_subsref(obj: Value, subscript: Value) -> BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(obj, path, gridded_interpolant_read_step, None)
        .await
}

async fn gridded_interpolant_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<Value> {
    let payload = step.selector_value()?;
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => gridded_member(obj, payload),
        crate::object::indexing::ObjectIndexKind::Paren => {
            let spec = object_to_spec(&obj)?;
            let query_args = payload_to_args(payload)?;
            ensure_query_integer_extensions(&query_args)?;
            let query_args = gather_values(query_args).await?;
            evaluate_interpolant(&spec, query_args)
        }
        crate::object::indexing::ObjectIndexKind::Brace => {
            Err(invalid("brace indexing is not supported"))
        }
    }
}
