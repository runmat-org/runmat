//! MATLAB-compatible `spline` builtin backed by piecewise-polynomial structs.

use runmat_builtins::{ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{
    build_spline_pp, evaluate_pp, interp_error, pp_to_value, query_points, series_from_values,
    Extrapolation,
};

const NAME: &str = "spline";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::spline")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("cubic-spline"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Spline coefficient construction and evaluation currently run on the CPU reference path after gathering GPU inputs.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::spline"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Spline builds a piecewise-polynomial representation and is not fused.",
};

fn spline_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.len() {
        0 | 1 => Type::Unknown,
        2 => Type::Struct { known_fields: None },
        _ => match args.get(2) {
            Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
            Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
                shape: shape.clone(),
            },
            _ => Type::tensor(),
        },
    }
}

#[runtime_builtin(
    name = "spline",
    category = "math/interpolation",
    summary = "Cubic spline interpolation and piecewise-polynomial construction.",
    keywords = "spline,cubic interpolation,pp,ppval",
    accel = "sink",
    sink = true,
    type_resolver(spline_type),
    builtin_path = "crate::builtins::math::interpolation::spline"
)]
async fn spline_builtin(x: Value, y: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let series = series_from_values(x, y, NAME).await?;
    let pp = build_spline_pp(&series, NAME)?;
    match rest.len() {
        0 => pp_to_value(pp, NAME),
        1 => {
            let query = query_points(rest.into_iter().next().expect("query"), NAME).await?;
            evaluate_pp(&pp, &query, &Extrapolation::Extrapolate, NAME)
        }
        _ => Err(interp_error(NAME, "spline: too many input arguments")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{StructValue, Tensor};

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    #[test]
    fn spline_two_arg_returns_pp_struct() {
        let value = block_on(spline_builtin(
            row(&[1.0, 2.0, 3.0]),
            row(&[1.0, 4.0, 9.0]),
            vec![],
        ))
        .expect("spline");
        let Value::Struct(StructValue { fields }) = value else {
            panic!("expected struct");
        };
        assert!(fields.contains_key("breaks"));
        assert!(fields.contains_key("coefs"));
    }

    #[test]
    fn spline_three_arg_evaluates() {
        let value = block_on(spline_builtin(
            row(&[1.0, 2.0, 3.0]),
            row(&[1.0, 4.0, 9.0]),
            vec![Value::Num(1.5)],
        ))
        .expect("spline");
        let Value::Num(result) = value else {
            panic!("expected scalar");
        };
        assert!((result - 2.25).abs() < 1e-10);
    }
}
