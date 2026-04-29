//! Evaluate piecewise-polynomial structs produced by `spline` and `pchip`.

use runmat_builtins::{ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{evaluate_pp, pp_from_value, query_points, Extrapolation};

const NAME: &str = "ppval";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::ppval")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("piecewise-polynomial-eval"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Initial implementation evaluates pp structs on the CPU after gathering GPU query points.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::interpolation::ppval")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "ppval is currently a runtime sink.",
};

fn ppval_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.get(1) {
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        _ => Type::tensor(),
    }
}

#[runtime_builtin(
    name = "ppval",
    category = "math/interpolation",
    summary = "Evaluate a piecewise-polynomial structure at query points.",
    keywords = "ppval,piecewise polynomial,spline,pchip",
    accel = "sink",
    sink = true,
    type_resolver(ppval_type),
    builtin_path = "crate::builtins::math::interpolation::ppval"
)]
async fn ppval_builtin(pp: Value, xq: Value) -> crate::BuiltinResult<Value> {
    let parsed = pp_from_value(pp, NAME).await?;
    let query = query_points(xq, NAME).await?;
    evaluate_pp(&parsed, &query, &Extrapolation::Extrapolate, NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::interpolation::pp::{build_spline_pp, pp_to_value, NumericSeries};
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn ppval_evaluates_spline_struct() {
        let series = NumericSeries {
            x: vec![1.0, 2.0, 3.0],
            y: vec![1.0, 4.0, 9.0],
            series: 1,
            trailing_shape: Vec::new(),
        };
        let pp = pp_to_value(
            build_spline_pp(&series, "spline").expect("spline"),
            "spline",
        )
        .expect("pp");
        let query = Value::Tensor(Tensor::new(vec![1.5, 2.5], vec![1, 2]).expect("tensor"));
        let value = block_on(ppval_builtin(pp, query)).expect("ppval");
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert!((tensor.data[0] - 2.25).abs() < 1e-10);
        assert!((tensor.data[1] - 6.25).abs() < 1e-10);
    }
}
