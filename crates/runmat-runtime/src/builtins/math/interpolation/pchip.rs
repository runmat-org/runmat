//! MATLAB-compatible `pchip` builtin for shape-preserving cubic interpolation.

use runmat_builtins::{ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{
    build_pchip_pp, evaluate_pp, interp_error, pp_to_value, query_points, series_from_values,
    Extrapolation,
};

const NAME: &str = "pchip";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::pchip")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("pchip"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "PCHIP coefficient construction and evaluation currently run on the CPU reference path after gathering GPU inputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::interpolation::pchip")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "PCHIP builds a piecewise-polynomial representation and is not fused.",
};

fn pchip_type(args: &[Type], _ctx: &ResolveContext) -> Type {
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
    name = "pchip",
    category = "math/interpolation",
    summary = "Shape-preserving piecewise cubic Hermite interpolation.",
    keywords = "pchip,shape preserving,cubic hermite,interpolation,ppval",
    accel = "sink",
    sink = true,
    type_resolver(pchip_type),
    builtin_path = "crate::builtins::math::interpolation::pchip"
)]
async fn pchip_builtin(x: Value, y: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let series = series_from_values(x, y, NAME).await?;
    let pp = build_pchip_pp(&series, NAME)?;
    match rest.len() {
        0 => pp_to_value(pp, NAME),
        1 => {
            let query = query_points(rest.into_iter().next().expect("query"), NAME).await?;
            evaluate_pp(&pp, &query, &Extrapolation::Extrapolate, NAME)
        }
        _ => Err(interp_error(NAME, "pchip: too many input arguments")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    #[test]
    fn pchip_three_arg_evaluates_monotone_data() {
        let value = block_on(pchip_builtin(
            row(&[1.0, 2.0, 3.0, 4.0]),
            row(&[0.0, 1.0, 1.5, 1.75]),
            vec![row(&[1.5, 2.5, 3.5])],
        ))
        .expect("pchip");
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert!(tensor.data.windows(2).all(|pair| pair[1] >= pair[0]));
    }
}
