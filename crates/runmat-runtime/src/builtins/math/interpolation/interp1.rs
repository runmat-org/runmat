//! MATLAB-compatible `interp1` builtin for dense real numeric data.

use runmat_builtins::{ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{
    build_pchip_pp, build_spline_pp, evaluate_linear_or_nearest, evaluate_pp,
    implicit_series_from_values, interp_error, parse_extrapolation, parse_method, query_points,
    series_from_values, Extrapolation, InterpMethod,
};

const NAME: &str = "interp1";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::interp1")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("interpolation-1d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Initial implementation gathers GPU inputs to the CPU reference path. Provider kernels can later accelerate linear and nearest evaluation.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::interp1"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Interpolation is currently a runtime sink.",
};

fn interp1_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let query = match args.len() {
        0 | 1 => return Type::tensor(),
        2 => args.get(1),
        _ => args.get(2),
    };
    match query {
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        _ => Type::tensor(),
    }
}

#[runtime_builtin(
    name = "interp1",
    category = "math/interpolation",
    summary = "One-dimensional interpolation for sampled data.",
    keywords = "interp1,interpolation,linear,nearest,spline,pchip",
    accel = "sink",
    sink = true,
    type_resolver(interp1_type),
    builtin_path = "crate::builtins::math::interpolation::interp1"
)]
async fn interp1_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedInterp1::parse(args).await?;
    match parsed.method {
        InterpMethod::Linear | InterpMethod::Nearest => evaluate_linear_or_nearest(
            &parsed.series,
            &parsed.query,
            parsed.method,
            &parsed.extrap,
            NAME,
        ),
        InterpMethod::Spline => {
            let pp = build_spline_pp(&parsed.series, NAME)?;
            evaluate_pp(&pp, &parsed.query, &parsed.extrap_for_cubic(), NAME)
        }
        InterpMethod::Pchip => {
            let pp = build_pchip_pp(&parsed.series, NAME)?;
            evaluate_pp(&pp, &parsed.query, &parsed.extrap_for_cubic(), NAME)
        }
    }
}

struct ParsedInterp1 {
    series: super::pp::NumericSeries,
    query: super::pp::QueryPoints,
    method: InterpMethod,
    extrap: Extrapolation,
}

impl ParsedInterp1 {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        if args.len() < 2 {
            return Err(interp_error(
                NAME,
                "interp1: expected at least Y and Xq arguments",
            ));
        }

        let mut method = InterpMethod::Linear;
        let mut extrap = Extrapolation::Nan;
        let (series, query, options) = if args.len() == 2 || third_arg_is_option(&args) {
            let mut iter = args.into_iter();
            let y = iter.next().expect("Y argument");
            let xq = iter.next().expect("Xq argument");
            let series = implicit_series_from_values(y, NAME).await?;
            let query = query_points(xq, NAME).await?;
            (series, query, iter.collect::<Vec<_>>())
        } else {
            let mut iter = args.into_iter();
            let x = iter.next().expect("X argument");
            let y = iter.next().expect("Y argument");
            let xq = iter.next().expect("Xq argument");
            let series = series_from_values(x, y, NAME).await?;
            let query = query_points(xq, NAME).await?;
            (series, query, iter.collect::<Vec<_>>())
        };

        for option in &options {
            if let Some(parsed) = parse_extrapolation(option, NAME).await? {
                extrap = parsed;
                continue;
            }
            if let Some(parsed) = parse_method(option, NAME)? {
                method = parsed;
                continue;
            }
            return Err(interp_error(
                NAME,
                "interp1: unsupported interpolation option",
            ));
        }

        Ok(Self {
            series,
            query,
            method,
            extrap,
        })
    }

    fn extrap_for_cubic(&self) -> Extrapolation {
        self.extrap.clone()
    }
}

fn third_arg_is_option(args: &[Value]) -> bool {
    args.get(2)
        .and_then(|value| crate::builtins::common::random_args::keyword_of(value))
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(interp1_builtin(args))
    }

    #[test]
    fn interp1_linear_midpoints() {
        let result = run(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[10.0, 20.0, 40.0]),
            row(&[1.5, 2.5]),
        ])
        .expect("interp1");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.data, vec![15.0, 30.0]);
    }

    #[test]
    fn interp1_nearest() {
        let result = run(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[10.0, 20.0, 40.0]),
            row(&[1.2, 2.8]),
            Value::String("nearest".to_string()),
        ])
        .expect("interp1");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.data, vec![10.0, 40.0]);
    }

    #[test]
    fn interp1_default_out_of_range_is_nan() {
        let result =
            run(vec![row(&[1.0, 2.0]), row(&[10.0, 20.0]), Value::Num(0.0)]).expect("interp1");
        let Value::Num(value) = result else {
            panic!("expected scalar");
        };
        assert!(value.is_nan());
    }

    #[test]
    fn interp1_extrapolates_when_requested() {
        let result = run(vec![
            row(&[1.0, 2.0]),
            row(&[10.0, 20.0]),
            Value::Num(0.0),
            Value::String("extrap".to_string()),
        ])
        .expect("interp1");
        assert_eq!(result, Value::Num(0.0));
    }
}
