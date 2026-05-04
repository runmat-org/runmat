//! MATLAB-compatible `ode45` builtin.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::ode::common::{
    build_ode_output, ode_options_from_struct, parse_ode_input, parse_options, solve_ode, OdeMethod,
};
use crate::builtins::math::ode::type_resolvers::ode_solution_type;
use crate::BuiltinResult;

const NAME: &str = "ode45";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::ode::ode45")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ode45",
    op_kind: GpuOpKind::Custom("ode-solve"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Adaptive ODE integration runs on the host. RHS callbacks may call GPU-aware builtins.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::ode::ode45")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ode45",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "ODE integration repeatedly invokes user callbacks and terminates fusion planning.",
};

#[runtime_builtin(
    name = "ode45",
    category = "math/ode",
    summary = "Solve nonstiff ODE systems with an adaptive Dormand-Prince 5(4) method.",
    keywords = "ode45,ode,nonstiff,dormand-prince,adaptive step",
    accel = "sink",
    type_resolver(ode_solution_type),
    builtin_path = "crate::builtins::math::ode::ode45"
)]
async fn ode45_builtin(
    function: Value,
    tspan: Value,
    y0: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(crate::builtins::math::ode::common::ode_error(
            NAME,
            "ode45: too many input arguments",
        ));
    }
    let options = parse_options(NAME, rest.first())?;
    let opts = ode_options_from_struct(NAME, options.as_ref())?;
    let input = parse_ode_input(NAME, tspan, y0).await?;
    let result = solve_ode(NAME, OdeMethod::Ode45, &function, &input, &opts).await?;
    build_ode_output(NAME, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;
    use std::sync::Arc;

    #[test]
    fn ode45_scalar_decay_returns_reasonable_final_value() {
        let _guard = crate::user_functions::install_user_function_invoker(Some(Arc::new(
            move |_name, args| {
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(-y)) })
            },
        )));

        let out = block_on(ode45_builtin(
            Value::FunctionHandle("decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap();

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.cols(), 1);
                let last = t.data[t.rows() - 1];
                assert!((last - (-1.0_f64).exp()).abs() < 5.0e-3);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }
}
