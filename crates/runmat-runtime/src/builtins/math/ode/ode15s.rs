//! MATLAB-compatible `ode15s` builtin.

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

const NAME: &str = "ode15s";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::ode::ode15s")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ode15s",
    op_kind: GpuOpKind::Custom("ode-solve-stiff"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Stiff ODE integration runs on the host. RHS callbacks may call GPU-aware builtins.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::ode::ode15s")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ode15s",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "ODE integration repeatedly invokes user callbacks and terminates fusion planning.",
};

#[runtime_builtin(
    name = "ode15s",
    category = "math/ode",
    summary = "Solve stiff ODE systems with an adaptive implicit host-side integrator.",
    keywords = "ode15s,ode,stiff,implicit,adaptive step",
    accel = "sink",
    type_resolver(ode_solution_type),
    builtin_path = "crate::builtins::math::ode::ode15s"
)]
async fn ode15s_builtin(
    function: Value,
    tspan: Value,
    y0: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(crate::builtins::math::ode::common::ode_error(
            NAME,
            "ode15s: too many input arguments",
        ));
    }
    let options = parse_options(NAME, rest.first())?;
    let opts = ode_options_from_struct(NAME, options.as_ref())?;
    let input = parse_ode_input(NAME, tspan, y0).await?;
    let result = solve_ode(NAME, OdeMethod::Ode15s, &function, &input, &opts).await?;
    build_ode_output(NAME, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{StructValue, Tensor};
    use std::sync::Arc;

    #[test]
    fn ode15s_handles_linear_stiff_decay() {
        let _guard = crate::user_functions::install_user_function_invoker(Some(Arc::new(
            move |_name, args| {
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(-15.0 * y)) })
            },
        )));

        let out = block_on(ode15s_builtin(
            Value::FunctionHandle("stiff_decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap();

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.cols(), 1);
                let last = t.data[t.rows() - 1];
                assert!(last.is_finite());
                assert!(last > 0.0);
                assert!(last < 0.1);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn ode15s_accepts_picard_unstable_stiff_step_with_newton() {
        let _guard = crate::user_functions::install_user_function_invoker(Some(Arc::new(
            move |_name, args| {
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(-1000.0 * y)) })
            },
        )));
        let mut options = StructValue::new();
        options.insert("RelTol", Value::Num(1.0e6));
        options.insert("AbsTol", Value::Num(1.0e6));
        options.insert("InitialStep", Value::Num(0.1));
        options.insert("MaxStep", Value::Num(0.1));
        options.insert("MaxSteps", Value::Num(2.0));

        let out = block_on(ode15s_builtin(
            Value::FunctionHandle("very_stiff_decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 0.1], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            vec![Value::Struct(options)],
        ))
        .unwrap();

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.cols(), 1);
                let last = t.data[t.rows() - 1];
                assert!(last.is_finite());
                assert!(last > 0.0);
                assert!(last < 0.02);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }
}
