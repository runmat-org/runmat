//! MATLAB-compatible `ode23` builtin.

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

const NAME: &str = "ode23";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::ode::ode23")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ode23",
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::ode::ode23")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ode23",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "ODE integration repeatedly invokes user callbacks and terminates fusion planning.",
};

#[runtime_builtin(
    name = "ode23",
    category = "math/ode",
    summary = "Solve nonstiff ODE systems with an adaptive Bogacki-Shampine 3(2) method.",
    keywords = "ode23,ode,nonstiff,bogacki-shampine,adaptive step",
    accel = "sink",
    type_resolver(ode_solution_type),
    builtin_path = "crate::builtins::math::ode::ode23"
)]
async fn ode23_builtin(
    function: Value,
    tspan: Value,
    y0: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(crate::builtins::math::ode::common::ode_error(
            NAME,
            "ode23: too many input arguments",
        ));
    }
    let options = parse_options(NAME, rest.first())?;
    let opts = ode_options_from_struct(NAME, options.as_ref())?;
    let input = parse_ode_input(NAME, tspan, y0).await?;
    let result = solve_ode(NAME, OdeMethod::Ode23, &function, &input, &opts).await?;
    build_ode_output(NAME, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;
    use std::sync::Arc;

    #[test]
    fn ode23_supports_two_output_form() {
        let _guard = crate::user_functions::install_user_function_invoker(Some(Arc::new(
            move |_name, args| {
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(-y)) })
            },
        )));

        let _out_guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(ode23_builtin(
            Value::FunctionHandle("decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0], vec![1, 3]).unwrap()),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap();

        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }
}
