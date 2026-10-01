//! Evaluate piecewise-polynomial structs produced by `spline` and `pchip`.

use runmat_builtins::{
    BuiltinErrorDescriptor, PPVAL_DESCRIPTOR, PPVAL_ERROR_INTERNAL, PPVAL_ERROR_INVALID_ARGUMENT,
    PPVAL_ERROR_INVALID_INPUT, PPVAL_INTEGER_QUERY_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{evaluate_pp, pp_from_value, query_points, Extrapolation};
use crate::{build_runtime_error, RuntimeError};

const NAME: &str = "ppval";

fn ppval_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

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

#[runtime_builtin(
    name = "ppval",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::interpolation::ppval"
)]
async fn ppval_builtin(pp: Value, xq: Value) -> crate::BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&xq, NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &xq,
        &PPVAL_INTEGER_QUERY_EXTENSION,
        NAME,
        "query point",
    )
    .await?;
    let parsed = pp_from_value(pp, NAME).await.map_err(|err| {
        ppval_error_with_message(err.message().to_string(), &PPVAL_ERROR_INVALID_ARGUMENT)
    })?;
    let query = query_points(xq, NAME).await.map_err(|err| {
        ppval_error_with_message(err.message().to_string(), &PPVAL_ERROR_INVALID_INPUT)
    })?;
    evaluate_pp(&parsed, &query, &Extrapolation::Extrapolate, NAME)
        .map_err(|err| ppval_error_with_message(err.message().to_string(), &PPVAL_ERROR_INTERNAL))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::interpolation::pp::{build_spline_pp, pp_to_value, NumericSeries};
    use futures::executor::block_on;
    use runmat_value::Tensor;

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
        assert!((tensor.materialize_f64()[0] - 2.25).abs() < 1e-10);
        assert!((tensor.materialize_f64()[1] - 6.25).abs() < 1e-10);
    }

    #[test]
    fn ppval_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = PPVAL_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["Vq = ppval(pp, Xq)"]);
    }

    #[test]
    fn ppval_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = PPVAL_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.PPVAL.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.PPVAL.INVALID_INPUT"));
        assert!(codes.contains(&"RM.PPVAL.INTERNAL"));
    }

    #[test]
    fn ppval_invalid_first_arg_uses_stable_identifier() {
        let err = block_on(ppval_builtin(
            Value::Num(42.0),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("tensor")),
        ))
        .expect_err("expected ppval argument error");
        assert_eq!(err.identifier(), PPVAL_ERROR_INVALID_ARGUMENT.identifier);
    }
}
