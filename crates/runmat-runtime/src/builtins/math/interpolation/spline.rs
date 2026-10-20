//! MATLAB-compatible `spline` builtin backed by piecewise-polynomial structs.

use runmat_builtins::{
    BuiltinErrorDescriptor, SPLINE_ERROR_INTERNAL, SPLINE_ERROR_INVALID_ARGUMENT,
    SPLINE_ERROR_INVALID_INPUT, SPLINE_INTEGER_XQ_EXTENSION, SPLINE_INTEGER_X_EXTENSION,
    SPLINE_INTEGER_Y_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{
    build_spline_pp_with_slopes, evaluate_pp, is_spline_execution_failure, pp_to_value,
    query_points, spline_samples_from_values, Extrapolation, PpOutputLayout, SampleAxis,
};
use crate::{build_runtime_error, RuntimeError};

const NAME: &str = "spline";

fn spline_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn spline_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    spline_error_with_message(
        format!(
            "{}: {}",
            SPLINE_ERROR_INVALID_ARGUMENT.message,
            detail.as_ref()
        ),
        &SPLINE_ERROR_INVALID_ARGUMENT,
    )
}

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

#[runtime_builtin(
    name = "spline",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::interpolation::spline",
    context = true
)]
async fn spline_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    y: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &x,
        &SPLINE_INTEGER_X_EXTENSION,
        NAME,
        "sample location",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &y,
        &SPLINE_INTEGER_Y_EXTENSION,
        NAME,
        "sample value",
    )
    .await?;
    if let Some(query) = rest.first() {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            query,
            &SPLINE_INTEGER_XQ_EXTENSION,
            NAME,
            "query point",
        )
        .await?;
    }
    let samples = spline_samples_from_values(x, y, NAME)
        .await
        .map_err(|err| {
            spline_error_with_message(err.message().to_string(), &SPLINE_ERROR_INVALID_INPUT)
        })?;
    let output_layout = match samples.series.sample_axis {
        SampleAxis::First => PpOutputLayout::QueryLeading,
        SampleAxis::Last => PpOutputLayout::ComponentLeading,
    };
    let mut work = runtime
        .as_ref()
        .map(crate::context::RuntimeContext::cpu_cooperation);
    let pp = build_spline_pp_with_slopes(
        &samples.series,
        samples.endpoint_slopes.as_deref(),
        NAME,
        &mut work,
    )
    .await
    .map_err(|err| {
        if is_spline_execution_failure(&err) {
            crate::builtins::common::map_control_flow_with_builtin(err, NAME)
        } else {
            spline_error_with_message(err.message().to_string(), &SPLINE_ERROR_INTERNAL)
        }
    })?;
    match rest.len() {
        0 => pp_to_value(pp, NAME).map_err(|err| {
            spline_error_with_message(err.message().to_string(), &SPLINE_ERROR_INTERNAL)
        }),
        1 => {
            let query = query_points(rest.into_iter().next().expect("query"), NAME)
                .await
                .map_err(|err| {
                    spline_error_with_message(
                        err.message().to_string(),
                        &SPLINE_ERROR_INVALID_INPUT,
                    )
                })?;
            evaluate_pp(
                &pp,
                &query,
                &Extrapolation::Extrapolate,
                output_layout,
                NAME,
            )
            .map_err(|err| {
                spline_error_with_message(err.message().to_string(), &SPLINE_ERROR_INTERNAL)
            })
        }
        _ => Err(spline_invalid_argument("too many input arguments")),
    }
}

#[cfg(test)]
async fn spline_builtin(x: Value, y: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    spline_builtin_registered(
        None,
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::SPLINE_DESCRIPTOR;
    use runmat_value::{StructValue, Tensor};

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

    #[test]
    fn spline_vector_endpoint_slopes_reproduce_cubic_and_column_vector_form() {
        let x = row(&[0.0, 1.0, 2.0]);
        for y_shape in [vec![1, 5], vec![5, 1]] {
            let y = Value::Tensor(Tensor::new(vec![0.0, 0.0, 1.0, 8.0, 12.0], y_shape).unwrap());
            let value = block_on(spline_builtin(x.clone(), y, vec![row(&[0.5, 1.5])]))
                .expect("clamped spline");
            let Value::Tensor(output) = value else {
                panic!("vector query must return a tensor");
            };
            assert_eq!(output.shape(), vec![1, 2]);
            assert!((output.materialize_f64()[0] - 0.125).abs() < 1e-12);
            assert!((output.materialize_f64()[1] - 3.375).abs() < 1e-12);
        }
    }

    #[test]
    fn spline_array_endpoint_slopes_apply_to_each_component() {
        let y = Value::Tensor(
            Tensor::new(
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 3.0, 8.0, 17.0, 12.0, 24.0],
                vec![2, 5],
            )
            .unwrap(),
        );
        let value = block_on(spline_builtin(
            row(&[0.0, 1.0, 2.0]),
            y,
            vec![row(&[0.5, 1.5])],
        ))
        .expect("array clamped spline");
        let Value::Tensor(output) = value else {
            panic!("array-valued spline must return a tensor");
        };
        assert_eq!(output.shape(), vec![2, 2]);
        let expected = [0.125, 1.25, 3.375, 7.75];
        for (actual, expected) in output.materialize_f64().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn spline_preserves_nonsquare_array_component_axes() {
        let data = (0..4)
            .flat_map(|sample| (0..6).map(move |component| (component * 100 + sample) as f64))
            .collect();
        let value = block_on(spline_builtin(
            row(&[1.0, 2.0, 3.0, 4.0]),
            Value::Tensor(Tensor::new(data, vec![2, 3, 4]).unwrap()),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
            )],
        ))
        .expect("array-valued spline");
        let Value::Tensor(output) = value else {
            panic!("array-valued spline must return a tensor");
        };
        assert_eq!(output.shape(), vec![2, 3, 2, 2]);
        let values = output.materialize_f64();
        for query in 0..4 {
            for component in 0..6 {
                assert!(
                    (values[component + query * 6] - (component * 100 + query) as f64).abs() < 1e-9
                );
            }
        }
    }

    #[test]
    fn spline_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = SPLINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"pp = spline(X, Y)"));
        assert!(labels.contains(&"Vq = spline(X, Y, Xq)"));
    }

    #[test]
    fn spline_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = SPLINE_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.SPLINE.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.SPLINE.INVALID_INPUT"));
        assert!(codes.contains(&"RM.SPLINE.INTERNAL"));
    }

    #[test]
    fn spline_too_many_args_uses_stable_identifier() {
        let err = block_on(spline_builtin(
            row(&[1.0, 2.0]),
            row(&[1.0, 4.0]),
            vec![Value::Num(1.5), Value::Num(2.5)],
        ))
        .expect_err("expected spline argument error");
        assert_eq!(err.identifier(), SPLINE_ERROR_INVALID_ARGUMENT.identifier);
    }
}
