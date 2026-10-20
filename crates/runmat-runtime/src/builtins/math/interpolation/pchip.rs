//! MATLAB-compatible `pchip` builtin for shape-preserving cubic interpolation.

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, PCHIP_ERROR_INTERNAL,
    PCHIP_ERROR_INVALID_ARGUMENT, PCHIP_ERROR_INVALID_INPUT, PCHIP_INTEGER_QUERY_EXTENSION,
    PCHIP_INTEGER_SAMPLE_LOCATIONS_EXTENSION, PCHIP_INTEGER_SAMPLE_VALUES_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{
    build_pchip_pp, evaluate_pp, pp_to_value, query_points, series_from_values_on_axis,
    Extrapolation, PpOutputLayout, SampleAxis,
};
use crate::{build_runtime_error, RuntimeError};

const NAME: &str = "pchip";

fn pchip_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn pchip_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    pchip_error_with_message(
        format!(
            "{}: {}",
            PCHIP_ERROR_INVALID_ARGUMENT.message,
            detail.as_ref()
        ),
        &PCHIP_ERROR_INVALID_ARGUMENT,
    )
}

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

#[runtime_builtin(
    name = "pchip",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::interpolation::pchip"
)]
async fn pchip_builtin(
    x: Value,
    y: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    ensure_pchip_integer_boundary(&x, &PCHIP_INTEGER_SAMPLE_LOCATIONS_EXTENSION, "X").await?;
    ensure_pchip_integer_boundary(&y, &PCHIP_INTEGER_SAMPLE_VALUES_EXTENSION, "Y").await?;
    if let Some(query) = rest.first() {
        ensure_pchip_integer_boundary(query, &PCHIP_INTEGER_QUERY_EXTENSION, "Xq").await?;
    }
    let series = series_from_values_on_axis(x, y, SampleAxis::Last, NAME)
        .await
        .map_err(|err| {
            pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INVALID_INPUT)
        })?;
    let output_layout = match series.sample_axis {
        SampleAxis::First => PpOutputLayout::QueryLeading,
        SampleAxis::Last => PpOutputLayout::ComponentLeading,
    };
    let pp = build_pchip_pp(&series, NAME).map_err(|err| {
        pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INTERNAL)
    })?;
    match rest.len() {
        0 => pp_to_value(pp, NAME).map_err(|err| {
            pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INTERNAL)
        }),
        1 => {
            let query = query_points(rest.into_iter().next().expect("query"), NAME)
                .await
                .map_err(|err| {
                    pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INVALID_INPUT)
                })?;
            evaluate_pp(
                &pp,
                &query,
                &Extrapolation::Extrapolate,
                output_layout,
                NAME,
            )
            .map_err(|err| {
                pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INTERNAL)
            })
        }
        _ => Err(pchip_invalid_argument("too many input arguments")),
    }
}

async fn ensure_pchip_integer_boundary(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
    role: &str,
) -> crate::BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64_async, value_has_native_integer_class,
    };
    if !value_has_native_integer_class(value) {
        return Ok(());
    }
    crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)?;
    if !native_integer_value_is_exact_f64_async(value).await? {
        return Err(pchip_error_with_message(
            format!("pchip: integer {role} values must be exactly representable as double"),
            &PCHIP_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::PCHIP_DESCRIPTOR;
    use runmat_value::Tensor;

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    #[test]
    fn pchip_three_arg_evaluates_monotone_data() {
        let value = block_on(pchip_builtin(
            row(&[1.0, 2.0, 3.0, 4.0]),
            row(&[0.0, 1.0, 1.5, 1.75]),
            crate::builtins::common::test_support::builtin_arguments(vec![row(&[1.5, 2.5, 3.5])]),
        ))
        .expect("pchip");
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert!(tensor
            .materialize_f64()
            .windows(2)
            .all(|pair| pair[1] >= pair[0]));
    }

    #[test]
    fn pchip_documented_example_has_checked_values_and_class() {
        let value = block_on(pchip_builtin(
            row(&[1.0, 2.0, 3.0]),
            row(&[1.0, 4.0, 9.0]),
            crate::builtins::common::test_support::builtin_arguments(vec![row(&[1.5, 2.5])]),
        ))
        .expect("pchip values");
        let Value::Tensor(output) = value else {
            panic!("vector query must return a double tensor");
        };
        assert_eq!(output.materialize_f64(), vec![2.28125, 6.21875]);
    }

    #[test]
    fn pchip_uses_last_sample_axis_and_component_leading_output() {
        let data = (0..4)
            .flat_map(|sample| (0..6).map(move |component| (component * 100 + sample) as f64))
            .collect();
        let value = block_on(pchip_builtin(
            row(&[1.0, 2.0, 3.0, 4.0]),
            Value::Tensor(Tensor::new(data, vec![2, 3, 4]).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
            )]),
        ))
        .expect("array-valued pchip");
        let Value::Tensor(output) = value else {
            panic!("array-valued pchip must return a tensor");
        };
        assert_eq!(output.shape(), vec![2, 3, 2, 2]);
        let values = output.materialize_f64();
        for query in 0..4 {
            for component in 0..6 {
                assert_eq!(
                    values[component + query * 6],
                    (component * 100 + query) as f64
                );
            }
        }
    }

    #[test]
    fn pchip_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = PCHIP_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"pp = pchip(X, Y)"));
        assert!(labels.contains(&"Vq = pchip(X, Y, Xq)"));
    }

    #[test]
    fn pchip_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = PCHIP_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.PCHIP.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.PCHIP.INVALID_INPUT"));
        assert!(codes.contains(&"RM.PCHIP.INTERNAL"));
    }

    #[test]
    fn pchip_too_many_args_uses_stable_identifier() {
        let err = block_on(pchip_builtin(
            row(&[1.0, 2.0]),
            row(&[1.0, 4.0]),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.5),
                Value::Num(2.5),
            ]),
        ))
        .expect_err("expected pchip argument error");
        assert_eq!(err.identifier(), PCHIP_ERROR_INVALID_ARGUMENT.identifier);
    }
}
