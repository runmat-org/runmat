//! Evaluate piecewise-polynomial structs produced by `spline` and `pchip`.

use runmat_builtins::{
    BuiltinErrorDescriptor, PPVAL_ERROR_INTERNAL, PPVAL_ERROR_INVALID_ARGUMENT,
    PPVAL_ERROR_INVALID_INPUT, PPVAL_INTEGER_QUERY_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{evaluate_pp, pp_from_value, query_points, Extrapolation, PpOutputLayout};
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
    evaluate_pp(
        &parsed,
        &query,
        &Extrapolation::Extrapolate,
        PpOutputLayout::ComponentLeading,
        NAME,
    )
    .map_err(|err| ppval_error_with_message(err.message().to_string(), &PPVAL_ERROR_INTERNAL))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::interpolation::pp::{build_spline_pp, pp_to_value, NumericSeries};
    use futures::executor::block_on;
    use runmat_builtins::PPVAL_DESCRIPTOR;
    use runmat_value::Tensor;

    #[test]
    fn ppval_evaluates_spline_struct() {
        let series = NumericSeries {
            x: vec![1.0, 2.0, 3.0],
            y: vec![1.0, 4.0, 9.0],
            series: 1,
            component_shape: Vec::new(),
            sample_axis: super::super::pp::SampleAxis::Last,
        };
        let pp = pp_to_value(
            block_on(build_spline_pp(&series, "spline", &mut ())).expect("spline"),
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
    fn ppval_evaluates_nonsquare_array_valued_struct() {
        let samples = (0..4)
            .flat_map(|sample| (0..6).map(move |component| (component * 100 + sample) as f64))
            .collect();
        let series = block_on(super::super::pp::series_from_values_on_axis(
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap()),
            Value::Tensor(Tensor::new(samples, vec![2, 3, 4]).unwrap()),
            super::super::pp::SampleAxis::Last,
            "pchip",
        ))
        .unwrap();
        let pp = pp_to_value(
            super::super::pp::build_pchip_pp(&series, "pchip").unwrap(),
            "pchip",
        )
        .unwrap();
        let query = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let Value::Tensor(output) = block_on(ppval_builtin(pp, query)).unwrap() else {
            panic!("array-valued ppval must return a tensor");
        };
        assert_eq!(output.shape(), vec![2, 3, 2, 2]);
        let values = output.materialize_f64();
        for sample in 0..4 {
            for component in 0..6 {
                assert_eq!(
                    values[component + sample * 6],
                    (component * 100 + sample) as f64
                );
            }
        }
    }

    #[test]
    fn ppval_reads_piece_major_rows_from_external_vector_valued_pp() {
        let mut pp = runmat_value::UnboundedStructValueBuilder::new();
        pp.insert("form", Value::String("pp".into()));
        pp.insert(
            "breaks",
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap()),
        );
        // Rows are component 1 then component 2 within each interval;
        // columns are the linear and constant coefficients.
        pp.insert(
            "coefs",
            Value::Tensor(
                Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], vec![4, 2]).unwrap(),
            ),
        );
        pp.insert("pieces", Value::Num(2.0));
        pp.insert("order", Value::Num(2.0));
        pp.insert("dim", Value::Num(2.0));
        let query = Value::Tensor(Tensor::new(vec![0.5, 1.5], vec![1, 2]).unwrap());
        let Value::Tensor(output) =
            block_on(ppval_builtin(Value::Struct(pp.finish()), query)).unwrap()
        else {
            panic!("two-component pp must return a tensor");
        };
        assert_eq!(output.shape(), vec![2, 2]);
        assert_eq!(output.materialize_f64(), vec![10.5, 21.0, 31.5, 42.0]);
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
