//! Pole-zero map plotting and extraction for SISO transfer-function and state-space models.

use num_complex::Complex64;
use runmat_builtins::{
    BuiltinErrorDescriptor, PZMAP_ERROR_INTERNAL, PZMAP_ERROR_INVALID_ARGUMENT,
    PZMAP_ERROR_INVALID_MODEL, PZMAP_ERROR_PLOT_FAILED, PZMAP_ERROR_UNSUPPORTED_MODEL,
};
#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerInputAvailability, PZMAP_DESCRIPTOR, PZMAP_INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::model_polynomial::ss_object_to_tf_async;
use crate::builtins::control::tf_model::{output_complex_column, TfModel, SS_CLASS, TF_CLASS};
use crate::builtins::plotting::state::presentation_failure_kind;
use crate::builtins::plotting::style::{parse_line_style_args, LineStyleParseOptions};
use crate::{BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "pzmap";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::pzmap")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pzmap",
    op_kind: GpuOpKind::Custom("control-pole-zero-map"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "pzmap computes roots from host-side transfer-function metadata and plots through host rendering.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::pzmap")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pzmap",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "pzmap is model analysis and plotting; it terminates numeric fusion chains.",
};

#[runtime_builtin(
    name = "pzmap",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::pzmap"
)]
async fn pzmap_builtin(sys: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(pzmap_error(
            "pzmap: too many output arguments",
            &PZMAP_ERROR_INVALID_ARGUMENT,
        ));
    }
    if is_plot_form_call() {
        plot_pole_zero_map_statement(sys, rest).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    if !rest.is_empty() {
        return Err(pzmap_error(
            "pzmap: output forms support exactly one system",
            &PZMAP_ERROR_INVALID_ARGUMENT,
        ));
    }

    let eval = PoleZeroMap::from_value_async(sys).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            render_pole_zero_map(&eval, None, 0).await?;
            return Ok(Value::OutputList(Vec::new()));
        }
        return match out_count {
            1 => Ok(Value::OutputList(vec![eval.poles_value()?])),
            2 => Ok(Value::OutputList(eval.outputs()?)),
            _ => Err(pzmap_error(
                "pzmap: too many output arguments",
                &PZMAP_ERROR_INVALID_ARGUMENT,
            )),
        };
    }

    eval.poles_value()
}

fn is_plot_form_call() -> bool {
    matches!(crate::output_count::current_output_count(), Some(0))
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
}

async fn plot_pole_zero_map_statement(first_sys: Value, rest: Vec<Value>) -> BuiltinResult<()> {
    let mut systems = vec![(first_sys, None)];
    for arg in rest {
        if is_plot_style_arg(&arg) {
            if let Some((_, style)) = systems.last_mut() {
                if style.is_some() {
                    return Err(pzmap_error(
                        "pzmap: only one style argument is supported per system",
                        &PZMAP_ERROR_INVALID_ARGUMENT,
                    ));
                }
                parse_line_style_args(
                    std::slice::from_ref(&arg),
                    &LineStyleParseOptions::generic(BUILTIN_NAME),
                )
                .map_err(|err| {
                    pzmap_error(err.message().to_string(), &PZMAP_ERROR_INVALID_ARGUMENT)
                })?;
                *style = Some(arg);
                continue;
            }
        }
        if is_dynamic_model_object(&arg) {
            systems.push((arg, None));
            continue;
        }
        return Err(pzmap_error(
            "pzmap: statement-form plots accept one or more tf or ss systems with optional styles",
            &PZMAP_ERROR_INVALID_ARGUMENT,
        ));
    }

    let mut args = Vec::new();
    for (idx, (system, style)) in systems.into_iter().enumerate() {
        let eval = PoleZeroMap::from_value_async(system).await?;
        push_pole_zero_map_series(&mut args, &eval, style.as_ref(), idx)?;
    }
    render_pole_zero_map_args(args).await
}

fn is_dynamic_model_object(value: &Value) -> bool {
    matches!(value, Value::Object(object) if object.is_class(TF_CLASS) || object.is_class(SS_CLASS))
}

fn is_plot_style_arg(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

#[derive(Clone, Debug)]
struct PoleZeroMap {
    poles: Vec<Complex64>,
    zeros: Vec<Complex64>,
}

impl PoleZeroMap {
    async fn from_value_async(value: Value) -> BuiltinResult<Self> {
        let model = match value {
            Value::Object(object) if object.is_class(TF_CLASS) => {
                TfModel::from_value_async(Value::Object(object), BUILTIN_NAME).await?
            }
            Value::Object(object) if object.is_class(SS_CLASS) => {
                ss_object_to_tf_async(object, BUILTIN_NAME).await?
            }
            Value::Object(object) => {
                return Err(pzmap_error(
                    format!(
                        "pzmap: unsupported model class '{}'; supported classes are tf and ss",
                        object.class_name
                    ),
                    &PZMAP_ERROR_UNSUPPORTED_MODEL,
                ));
            }
            other => {
                return Err(pzmap_error(
                    format!("pzmap: expected a tf or ss object, got {other:?}"),
                    &PZMAP_ERROR_INVALID_MODEL,
                ));
            }
        };
        if model.input_delay != 0.0 || model.output_delay != 0.0 {
            return Err(pzmap_error(
                "pzmap: transfer functions with input or output delays are not supported",
                &PZMAP_ERROR_UNSUPPORTED_MODEL,
            ));
        }
        Ok(Self {
            poles: model.poles().map_err(map_model_error)?,
            zeros: model.zeros().map_err(map_model_error)?,
        })
    }

    fn outputs(&self) -> BuiltinResult<Vec<Value>> {
        Ok(vec![self.poles_value()?, self.zeros_value()?])
    }

    fn poles_value(&self) -> BuiltinResult<Value> {
        output_complex_column(self.poles.clone(), BUILTIN_NAME)
    }

    fn zeros_value(&self) -> BuiltinResult<Value> {
        output_complex_column(self.zeros.clone(), BUILTIN_NAME)
    }
}

async fn render_pole_zero_map(
    eval: &PoleZeroMap,
    style: Option<&Value>,
    series_index: usize,
) -> BuiltinResult<()> {
    let mut args = Vec::new();
    push_pole_zero_map_series(&mut args, eval, style, series_index)?;
    render_pole_zero_map_args(args).await
}

fn push_pole_zero_map_series(
    args: &mut Vec<Value>,
    eval: &PoleZeroMap,
    style: Option<&Value>,
    series_index: usize,
) -> BuiltinResult<()> {
    let color = style_color(style, series_index)?;
    push_marker_series(args, &eval.poles, color, 'x')?;
    push_marker_series(args, &eval.zeros, color, 'o')?;
    Ok(())
}

async fn render_pole_zero_map_args(args: Vec<Value>) -> BuiltinResult<()> {
    if args.is_empty() {
        return Ok(());
    }
    if let Err(err) = crate::call_builtin_async("plot", &args).await {
        if presentation_failure_kind(&err).is_some() {
            return Ok(());
        }
        return Err(crate::builtins::plotting::state::with_render_cause(
            pzmap_error(
                format!("pzmap: plotting failed: {}", err.message()),
                &PZMAP_ERROR_PLOT_FAILED,
            ),
            err,
        ));
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Pole-Zero Map")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Real Axis")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Imaginary Axis")]).await;
    let _ = crate::call_builtin_async("grid", &[Value::from("on")]).await;
    Ok(())
}

fn push_marker_series(
    args: &mut Vec<Value>,
    values: &[Complex64],
    color: glam::Vec4,
    marker: char,
) -> BuiltinResult<()> {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for value in values {
        if value.re.is_finite() && value.im.is_finite() {
            x.push(value.re);
            y.push(value.im);
        }
    }
    if !x.is_empty() {
        args.push(column_tensor(x)?);
        args.push(column_tensor(y)?);
        args.push(Value::from(marker.to_string()));
        args.push(Value::from("Color"));
        args.push(color_value(color)?);
    }
    Ok(())
}

fn style_color(style: Option<&Value>, series_index: usize) -> BuiltinResult<glam::Vec4> {
    if let Some(style) = style {
        let parsed = parse_line_style_args(
            std::slice::from_ref(style),
            &LineStyleParseOptions::generic(BUILTIN_NAME),
        )
        .map_err(|err| pzmap_error(err.message().to_string(), &PZMAP_ERROR_INVALID_ARGUMENT))?;
        if parsed.color_explicit {
            return Ok(parsed.appearance.color);
        }
    }
    Ok(crate::builtins::plotting::state::line_color_for_series_index(series_index))
}

fn color_value(color: glam::Vec4) -> BuiltinResult<Value> {
    Tensor::new(
        vec![color.x as f64, color.y as f64, color.z as f64],
        vec![1, 3],
    )
    .map(Value::Tensor)
    .map_err(|err| {
        pzmap_error(
            format!("pzmap: failed to build color value: {err}"),
            &PZMAP_ERROR_INTERNAL,
        )
    })
}

fn column_tensor(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    Tensor::new(data, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(|err| {
            pzmap_error(
                format!("pzmap: failed to build plot vector: {err}"),
                &PZMAP_ERROR_INTERNAL,
            )
        })
}

fn map_model_error(err: RuntimeError) -> RuntimeError {
    pzmap_error(err.message().to_string(), &PZMAP_ERROR_INTERNAL)
}

fn pzmap_error(message: impl Into<String>, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::ObjectInstance;

    #[test]
    fn invalid_style_rejects_before_model_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let den = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &den).unwrap();
            let mut model = ObjectInstance::new(TF_CLASS.to_string());
            model.properties.insert("Numerator".into(), Value::Num(1.0));
            model
                .properties
                .insert("Denominator".into(), Value::GpuTensor(resident));
            for name in ["Ts", "InputDelay", "OutputDelay"] {
                model.properties.insert(name.into(), Value::Num(0.0));
            }
            provider.reset_telemetry();
            let error = block_on(plot_pole_zero_map_statement(
                Value::Object(model),
                vec![Value::from("not-a-style")],
            ))
            .expect_err("invalid style");
            assert_eq!(error.identifier(), Some("RunMat:pzmap:InvalidArgument"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn unrelated_resident_property_is_not_downloaded_for_data_form() {
        let unrelated = Tensor::new(vec![5.0, 6.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &unrelated)
                    .expect("upload unrelated property");
            let mut model = ObjectInstance::new(TF_CLASS.to_string());
            model.properties.insert("Numerator".into(), Value::Num(2.0));
            model
                .properties
                .insert("Denominator".into(), Value::Num(1.0));
            model.properties.insert("Ts".into(), Value::Num(0.0));
            model
                .properties
                .insert("InputDelay".into(), Value::Num(0.0));
            model
                .properties
                .insert("OutputDelay".into(), Value::Num(0.0));
            model
                .properties
                .insert("Unrelated".into(), Value::GpuTensor(resident));
            let _outputs = crate::output_count::push_output_count(Some(2));
            provider.reset_telemetry();
            let Value::OutputList(values) =
                block_on(pzmap_builtin(Value::Object(model), vec![])).expect("static gain")
            else {
                panic!("expected two outputs");
            };
            assert_eq!(values.len(), 2);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    fn tf(num: Vec<f64>, den: Vec<f64>) -> Value {
        block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
                Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
            ],
        ))
        .expect("tf")
    }

    fn ss(a: Value, b: Value, c: Value, d: Value) -> Value {
        block_on(crate::call_builtin_async("ss", &[a, b, c, d])).expect("ss")
    }

    fn run_pzmap(sys: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(pzmap_builtin(sys, rest))
    }

    fn tensor(value: &Value) -> &Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn descriptor_signatures_cover_output_forms() {
        let labels = PZMAP_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"pzmap(sys)"));
        assert!(labels.contains(&"p = pzmap(sys)"));
        assert!(labels.contains(&"[p,z] = pzmap(sys)"));
        assert_eq!(PZMAP_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            PZMAP_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
    }

    #[test]
    fn two_output_call_returns_poles_and_zeros() {
        let sys = tf(vec![1.0, 3.0, 2.0], vec![1.0, 4.0]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = run_pzmap(sys, Vec::new()).expect("pzmap");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);

        let poles = tensor(&outputs[0]);
        assert_eq!(poles.shape, vec![1, 1]);
        assert!((poles.materialize_f64()[0] + 4.0).abs() < 1.0e-8);

        let zeros = tensor(&outputs[1]);
        assert_eq!(zeros.shape, vec![2, 1]);
        assert!(zeros
            .materialize_f64()
            .iter()
            .any(|z| (*z + 1.0).abs() < 1.0e-8));
        assert!(zeros
            .materialize_f64()
            .iter()
            .any(|z| (*z + 2.0).abs() < 1.0e-8));
    }

    #[test]
    fn complex_poles_are_returned_as_complex_column() {
        let sys = tf(vec![1.0], vec![1.0, 0.0, 1.0]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = run_pzmap(sys, Vec::new()).expect("pzmap");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        let Value::ComplexTensor(poles) = &outputs[0] else {
            panic!("expected complex poles");
        };
        assert_eq!(poles.shape, vec![2, 1]);
        assert!(poles
            .materialize_f64()
            .iter()
            .all(|(re, _)| re.abs() < 1.0e-8));
        assert!(poles
            .materialize_f64()
            .iter()
            .any(|(_, im)| (*im - 1.0).abs() < 1.0e-8));
        assert!(poles
            .materialize_f64()
            .iter()
            .any(|(_, im)| (*im + 1.0).abs() < 1.0e-8));
        assert_eq!(tensor(&outputs[1]).shape, vec![0, 1]);
    }

    #[test]
    fn statement_form_plots_without_error() {
        let sys = tf(vec![1.0, 2.0], vec![1.0, 3.0, 4.0]);
        let _guard = crate::output_count::push_output_count(Some(0));
        let result = run_pzmap(sys, Vec::new()).expect("pzmap");
        assert!(matches!(result, Value::OutputList(outputs) if outputs.is_empty()));
    }

    #[test]
    fn statement_form_accepts_multiple_systems() {
        let first = tf(vec![1.0], vec![1.0, 1.0]);
        let second = tf(vec![1.0, 2.0], vec![1.0, 3.0, 4.0]);
        let _guard = crate::output_count::push_output_count(Some(0));
        let result = run_pzmap(first, vec![second]).expect("pzmap");
        assert!(matches!(result, Value::OutputList(outputs) if outputs.is_empty()));
    }

    #[test]
    fn statement_form_accepts_styles_per_system() {
        let _ = crate::builtins::plotting::clear_figure(None);
        let first = tf(vec![1.0, 2.0], vec![1.0, 3.0]);
        let second = tf(vec![1.0, 4.0], vec![1.0, 5.0]);
        let _guard = crate::output_count::push_output_count(Some(0));
        let result =
            run_pzmap(first, vec![Value::from("r"), second, Value::from("b--")]).expect("pzmap");
        assert!(matches!(result, Value::OutputList(outputs) if outputs.is_empty()));
    }

    #[test]
    fn marker_series_pair_poles_and_zeros_with_same_color() {
        let eval = PoleZeroMap {
            poles: vec![Complex64::new(-1.0, 0.0)],
            zeros: vec![Complex64::new(-2.0, 0.0)],
        };
        let mut args = Vec::new();
        push_pole_zero_map_series(&mut args, &eval, Some(&Value::from("r--")), 0)
            .expect("series args");
        assert_eq!(args.len(), 10);
        assert_eq!(args[2], Value::from("x"));
        assert_eq!(args[3], Value::from("Color"));
        assert_eq!(args[7], Value::from("o"));
        assert_eq!(args[8], Value::from("Color"));
        match (&args[4], &args[9]) {
            (Value::Tensor(pole_color), Value::Tensor(zero_color)) => {
                assert_eq!(pole_color.materialize_f64(), zero_color.materialize_f64());
            }
            other => panic!("expected color tensors, got {other:?}"),
        }
    }

    #[test]
    fn statement_form_rejects_invalid_style_tokens() {
        let sys = tf(vec![1.0], vec![1.0, 1.0]);
        let _guard = crate::output_count::push_output_count(Some(0));
        let err = run_pzmap(sys, vec![Value::from("foo")]).expect_err("invalid style");
        assert!(err.message().contains("unrecognised style token"));
        assert_eq!(err.identifier(), PZMAP_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn state_space_model_returns_poles_and_zeros() {
        let sys = ss(
            Value::Num(-1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(0.0),
        );
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = run_pzmap(sys, Vec::new()).expect("pzmap");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);
        assert_eq!(tensor(&outputs[0]).materialize_f64(), vec![-1.0]);
        assert_eq!(tensor(&outputs[1]).shape, vec![0, 1]);
    }

    #[test]
    fn non_model_input_uses_pzmap_invalid_model_identifier() {
        let err = run_pzmap(Value::Num(1.0), Vec::new()).expect_err("should fail");
        assert!(err.message().contains("expected a tf or ss object"));
        assert_eq!(err.identifier(), PZMAP_ERROR_INVALID_MODEL.identifier);
    }

    #[test]
    fn output_form_rejects_extra_arguments() {
        let sys = tf(vec![1.0], vec![1.0, 1.0]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let err = run_pzmap(sys, vec![Value::Num(1.0)]).expect_err("should fail");
        assert!(err.message().contains("exactly one system"));
        assert_eq!(err.identifier(), PZMAP_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn too_many_outputs_are_rejected() {
        let sys = tf(vec![1.0], vec![1.0, 1.0]);
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = run_pzmap(sys, Vec::new()).expect_err("should fail");
        assert!(err.message().contains("too many output arguments"));
        assert_eq!(err.identifier(), PZMAP_ERROR_INVALID_ARGUMENT.identifier);
    }
}
