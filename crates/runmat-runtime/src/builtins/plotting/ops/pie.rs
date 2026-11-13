use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{PIE_ERROR_INVALID_ARGUMENT, PIE_INTEGER_VALUES_EXTENSION};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::PieChart;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;

use crate::{build_runtime_error, RuntimeError};

use super::common::gather_tensor_from_gpu_async;
use super::op_common::value_as_text_string;
use super::state::{render_active_plot, PlotRenderOptions};

const BUILTIN_NAME: &str = "pie";

fn pie_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn pie_invalid(detail: impl AsRef<str>) -> RuntimeError {
    pie_error_with_detail(&PIE_ERROR_INVALID_ARGUMENT, detail)
}

fn map_pie_invalid(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = pie_error_with_detail(&PIE_ERROR_INVALID_ARGUMENT, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::pie")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pie",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "pie is a plotting sink; GPU inputs may remain on device until host fallback is needed for pie geometry generation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::pie")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pie",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "pie terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "pie",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::pie"
)]
pub async fn pie_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "pie: too many output arguments",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "pie")?;
    let _arguments = args.fork();
    work.finish(pie_impl(args, &work).await)
}
async fn pie_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (target_axes, args) = parse_axes_target(args).map_err(map_pie_invalid)?;
    super::op_common::apply_axes_target(target_axes, BUILTIN_NAME)?;
    if let Some(values) = args.first() {
        crate::builtins::common::validation::reject_typed_complex_integer(values, BUILTIN_NAME)?;
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            values,
            &PIE_INTEGER_VALUES_EXTENSION,
            BUILTIN_NAME,
            "slice",
        )
        .await?;
    }
    let (values, explode, labels) = parse_pie_args(args).await.map_err(map_pie_invalid)?;
    let mut chart = PieChart::new(values, None).map_err(|e| pie_invalid(&e))?;
    if let Some(explode) = explode {
        chart = chart.with_explode(explode);
    }
    if let Some(labels) = labels {
        match labels {
            PieLabelsArg::Explicit(labels) => {
                chart = chart.with_slice_labels(labels);
            }
            PieLabelsArg::Format(fmt) => {
                chart = chart.with_label_format(fmt);
            }
        }
    }
    let mut chart = Some(chart);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            axis_equal: true,
            grid: false,
            ..Default::default()
        },
        move |figure, axes| {
            let plot_index =
                figure.add_pie_chart_on_axes(chart.take().expect("pie consumed once"), axes);
            // MATLAB's pie hides its axes (as `axis off`).
            let mut style = figure
                .axes_metadata(axes)
                .map(|meta| meta.axes_style.clone())
                .unwrap_or_default();
            style.visible = false;
            figure.set_axes_style(axes, style);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_pie_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_pie_invalid(err));
    }
    Ok(handle)
}

enum PieLabelsArg {
    Explicit(Vec<String>),
    Format(String),
}

async fn parse_pie_args(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<(Vec<f64>, Option<Vec<bool>>, Option<PieLabelsArg>)> {
    if args.is_empty() {
        return Err(pie_invalid("expected values input"));
    }
    let values = tensor_from_value(args[0].clone()).await?;
    let values = tensor_utils::tensor_into_values_f64(values);
    if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(pie_invalid("values must be finite and nonnegative"));
    }
    let mut explode: Option<Vec<bool>> = None;
    let mut labels: Option<PieLabelsArg> = None;
    for arg in args.into_iter().skip(1) {
        if explode.is_none()
            && matches!(
                arg,
                Value::Num(_)
                    | Value::Int(_)
                    | Value::Bool(_)
                    | Value::Tensor(_)
                    | Value::LogicalArray(_)
                    | Value::GpuTensor(_)
            )
        {
            let t = tensor_from_value(arg.clone()).await?;
            let explode_values = tensor_utils::tensor_values_f64(&t);
            if explode_values.len() == values.len() && explode_values.iter().all(|v| v.is_finite())
            {
                explode = Some(explode_values.into_iter().map(|v| v != 0.0).collect());
                continue;
            }
        }
        labels = Some(parse_labels(arg, values.len())?);
    }
    if let Some(explode) = explode.as_ref() {
        if explode.len() != values.len() {
            return Err(pie_invalid("explode vector must match values length"));
        }
    }
    if let Some(PieLabelsArg::Explicit(labels)) = labels.as_ref() {
        if labels.len() != values.len() {
            return Err(pie_invalid("labels must match values length"));
        }
    }
    Ok((values, explode, labels))
}

fn parse_axes_target(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<(super::op_common::AxesTarget, crate::call::BuiltinArguments)> {
    super::op_common::split_leading_axes_arguments(args, BUILTIN_NAME)
}

async fn tensor_from_value(value: Value) -> crate::BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await,
        other => Tensor::try_from(&other).map_err(|e| pie_invalid(&e)),
    }
}

fn parse_labels(value: Value, value_len: usize) -> crate::BuiltinResult<PieLabelsArg> {
    match value {
        Value::StringArray(arr) => Ok(PieLabelsArg::Explicit(
            arr.data()
                .iter()
                .map(|text| {
                    text.as_str().map(str::to_owned).ok_or_else(|| {
                        pie_error_with_detail(
                            &PIE_ERROR_INVALID_ARGUMENT,
                            "pie: labels must be text",
                        )
                    })
                })
                .collect::<crate::BuiltinResult<Vec<_>>>()?,
        )),
        Value::Cell(cell) => {
            let mut labels = Vec::new();
            for row in 0..cell.rows {
                for col in 0..cell.cols {
                    let v = cell.get(row, col).map_err(|e| pie_invalid(&e))?;
                    labels.push(
                        value_as_text_string(&v)
                            .map_err(|source| {
                                crate::builtins::plotting::plotting_error_with_source(
                                    "pie",
                                    source.to_string(),
                                    source,
                                )
                            })?
                            .ok_or_else(|| pie_invalid("labels must be strings"))?,
                    );
                }
            }
            Ok(PieLabelsArg::Explicit(labels))
        }
        other => {
            let text = value_as_text_string(&other)
                .map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "pie",
                        source.to_string(),
                        source,
                    )
                })?
                .ok_or_else(|| pie_invalid("labels must be strings"))?;
            if value_len > 1 && text.contains('%') {
                Ok(PieLabelsArg::Format(text))
            } else {
                Ok(PieLabelsArg::Explicit(vec![text]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    async fn pie_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::pie_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("pie test vector")
    }

    fn int_vec_tensor(data: Vec<i16>) -> Tensor {
        Tensor::new_integer(
            runmat_value::IntegerStorage::I16(data.clone()),
            vec![data.len()],
        )
        .expect("integer tensor")
    }

    #[test]
    fn pie_builds_chart_with_labels_and_explode() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(pie_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0, 3.0])),
            Value::Tensor(vec_tensor(&[0.0, 1.0, 0.0])),
            Value::StringArray(
                runmat_value::StringArray::new(
                    vec!["A".into(), "B".into(), "C".into()],
                    vec![1, 3],
                )
                .unwrap(),
            ),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Pie(pie) = fig.plots().next().unwrap() else {
            panic!("expected pie");
        };
        assert_eq!(pie.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(pie.slice_labels, vec!["A", "B", "C"]);
        assert_eq!(pie.explode, vec![false, true, false]);
    }

    #[test]
    fn pie_values_and_explode_read_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        futures::executor::block_on(pie_builtin(vec![
            Value::Tensor(int_vec_tensor(vec![1, 2, 3])),
            Value::Tensor(int_vec_tensor(vec![0, 1, 0])),
        ]))
        .unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Pie(pie) = fig.plots().next().unwrap() else {
            panic!("expected pie");
        };
        assert_eq!(pie.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(pie.explode, vec![false, true, false]);
    }

    #[test]
    fn pie_supports_axes_target_and_validates_lengths() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            current_figure_handle(),
            1,
        ));

        let _ = futures::executor::block_on(pie_builtin(vec![
            ax,
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["Left".into(), "Right".into()], vec![1, 2])
                    .unwrap(),
            ),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Pie(_)));
        assert_eq!(fig.plot_axes_indices()[0], 1);

        let err = futures::executor::block_on(pie_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["Only".into()], vec![1, 1]).unwrap(),
            ),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("labels must match values length"));
    }

    #[test]
    fn pie_rejects_negative_values() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let err =
            futures::executor::block_on(pie_builtin(vec![Value::Tensor(vec_tensor(&[1.0, -1.0]))]))
                .unwrap_err();
        assert!(err.to_string().contains("nonnegative"));
    }

    #[test]
    fn pie_supports_format_string_labels_and_nonbinary_explode() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(pie_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[0.0, 3.0])),
            Value::String("%.1f%%".into()),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Pie(pie) = fig.plots().next().unwrap() else {
            panic!("expected pie");
        };
        assert_eq!(pie.explode, vec![false, true]);
        assert_eq!(pie.label_format.as_deref(), Some("%.1f%%"));
        let labels = pie
            .slice_meta()
            .into_iter()
            .map(|s| s.label)
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["33.3%", "66.7%"]);
    }

    #[test]
    fn pie_descriptor_includes_core_signatures() {
        let labels: Vec<&str> = runmat_builtins::PIE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = pie(X)"));
        assert!(labels.contains(&"h = pie(X, explode)"));
        assert!(labels.contains(&"h = pie(X, explode, labels)"));
    }

    #[test]
    fn pie_missing_values_uses_stable_identifier() {
        let err = futures::executor::block_on(pie_builtin(Vec::new()))
            .expect_err("expected pie argument validation error");
        assert_eq!(err.identifier(), PIE_ERROR_INVALID_ARGUMENT.identifier);
    }
}
