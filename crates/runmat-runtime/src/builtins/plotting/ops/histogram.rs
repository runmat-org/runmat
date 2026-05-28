use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::BarChart;
use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

use super::bar::apply_bar_style;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_bar_style_args, BarStyleDefaults};

const BUILTIN_NAME: &str = "histogram";
const HIST_BAR_WIDTH: f32 = 0.95;
const HIST_DEFAULT_COLOR: glam::Vec4 = glam::Vec4::new(0.15, 0.5, 0.8, 0.95);
const HIST_DEFAULT_LABEL: &str = "Frequency";

const HISTOGRAM_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created histogram chart.",
}];

const HISTOGRAM_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input sample data.",
}];

const HISTOGRAM_INPUTS_X_BINS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input sample data.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bin count scalar or explicit edge vector.",
    },
];

const HISTOGRAM_INPUTS_X_NAMEVALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input sample data.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value pairs for histogram options and styling.",
    },
];

const HISTOGRAM_INPUTS_X_BINS_NAMEVALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input sample data.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bin count scalar or explicit edge vector.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional name/value pairs for options and styling.",
    },
];

const HISTOGRAM_INPUTS_AX_X: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input sample data.",
    },
];

const HISTOGRAM_INPUTS_AX_X_BINS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input sample data.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bin count scalar or explicit edge vector.",
    },
];

const HISTOGRAM_INPUTS_AX_X_NAMEVALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input sample data.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value pairs for options and styling.",
    },
];

const HISTOGRAM_INPUTS_AX_X_BINS_NAMEVALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input sample data.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bin count scalar or explicit edge vector.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional name/value pairs for options and styling.",
    },
];

const HISTOGRAM_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = histogram(X)",
        inputs: &HISTOGRAM_INPUTS_X,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(X, bins)",
        inputs: &HISTOGRAM_INPUTS_X_BINS,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(X, Name, Value, ...)",
        inputs: &HISTOGRAM_INPUTS_X_NAMEVALUE,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(X, bins, Name, Value, ...)",
        inputs: &HISTOGRAM_INPUTS_X_BINS_NAMEVALUE,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(ax, X)",
        inputs: &HISTOGRAM_INPUTS_AX_X,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(ax, X, bins)",
        inputs: &HISTOGRAM_INPUTS_AX_X_BINS,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(ax, X, Name, Value, ...)",
        inputs: &HISTOGRAM_INPUTS_AX_X_NAMEVALUE,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(ax, X, bins, Name, Value, ...)",
        inputs: &HISTOGRAM_INPUTS_AX_X_BINS_NAMEVALUE,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
];

const HISTOGRAM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTOGRAM.INVALID_ARGUMENT",
    identifier: Some("RunMat:histogram:InvalidArgument"),
    when: "Histogram input arrays, bins, or name/value options are malformed or incompatible.",
    message: "histogram: invalid argument",
};

const HISTOGRAM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTOGRAM.INTERNAL",
    identifier: Some("RunMat:histogram:Internal"),
    when: "Internal render preparation or histogram state registration fails.",
    message: "histogram: internal operation failed",
};

const HISTOGRAM_ERRORS: [BuiltinErrorDescriptor; 2] =
    [HISTOGRAM_ERROR_INVALID_ARGUMENT, HISTOGRAM_ERROR_INTERNAL];

pub const HISTOGRAM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HISTOGRAM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HISTOGRAM_ERRORS,
};

fn histogram_descriptor_error(
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

fn map_histogram_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    histogram_descriptor_error(&HISTOGRAM_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_histogram_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    histogram_descriptor_error(&HISTOGRAM_ERROR_INTERNAL, err.message)
}

fn histogram_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    histogram_descriptor_error(&HISTOGRAM_ERROR_INVALID_ARGUMENT, detail)
}

fn histogram_internal(detail: impl AsRef<str>) -> RuntimeError {
    histogram_descriptor_error(&HISTOGRAM_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::histogram")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "histogram",
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
    notes: "Histogram rendering terminates fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::histogram")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "histogram",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "histogram terminates fusion graphs and produces plotting output.",
};

#[runtime_builtin(
    name = "histogram",
    category = "plotting",
    summary = "Plot a MATLAB-compatible histogram.",
    keywords = "histogram,hist,histcounts,frequency",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::histogram::HISTOGRAM_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::histogram"
)]
pub async fn histogram_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target_axes, data, rest) = parse_histogram_call(args)?;
    let (histcounts_args, style_args) = split_histogram_args(&rest);
    let eval = crate::builtins::stats::hist::histcounts::evaluate(data, &histcounts_args)
        .await
        .map_err(map_histogram_invalid_argument)?;
    let (counts_value, edges_value) = eval.into_pair();
    let counts = Tensor::try_from(&counts_value)
        .map_err(|e| histogram_invalid_argument(format!("cannot convert counts tensor: {e}")))?;
    let edges = Tensor::try_from(&edges_value)
        .map_err(|e| histogram_invalid_argument(format!("cannot convert edge tensor: {e}")))?;

    let defaults = BarStyleDefaults::new(HIST_DEFAULT_COLOR, HIST_BAR_WIDTH);
    let style = parse_bar_style_args(BUILTIN_NAME, &style_args, defaults)
        .map_err(map_histogram_invalid_argument)?;
    let labels = histogram_labels_from_edges(&edges.data);
    let mut chart = BarChart::new(labels, counts.data.clone())
        .map_err(|e| histogram_internal(format!("chart construction failed: {e}")))?;
    chart.set_histogram_bin_edges(edges.data.clone());
    apply_bar_style(&mut chart, &style, HIST_DEFAULT_LABEL);

    let normalization = infer_normalization(&histcounts_args);
    let y_label = match normalization.as_str() {
        "count" => "Count",
        "probability" => "Probability",
        "percentage" => "Percentage",
        "countdensity" => "CountDensity",
        "pdf" => "PDF",
        "cdf" => "CDF",
        _ => HIST_DEFAULT_LABEL,
    };

    let mut chart_opt = Some(chart);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Histogram",
            x_label: "Bin",
            y_label,
            ..Default::default()
        },
        move |figure, axes| {
            let axes = target_axes.unwrap_or(axes);
            let plot_index = figure.add_bar_chart_on_axes(
                chart_opt.take().expect("histogram chart consumed once"),
                axes,
            );
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_histogram_handle(
        figure_handle,
        axes,
        plot_index,
        edges.data.clone(),
        counts.data.clone(),
        normalization.clone(),
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_histogram_internal(err));
    }
    Ok(handle)
}

fn parse_histogram_call(
    args: Vec<Value>,
) -> crate::BuiltinResult<(Option<usize>, Value, Vec<Value>)> {
    if args.is_empty() {
        return Err(histogram_invalid_argument("expected data input"));
    }
    let mut it = args.into_iter();
    let first = it.next().unwrap();
    if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(_, axes)) =
        crate::builtins::plotting::properties::resolve_plot_handle(&first, BUILTIN_NAME)
    {
        let data = it
            .next()
            .ok_or_else(|| histogram_invalid_argument("expected data after axes handle"))?;
        return Ok((Some(axes), data, it.collect()));
    }
    Ok((None, first, it.collect()))
}

fn split_histogram_args(args: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let mut hist_args = Vec::new();
    let mut style_args = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        let Some(key) = value_as_string(&args[idx]) else {
            hist_args.extend_from_slice(&args[idx..]);
            break;
        };
        if idx + 1 >= args.len() {
            hist_args.push(args[idx].clone());
            break;
        }
        let lower = key.trim().to_ascii_lowercase();
        let target = match lower.as_str() {
            "binedges" | "numbins" | "binwidth" | "binlimits" | "normalization" | "binmethod" => {
                &mut hist_args
            }
            _ => &mut style_args,
        };
        target.push(args[idx].clone());
        target.push(args[idx + 1].clone());
        idx += 2;
    }
    (hist_args, style_args)
}

fn histogram_labels_from_edges(edges: &[f64]) -> Vec<String> {
    edges
        .windows(2)
        .map(|pair| format!("[{:.3}, {:.3})", pair[0], pair[1]))
        .collect()
}

fn infer_normalization(args: &[Value]) -> String {
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if let Some(key) = value_as_string(&args[idx]) {
            if key.trim().eq_ignore_ascii_case("Normalization") {
                if let Some(v) = value_as_string(&args[idx + 1]) {
                    return v.trim().to_ascii_lowercase();
                }
            }
        }
        idx += 2;
    }
    "count".to_string()
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(chars) => Some(chars.data.iter().collect()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    #[test]
    fn histogram_accepts_edges_vector_semantics() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let out = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(tensor_from(&[0.1, 0.2, 0.9, 1.1])),
            Value::Tensor(Tensor {
                data: vec![0.0, 1.0, 2.0],
                shape: vec![1, 3],
                rows: 1,
                cols: 3,
                dtype: runmat_builtins::NumericDType::F64,
            }),
        ]));
        let out = out.unwrap();
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(out), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.data, vec![3.0, 1.0]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Bar(_)));
    }

    #[test]
    fn histogram_supports_extended_normalizations() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let out = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])),
            Value::String("NumBins".into()),
            Value::Num(3.0),
            Value::String("Normalization".into()),
            Value::String("cdf".into()),
        ]));
        let out = out.unwrap();
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(out), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.data.last().copied().unwrap_or_default(), 1.0);
    }

    #[test]
    fn histogram_supports_axes_target_and_histcounts_semantics() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let out = futures::executor::block_on(histogram_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[
                0.0,
                1.0,
                2.0,
                100.0,
                f64::NAN,
                f64::INFINITY,
            ])),
            Value::Tensor(Tensor {
                data: vec![0.0, 1.0, 2.0, 3.0],
                shape: vec![1, 4],
                rows: 1,
                cols: 4,
                dtype: runmat_builtins::NumericDType::F64,
            }),
        ]))
        .unwrap();
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(out), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.data, vec![1.0, 1.0, 1.0]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
    }

    #[test]
    fn histogram_supports_binmethod_auto_and_countdensity() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let out = futures::executor::block_on(histogram_builtin(vec![
            Value::Tensor(tensor_from(&[1.0, 1.1, 2.0, 2.1, 2.2, 5.0])),
            Value::String("BinMethod".into()),
            Value::String("auto".into()),
            Value::String("Normalization".into()),
            Value::String("countdensity".into()),
        ]))
        .unwrap();
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(out), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert!(!counts.data.is_empty());
        assert!(counts.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn histogram_returns_object_handle_with_get_set_semantics() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle =
            futures::executor::block_on(histogram_builtin(vec![Value::Tensor(tensor_from(&[
                1.0, 2.0, 3.0,
            ]))]))
            .unwrap();
        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("histogram".into()));
        set_builtin(vec![
            Value::Num(handle),
            Value::String("Normalization".into()),
            Value::String("cdf".into()),
        ])
        .unwrap();
        let norm = get_builtin(vec![
            Value::Num(handle),
            Value::String("Normalization".into()),
        ])
        .unwrap();
        assert_eq!(norm, Value::String("cdf".into()));
    }

    #[test]
    fn histogram_supports_displayname_property() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle =
            futures::executor::block_on(histogram_builtin(vec![Value::Tensor(tensor_from(&[
                1.0, 2.0, 3.0,
            ]))]))
            .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayName".into()),
            Value::String("Process A".into()),
        ])
        .unwrap();
        let display_name = get_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayName".into()),
        ])
        .unwrap();
        assert_eq!(display_name, Value::String("Process A".into()));

        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        let Value::Struct(props) = props else {
            panic!("expected struct from get(handle)");
        };
        assert_eq!(
            props.fields.get("DisplayName"),
            Some(&Value::String("Process A".into()))
        );
    }

    #[test]
    fn histogram_descriptor_includes_axes_signature() {
        let labels: Vec<&str> = HISTOGRAM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = histogram(X)"));
        assert!(labels.contains(&"h = histogram(X, bins)"));
        assert!(labels.contains(&"h = histogram(ax, X, Name, Value, ...)"));
    }

    #[test]
    fn histogram_missing_data_uses_stable_identifier() {
        let err = futures::executor::block_on(histogram_builtin(Vec::new()))
            .expect_err("expected histogram argument validation error");
        assert_eq!(
            err.identifier(),
            HISTOGRAM_ERROR_INVALID_ARGUMENT.identifier
        );
    }
}
