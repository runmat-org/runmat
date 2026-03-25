use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{LineMarkerAppearance, StemPlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::string_type;

use super::common::numeric_pair;
use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{
    marker_metadata_from_appearance, parse_line_style_args, LineAppearance, LineStyleParseOptions,
};

const BUILTIN_NAME: &str = "stem";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::stem")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "stem",
    op_kind: GpuOpKind::Custom("plot-render"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "stem is a plotting sink; GPU inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::stem")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "stem",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "stem performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "stem",
    category = "plotting",
    summary = "Render MATLAB-compatible stem plots.",
    keywords = "stem,plotting,discrete",
    sink = true,
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::stem"
)]
pub fn stem_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    let (x, y, rest) = parse_stem_args(args)?;
    let parsed = parse_stem_style_args(&rest)?;
    let mut x_input = Some(NumericInput::from_value(x, BUILTIN_NAME)?);
    let mut y_input = Some(NumericInput::from_value(y, BUILTIN_NAME)?);
    let opts = PlotRenderOptions {
        title: "Stem",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let x = x_input
            .take()
            .expect("stem x consumed once")
            .into_tensor(BUILTIN_NAME)?;
        let y = y_input
            .take()
            .expect("stem y consumed once")
            .into_tensor(BUILTIN_NAME)?;
        let (x, y) = numeric_pair(x, y, BUILTIN_NAME)?;
        let mut plot = StemPlot::new(x, y)
            .map_err(|e| plotting_error(BUILTIN_NAME, format!("stem: {e}")))?
            .with_style(
                parsed.appearance.color,
                parsed.appearance.line_width,
                parsed.appearance.line_style,
                parsed.baseline,
            )
            .with_baseline_style(parsed.appearance.color, parsed.baseline_visible)
            .with_label(parsed.label.clone().unwrap_or_else(|| "Data".into()));
        if let Some(marker) = parsed.marker.clone() {
            plot.set_marker(Some(marker));
        }
        figure.add_stem_plot_on_axes(plot, axes);
        Ok(())
    })
}

struct ParsedStemStyle {
    appearance: LineAppearance,
    marker: Option<LineMarkerAppearance>,
    label: Option<String>,
    baseline: f64,
    baseline_visible: bool,
}

fn parse_stem_style_args(args: &[Value]) -> crate::BuiltinResult<ParsedStemStyle> {
    let parsed = parse_line_style_args(args, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    let marker = marker_metadata_from_appearance(&parsed.appearance);
    let mut baseline = 0.0;
    let mut baseline_visible = true;
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]) {
            let lower = key.trim().to_ascii_lowercase();
            match lower.as_str() {
                "basevalue" => {
                    baseline = super::style::value_as_f64(&args[idx + 1]).ok_or_else(|| {
                        plotting_error(BUILTIN_NAME, "stem: BaseValue must be numeric")
                    })?;
                }
                "baseline" => {
                    baseline_visible =
                        super::style::value_as_bool(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "stem: BaseLine must be logical")
                        })?;
                }
                "filled" => {
                    // accepted for compatibility through marker face handling
                }
                _ => {}
            }
        }
        idx += 2;
    }
    Ok(ParsedStemStyle {
        appearance: parsed.appearance,
        marker,
        label: parsed.label,
        baseline,
        baseline_visible,
    })
}

fn parse_stem_args(args: Vec<Value>) -> crate::BuiltinResult<(Value, Value, Vec<Value>)> {
    if args.is_empty() {
        return Err(plotting_error(
            BUILTIN_NAME,
            "stem: expected at least Y data",
        ));
    }
    let mut it = args.into_iter();
    let first = it.next().unwrap();
    let Some(second) = it.next() else {
        let y = first;
        let y_tensor =
            Tensor::try_from(&y).map_err(|e| plotting_error(BUILTIN_NAME, format!("stem: {e}")))?;
        let len = y_tensor.data.len();
        let x = Value::Tensor(Tensor {
            data: (1..=len).map(|i| i as f64).collect(),
            shape: vec![len],
            rows: len,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        });
        return Ok((x, y, Vec::new()));
    };
    if matches!(second, Value::String(_) | Value::CharArray(_)) {
        let y = first;
        let y_tensor =
            Tensor::try_from(&y).map_err(|e| plotting_error(BUILTIN_NAME, format!("stem: {e}")))?;
        let len = y_tensor.data.len();
        let x = Value::Tensor(Tensor {
            data: (1..=len).map(|i| i as f64).collect(),
            shape: vec![len],
            rows: len,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        });
        let mut rest = vec![second];
        rest.extend(it);
        return Ok((x, y, rest));
    }
    Ok((first, second, it.collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn stem_builds_plot_from_y_only() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = stem_builtin(vec![Value::Tensor(tensor_from(&[1.0, 2.0, 3.0]))]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Stem(_)));
    }
}
