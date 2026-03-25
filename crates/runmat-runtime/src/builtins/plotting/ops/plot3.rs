use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_plot::plots::Line3Plot;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::string_type;

use super::common::numeric_triplet;
use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_line_style_args, LineAppearance, LineStyleParseOptions};

const BUILTIN_NAME: &str = "plot3";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::plot3")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "plot3",
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
    notes: "plot3 is a plotting sink; GPU inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::plot3")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "plot3",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "plot3 performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "plot3",
    category = "plotting",
    summary = "Create MATLAB-compatible 3-D line plots.",
    keywords = "plot3,line,3d,visualization",
    sink = true,
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::plot3"
)]
pub async fn plot3_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    let (x, y, z, rest) = parse_plot3_args(args)?;
    let appearance =
        parse_line_style_args(&rest, &LineStyleParseOptions::generic(BUILTIN_NAME))?.appearance;
    let x = NumericInput::from_value(x, BUILTIN_NAME)?;
    let y = NumericInput::from_value(y, BUILTIN_NAME)?;
    let z = NumericInput::from_value(z, BUILTIN_NAME)?;

    let opts = PlotRenderOptions {
        title: "3-D Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };

    let mut inputs = Some((x, y, z, appearance));
    render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let (x, y, z, appearance) = inputs.take().expect("plot3 consumed once");
        let x = x.into_tensor(BUILTIN_NAME)?;
        let y = y.into_tensor(BUILTIN_NAME)?;
        let z = z.into_tensor(BUILTIN_NAME)?;
        let (x, y, z) = numeric_triplet(x, y, z, BUILTIN_NAME)?;
        let plot = build_line3_plot(x, y, z, &appearance)?;
        figure.add_line3_plot_on_axes(plot, axes);
        Ok(())
    })
}

fn parse_plot3_args(args: Vec<Value>) -> crate::BuiltinResult<(Value, Value, Value, Vec<Value>)> {
    if args.len() < 3 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "plot3: expected X, Y, and Z inputs",
        ));
    }
    let mut it = args.into_iter();
    let x = it.next().unwrap();
    let y = it.next().unwrap();
    let z = it.next().unwrap();
    Ok((x, y, z, it.collect()))
}

fn build_line3_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    appearance: &LineAppearance,
) -> crate::BuiltinResult<Line3Plot> {
    Ok(Line3Plot::new(x, y, z)
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("plot3: {e}")))?
        .with_style(
            appearance.color,
            appearance.line_width,
            appearance.line_style,
        )
        .with_label("Data"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::{NumericDType, Tensor};
    use runmat_plot::plots::PlotElement;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: NumericDType::F64,
        }
    }

    #[test]
    fn plot3_builds_line3_plot() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = futures::executor::block_on(plot3_builtin(vec![
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Line3(_)));
    }
}
