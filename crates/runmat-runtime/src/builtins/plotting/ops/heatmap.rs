use std::cell::RefCell;
use std::rc::Rc;

use runmat_builtins::{StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};

use super::common::{gather_tensor_from_gpu_async, SurfaceDataInput};
use super::op_common::surface_inputs::AxisSource;
use super::op_common::value_as_text_string;
use super::state::{color_limits_snapshot, render_active_plot, PlotRenderOptions};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const BUILTIN_NAME: &str = "heatmap";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::heatmap")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "heatmap",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "heatmap is a plotting sink; inputs are gathered to build labeled HeatmapChart state.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::heatmap")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "heatmap",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "heatmap terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "heatmap",
    category = "plotting",
    summary = "Create a MATLAB-compatible heatmap chart.",
    keywords = "heatmap,plotting,chart,colormap,matrix visualization",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::heatmap"
)]
pub async fn heatmap_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let ParsedHeatmap {
        x_labels,
        y_labels,
        color_data,
        rest,
    } = parse_heatmap_args(args).await?;

    crate::builtins::plotting::properties::validate_heatmap_property_pairs(
        &rest,
        x_labels.len(),
        y_labels.len(),
        BUILTIN_NAME,
    )?;

    let rows = color_data.rows;
    let cols = color_data.cols;
    let render_data = transpose_for_surface(&color_data);
    let x_axis = AxisSource::Host(default_axis(cols));
    let y_axis = AxisSource::Host(default_axis(rows));
    let color_limits = color_limits_snapshot();
    let mut surface = super::image::build_indexed_image_surface(
        &SurfaceDataInput::Host(render_data),
        &x_axis,
        &y_axis,
        ColorMap::Parula,
        color_limits,
    )
    .await?;
    surface = surface
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::None);
    if color_limits.is_some() {
        surface = surface.with_color_limits(color_limits);
    }

    let mut surface = Some(surface);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let render_x_labels = x_labels.clone();
    let render_y_labels = y_labels.clone();
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "",
            x_label: "",
            y_label: "",
            axis_equal: true,
            ..Default::default()
        },
        move |figure, axes| {
            let plot_index = figure.add_surface_plot_on_axes(
                surface.take().expect("heatmap plot consumed once"),
                axes,
            );
            figure.set_axes_colorbar_enabled(axes, true);
            figure.set_axes_tick_labels(
                axes,
                Some(render_x_labels.clone()),
                Some(render_y_labels.clone()),
            );
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_heatmap_handle(
        figure_handle,
        axes,
        plot_index,
        x_labels,
        y_labels,
        color_data,
    );
    if !rest.is_empty() {
        let plot_handle = crate::builtins::plotting::properties::resolve_plot_handle(
            &Value::Num(handle),
            BUILTIN_NAME,
        )?;
        crate::builtins::plotting::properties::set_properties(plot_handle, &rest, BUILTIN_NAME)?;
    }
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

struct ParsedHeatmap {
    x_labels: Vec<String>,
    y_labels: Vec<String>,
    color_data: Tensor,
    rest: Vec<Value>,
}

async fn parse_heatmap_args(args: Vec<Value>) -> crate::BuiltinResult<ParsedHeatmap> {
    match args.len() {
        0 => Err(crate::builtins::plotting::plotting_error(
            BUILTIN_NAME,
            "heatmap: expected CData or XValues,YValues,CData input",
        )),
        1 => {
            let color_data = cdata_tensor(args.into_iter().next().expect("one arg")).await?;
            let x_labels = default_labels(color_data.cols);
            let y_labels = default_labels(color_data.rows);
            Ok(ParsedHeatmap {
                x_labels,
                y_labels,
                color_data,
                rest: Vec::new(),
            })
        }
        2 => Err(crate::builtins::plotting::plotting_error(
            BUILTIN_NAME,
            "heatmap: expected CData or XValues,YValues,CData input",
        )),
        _ => {
            let mut it = args.into_iter();
            let x = it.next().expect("x labels");
            let y = it.next().expect("y labels");
            let c = it.next().expect("cdata");
            let rest: Vec<Value> = it.collect();
            let color_data = cdata_tensor(c).await?;
            let x_labels = labels_from_value(&x, color_data.cols, "XValues")?;
            let y_labels = labels_from_value(&y, color_data.rows, "YValues")?;
            Ok(ParsedHeatmap {
                x_labels,
                y_labels,
                color_data,
                rest,
            })
        }
    }
}

async fn cdata_tensor(value: Value) -> crate::BuiltinResult<Tensor> {
    let tensor = match value {
        Value::GpuTensor(handle) => gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?,
        other => Tensor::try_from(&other).map_err(|e| {
            crate::builtins::plotting::plotting_error(BUILTIN_NAME, format!("heatmap: {e}"))
        })?,
    };
    if tensor.rows == 0 || tensor.cols == 0 {
        return Err(crate::builtins::plotting::plotting_error(
            BUILTIN_NAME,
            "heatmap: CData must contain at least a 2-D grid",
        ));
    }
    Ok(tensor)
}

fn labels_from_value(
    value: &Value,
    expected_len: usize,
    axis_name: &str,
) -> crate::BuiltinResult<Vec<String>> {
    let labels = match value {
        Value::StringArray(StringArray { data, .. }) => data.clone(),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| {
                value_as_text_string(item).ok_or_else(|| {
                    crate::builtins::plotting::plotting_error(
                        BUILTIN_NAME,
                        format!("heatmap: {axis_name} cell values must be text"),
                    )
                })
            })
            .collect::<crate::BuiltinResult<Vec<_>>>()?,
        Value::CharArray(chars) if chars.rows == 1 => vec![chars.data.iter().collect()],
        Value::String(text) => vec![text.clone()],
        Value::Tensor(tensor) => tensor.data.iter().map(|v| v.to_string()).collect(),
        Value::Int(i) => vec![i.to_i64().to_string()],
        Value::Num(v) => vec![v.to_string()],
        other => {
            return Err(crate::builtins::plotting::plotting_error(
                BUILTIN_NAME,
                format!("heatmap: unsupported {axis_name} value {other:?}"),
            ))
        }
    };
    if labels.len() != expected_len {
        return Err(crate::builtins::plotting::plotting_error(
            BUILTIN_NAME,
            format!("heatmap: {axis_name} must have {expected_len} labels"),
        ));
    }
    Ok(labels)
}

fn default_labels(len: usize) -> Vec<String> {
    (1..=len).map(|idx| idx.to_string()).collect()
}

fn default_axis(len: usize) -> Vec<f64> {
    (1..=len).map(|idx| idx as f64).collect()
}

fn transpose_for_surface(tensor: &Tensor) -> Tensor {
    let mut data = vec![0.0; tensor.data.len()];
    for row in 0..tensor.rows {
        for col in 0..tensor.cols {
            let src = row + tensor.rows * col;
            let dst = col + tensor.cols * row;
            data[dst] = tensor.data[src];
        }
    }
    Tensor {
        data,
        shape: vec![tensor.cols, tensor.rows],
        rows: tensor.cols,
        cols: tensor.rows,
        dtype: tensor.dtype,
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
    use runmat_builtins::{CellArray, NumericDType, Value};
    use runmat_plot::plots::PlotElement;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor {
            data,
            shape: vec![rows, cols],
            rows,
            cols,
            dtype: NumericDType::F64,
        }
    }

    #[test]
    fn heatmap_cdata_builds_heatmap_handle() {
        let _guard = setup();
        let handle = futures::executor::block_on(heatmap_builtin(vec![Value::Tensor(tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            2,
            3,
        ))]))
        .expect("heatmap should render");
        assert!(handle.is_finite());

        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert!(surface.flatten_z);
        assert!(surface.image_mode);
        assert_eq!(surface.x_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(surface.y_data, vec![1.0, 2.0]);
        assert!(fig.axes_metadata(0).unwrap().colorbar_enabled);
    }

    #[test]
    fn heatmap_accepts_labels_and_exposes_chart_properties() {
        let _guard = setup();
        let x = CellArray::new(
            vec![
                Value::String("Small".into()),
                Value::String("Medium".into()),
                Value::String("Large".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let y = CellArray::new(
            vec![
                Value::String("Green".into()),
                Value::String("Red".into()),
                Value::String("Blue".into()),
                Value::String("Gray".into()),
            ],
            1,
            4,
        )
        .unwrap();
        let cdata = tensor(
            vec![
                45.0, 43.0, 32.0, 23.0, 60.0, 54.0, 94.0, 95.0, 32.0, 76.0, 68.0, 58.0,
            ],
            4,
            3,
        );
        let handle = futures::executor::block_on(heatmap_builtin(vec![
            Value::Cell(x),
            Value::Cell(y),
            Value::Tensor(cdata),
        ]))
        .expect("heatmap should render");

        set_builtin(vec![
            Value::Num(handle),
            Value::String("Title".into()),
            Value::String("T-Shirt Orders".into()),
            Value::String("XLabel".into()),
            Value::String("Sizes".into()),
            Value::String("YLabel".into()),
            Value::String("Colors".into()),
        ])
        .unwrap();

        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Title".into())]).unwrap(),
            Value::String("T-Shirt Orders".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("XLabel".into())]).unwrap(),
            Value::String("Sizes".into())
        );
        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(
            meta.x_tick_labels.as_ref().unwrap(),
            &vec![
                "Small".to_string(),
                "Medium".to_string(),
                "Large".to_string()
            ]
        );
        assert_eq!(
            meta.y_tick_labels.as_ref().unwrap(),
            &vec![
                "Green".to_string(),
                "Red".to_string(),
                "Blue".to_string(),
                "Gray".to_string()
            ]
        );
        let labels = get_builtin(vec![
            Value::Num(handle),
            Value::String("XDisplayLabels".into()),
        ])
        .unwrap();
        let Value::StringArray(labels) = labels else {
            panic!("expected string array");
        };
        assert_eq!(labels.data, vec!["Small", "Medium", "Large"]);

        set_builtin(vec![
            Value::Num(handle),
            Value::String("XDisplayLabels".into()),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::String("S".into()),
                        Value::String("M".into()),
                        Value::String("L".into()),
                    ],
                    1,
                    3,
                )
                .unwrap(),
            ),
            Value::String("YDisplayLabels".into()),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::String("G".into()),
                        Value::String("R".into()),
                        Value::String("B".into()),
                        Value::String("Y".into()),
                    ],
                    1,
                    4,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(
            meta.x_tick_labels.as_ref().unwrap(),
            &vec!["S".to_string(), "M".to_string(), "L".to_string()]
        );
        assert_eq!(
            meta.y_tick_labels.as_ref().unwrap(),
            &vec![
                "G".to_string(),
                "R".to_string(),
                "B".to_string(),
                "Y".to_string()
            ]
        );
    }

    #[test]
    fn heatmap_rejects_bad_property_pairs_before_mutating_figure() {
        let _guard = setup();
        let before = clone_figure(current_figure_handle())
            .map(|figure| figure.plots().count())
            .unwrap_or(0);

        let err = futures::executor::block_on(heatmap_builtin(vec![
            Value::Cell(
                CellArray::new(
                    vec![Value::String("A".into()), Value::String("B".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("C".into()), Value::String("D".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
            Value::Tensor(tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
            Value::String("NotAHeatmapProperty".into()),
            Value::Num(1.0),
        ]))
        .expect_err("invalid property should fail");
        assert!(err.to_string().contains("unsupported heatmap property"));

        let after = clone_figure(current_figure_handle())
            .map(|figure| figure.plots().count())
            .unwrap_or(0);
        assert_eq!(after, before);
    }
}
