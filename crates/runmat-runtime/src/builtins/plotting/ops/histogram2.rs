use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    HISTOGRAM2_ERROR_INTERNAL as ERROR_INTERNAL,
    HISTOGRAM2_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
};
use std::cell::RefCell;
use std::rc::Rc;

use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_arguments};
use crate::builtins::plotting::state::{
    color_limits_snapshot, render_active_plot, PlotRenderOptions,
};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "histogram2";

const DEFAULT_FACE_ALPHA: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Histogram2DisplayStyle {
    Bar3,
    Tile,
}

impl Histogram2DisplayStyle {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Bar3 => "bar3",
            Self::Tile => "tile",
        }
    }
}

#[derive(Clone, Debug)]
struct Histogram2Options {
    histcounts_args: Vec<Value>,
    display_style: Histogram2DisplayStyle,
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<String>,
    normalization: String,
}

pub(crate) struct Histogram2Chart {
    pub(crate) values: Tensor,
    pub(crate) raw_counts: Tensor,
    pub(crate) x_bin_edges: Vec<f64>,
    pub(crate) y_bin_edges: Vec<f64>,
    pub(crate) surface: SurfacePlot,
}

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INVALID_ARGUMENT, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INTERNAL, message)
}

fn map_plot_error(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        crate::build_runtime_error(format!("histogram2: {}", err.message))
            .with_builtin(NAME)
            .with_identifier(ERROR_INVALID_ARGUMENT.identifier.unwrap())
            .with_source(err)
            .build()
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::histogram2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "histogram2",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Omit,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "histogram2 is a plotting sink built on histcounts2; current builds materialise the 2-D bin grid before rendering.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::histogram2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "histogram2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "histogram2 terminates fusion graphs and produces plotting output.",
};

#[runtime_builtin(
    name = "histogram2",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::histogram2"
)]
pub async fn histogram2_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "histogram2: too many output arguments",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "histogram2")?;
    let _arguments = args.fork();
    work.finish(histogram2_impl(args, &work).await)
}
async fn histogram2_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (axes_target, args) = split_leading_axes_arguments(args, NAME).map_err(map_plot_error)?;
    work.check()?;
    apply_axes_target(axes_target, NAME).map_err(map_plot_error)?;

    if args.len() < 2 {
        return Err(invalid("histogram2: expected X and Y inputs"));
    }
    let x = args[0].clone();
    let y = args[1].clone();
    let retained_data = histogram2_data_value(&x, &y).await?;
    let normalization_denominator = retained_data.rows() as f64;
    let rest = args.into_range(2..)?;
    let options = parse_options(rest)?;
    crate::builtins::stats::hist::histcounts2::ensure_histcounts2_public_syntax(
        crate::call::FunctionArgumentView::from_slice(&options.histcounts_args),
    )?;

    let chart = build_chart(x.clone(), y.clone(), &options, normalization_denominator).await?;
    work.check()?;
    render_histogram2(
        options,
        chart,
        Some(retained_data),
        normalization_denominator,
    )
}

fn parse_options(args: crate::call::BuiltinArguments) -> BuiltinResult<Histogram2Options> {
    let mut histcounts_args = Vec::new();
    let mut display_style = Histogram2DisplayStyle::Bar3;
    let mut show_empty_bins = false;
    let mut face_alpha = DEFAULT_FACE_ALPHA;
    let mut display_name = None;
    let mut normalization = "count".to_string();
    let mut index = 0usize;

    while index < args.len() {
        if let Some(key) = value_as_string(&args[index]).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "histogram2",
                source.to_string(),
                source,
            )
        })? {
            if index + 1 >= args.len() {
                return Err(invalid(format!(
                    "histogram2: missing value for option '{key}'"
                )));
            }
            let value = &args[index + 1];
            match key.trim().to_ascii_lowercase().as_str() {
                "displaystyle" => {
                    display_style = parse_display_style(value)?;
                }
                "showemptybins" => {
                    show_empty_bins = option_bool(value, "ShowEmptyBins")?;
                }
                "facealpha" => {
                    face_alpha = option_scalar(value, "FaceAlpha")?;
                    validate_face_alpha(face_alpha)?;
                }
                "displayname" => {
                    display_name = Some(
                        tensor::value_to_string(value)
                            .ok_or_else(|| invalid("histogram2: DisplayName must be text"))?,
                    );
                }
                "normalization" => {
                    normalization = tensor::value_to_string(value)
                        .ok_or_else(|| invalid("histogram2: Normalization must be text"))?
                        .trim()
                        .to_ascii_lowercase();
                    histcounts_args.push(args[index].clone());
                    histcounts_args.push(value.clone());
                }
                "numbins" | "xbinwidth" | "ybinwidth" | "binwidth" | "xbinlimits"
                | "ybinlimits" | "binlimits" | "xbinedges" | "ybinedges" | "binmethod"
                | "xbinmethod" | "ybinmethod" => {
                    histcounts_args.push(args[index].clone());
                    histcounts_args.push(value.clone());
                }
                "facecolor" | "edgecolor" | "edgealpha" | "linestyle" => {
                    // Accepted as rendering style compatibility; RunMat's SurfacePlot backend maps
                    // histogram2 through colormapped surfaces and does not yet expose per-bin edges.
                }
                other => {
                    return Err(invalid(format!(
                        "histogram2: unrecognised option '{other}'"
                    )));
                }
            }
            index += 2;
        } else {
            histcounts_args.push(args[index].clone());
            index += 1;
        }
    }

    validate_normalization(&normalization)?;
    Ok(Histogram2Options {
        histcounts_args,
        display_style,
        show_empty_bins,
        face_alpha,
        display_name,
        normalization,
    })
}

async fn build_chart(
    x: Value,
    y: Value,
    options: &Histogram2Options,
    normalization_denominator: f64,
) -> BuiltinResult<Histogram2Chart> {
    let raw_args = histcounts_count_args(&options.histcounts_args);
    let raw_eval = crate::builtins::stats::hist::histcounts2::evaluate(
        x,
        y,
        crate::call::FunctionArgumentView::from_slice(&raw_args),
    )
    .await
    .map_err(map_histcounts_error)?;
    let (raw_counts_value, x_edges_value, y_edges_value) = raw_eval.into_triple();
    let raw_counts = Tensor::try_from(&raw_counts_value)
        .map_err(|err| invalid(format!("histogram2: cannot convert raw bin counts: {err}")))?;
    let x_edges = Tensor::try_from(&x_edges_value).map_err(|err| {
        invalid(format!(
            "histogram2: cannot convert XBinEdges tensor: {err}"
        ))
    })?;
    let y_edges = Tensor::try_from(&y_edges_value).map_err(|err| {
        invalid(format!(
            "histogram2: cannot convert YBinEdges tensor: {err}"
        ))
    })?;
    let x_edges = tensor::tensor_into_values_f64(x_edges);
    let y_edges = tensor::tensor_into_values_f64(y_edges);
    let values = apply_normalization_2d(
        &raw_counts,
        &x_edges,
        &y_edges,
        &options.normalization,
        normalization_denominator,
    )?;

    let surface = build_surface_from_values(
        &values,
        &x_edges,
        &y_edges,
        options.display_style,
        options.show_empty_bins,
        options.face_alpha,
        options.display_name.as_deref(),
        color_limits_snapshot(),
    )?;
    Ok(Histogram2Chart {
        values,
        raw_counts,
        x_bin_edges: x_edges,
        y_bin_edges: y_edges,
        surface,
    })
}

fn map_histcounts_error(err: RuntimeError) -> RuntimeError {
    invalid(err.message)
}

fn histcounts_count_args(args: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        if crate::builtins::plotting::style::value_text_equals_ascii(
            &args[index],
            "Normalization",
            true,
        ) {
            index += 2;
        } else {
            out.push(args[index].clone());
            index += 1;
        }
    }
    out
}

pub(crate) fn build_surface_from_values(
    values: &Tensor,
    x_edges: &[f64],
    y_edges: &[f64],
    display_style: Histogram2DisplayStyle,
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<&str>,
    color_limits: Option<(f64, f64)>,
) -> BuiltinResult<SurfacePlot> {
    if values.shape().len() < 2 || x_edges.len() < 2 || y_edges.len() < 2 {
        return Err(internal("histogram2: invalid 2-D histogram grid"));
    }
    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    if tensor::tensor_element_len(values) != x_bins * y_bins {
        return Err(internal(
            "histogram2: Values shape does not match bin edges",
        ));
    }
    validate_face_alpha(face_alpha)?;

    let mut z_grid = vec![vec![0.0; y_bins]; x_bins];
    for (ix, row) in z_grid.iter_mut().enumerate().take(x_bins) {
        for (iy, cell) in row.iter_mut().enumerate().take(y_bins) {
            let mut value = tensor::tensor_value_f64(values, ix + iy * x_bins);
            if !show_empty_bins && value == 0.0 {
                value = f64::NAN;
            }
            *cell = value;
        }
    }

    let mut surface = SurfacePlot::new(bin_centers(x_edges), bin_centers(y_edges), z_grid)
        .map_err(|err| internal(format!("histogram2: {err}")))?
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::None)
        .with_alpha(face_alpha as f32);
    if display_style == Histogram2DisplayStyle::Tile {
        surface = surface.with_flatten_z(true).with_image_mode(true);
    }
    if let Some(display_name) = display_name {
        surface = surface.with_label(display_name.to_string());
    }
    if color_limits.is_some() {
        surface = surface.with_color_limits(color_limits);
    }
    Ok(surface)
}

fn render_histogram2(
    options: Histogram2Options,
    chart: Histogram2Chart,
    data: Option<Tensor>,
    normalization_denominator: f64,
) -> BuiltinResult<f64> {
    let mut surface = Some(chart.surface);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result =
        render_active_plot(NAME, PlotRenderOptions::default(), move |figure, axes| {
            let plot_index = figure.add_surface_plot_on_axes(
                surface.take().expect("histogram2 plot consumed once"),
                axes,
            );
            figure.set_axes_colorbar_enabled(axes, true);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_histogram2_handle(
        figure_handle,
        axes,
        plot_index,
        chart.values,
        chart.raw_counts,
        chart.x_bin_edges,
        chart.y_bin_edges,
        options.normalization,
        normalization_denominator,
        options.display_style,
        options.show_empty_bins,
        options.face_alpha,
        options.display_name,
        data,
    );
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(internal(err.message));
    }
    Ok(handle)
}

async fn histogram2_data_value(x: &Value, y: &Value) -> BuiltinResult<Tensor> {
    let x = match x {
        Value::GpuTensor(handle) => {
            crate::builtins::common::gpu_helpers::gather_tensor_async(handle)
                .await
                .map_err(map_histcounts_error)?
        }
        _ => tensor::value_into_tensor_for(NAME, x.clone())
            .map_err(|err| invalid(format!("histogram2: cannot retain X data: {err}")))?,
    };
    let y = match y {
        Value::GpuTensor(handle) => {
            crate::builtins::common::gpu_helpers::gather_tensor_async(handle)
                .await
                .map_err(map_histcounts_error)?
        }
        _ => tensor::value_into_tensor_for(NAME, y.clone())
            .map_err(|err| invalid(format!("histogram2: cannot retain Y data: {err}")))?,
    };
    if x.len() != y.len() {
        return Err(invalid(
            "histogram2: X and Y must contain the same number of elements",
        ));
    }
    let len = x.len();
    let x = x
        .reshape(vec![len, 1])
        .map_err(|err| invalid(format!("histogram2: cannot reshape X data: {err}")))?;
    let y = y
        .reshape(vec![len, 1])
        .map_err(|err| invalid(format!("histogram2: cannot reshape Y data: {err}")))?;
    crate::builtins::common::concatenation::hcat_matrices(&x, &y)
}

pub(crate) fn apply_normalization_2d(
    raw_counts: &Tensor,
    x_edges: &[f64],
    y_edges: &[f64],
    norm: &str,
    normalization_denominator: f64,
) -> BuiltinResult<Tensor> {
    validate_normalization(norm)?;
    let x_bins = x_edges.len().saturating_sub(1);
    let y_bins = y_edges.len().saturating_sub(1);
    let counts = raw_counts
        .as_f64_slice()
        .ok_or_else(|| internal("histogram2: raw bin counts must be double"))?;
    let total = normalization_denominator;
    let x_widths: Vec<f64> = x_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let y_widths: Vec<f64> = y_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let values = match norm {
        "count" => counts.to_vec(),
        "probability" => {
            if total > 0.0 {
                counts.iter().map(|&c| c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        "percentage" => {
            if total > 0.0 {
                counts.iter().map(|&c| 100.0 * c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        "countdensity" => density_counts(counts, &x_widths, &y_widths, x_bins, y_bins, 1.0),
        "pdf" | "probabilitydensity" => {
            if total > 0.0 {
                density_counts(counts, &x_widths, &y_widths, x_bins, y_bins, total)
            } else {
                vec![0.0; counts.len()]
            }
        }
        "cumcount" => cumulative_counts(counts, x_bins, y_bins, None),
        "cdf" => cumulative_counts(counts, x_bins, y_bins, Some(total)),
        other => {
            return Err(invalid(format!(
                "histogram2: unsupported normalization '{other}'"
            )))
        }
    };
    Tensor::new(values, raw_counts.shape().to_vec())
        .map_err(|err| internal(format!("histogram2: {err}")))
}

fn density_counts(
    counts: &[f64],
    x_widths: &[f64],
    y_widths: &[f64],
    x_bins: usize,
    y_bins: usize,
    total_scale: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; counts.len()];
    for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
        for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
            let idx = ix + iy * x_bins;
            let area = x_width * y_width;
            out[idx] = if area > 0.0 {
                counts[idx] / (total_scale * area)
            } else {
                0.0
            };
        }
    }
    out
}

fn cumulative_counts(
    counts: &[f64],
    x_bins: usize,
    y_bins: usize,
    divisor: Option<f64>,
) -> Vec<f64> {
    let mut out = vec![0.0; counts.len()];
    for y in 0..y_bins {
        for x in 0..x_bins {
            let idx = x + y * x_bins;
            let left = if x > 0 { out[idx - 1] } else { 0.0 };
            let below = if y > 0 { out[idx - x_bins] } else { 0.0 };
            let diagonal = if x > 0 && y > 0 {
                out[idx - x_bins - 1]
            } else {
                0.0
            };
            out[idx] = counts[idx] + left + below - diagonal;
        }
    }
    if let Some(divisor) = divisor {
        if divisor > 0.0 {
            for value in &mut out {
                *value /= divisor;
            }
        } else {
            out.fill(0.0);
        }
    }
    out
}

pub(crate) fn validate_normalization(norm: &str) -> BuiltinResult<()> {
    match norm.trim().to_ascii_lowercase().as_str() {
        "count" | "probability" | "percentage" | "countdensity" | "pdf" | "probabilitydensity"
        | "cumcount" | "cdf" => Ok(()),
        other => Err(invalid(format!(
            "histogram2: unsupported normalization '{other}'"
        ))),
    }
}

pub(crate) fn validate_face_alpha(value: f64) -> BuiltinResult<()> {
    if !(0.0..=1.0).contains(&value) || !value.is_finite() {
        return Err(invalid(
            "histogram2: FaceAlpha must be in the interval [0,1]",
        ));
    }
    Ok(())
}

pub(crate) fn option_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid(format!("histogram2: {name} must be 0 or 1"))),
        };
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) => numeric_bool(*value, name),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            numeric_bool(tensor::tensor_value_f64(tensor, 0), name)
        }
        _ => {
            if let Some(text) = tensor::value_to_string(value) {
                match text.trim().to_ascii_lowercase().as_str() {
                    "on" | "true" | "1" => Ok(true),
                    "off" | "false" | "0" => Ok(false),
                    _ => Err(invalid(format!("histogram2: {name} must be logical"))),
                }
            } else {
                Err(invalid(format!("histogram2: {name} must be logical")))
            }
        }
    }
}

fn numeric_bool(value: f64, name: &str) -> BuiltinResult<bool> {
    if value == 0.0 {
        Ok(false)
    } else if value == 1.0 {
        Ok(true)
    } else {
        Err(invalid(format!("histogram2: {name} must be 0 or 1")))
    }
}

pub(crate) fn option_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_value_f64(tensor, 0))
        }
        _ => Err(invalid(format!("histogram2: {name} must be a scalar"))),
    }
}

pub(crate) fn parse_display_style(value: &Value) -> BuiltinResult<Histogram2DisplayStyle> {
    let text = tensor::value_to_string(value)
        .ok_or_else(|| invalid("histogram2: DisplayStyle must be text"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "bar3" => Ok(Histogram2DisplayStyle::Bar3),
        "tile" => Ok(Histogram2DisplayStyle::Tile),
        other => Err(invalid(format!(
            "histogram2: unsupported DisplayStyle '{other}'"
        ))),
    }
}

fn bin_centers(edges: &[f64]) -> Vec<f64> {
    edges
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

fn value_as_string(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    crate::builtins::plotting::style::value_as_string(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse_options(args: Vec<Value>) -> BuiltinResult<Histogram2Options> {
        super::parse_options(crate::builtins::common::test_support::builtin_arguments(
            args,
        ))
    }

    async fn histogram2_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::histogram2_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::figure::PlotElement;

    fn vec_tensor(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
    }

    fn int_matrix(values: Vec<i16>, rows: usize, cols: usize) -> Tensor {
        Tensor::new_integer(runmat_value::IntegerStorage::I16(values), vec![rows, cols])
            .expect("integer matrix")
    }

    fn double_values(tensor: &Tensor) -> &[f64] {
        tensor.as_f64_slice().expect("expected double tensor")
    }

    #[test]
    fn histogram2_option_scalar_reads_typed_integer_storage_exactly() {
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();

        assert_eq!(
            option_scalar(&Value::Tensor(tensor), "FaceAlpha").unwrap(),
            1.0
        );

        for bins in [Value::Num(4.0), Value::Int(runmat_value::IntValue::U8(4))] {
            let parsed = parse_options(vec![
                bins.clone(),
                Value::String("ShowEmptyBins".into()),
                Value::Bool(true),
            ])
            .expect("numeric bin count remains positional");
            assert_eq!(parsed.histcounts_args, vec![bins]);
            assert!(parsed.show_empty_bins);
        }
    }

    #[test]
    fn histogram2_boolean_option_reads_all_integer_storage_classes_exactly() {
        let storages = [
            runmat_value::IntegerStorage::I8(vec![1]),
            runmat_value::IntegerStorage::I16(vec![1]),
            runmat_value::IntegerStorage::I32(vec![1]),
            runmat_value::IntegerStorage::I64(vec![1]),
            runmat_value::IntegerStorage::U8(vec![1]),
            runmat_value::IntegerStorage::U16(vec![1]),
            runmat_value::IntegerStorage::U32(vec![1]),
            runmat_value::IntegerStorage::U64(vec![1]),
        ];

        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("logical");
            assert!(option_bool(&Value::Tensor(tensor), "ShowEmptyBins").expect("logical"));
        }
    }

    #[test]
    fn histogram2_surface_values_read_typed_integer_storage_exactly() {
        let surface = build_surface_from_values(
            &int_matrix(vec![1, 2, 3, 4], 2, 2),
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 2.0],
            Histogram2DisplayStyle::Bar3,
            true,
            1.0,
            None,
            None,
        )
        .expect("surface");

        assert_eq!(
            surface.z_data.as_ref().unwrap(),
            &vec![vec![1.0, 3.0], vec![2.0, 4.0]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn histogram2_retained_low_rank_data_has_two_columns() {
        for (shape, x_values, y_values) in [
            (Vec::new(), vec![0.25], vec![0.75]),
            (vec![2], vec![0.25, 1.25], vec![0.75, 1.75]),
        ] {
            let x = Value::Tensor(Tensor::new(x_values.clone(), shape.clone()).unwrap());
            let y = Value::Tensor(Tensor::new(y_values.clone(), shape).unwrap());
            let data = futures::executor::block_on(histogram2_data_value(&x, &y))
                .expect("retained histogram data");

            assert_eq!(data.shape(), &[x_values.len(), 2]);
            assert_eq!(data.rows(), x_values.len());
            assert_eq!(data.materialize_f64(), [x_values, y_values].concat());
        }
    }

    #[test]
    fn histogram2_counts_edges_and_registers_chart_state() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(histogram2_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(values.shape(), &[2, 2]);
        assert_eq!(double_values(&values), &[2.0, 0.0, 0.0, 2.0]);

        let figure = clone_figure(current_figure_handle()).unwrap();
        let plot = figure.plots().next().unwrap();
        let PlotElement::Surface(surface) = plot else {
            panic!("expected surface plot");
        };
        assert!(!surface.image_mode);
        assert!(!surface.flatten_z);
        assert_eq!(surface.x_data, vec![0.25, 0.75]);
        assert_eq!(surface.y_data, vec![0.25, 0.75]);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("ShowEmptyBins".into())
            ])
            .unwrap(),
            Value::Bool(false)
        );
        let data = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Data".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(data.shape(), &[4, 2]);
        assert_eq!(
            double_values(&data),
            &[0.0, 0.2, 0.8, 1.0, 0.0, 0.1, 0.9, 1.0]
        );
    }

    #[test]
    fn histogram2_supports_tile_style_and_normalization() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(histogram2_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::String("NumBins".into()),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
            Value::String("Normalization".into()),
            Value::String("probability".into()),
            Value::String("DisplayStyle".into()),
            Value::String("tile".into()),
            Value::String("ShowEmptyBins".into()),
            Value::String("off".into()),
            Value::String("FaceAlpha".into()),
            Value::Num(0.5),
            Value::String("DisplayName".into()),
            Value::String("density".into()),
        ]))
        .unwrap();
        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(double_values(&values), &[0.5, 0.0, 0.0, 0.5]);
        assert!(matches!(
            get_builtin(vec![Value::Num(handle), Value::String("DisplayStyle".into())]).unwrap(),
            Value::String(style) if style == "tile"
        ));
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        assert!(surface.image_mode);
        assert!(surface.flatten_z);
        assert_eq!(surface.alpha, 0.5);
        assert_eq!(surface.label.as_deref(), Some("density"));
    }

    #[test]
    fn histogram2_set_updates_display_and_normalization_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(histogram2_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("Normalization".into()),
            Value::String("cdf".into()),
            Value::String("DisplayStyle".into()),
            Value::String("tile".into()),
            Value::String("ShowEmptyBins".into()),
            Value::String("off".into()),
        ])
        .unwrap();
        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(double_values(&values), &[0.5, 0.5, 0.5, 1.0]);
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        assert!(surface.image_mode);
    }

    #[test]
    fn histogram2_normalization_uses_full_cardinality_and_preserves_raw_counts() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(histogram2_builtin(vec![
            vec_tensor(&[0.25, 1.25, 99.0, f64::NAN]),
            vec_tensor(&[0.25, 1.25, 99.0, 0.25]),
            Value::String("XBinEdges".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap()),
            Value::String("YBinEdges".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap()),
            Value::String("Normalization".into()),
            Value::String("percentage".into()),
        ]))
        .unwrap();
        let raw = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(double_values(&raw), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(double_values(&values), &[25.0, 0.0, 0.0, 25.0]);
    }

    #[test]
    fn histogram2_rejects_nonscalar_face_alpha_updates() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(histogram2_builtin(vec![
            vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
            vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        let err = set_builtin(vec![
            Value::Num(handle),
            Value::String("FaceAlpha".into()),
            Value::Tensor(Tensor::new(vec![0.25, 0.5], vec![1, 2]).unwrap()),
        ])
        .expect_err("non-scalar FaceAlpha should be rejected");
        assert!(err.message.contains("FaceAlpha must be a scalar"));
    }
}
