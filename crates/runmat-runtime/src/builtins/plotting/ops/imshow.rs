use std::cell::RefCell;
use std::io::Cursor;
#[cfg(target_arch = "wasm32")]
use std::path::{Path, PathBuf};
use std::rc::Rc;

use runmat_builtins::{CharArray, NumericDType, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use super::common::SurfaceDataInput;
use super::state::{render_active_plot, PlotRenderOptions};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const BUILTIN_NAME: &str = "imshow";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::imshow")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imshow",
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
    notes: "imshow is a plotting sink; grayscale GPU inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::imshow")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imshow",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "imshow terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "imshow",
    category = "plotting",
    summary = "Display grayscale, binary, truecolor, or file-backed images.",
    keywords = "imshow,plotting,image,grayscale,rgb,binary",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::imshow"
)]
pub async fn imshow_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (image_value, range) = parse_args(args).await?;

    if let Some(path) = string_like_path(&image_value)? {
        if range != DisplayRange::Default {
            return Err(imshow_error(
                "imshow: display ranges are only supported for numeric image data",
            ));
        }
        let tensor = tensor_from_image_file(&path).await?;
        return render_truecolor_tensor(tensor).await;
    }

    let image_value = normalize_image_value(image_value)?;
    if is_truecolor_value(&image_value) {
        if range != DisplayRange::Default {
            return Err(imshow_error(
                "imshow: display ranges are only supported for grayscale image data",
            ));
        }
        let tensor = tensor_from_value(image_value).await?;
        return render_truecolor_tensor(tensor).await;
    }

    render_grayscale_value(image_value, range).await
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DisplayRange {
    Default,
    Auto,
    Limits(f64, f64),
}

async fn parse_args(args: Vec<Value>) -> crate::BuiltinResult<(Value, DisplayRange)> {
    match args.len() {
        0 => Err(imshow_error("imshow: expected image data or filename")),
        1 => Ok((
            args.into_iter().next().expect("one arg"),
            DisplayRange::Default,
        )),
        2 => {
            let mut it = args.into_iter();
            let image_value = it.next().expect("image arg");
            let range_value = it.next().expect("range arg");
            Ok((image_value, parse_display_range(range_value).await?))
        }
        _ => Err(imshow_error("imshow: too many input arguments")),
    }
}

async fn parse_display_range(value: Value) -> crate::BuiltinResult<DisplayRange> {
    let host = crate::gather_if_needed_async(&value)
        .await
        .map_err(|flow| {
            crate::builtins::common::map_control_flow_with_builtin(flow, BUILTIN_NAME)
        })?;
    match host {
        Value::Tensor(tensor) if tensor.data.is_empty() => Ok(DisplayRange::Auto),
        Value::Tensor(tensor) if tensor.data.len() == 2 => {
            let lo = tensor.data[0];
            let hi = tensor.data[1];
            validate_display_limits(lo, hi)
        }
        _ => Err(imshow_error(
            "imshow: display range must be [] or a two-element numeric vector",
        )),
    }
}

fn validate_display_limits(lo: f64, hi: f64) -> crate::BuiltinResult<DisplayRange> {
    if !lo.is_finite() || !hi.is_finite() || lo >= hi {
        return Err(imshow_error(
            "imshow: display range must contain finite increasing limits",
        ));
    }
    Ok(DisplayRange::Limits(lo, hi))
}

fn normalize_image_value(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::LogicalArray(array) => tensor::logical_to_tensor(&array)
            .map(Value::Tensor)
            .map_err(|err| imshow_error(format!("imshow: {err}"))),
        Value::Bool(flag) => Tensor::new(vec![if flag { 1.0 } else { 0.0 }], vec![1, 1])
            .map(Value::Tensor)
            .map_err(|err| imshow_error(format!("imshow: {err}"))),
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map(Value::Tensor)
            .map_err(|err| imshow_error(format!("imshow: {err}"))),
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1])
            .map(Value::Tensor)
            .map_err(|err| imshow_error(format!("imshow: {err}"))),
        other => Ok(other),
    }
}

fn is_truecolor_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor
            .shape
            .get(2)
            .copied()
            .is_some_and(|c| c == 3 || c == 4),
        Value::GpuTensor(handle) => handle
            .shape
            .get(2)
            .copied()
            .is_some_and(|c| c == 3 || c == 4),
        _ => false,
    }
}

async fn tensor_from_value(value: Value) -> crate::BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => {
            super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await
        }
        other => Tensor::try_from(&other).map_err(|err| imshow_error(format!("imshow: {err}"))),
    }
}

async fn render_grayscale_value(value: Value, range: DisplayRange) -> crate::BuiltinResult<f64> {
    let c_input = SurfaceDataInput::from_value(value, BUILTIN_NAME)?;
    let (rows, cols) = c_input.grid_shape(BUILTIN_NAME)?;
    let x_axis = default_axis_vec(cols);
    let y_axis = default_image_y_axis(rows);
    let color_limits = display_range_for_input(&c_input, range).await?;
    let mut surface = build_grayscale_image_surface(c_input, x_axis, y_axis, color_limits).await?;

    surface = surface
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_colormap(ColorMap::Gray)
        .with_shading(ShadingMode::None)
        .with_color_limits(Some(color_limits));
    render_surface(surface, Some(color_limits)).await
}

async fn render_truecolor_tensor(tensor: Tensor) -> crate::BuiltinResult<f64> {
    let (rows, cols) = truecolor_shape(&tensor)?;
    let x_host = default_axis_vec(cols);
    let y_host = default_image_y_axis(rows);
    let surface = build_truecolor_image_surface(tensor, x_host, y_host)?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_shading(ShadingMode::None);
    render_surface(surface, None).await
}

async fn render_surface(
    surface: SurfacePlot,
    color_limits: Option<(f64, f64)>,
) -> crate::BuiltinResult<f64> {
    let mut surface = Some(surface);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let opts = PlotRenderOptions {
        title: "Image",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        if let Some(limits) = color_limits {
            figure.set_axes_color_limits(axes, Some(limits));
        }
        let plot_index = figure
            .add_surface_plot_on_axes(surface.take().expect("imshow plot consumed once"), axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_image_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

async fn display_range_for_input(
    input: &SurfaceDataInput,
    range: DisplayRange,
) -> crate::BuiltinResult<(f64, f64)> {
    match range {
        DisplayRange::Default => {
            let limits = match input {
                SurfaceDataInput::Host(tensor) => dtype_default_limits(tensor.dtype),
                SurfaceDataInput::Gpu(_) => (0.0, 1.0),
            };
            Ok(limits)
        }
        DisplayRange::Limits(lo, hi) => Ok((lo, hi)),
        DisplayRange::Auto => {
            let (lo, hi) = match input {
                SurfaceDataInput::Host(tensor) => finite_tensor_bounds(tensor),
                SurfaceDataInput::Gpu(handle) => {
                    let (lo, hi) =
                        super::gpu_helpers::axis_bounds_async(handle, BUILTIN_NAME).await?;
                    (lo as f64, hi as f64)
                }
            };
            Ok(expand_degenerate_limits(lo, hi))
        }
    }
}

fn finite_tensor_bounds(tensor: &Tensor) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &value in &tensor.data {
        if value.is_finite() {
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    if lo.is_finite() && hi.is_finite() {
        (lo, hi)
    } else {
        (0.0, 1.0)
    }
}

fn expand_degenerate_limits(lo: f64, hi: f64) -> (f64, f64) {
    if lo < hi {
        return (lo, hi);
    }
    if lo > 0.0 {
        (0.0, lo)
    } else if lo < 0.0 {
        (lo, 0.0)
    } else {
        (0.0, 1.0)
    }
}

fn dtype_default_limits(dtype: NumericDType) -> (f64, f64) {
    match dtype {
        NumericDType::U8 => (0.0, 255.0),
        NumericDType::U16 => (0.0, 65535.0),
        NumericDType::F32 | NumericDType::F64 => (0.0, 1.0),
    }
}

fn default_axis_vec(len: usize) -> Vec<f64> {
    (1..=len).map(|idx| idx as f64).collect()
}

fn default_image_y_axis(rows: usize) -> Vec<f64> {
    (1..=rows).rev().map(|idx| idx as f64).collect()
}

fn truecolor_shape(tensor: &Tensor) -> crate::BuiltinResult<(usize, usize)> {
    let rows = tensor.shape.first().copied().unwrap_or(tensor.rows);
    let cols = tensor.shape.get(1).copied().unwrap_or(tensor.cols);
    let channels = tensor.shape.get(2).copied().unwrap_or(1);
    if rows == 0 || cols == 0 || (channels != 3 && channels != 4) {
        return Err(imshow_error(
            "imshow: truecolor image data must be MxNx3 or MxNx4",
        ));
    }
    let expected_len = rows * cols * channels;
    if tensor.data.len() != expected_len {
        return Err(imshow_error("imshow: truecolor image data length mismatch"));
    }
    Ok((rows, cols))
}

fn build_truecolor_image_surface(
    tensor: Tensor,
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
) -> crate::BuiltinResult<SurfacePlot> {
    let image_rows = tensor.shape.first().copied().unwrap_or(tensor.rows);
    let image_cols = tensor.shape.get(1).copied().unwrap_or(tensor.cols);
    if x_axis.len() != image_cols || y_axis.len() != image_rows {
        return Err(imshow_error(
            "imshow: truecolor image axes must match image columns and rows",
        ));
    }
    let channels = tensor.shape.get(2).copied().unwrap_or(3);
    let scale = match tensor.dtype {
        NumericDType::U8 => 1.0f32 / 255.0,
        NumericDType::U16 => 1.0f32 / 65535.0,
        NumericDType::F32 | NumericDType::F64 => 1.0,
    };
    let mut grid = vec![vec![glam::Vec4::ZERO; image_rows]; image_cols];
    for row in 0..image_rows {
        for col in 0..image_cols {
            let base = row + image_rows * col;
            let r = tensor.data[base] as f32 * scale;
            let g = tensor.data[base + image_rows * image_cols] as f32 * scale;
            let b = tensor.data[base + 2 * image_rows * image_cols] as f32 * scale;
            let a = if channels == 4 {
                tensor.data[base + 3 * image_rows * image_cols] as f32 * scale
            } else {
                1.0
            };
            grid[col][row] = glam::Vec4::new(r, g, b, a);
        }
    }
    let z = vec![vec![0.0; image_rows]; image_cols];
    Ok(SurfacePlot::new(x_axis, y_axis, z)
        .map_err(|err| imshow_error(format!("imshow: {err}")))?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_color_grid(grid)
        .with_shading(ShadingMode::None))
}

async fn build_grayscale_image_surface(
    input: SurfaceDataInput,
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    color_limits: (f64, f64),
) -> crate::BuiltinResult<SurfacePlot> {
    let tensor = match input {
        SurfaceDataInput::Host(tensor) => tensor,
        SurfaceDataInput::Gpu(handle) => {
            super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
        }
    };
    let grid = tensor_to_image_grid(tensor, x_axis.len(), y_axis.len())?;
    let color_grid = grayscale_color_grid(&grid, color_limits);
    Ok(SurfacePlot::new(x_axis, y_axis, grid)
        .map_err(|err| imshow_error(format!("imshow: {err}")))?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_colormap(ColorMap::Gray)
        .with_color_grid(color_grid)
        .with_shading(ShadingMode::None)
        .with_color_limits(Some(color_limits)))
}

fn tensor_to_image_grid(
    tensor: Tensor,
    cols: usize,
    rows: usize,
) -> crate::BuiltinResult<Vec<Vec<f64>>> {
    if tensor.data.len() != rows * cols {
        return Err(imshow_error(format!(
            "imshow: image data must contain exactly {} values ({}x{})",
            rows * cols,
            rows,
            cols
        )));
    }
    let mut grid = vec![vec![0.0; rows]; cols];
    for row in 0..rows {
        for (col, col_vec) in grid.iter_mut().enumerate().take(cols) {
            col_vec[row] = tensor.data[row + rows * col];
        }
    }
    Ok(grid)
}

fn grayscale_color_grid(grid: &[Vec<f64>], color_limits: (f64, f64)) -> Vec<Vec<glam::Vec4>> {
    grid.iter()
        .map(|col| {
            col.iter()
                .map(|&value| {
                    let gray = grayscale_level(value, color_limits);
                    glam::Vec4::new(gray, gray, gray, 1.0)
                })
                .collect()
        })
        .collect()
}

fn grayscale_level(value: f64, color_limits: (f64, f64)) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let (lo, hi) = color_limits;
    let denom = hi - lo;
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    ((value - lo) / denom).clamp(0.0, 1.0) as f32
}

async fn tensor_from_image_file(path: &str) -> crate::BuiltinResult<Tensor> {
    let bytes = read_image_file_bytes(path).await?;
    let reader = image::io::Reader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|err| {
            imshow_error(format!(
                "imshow: unable to identify image file '{path}' ({err})"
            ))
        })?;
    let image = reader.decode().map_err(|err| {
        imshow_error(format!(
            "imshow: unable to decode image file '{path}' ({err})"
        ))
    })?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let rows = height as usize;
    let cols = width as usize;
    let channels = 4;
    let plane = rows * cols;
    let mut data = vec![0.0; plane * channels];
    for row in 0..rows {
        for col in 0..cols {
            let pixel = rgba.get_pixel(col as u32, row as u32).0;
            let base = row + rows * col;
            data[base] = f64::from(pixel[0]) / 255.0;
            data[base + plane] = f64::from(pixel[1]) / 255.0;
            data[base + 2 * plane] = f64::from(pixel[2]) / 255.0;
            data[base + 3 * plane] = f64::from(pixel[3]) / 255.0;
        }
    }
    Tensor::new_with_dtype(data, vec![rows, cols, channels], NumericDType::F64)
        .map_err(|err| imshow_error(format!("imshow: {err}")))
}

async fn read_image_file_bytes(path: &str) -> crate::BuiltinResult<Vec<u8>> {
    match runmat_filesystem::read_async(path).await {
        Ok(bytes) => Ok(bytes),
        Err(fs_err) => {
            #[cfg(target_arch = "wasm32")]
            {
                match read_unique_virtual_file_by_name(path).await {
                    Ok(Some(bytes)) => return Ok(bytes),
                    Ok(None) => {}
                    Err(search_err) => {
                        return Err(imshow_error(format!(
                            "imshow: unable to read image file '{path}' ({fs_err}; virtual filesystem search failed: {search_err})"
                        )));
                    }
                }
                Err(imshow_error(format!(
                    "imshow: unable to read image file '{path}' ({fs_err}; no unique uploaded file named '{path}' was found in the virtual filesystem)"
                )))
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                Err(imshow_error(format!(
                    "imshow: unable to read image file '{path}' ({fs_err})"
                )))
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn read_unique_virtual_file_by_name(path: &str) -> Result<Option<Vec<u8>>, String> {
    if !is_simple_relative_filename(path) {
        return Ok(None);
    }
    let matches = find_virtual_files_named(path).await?;
    match matches.as_slice() {
        [] => Ok(None),
        [found] => runmat_filesystem::read_async(found)
            .await
            .map(Some)
            .map_err(|err| format!("matched '{}', but read failed: {err}", found.display())),
        _ => Err(format!(
            "filename is ambiguous; matches: {}",
            matches
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

#[cfg(target_arch = "wasm32")]
async fn find_virtual_files_named(file_name: &str) -> Result<Vec<PathBuf>, String> {
    const MAX_VISITED_DIRS: usize = 4096;

    let mut matches = Vec::new();
    let mut stack = vec![PathBuf::from("/")];
    let mut visited = 0usize;
    while let Some(dir) = stack.pop() {
        visited += 1;
        if visited > MAX_VISITED_DIRS {
            return Err("virtual filesystem search exceeded directory limit".to_string());
        }
        let entries = match runmat_filesystem::read_dir_async(&dir).await {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries {
            let name = entry.file_name().to_string_lossy();
            if entry.is_dir() {
                stack.push(entry.path().to_path_buf());
            } else if name == file_name {
                matches.push(entry.path().to_path_buf());
            }
        }
    }
    Ok(matches)
}

#[cfg(target_arch = "wasm32")]
fn is_simple_relative_filename(path: &str) -> bool {
    let path = Path::new(path);
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
        && path.components().count() == 1
}

fn string_like_path(value: &Value) -> crate::BuiltinResult<Option<String>> {
    match value {
        Value::String(s) => Ok(Some(s.clone())),
        Value::StringArray(array) => scalar_string_array(array).map(Some),
        Value::CharArray(chars) => scalar_char_array(chars).map(Some),
        _ => Ok(None),
    }
}

fn scalar_string_array(array: &StringArray) -> crate::BuiltinResult<String> {
    if array.data.len() == 1 {
        Ok(array.data[0].clone())
    } else {
        Err(imshow_error("imshow: filename must be a string scalar"))
    }
}

fn scalar_char_array(chars: &CharArray) -> crate::BuiltinResult<String> {
    if chars.rows == 1 {
        Ok(chars.data.iter().collect())
    } else {
        Err(imshow_error(
            "imshow: filename must be a character row vector",
        ))
    }
}

fn imshow_error(message: impl Into<String>) -> crate::RuntimeError {
    crate::builtins::plotting::plotting_error(BUILTIN_NAME, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::LogicalArray;
    use runmat_plot::plots::PlotElement;

    fn matrix(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor {
            data,
            shape: vec![rows, cols],
            rows,
            cols,
            dtype: NumericDType::F64,
        }
    }

    fn reset() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn imshow_repro_displays_ones_as_grayscale_image() {
        let _guard = reset();
        let handle = futures::executor::block_on(imshow_builtin(vec![Value::Tensor(matrix(
            vec![1.0; 100 * 100],
            100,
            100,
        ))]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert!(surface.flatten_z);
        assert!(surface.image_mode);
        assert_eq!(surface.colormap, ColorMap::Gray);
        assert_eq!(surface.color_limits, Some((0.0, 1.0)));
        let grid = surface.color_grid.as_ref().expect("grayscale color grid");
        assert_eq!(grid[0][0], glam::Vec4::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap(),
            Value::String("image".into())
        );
    }

    #[test]
    fn imshow_is_registered_with_dispatcher() {
        let _guard = reset();
        let result = futures::executor::block_on(crate::call_builtin_async(
            BUILTIN_NAME,
            &[Value::Tensor(matrix(vec![1.0; 4], 2, 2))],
        ))
        .expect("imshow should be registered");
        assert!(matches!(result, Value::Num(handle) if handle.is_finite()));
    }

    #[test]
    fn imshow_accepts_explicit_display_range() {
        let _guard = reset();
        let range = Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap();
        futures::executor::block_on(imshow_builtin(vec![
            Value::Tensor(matrix(vec![10.0, 15.0, 20.0, 12.0], 2, 2)),
            Value::Tensor(range),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.color_limits, Some((10.0, 20.0)));
    }

    #[test]
    fn imshow_empty_display_range_auto_scales() {
        let _guard = reset();
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        futures::executor::block_on(imshow_builtin(vec![
            Value::Tensor(matrix(vec![10.0, 15.0, 20.0, 12.0], 2, 2)),
            Value::Tensor(empty),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.color_limits, Some((10.0, 20.0)));
    }

    #[test]
    fn imshow_uint8_grayscale_defaults_to_0_255() {
        let _guard = reset();
        let tensor = Tensor {
            data: vec![0.0, 128.0, 200.0, 255.0],
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
            dtype: NumericDType::U8,
        };
        futures::executor::block_on(imshow_builtin(vec![Value::Tensor(tensor)])).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.color_limits, Some((0.0, 255.0)));
        let grid = surface.color_grid.as_ref().expect("grayscale color grid");
        assert_eq!(grid.len(), 2);
        assert_eq!(grid[0].len(), 2);
        assert_eq!(grid[0][0], glam::Vec4::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(grid[1][1], glam::Vec4::new(1.0, 1.0, 1.0, 1.0));
        let mid = grid[0][1];
        assert_eq!(mid.x, mid.y);
        assert_eq!(mid.y, mid.z);
        assert!((mid.x - (128.0 / 255.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn imshow_uint16_grayscale_defaults_to_0_65535() {
        let _guard = reset();
        let tensor = Tensor {
            data: vec![0.0, 1000.0, 40000.0, 65535.0],
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
            dtype: NumericDType::U16,
        };
        futures::executor::block_on(imshow_builtin(vec![Value::Tensor(tensor)])).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.color_limits, Some((0.0, 65535.0)));
    }

    #[test]
    fn imshow_logical_image_uses_binary_limits() {
        let _guard = reset();
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        futures::executor::block_on(imshow_builtin(vec![Value::LogicalArray(logical)])).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.color_limits, Some((0.0, 1.0)));
        assert_eq!(surface.colormap, ColorMap::Gray);
    }

    #[test]
    fn imshow_truecolor_builds_color_grid() {
        let _guard = reset();
        let tensor = Tensor {
            data: vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            shape: vec![2, 2, 3],
            rows: 2,
            cols: 2,
            dtype: NumericDType::F64,
        };
        futures::executor::block_on(imshow_builtin(vec![Value::Tensor(tensor)])).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert!(surface.image_mode);
        let grid = surface.color_grid.as_ref().expect("color grid");
        assert_eq!(surface.x_data.len(), 2);
        assert_eq!(surface.y_data.len(), 2);
        assert_eq!(surface.y_data, vec![2.0, 1.0]);
        assert_eq!(grid.len(), 2);
        assert_eq!(grid[0].len(), 2);
        assert_eq!(surface.color_limits, None);
    }

    #[test]
    fn imshow_decodes_filename_input_without_rotating() {
        let _guard = reset();
        let mut path = std::env::temp_dir();
        path.push(format!("runmat_imshow_{}.png", std::process::id()));
        let mut image = image::RgbaImage::new(3, 2);
        image.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        image.put_pixel(1, 0, image::Rgba([0, 255, 0, 255]));
        image.put_pixel(2, 0, image::Rgba([0, 0, 255, 255]));
        image.put_pixel(0, 1, image::Rgba([0, 255, 255, 255]));
        image.put_pixel(1, 1, image::Rgba([255, 0, 255, 255]));
        image.put_pixel(2, 1, image::Rgba([255, 255, 0, 255]));
        image.save(&path).expect("save png");

        futures::executor::block_on(imshow_builtin(vec![Value::String(
            path.to_string_lossy().to_string(),
        )]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.x_data.len(), 3);
        assert_eq!(surface.y_data.len(), 2);
        assert_eq!(surface.x_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(surface.y_data, vec![2.0, 1.0]);
        let grid = surface.color_grid.as_ref().expect("color grid");
        assert_eq!(grid.len(), 3);
        assert_eq!(grid[0].len(), 2);
        assert_eq!(grid[0][0], glam::Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(grid[1][0], glam::Vec4::new(0.0, 1.0, 0.0, 1.0));
        assert_eq!(grid[2][0], glam::Vec4::new(0.0, 0.0, 1.0, 1.0));
        assert_eq!(grid[0][1], glam::Vec4::new(0.0, 1.0, 1.0, 1.0));
        assert_eq!(grid[1][1], glam::Vec4::new(1.0, 0.0, 1.0, 1.0));
        assert_eq!(grid[2][1], glam::Vec4::new(1.0, 1.0, 0.0, 1.0));

        let _ = test_support::fs::remove_file(&path);
    }
}
