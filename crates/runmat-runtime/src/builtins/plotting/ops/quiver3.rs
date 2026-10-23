//! MATLAB-compatible `quiver3` builtin.
use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::QUIVER3_DESCRIPTOR;

use glam::Vec4;
use runmat_macros::runtime_builtin;
use runmat_plot::plots::NumericPlotData;
use runmat_plot::plots::QuiverPlot;
use runmat_value::{Tensor, Value};
use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_helpers;

use crate::{build_runtime_error, RuntimeError};

use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_line_style_args, value_as_f64, LineStyleParseOptions};

const BUILTIN_NAME: &str = "quiver3";

type Quiver3Args = (
    super::op_common::AxesTarget,
    Value,
    Value,
    Value,
    Value,
    Value,
    Value,
    crate::call::BuiltinArguments,
);
type Quiver3Components = (
    NumericPlotData,
    NumericPlotData,
    NumericPlotData,
    NumericPlotData,
    NumericPlotData,
    NumericPlotData,
);

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::quiver3")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "quiver3",
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
    notes: "quiver3 is a 3-D plotting sink. GPU inputs are gathered to build renderer geometry; a direct WGPU packer is tracked by the GPU fast-path audit.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::quiver3")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "quiver3",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "quiver3 performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "quiver3",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::quiver3"
)]
pub async fn quiver3_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "quiver3: expected at most one output",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "quiver3")?;
    let _arguments = args.fork();
    work.finish(quiver3_impl(args, &work).await)
}

async fn quiver3_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (target_axes, x, y, z, u, v, w, rest) =
        parse_quiver3_args(args).map_err(map_quiver3_invalid_argument)?;
    let parsed = parse_quiver3_style_args(rest.view()).map_err(map_quiver3_invalid_argument)?;
    let [x, y, z, u, v, w] = [x, y, z, u, v, w].map(|value| {
        NumericInput::from_value(value, BUILTIN_NAME).map_err(map_quiver3_invalid_argument)
    });
    // Validate all representations before the first potentially pending download.
    let (x, y, z, u, v, w) = (x?, y?, z?, u?, v?, w?);
    let x_tensor = x
        .into_tensor_async(BUILTIN_NAME)
        .await
        .map_err(map_quiver3_invalid_argument)?;
    let y_tensor = y
        .into_tensor_async(BUILTIN_NAME)
        .await
        .map_err(map_quiver3_invalid_argument)?;
    let z_tensor = z
        .into_tensor_async(BUILTIN_NAME)
        .await
        .map_err(map_quiver3_invalid_argument)?;
    let u_tensor = u
        .into_tensor_async(BUILTIN_NAME)
        .await
        .map_err(map_quiver3_invalid_argument)?;
    let v_tensor = v
        .into_tensor_async(BUILTIN_NAME)
        .await
        .map_err(map_quiver3_invalid_argument)?;
    let w_tensor = w
        .into_tensor_async(BUILTIN_NAME)
        .await
        .map_err(map_quiver3_invalid_argument)?;
    let (x_vals, y_vals, z_vals, u_vals, v_vals, w_vals) =
        materialize_quiver3_components(x_tensor, y_tensor, z_tensor, u_tensor, v_tensor, w_tensor)
            .map_err(map_quiver3_invalid_argument)?;
    let label = parsed.label.clone().unwrap_or_else(|| "Data".into());
    let mut plot = QuiverPlot::from_numeric_data3d(x_vals, y_vals, z_vals, u_vals, v_vals, w_vals)
        .map_err(|err| plotting_error(BUILTIN_NAME, err))?
        .with_style(
            parsed.color,
            parsed.line_width,
            parsed.scale,
            parsed.head_size,
        )
        .with_label(label);
    if let Some(visible) = parsed.visible {
        plot.set_visible(visible);
    }
    let mut prepared = Some(plot);
    let opts = PlotRenderOptions::default();
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    work.check()?;
    super::op_common::apply_axes_target(target_axes, BUILTIN_NAME)?;
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let axes = target_axes.map_or(axes, |(_, index)| index);
        let plot = prepared
            .take()
            .expect("quiver3 prepared plot consumed once");
        let plot_index = figure.add_quiver_plot_on_axes(plot, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_quiver3_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_quiver3_internal(err));
    }
    Ok(handle)
}

#[derive(Clone)]
struct ParsedQuiver3Style {
    visible: Option<bool>,
    color: Vec4,
    line_width: f32,
    label: Option<String>,
    scale: f32,
    head_size: f32,
}

fn parse_quiver3_style_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<ParsedQuiver3Style> {
    let mut filtered = Vec::new();
    let mut scale = 1.0f32;
    let mut head_size = 0.1f32;
    let mut idx = 0usize;
    if let Some(first) = args.first().and_then(value_as_f64) {
        scale = first as f32;
        idx = 1;
    }
    while idx < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "quiver3",
                source.to_string(),
                source,
            )
        })? {
            let key = key.trim().to_ascii_lowercase();
            if idx + 1 < args.len() {
                match key.as_str() {
                    "autoscalefactor" | "scale" => {
                        scale = value_as_f64(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "quiver3: AutoScaleFactor must be numeric")
                        })? as f32;
                        idx += 2;
                        continue;
                    }
                    "maxheadsize" | "headsize" => {
                        head_size = value_as_f64(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "quiver3: MaxHeadSize must be numeric")
                        })? as f32;
                        idx += 2;
                        continue;
                    }
                    _ => {}
                }
            }
        }
        filtered.push(args[idx].clone());
        idx += 1;
    }
    let parsed = parse_line_style_args(&filtered, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    Ok(ParsedQuiver3Style {
        visible: parsed.appearance.visible,
        color: parsed.appearance.color,
        line_width: parsed.appearance.line_width,
        label: parsed.label,
        scale,
        head_size,
    })
}

fn parse_quiver3_args(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Quiver3Args> {
    let (target_axes, args) = if args
        .first()
        .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        (None, args)
    } else {
        super::op_common::split_leading_axes_arguments(args, BUILTIN_NAME)?
    };
    if args.len() < 4 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver3: insufficient coordinate/vector inputs",
        ));
    }
    let explicit = args.len() >= 6 && args.iter().take(6).all(is_numeric_like);
    let (x, y, z, u, v, w) = if explicit {
        (
            args[0].clone(),
            args[1].clone(),
            args[2].clone(),
            args[3].clone(),
            args[4].clone(),
            args[5].clone(),
        )
    } else {
        let (x, y) = default_quiver3_grid_from_value(&args[0])?;
        (
            Value::Tensor(x),
            Value::Tensor(y),
            args[0].clone(),
            args[1].clone(),
            args[2].clone(),
            args[3].clone(),
        )
    };
    Ok((
        target_axes,
        x,
        y,
        z,
        u,
        v,
        w,
        args.into_range(if explicit { 6 } else { 4 }..)?,
    ))
}

fn is_numeric_like(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

fn default_quiver3_grid_from_value(value: &Value) -> crate::BuiltinResult<(Tensor, Tensor)> {
    let (rows, cols, len) = tensor_shape_from_value(value)?;
    let mut x = Vec::with_capacity(len);
    let mut y = Vec::with_capacity(len);
    for idx in 0..len {
        let row = idx % rows;
        let col = (idx / rows) % cols;
        x.push((col + 1) as f64);
        y.push((row + 1) as f64);
    }
    Ok((
        Tensor::new(x, vec![len]).expect("implicit quiver3 x grid"),
        Tensor::new(y, vec![len]).expect("implicit quiver3 y grid"),
    ))
}

fn quiver3_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

fn tensor_shape_from_value(value: &Value) -> crate::BuiltinResult<(usize, usize, usize)> {
    match value {
        Value::GpuTensor(handle) => {
            let rows = handle.shape().first().copied().unwrap_or(1).max(1);
            let cols = handle.shape().get(1).copied().unwrap_or(1).max(1);
            let len = handle.shape().iter().product::<usize>().max(1);
            Ok((rows, cols, len))
        }
        _ => {
            let tensor = Tensor::try_from(value)
                .map_err(|err| plotting_error(BUILTIN_NAME, format!("quiver3: {err}")))?;
            let (rows, cols) = quiver3_matrix_dimensions(&tensor);
            Ok((
                rows.max(1),
                cols.max(1),
                tensor_helpers::tensor_element_len(&tensor),
            ))
        }
    }
}

fn materialize_quiver3_components(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    u: Tensor,
    v: Tensor,
    w: Tensor,
) -> crate::BuiltinResult<Quiver3Components> {
    let x_len = tensor_helpers::tensor_element_len(&x);
    let y_len = tensor_helpers::tensor_element_len(&y);
    let z_len = tensor_helpers::tensor_element_len(&z);
    let u_len = tensor_helpers::tensor_element_len(&u);
    let v_len = tensor_helpers::tensor_element_len(&v);
    let w_len = tensor_helpers::tensor_element_len(&w);
    let (z_rows, z_cols) = quiver3_matrix_dimensions(&z);
    let z_shape = z.shape();
    let (u_rows, u_cols) = quiver3_matrix_dimensions(&u);
    let u_shape = u.shape();
    let (v_rows, v_cols) = quiver3_matrix_dimensions(&v);
    let v_shape = v.shape();
    let (w_rows, w_cols) = quiver3_matrix_dimensions(&w);
    let w_shape = w.shape();
    if u_rows != v_rows
        || u_cols != v_cols
        || u_len != v_len
        || u_shape != v_shape
        || u_rows != w_rows
        || u_cols != w_cols
        || u_len != w_len
        || u_shape != w_shape
    {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver3: U, V, and W inputs must have identical size",
        ));
    }
    if z_rows != u_rows || z_cols != u_cols || z_len != u_len || z_shape != u_shape {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver3: Z, U, V, and W inputs must have identical size",
        ));
    }
    if x_len == u_len && y_len == u_len {
        return quiver3_numeric_data(x, y, z, u, v, w);
    }
    let u_is_matrix = u_rows > 1 && u_cols > 1;
    if !u_is_matrix {
        if x_len != u_len || y_len != u_len {
            return Err(plotting_error(
                BUILTIN_NAME,
                "quiver3: X, Y, Z, U, V, and W vectors must have the same length",
            ));
        }
        return quiver3_numeric_data(x, y, z, u, v, w);
    }
    let rows = u_rows;
    let cols = u_cols;
    if x_len == rows * cols && y_len == rows * cols {
        return quiver3_numeric_data(x, y, z, u, v, w);
    }
    if x_len == cols && y_len == rows {
        let mut x_indices = Vec::with_capacity(rows * cols);
        let mut y_indices = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                x_indices.push(col);
                y_indices.push(row);
            }
        }
        let x = super::quiver::expand_quiver_axis_tensor(x, &x_indices, rows, cols, BUILTIN_NAME)?;
        let y = super::quiver::expand_quiver_axis_tensor(y, &y_indices, rows, cols, BUILTIN_NAME)?;
        return quiver3_numeric_data(x, y, z, u, v, w);
    }
    Err(plotting_error(
        BUILTIN_NAME,
        "quiver3: X and Y must match U/V/W as vectors or meshgrid-style coordinates",
    ))
}

fn quiver3_numeric_data(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    u: Tensor,
    v: Tensor,
    w: Tensor,
) -> crate::BuiltinResult<Quiver3Components> {
    Ok((
        super::common::numeric_plot_data(x)?,
        super::common::numeric_plot_data(y)?,
        super::common::numeric_plot_data(z)?,
        super::common::numeric_plot_data(u)?,
        super::common::numeric_plot_data(v)?,
        super::common::numeric_plot_data(w)?,
    ))
}

fn quiver3_error_with_detail(descriptor: &BuiltinErrorDescriptor, detail: String) -> RuntimeError {
    let message = if detail.trim().is_empty() {
        descriptor.message.to_string()
    } else {
        detail
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_quiver3_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = quiver3_error_with_detail(&QUIVER3_DESCRIPTOR.errors[0], err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_quiver3_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = quiver3_error_with_detail(&QUIVER3_DESCRIPTOR.errors[1], err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    async fn quiver3_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::quiver3_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use runmat_plot::plots::PlotElement;
    use runmat_value::IntegerStorage;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("quiver3 test vector")
    }

    fn int_vec_tensor(data: Vec<i16>) -> Tensor {
        Tensor::new_integer(IntegerStorage::I16(data.clone()), vec![data.len()])
            .expect("integer tensor")
    }

    fn mat_tensor(rows: usize, cols: usize, data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![rows, cols]).expect("quiver3 test matrix")
    }

    fn nd_tensor(shape: Vec<usize>, data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), shape).expect("quiver3 test tensor")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn quiver3_default_grid_preserves_dense_rank_boundaries() {
        let scalar = Value::Tensor(Tensor::new(vec![7.0], Vec::new()).unwrap());
        let (x, y) = default_quiver3_grid_from_value(&scalar).unwrap();
        assert_eq!(x.materialize_f64(), vec![1.0]);
        assert_eq!(y.materialize_f64(), vec![1.0]);

        let vector = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
        let (x, y) = default_quiver3_grid_from_value(&vector).unwrap();
        assert_eq!(x.materialize_f64(), vec![1.0, 2.0, 3.0]);
        assert_eq!(y.materialize_f64(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn quiver3_builds_default_grid_and_3d_vectors() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(mat_tensor(2, 2, &[10.0, 20.0, 30.0, 40.0])),
            Value::Tensor(mat_tensor(2, 2, &[1.0, 0.0, -1.0, 0.5])),
            Value::Tensor(mat_tensor(2, 2, &[0.0, 1.0, 0.5, -0.5])),
            Value::Tensor(mat_tensor(2, 2, &[2.0, 2.0, 1.0, 1.0])),
        ]))
        .unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver plot");
        };
        assert_eq!(quiver.x, vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(quiver.y, vec![1.0, 2.0, 1.0, 2.0]);
        assert_eq!(quiver.z.as_ref().unwrap(), &vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(quiver.w.as_ref().unwrap(), &vec![2.0, 2.0, 1.0, 1.0]);
    }

    #[test]
    fn quiver3_default_grid_covers_all_nd_elements() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let data: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        let handle = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(nd_tensor(vec![2, 2, 2], &data)),
            Value::Tensor(nd_tensor(vec![2, 2, 2], &[1.0; 8])),
            Value::Tensor(nd_tensor(vec![2, 2, 2], &[0.0; 8])),
            Value::Tensor(nd_tensor(vec![2, 2, 2], &[0.5; 8])),
        ]))
        .unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver plot");
        };
        assert_eq!(quiver.x.len(), 8);
        assert_eq!(quiver.y.len(), 8);
        assert_eq!(quiver.z.as_ref().unwrap().len(), 8);
        assert_eq!(quiver.x, vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
        assert_eq!(quiver.y, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn quiver3_rejects_equal_length_but_different_shapes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let err = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0, 3.0, 4.0])),
            Value::Tensor(mat_tensor(2, 2, &[1.0, 0.0, -1.0, 0.5])),
            Value::Tensor(mat_tensor(2, 2, &[0.0, 1.0, 0.5, -0.5])),
            Value::Tensor(mat_tensor(2, 2, &[2.0, 2.0, 1.0, 1.0])),
        ]))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:quiver3:InvalidArgument"));
        assert!(err.message.contains("identical size"));
    }

    #[test]
    fn quiver3_supports_explicit_coordinates_scale_style_and_get() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
            Value::Tensor(vec_tensor(&[4.0, 5.0])),
            Value::Tensor(vec_tensor(&[0.25, 0.5])),
            Value::Tensor(vec_tensor(&[0.75, 1.0])),
            Value::Tensor(vec_tensor(&[1.25, 1.5])),
            Value::Num(2.0),
            Value::String("r".into()),
            Value::String("DisplayName".into()),
            Value::String("field".into()),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver plot");
        };
        assert_eq!(quiver.scale, 2.0);
        assert_eq!(quiver.label.as_deref(), Some("field"));
        let z_data = get_builtin(vec![Value::Num(handle), Value::String("ZData".into())]).unwrap();
        let Value::Tensor(z_data) = z_data else {
            panic!("expected tensor zdata");
        };
        assert_eq!(z_data.materialize_f64(), vec![4.0, 5.0]);
    }

    #[test]
    fn quiver3_explicit_vectors_read_typed_integer_storage_exactly() {
        let wide = Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993));
        let (_, x, _, _, _, _, _, tail) = super::parse_quiver3_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                wide.clone(),
                wide.clone(),
                wide.clone(),
                wide.clone(),
                wide.clone(),
                wide.clone(),
                Value::from("r"),
            ]),
        )
        .expect("native integer scalars select explicit-six form");
        assert_eq!(x, wide);
        assert_eq!(tail.len(), 1);
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(int_vec_tensor(vec![0, 1])),
            Value::Tensor(int_vec_tensor(vec![2, 3])),
            Value::Tensor(int_vec_tensor(vec![4, 5])),
            Value::Tensor(int_vec_tensor(vec![6, 7])),
            Value::Tensor(int_vec_tensor(vec![8, 9])),
            Value::Tensor(int_vec_tensor(vec![10, 11])),
        ]))
        .unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver plot");
        };
        assert_eq!(quiver.x, vec![0.0, 1.0]);
        assert_eq!(quiver.y, vec![2.0, 3.0]);
        assert_eq!(quiver.z.as_ref().unwrap(), &vec![4.0, 5.0]);
        assert_eq!(quiver.u, vec![6.0, 7.0]);
        assert_eq!(quiver.v, vec![8.0, 9.0]);
        assert_eq!(quiver.w.as_ref().unwrap(), &vec![10.0, 11.0]);
    }

    #[test]
    fn quiver3_properties_preserve_wide_integer_source_storage() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let wide = (1_u64 << 53) + 1;
        let w = Tensor::new_integer(IntegerStorage::U64(vec![wide, wide + 2]), vec![2])
            .expect("wide w data");
        let handle = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(int_vec_tensor(vec![0, 1])),
            Value::Tensor(int_vec_tensor(vec![2, 3])),
            Value::Tensor(int_vec_tensor(vec![4, 5])),
            Value::Tensor(int_vec_tensor(vec![6, 7])),
            Value::Tensor(int_vec_tensor(vec![8, 9])),
            Value::Tensor(w),
        ]))
        .expect("quiver3");

        let value =
            get_builtin(vec![Value::Num(handle), Value::String("WData".into())]).expect("WData");
        let Value::Tensor(value) = value else {
            panic!("expected tensor WData")
        };
        assert_eq!(
            value.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide, wide + 2]))
        );
    }

    #[test]
    fn quiver3_rejects_mismatched_typed_integer_vector_lengths() {
        let err = materialize_quiver3_components(
            int_vec_tensor(vec![0, 1]),
            int_vec_tensor(vec![2]),
            int_vec_tensor(vec![4, 5]),
            int_vec_tensor(vec![6, 7]),
            int_vec_tensor(vec![8, 9]),
            int_vec_tensor(vec![10, 11]),
        )
        .expect_err("mismatched typed vectors should reject");

        assert!(format!("{err:?}").contains("same length"));
    }

    #[test]
    fn quiver3_descriptor_signatures_cover_supported_forms() {
        let labels: Vec<&str> = QUIVER3_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = quiver3(Z, U, V, W)"));
        assert!(labels.contains(&"h = quiver3(X, Y, Z, U, V, W)"));
        assert!(labels.contains(&"h = quiver3(Z, U, V, W, scaleOrStyleOrNameValue...)"));
        assert!(labels.contains(&"h = quiver3(X, Y, Z, U, V, W, scaleOrStyleOrNameValue...)"));
        assert!(labels.contains(&"h = quiver3(ax, Z, U, V, W)"));
        assert!(labels.contains(&"h = quiver3(ax, X, Y, Z, U, V, W)"));
        assert!(labels.contains(&"h = quiver3(ax, Z, U, V, W, scaleOrStyleOrNameValue...)"));
        assert!(labels.contains(&"h = quiver3(ax, X, Y, Z, U, V, W, scaleOrStyleOrNameValue...)"));
    }
}
