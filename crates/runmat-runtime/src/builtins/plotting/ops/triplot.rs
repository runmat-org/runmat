//! MATLAB-compatible `triplot` builtin.
use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{
    TRIPLOT_ERROR_INTERNAL, TRIPLOT_ERROR_INVALID_ARGUMENT, TRIPLOT_GPU_INPUT_EXTENSION,
    TRIPLOT_INTEGER_CONNECTIVITY_EXTENSION, TRIPLOT_INTEGER_COORDINATE_EXTENSION,
};

use runmat_macros::runtime_builtin;
use runmat_value::NumericScalar;
use runmat_value::{ObjectInstance, StructValue, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::op_common::line_inputs::NumericInput;
use super::op_common::{apply_axes_target, split_leading_axes_arguments, AxesTarget};
use super::plot::build_line_plot_for_builtin;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    current_axes_state, current_figure_handle, line_color_for_target_axes_series_index,
    register_line_handle, render_active_plot, PlotRenderOptions,
};
use super::style::{parse_line_style_args, value_as_bool, value_as_string, LineStyleParseOptions};

const BUILTIN_NAME: &str = "triplot";
type TriangulationArgs = (Value, Value, Value, crate::call::BuiltinArguments);

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::triplot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "triplot",
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
    notes: "triplot is a plotting sink that builds a triangle-edge line object from connectivity; inputs are gathered before line rendering.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::triplot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "triplot",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "triplot performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "triplot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::triplot"
)]
pub async fn triplot_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "triplot: expected at most one output",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "triplot")?;
    let _arguments = args.fork();
    work.finish(triplot_impl(args, &work).await)
}

async fn triplot_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (leading_axes, args) = split_leading_axes_arguments(args, BUILTIN_NAME)?;
    let parsed = parse_triplot_args(leading_axes, args).await?;

    let color_target = parsed.axes_target.unwrap_or_else(|| {
        let axes = current_axes_state();
        (axes.handle, axes.active_index)
    });
    let style = parse_line_style_args(
        &parsed.style_tokens,
        &LineStyleParseOptions::generic(BUILTIN_NAME),
    )
    .map_err(map_triplot_invalid)?;
    let mut appearance = style.appearance;
    if !style.color_explicit {
        appearance.color =
            line_color_for_target_axes_series_index(color_target.0, color_target.1, 0);
    }
    let label = style.label.unwrap_or_default();
    let mut plot = build_line_plot_for_builtin(
        BUILTIN_NAME,
        parsed.x_edges,
        parsed.y_edges,
        &label,
        &appearance,
    )
    .map_err(map_triplot_internal)?;
    if label.is_empty() {
        plot.label = None;
    }
    if let Some(visible) = parsed.visible {
        plot.set_visible(visible);
    }

    work.check()?;
    apply_axes_target(parsed.axes_target, BUILTIN_NAME).map_err(map_triplot_invalid)?;
    let figure_handle = current_figure_handle();
    let mut plot_slot = Some(plot);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let target_axes = parsed.axes_target.map(|(_, axes)| axes);
    work.check()?;
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions::default(),
        move |figure, axes| {
            let axes = target_axes.unwrap_or(axes);
            let plot = plot_slot
                .take()
                .expect("triplot plot consumed exactly once");
            let plot_index = figure.add_line_plot_on_axes(plot, axes);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );

    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = register_line_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_triplot_internal(err));
    }
    Ok(handle)
}

struct ParsedTriplot {
    axes_target: AxesTarget,
    x_edges: Vec<f64>,
    y_edges: Vec<f64>,
    style_tokens: Vec<Value>,
    visible: Option<bool>,
}

async fn parse_triplot_args(
    leading_axes: AxesTarget,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<ParsedTriplot> {
    if args.is_empty() {
        return Err(triplot_invalid("expected TRI,X,Y or triangulation input"));
    }

    let (mut axes_target, args) = peel_parent_value(leading_axes, args)?;
    let (tri, x, y, rest) = if let Some((tri, x, y, rest)) = parse_triangulation_struct(&args)? {
        (tri, x, y, rest)
    } else {
        if args.len() < 3 {
            return Err(triplot_invalid("expected TRI, X, and Y inputs"));
        }
        (
            args[0].clone(),
            args[1].clone(),
            args[2].clone(),
            args.range(3..)?,
        )
    };
    let StyleArgs {
        tokens,
        parent_axes,
        visible,
    } = extract_triplot_style_args(rest)?;
    if let Some(parent_axes) = parent_axes {
        axes_target = Some(parent_axes);
    }

    if crate::builtins::common::validation::value_contains_native_integer_class(&tri) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRIPLOT_INTEGER_CONNECTIVITY_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &x,
        &TRIPLOT_INTEGER_COORDINATE_EXTENSION,
        BUILTIN_NAME,
        "x coordinates",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &y,
        &TRIPLOT_INTEGER_COORDINATE_EXTENSION,
        BUILTIN_NAME,
        "y coordinates",
    )
    .await?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&tri)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&x)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&y)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRIPLOT_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    let tri = NumericInput::from_value(tri, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    let x = NumericInput::from_value(x, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    let y = NumericInput::from_value(y, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    let (x_edges, y_edges) = build_triangle_edges(&tri, &x, &y)?;
    Ok(ParsedTriplot {
        axes_target,
        x_edges,
        y_edges,
        style_tokens: tokens,
        visible,
    })
}

fn peel_parent_value(
    leading_axes: AxesTarget,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<(AxesTarget, crate::call::BuiltinArguments)> {
    if leading_axes.is_none() {
        if let Some(first) = args.first() {
            if let Ok(PlotHandle::Axes(figure, axes)) = resolve_plot_handle(first, BUILTIN_NAME) {
                return Ok((Some((figure, axes)), args.into_range(1..)?));
            }
        }
    }
    Ok((leading_axes, args))
}

fn parse_triangulation_struct(
    args: &crate::call::BuiltinArguments,
) -> BuiltinResult<Option<TriangulationArgs>> {
    let Some(first) = args.first() else {
        return Ok(None);
    };
    if !matches!(first, Value::Struct(_) | Value::Object(_)) {
        return Ok(None);
    }
    let tri = lookup_triangulation_value(first, "ConnectivityList")
        .or_else(|| lookup_triangulation_value(first, "triangulation"))
        .or_else(|| lookup_triangulation_value(first, "tri"))
        .or_else(|| lookup_triangulation_value(first, "Triangulation"))
        .cloned()
        .ok_or_else(|| triplot_invalid("triangulation input must contain ConnectivityList"))?;
    let points = lookup_triangulation_value(first, "Points")
        .or_else(|| lookup_triangulation_value(first, "vertices"))
        .or_else(|| lookup_triangulation_value(first, "X"))
        .cloned()
        .ok_or_else(|| triplot_invalid("triangulation input must contain Points"))?;
    let points = Tensor::try_from(&points)
        .map_err(|e| triplot_invalid(format!("Points must be numeric: {e}")))?;
    let (point_rows, point_cols) = triplot_matrix_dimensions(&points);
    if point_cols < 2 || point_rows == 0 {
        return Err(triplot_invalid(
            "Points must be an N-by-2 or N-by-more coordinate matrix",
        ));
    }
    let x = column_tensor(&points, 0)?;
    let y = column_tensor(&points, 1)?;
    Ok(Some((
        tri,
        Value::Tensor(x),
        Value::Tensor(y),
        args.range(1..)?,
    )))
}

fn lookup_triangulation_value<'a>(value: &'a Value, name: &str) -> Option<&'a Value> {
    match value {
        Value::Struct(st) => lookup_field_case_insensitive(st, name),
        Value::Object(object) => lookup_object_property_case_insensitive(object, name),
        _ => None,
    }
}

fn lookup_field_case_insensitive<'a>(st: &'a StructValue, name: &str) -> Option<&'a Value> {
    st.fields
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}

fn lookup_object_property_case_insensitive<'a>(
    object: &'a ObjectInstance,
    name: &str,
) -> Option<&'a Value> {
    object
        .properties()
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}

struct StyleArgs {
    tokens: Vec<Value>,
    parent_axes: AxesTarget,
    visible: Option<bool>,
}

fn extract_triplot_style_args(args: crate::call::BuiltinArguments) -> BuiltinResult<StyleArgs> {
    let mut tokens = Vec::new();
    let mut parent_axes = None;
    let mut visible = None;
    let mut idx = 0usize;
    while idx < args.len() {
        let token = args[idx].clone();
        let Some(name) = value_as_string(&token).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "triplot",
                source.to_string(),
                source,
            )
        })?
        else {
            return Err(triplot_invalid("style arguments must be strings"));
        };
        let lower = name.trim().to_ascii_lowercase();
        if lower == "parent" {
            if idx + 1 >= args.len() {
                return Err(triplot_invalid("Parent requires an axes handle"));
            }
            let PlotHandle::Axes(figure, axes) = resolve_plot_handle(&args[idx + 1], BUILTIN_NAME)?
            else {
                return Err(triplot_invalid("Parent must be an axes handle"));
            };
            parent_axes = Some((figure, axes));
            idx += 2;
            continue;
        }
        if lower == "visible" {
            if idx + 1 >= args.len() {
                return Err(triplot_invalid("Visible requires a value"));
            }
            visible = Some(
                value_as_bool(&args[idx + 1])
                    .ok_or_else(|| triplot_invalid("Visible must be logical or on/off"))?,
            );
            idx += 2;
            continue;
        }
        tokens.push(token);
        if idx == 0 && is_line_spec_token(&name) {
            idx += 1;
            continue;
        }
        if idx + 1 >= args.len() {
            return Err(triplot_invalid("name-value arguments must come in pairs"));
        }
        tokens.push(args[idx + 1].clone());
        idx += 2;
    }
    Ok(StyleArgs {
        tokens,
        parent_axes,
        visible,
    })
}

fn is_line_spec_token(token: &str) -> bool {
    !token.trim().is_empty()
        && token
            .chars()
            .all(|ch| "-:.+*ox^v<>sdphrgbcmykw".contains(ch))
}

fn build_triangle_edges(
    tri: &Tensor,
    x: &Tensor,
    y: &Tensor,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let (tri_rows, tri_cols) = triplot_matrix_dimensions(tri);
    if tri_cols != 3 {
        return Err(triplot_invalid("TRI must be an M-by-3 connectivity matrix"));
    }
    let point_count = tensor_utils::tensor_element_len(x);
    if !is_vector_tensor(x)
        || !is_vector_tensor(y)
        || point_count != tensor_utils::tensor_element_len(y)
    {
        return Err(triplot_invalid(
            "X and Y must be coordinate vectors with the same length",
        ));
    }
    if tri_rows == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if point_count == 0 {
        return Err(triplot_invalid("X and Y must contain at least one point"));
    }

    let x_values = tensor_utils::tensor_values_f64(x);
    let y_values = tensor_utils::tensor_values_f64(y);
    let edge_count = tri_rows
        .checked_mul(5)
        .ok_or_else(|| triplot_invalid("triangle edge count overflows"))?;
    let mut x_edges = Vec::new();
    let mut y_edges = Vec::new();
    x_edges.try_reserve_exact(edge_count).map_err(|cause| {
        crate::runtime_error::semantic_error_with_source(
            "RunMat:triplot:InvalidArgument",
            "triangle edge storage unavailable",
            cause,
        )
    })?;
    y_edges.try_reserve_exact(edge_count).map_err(|cause| {
        crate::runtime_error::semantic_error_with_source(
            "RunMat:triplot:InvalidArgument",
            "triangle edge storage unavailable",
            cause,
        )
    })?;
    for row in 0..tri_rows {
        let a = vertex_index_at(tri, row, point_count)?;
        let b = vertex_index_at(tri, row + tri_rows, point_count)?;
        let c = vertex_index_at(tri, row + 2 * tri_rows, point_count)?;
        for idx in [a, b, c, a] {
            x_edges.push(x_values[idx]);
            y_edges.push(y_values[idx]);
        }
        x_edges.push(f64::NAN);
        y_edges.push(f64::NAN);
    }
    Ok((x_edges, y_edges))
}

fn vertex_index_at(tensor: &Tensor, index: usize, point_count: usize) -> BuiltinResult<usize> {
    let value = tensor
        .numeric_value_at(index)
        .ok_or_else(|| triplot_invalid("TRI index is out of range"))?;
    if let Some(integer) = value.into_int_value() {
        let one_based = integer.try_to_usize().ok_or_else(|| {
            triplot_invalid("TRI indices must be positive integers in the runtime index range")
        })?;
        if one_based == 0 || one_based > point_count {
            return Err(triplot_invalid("TRI references a vertex outside X/Y"));
        }
        return Ok(one_based - 1);
    }
    match value {
        NumericScalar::F64(value) => vertex_index(value, point_count),
        NumericScalar::F32(value) => vertex_index(f64::from(value), point_count),
        _ => unreachable!("integer scalar handled above"),
    }
}

fn vertex_index(value: f64, point_count: usize) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(triplot_invalid(
            "TRI indices must be positive finite integers",
        ));
    }
    let idx = value as usize;
    if idx == 0 || idx > point_count {
        return Err(triplot_invalid("TRI references a vertex outside X/Y"));
    }
    Ok(idx - 1)
}

fn column_tensor(tensor: &Tensor, col: usize) -> BuiltinResult<Tensor> {
    let (rows, _) = triplot_matrix_dimensions(tensor);
    let indices = (0..rows).map(|row| row + col * rows).collect::<Vec<_>>();
    super::op_common::line_inputs::select_native(tensor, &indices, vec![rows, 1], BUILTIN_NAME)
}

fn triplot_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

fn is_vector_tensor(tensor: &Tensor) -> bool {
    let (rows, cols) = triplot_matrix_dimensions(tensor);
    rows == 1 || cols == 1
}

fn triplot_invalid(detail: impl AsRef<str>) -> RuntimeError {
    triplot_error_with_detail(&TRIPLOT_ERROR_INVALID_ARGUMENT, detail)
}

fn triplot_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let raw = detail.as_ref().trim();
    let normalized = raw.strip_prefix("triplot:").map(str::trim).unwrap_or(raw);
    let message = if normalized.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {}", error.message, normalized)
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_triplot_invalid(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = triplot_error_with_detail(&TRIPLOT_ERROR_INVALID_ARGUMENT, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_triplot_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = triplot_error_with_detail(&TRIPLOT_ERROR_INTERNAL, err.message.clone());
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
    async fn triplot_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::triplot_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::subplot::subplot_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clone_figure, current_figure_handle, reset_plot_state};
    use futures::executor::block_on;
    use runmat_plot::plots::PlotElement;

    fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![rows, cols]).expect("triplot test tensor"))
    }

    fn int_tensor(data: Vec<i16>, rows: usize, cols: usize) -> Tensor {
        Tensor::new_integer(runmat_value::IntegerStorage::I16(data), vec![rows, cols])
            .expect("integer tensor")
    }

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        guard
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn triplot_rank_one_connectivity_is_one_triangle_and_rank_zero_is_not_a_vector() {
        let tri = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let x = Tensor::new(vec![0.0, 1.0, 0.0], vec![3]).unwrap();
        let y = Tensor::new(vec![0.0, 0.0, 1.0], vec![3]).unwrap();
        let (x_edges, y_edges) = build_triangle_edges(&tri, &x, &y).unwrap();
        assert_eq!(&x_edges[..4], &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(&y_edges[..4], &[0.0, 0.0, 1.0, 0.0]);

        let scalar = Tensor::new(vec![1.0], Vec::new()).unwrap();
        assert!(!is_vector_tensor(&scalar));
        assert!(build_triangle_edges(&scalar, &x, &y)
            .expect_err("rank-zero TRI has no connectivity columns")
            .message()
            .contains("M-by-3"));
        assert!(build_triangle_edges(&tri, &scalar, &y)
            .expect_err("rank-zero X is not a coordinate vector")
            .message()
            .contains("coordinate vectors"));
    }

    #[test]
    fn triplot_edges_read_typed_integer_storage_exactly() {
        let (x_edges, y_edges) = build_triangle_edges(
            &int_tensor(vec![1, 2, 3], 1, 3),
            &int_tensor(vec![10, 20, 30], 3, 1),
            &int_tensor(vec![1, 2, 3], 3, 1),
        )
        .expect("edges");

        assert_eq!(x_edges[..4], [10.0, 20.0, 30.0, 10.0]);
        assert!(x_edges[4].is_nan());
        assert_eq!(y_edges[..4], [1.0, 2.0, 3.0, 1.0]);
        assert!(y_edges[4].is_nan());
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn triplot_connectivity_parser_preserves_wide_integer_indices_exactly() {
        let one_based = 9_007_199_254_740_993_u64;
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![one_based]),
            vec![1, 1],
        )
        .expect("wide connectivity");
        assert_eq!(
            vertex_index_at(&tensor, 0, one_based as usize).expect("exact index"),
            one_based as usize - 1
        );
    }

    #[test]
    fn triplot_integer_connectivity_and_coordinates_are_independently_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let connectivity_error = block_on(triplot_builtin(vec![
            Value::Tensor(int_tensor(vec![1, 2, 3], 1, 3)),
            tensor(&[0.0, 1.0, 0.0], 1, 3),
            tensor(&[0.0, 0.0, 1.0], 1, 3),
        ]))
        .expect_err("typed connectivity is an extension");
        assert_eq!(
            connectivity_error.identifier(),
            TRIPLOT_INTEGER_CONNECTIVITY_EXTENSION.error_identifier
        );

        let coordinate_error = block_on(triplot_builtin(vec![
            tensor(&[1.0, 2.0, 3.0], 1, 3),
            Value::Tensor(int_tensor(vec![0, 1, 0], 1, 3)),
            tensor(&[0.0, 0.0, 1.0], 1, 3),
        ]))
        .expect_err("typed coordinates are an extension");
        assert_eq!(
            coordinate_error.identifier(),
            TRIPLOT_INTEGER_COORDINATE_EXTENSION.error_identifier
        );
    }

    #[test]
    fn triplot_explicit_gpu_input_is_gated_before_gather() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 3],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle;
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let error = block_on(triplot_builtin(vec![
            Value::GpuTensor(handle),
            tensor(&[0.0, 1.0, 0.0], 1, 3),
            tensor(&[0.0, 0.0, 1.0], 1, 3),
        ]))
        .expect_err("MATLAB mode rejects explicit interactive gpuArray input");
        assert_eq!(
            error.identifier(),
            TRIPLOT_GPU_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn triplot_column_tensor_reads_typed_integer_storage_exactly() {
        use runmat_value::{IntegerStorage, NumericDType, NumericStorage};
        let column = column_tensor(&int_tensor(vec![1, 2, 3, 4], 2, 2), 1).unwrap();

        assert_eq!(column.materialize_f64(), vec![3.0, 4.0]);
        assert_eq!(column.numeric_dtype(), runmat_value::NumericDType::I16);
        assert_eq!(
            column.integer_storage(),
            Some(&runmat_value::IntegerStorage::I16(vec![3, 4]))
        );
        for (storage, dtype) in [
            (
                NumericStorage::F32(vec![1.0, 2.0, 3.0, 4.0]),
                NumericDType::F32,
            ),
            (
                NumericStorage::U64(vec![0, 1, 9_007_199_254_740_993, 9_007_199_254_740_995]),
                NumericDType::U64,
            ),
        ] {
            let input = Tensor::from_numeric_storage(storage, vec![2, 2]).unwrap();
            let before = input.clone();
            let output = column_tensor(&input, 1).unwrap();
            assert_eq!(output.numeric_dtype(), dtype);
            assert_eq!(output.shape(), [2, 1]);
            assert_eq!(input, before);
            if dtype == NumericDType::F32 {
                assert_eq!(output.as_f32_slice(), Some([3.0_f32, 4.0].as_slice()));
            } else {
                assert_eq!(
                    output.integer_storage(),
                    Some(&IntegerStorage::U64(vec![
                        9_007_199_254_740_993,
                        9_007_199_254_740_995
                    ]))
                );
            }
        }
    }

    #[test]
    fn triplot_builds_nan_separated_triangle_edges() {
        let _guard = setup();
        let handle = block_on(triplot_builtin(vec![
            tensor(&[1.0, 1.0, 2.0, 3.0, 3.0, 4.0], 2, 3),
            tensor(&[0.0, 1.0, 0.0, 1.0], 1, 4),
            tensor(&[0.0, 0.0, 1.0, 1.0], 1, 4),
            Value::String("r--".into()),
        ]))
        .unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Line(line) = fig.plots().next().unwrap() else {
            panic!("expected line plot");
        };
        let (x, y) = line.host_xy_f64().unwrap().unwrap();
        assert_eq!(x.len(), 10);
        assert_eq!(&x[0..4], &[0.0, 1.0, 0.0, 0.0]);
        assert!(x[4].is_nan());
        assert_eq!(&y[5..9], &[0.0, 1.0, 1.0, 0.0]);
        assert!(y[9].is_nan());
        assert_eq!(line.line_style, runmat_plot::plots::LineStyle::Dashed);
    }

    #[test]
    fn triplot_supports_axes_parent_visible_and_properties() {
        let _guard = setup();
        let ax = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let handle = block_on(triplot_builtin(vec![
            tensor(&[1.0, 2.0, 3.0], 1, 3),
            tensor(&[0.0, 1.0, 0.0], 1, 3),
            tensor(&[0.0, 0.0, 1.0], 1, 3),
            Value::String("Parent".into()),
            Value::Num(ax),
            Value::String("Visible".into()),
            Value::String("off".into()),
            Value::String("LineWidth".into()),
            Value::Num(2.0),
            Value::String("DisplayName".into()),
            Value::String("mesh".into()),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayName".into())
            ])
            .unwrap(),
            Value::String("mesh".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Visible".into())]).unwrap(),
            Value::String("off".into())
        );
        set_builtin(vec![
            Value::Num(handle),
            Value::String("LineWidth".into()),
            Value::Num(3.0),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("LineWidth".into())]).unwrap(),
            Value::Num(3.0)
        );
    }

    #[test]
    fn triplot_accepts_triangulation_like_struct() {
        let _guard = setup();
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields
            .insert("ConnectivityList".into(), tensor(&[1.0, 2.0, 3.0], 1, 3));
        st.fields.insert(
            "Points".into(),
            tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 2),
        );
        let handle = block_on(triplot_builtin(vec![Value::Struct(st.finish())])).unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Line(line) = fig.plots().next().unwrap() else {
            panic!("expected line plot");
        };
        let (x_data, y_data) = line.host_xy_f64().unwrap().unwrap();
        assert_eq!(x_data.len(), 5);
        assert_eq!(y_data.len(), 5);
        assert_eq!(x_data[0], x_data[3]);
        assert_eq!(y_data[0], y_data[3]);
        assert!(x_data[4].is_nan());
        assert!(y_data[4].is_nan());
        for (x, y) in x_data.iter().zip(&y_data).take(3) {
            let point = (*x, *y);
            assert!(
                point == (0.0, 0.0) || point == (1.0, 0.0) || point == (0.0, 1.0),
                "unexpected triangle vertex ({x}, {y})"
            );
        }
    }

    #[test]
    fn triplot_accepts_delaunaytri_object() {
        let _guard = setup();
        let dt = block_on(
            crate::builtins::geometry::triangulation::delaunay_tri_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![tensor(
                    &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    3,
                    2,
                )]),
            ),
        )
        .unwrap();
        let handle = block_on(triplot_builtin(vec![dt])).unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Line(line) = fig.plots().next().unwrap() else {
            panic!("expected line plot");
        };
        let (x_data, y_data) = line.host_xy_f64().unwrap().unwrap();
        assert_eq!(x_data.len(), 5);
        assert_eq!(y_data.len(), 5);
        assert_eq!(x_data[0], x_data[3]);
        assert_eq!(y_data[0], y_data[3]);
        assert!(x_data[4].is_nan());
        assert!(y_data[4].is_nan());
        for (x, y) in x_data.iter().zip(&y_data).take(3) {
            let point = (*x, *y);
            assert!(
                point == (0.0, 0.0) || point == (1.0, 0.0) || point == (0.0, 1.0),
                "unexpected triangle vertex ({x}, {y})"
            );
        }
    }

    #[test]
    fn triplot_rejects_out_of_range_connectivity() {
        let _guard = setup();
        let err = block_on(triplot_builtin(vec![
            tensor(&[1.0, 2.0, 4.0], 1, 3),
            tensor(&[0.0, 1.0, 0.0], 1, 3),
            tensor(&[0.0, 0.0, 1.0], 1, 3),
        ]))
        .expect_err("invalid vertex index should fail");
        assert_eq!(err.identifier(), TRIPLOT_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn triplot_allows_empty_connectivity() {
        let _guard = setup();
        let handle = block_on(triplot_builtin(vec![
            tensor(&[], 0, 3),
            tensor(&[0.0, 1.0, 0.0], 1, 3),
            tensor(&[0.0, 0.0, 1.0], 1, 3),
        ]))
        .unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Line(line) = fig.plots().next().unwrap() else {
            panic!("expected line plot");
        };
        let (x, y) = line.host_xy_f64().unwrap().unwrap();
        assert!(x.is_empty());
        assert!(y.is_empty());
    }
}
