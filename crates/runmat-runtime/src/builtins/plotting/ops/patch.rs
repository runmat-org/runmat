//! MATLAB-compatible `patch` builtin.

use glam::{Vec3, Vec4};
use runmat_builtins::{StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{PatchEdgeColorMode, PatchFaceColorMode, PatchPlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::BuiltinResult;

use super::common::gather_tensor_from_gpu;
use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{
    parse_color_value, value_as_bool, value_as_f64, value_as_string, LineStyleParseOptions,
};

const BUILTIN_NAME: &str = "patch";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::patch")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "patch",
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
    notes: "patch is a plotting sink. Initial implementation gathers gpuArray coordinate inputs, triangulates on the host, then renders through the shared GPU renderer.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::patch")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "patch",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "patch performs rendering and terminates fusion graphs.",
};

#[derive(Clone, Debug)]
struct PatchOptions {
    x_data: Option<Tensor>,
    y_data: Option<Tensor>,
    z_data: Option<Tensor>,
    faces: Option<Tensor>,
    vertices: Option<Tensor>,
    face_color: Vec4,
    edge_color: Vec4,
    face_color_mode: PatchFaceColorMode,
    edge_color_mode: PatchEdgeColorMode,
    face_alpha: f32,
    edge_alpha: f32,
    line_width: f32,
    label: Option<String>,
    visible: bool,
}

impl Default for PatchOptions {
    fn default() -> Self {
        Self {
            x_data: None,
            y_data: None,
            z_data: None,
            faces: None,
            vertices: None,
            face_color: Vec4::new(0.0, 0.447, 0.741, 1.0),
            edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            face_color_mode: PatchFaceColorMode::Color,
            edge_color_mode: PatchEdgeColorMode::Color,
            face_alpha: 1.0,
            edge_alpha: 1.0,
            line_width: 0.5,
            label: None,
            visible: true,
        }
    }
}

#[runtime_builtin(
    name = "patch",
    category = "plotting",
    summary = "Create MATLAB-compatible colored polygon patches.",
    keywords = "patch,plotting,polygon,faces,vertices",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::patch"
)]
pub fn patch_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (axes_target, args) = split_leading_axes_handle(args, BUILTIN_NAME)?;
    apply_axes_target(axes_target, BUILTIN_NAME)?;
    let mut plot = Some(parse_patch_plot(args)?);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Patch",
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
        move |figure, axes| {
            let patch = plot.take().expect("patch plot consumed once");
            let plot_index = figure.add_patch_plot_on_axes(patch, axes);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_patch_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

fn parse_patch_plot(args: Vec<Value>) -> BuiltinResult<PatchPlot> {
    if args.is_empty() {
        return Err(plotting_error(BUILTIN_NAME, "patch: expected input data"));
    }
    let mut opts = PatchOptions::default();
    let mut remaining = if let Some(Value::Struct(st)) = args.first() {
        apply_struct_options(&mut opts, st)?;
        args[1..].to_vec()
    } else {
        args
    };

    if remaining.first().and_then(value_as_string).is_some() {
        apply_property_pairs(&mut opts, &remaining)?;
    } else {
        apply_positional_data(&mut opts, &mut remaining)?;
        apply_property_pairs(&mut opts, &remaining)?;
    }

    let (vertices, faces) = if let (Some(faces), Some(vertices)) = (&opts.faces, &opts.vertices) {
        (vertices_from_tensor(vertices)?, faces_from_tensor(faces)?)
    } else {
        vertices_faces_from_xyz(&opts)?
    };

    let mut plot = PatchPlot::new(vertices, faces)
        .map_err(|err| plotting_error(BUILTIN_NAME, format!("patch: {err}")))?;
    plot.face_color = opts.face_color;
    plot.edge_color = opts.edge_color;
    plot.face_color_mode = opts.face_color_mode;
    plot.edge_color_mode = opts.edge_color_mode;
    plot.face_alpha = opts.face_alpha;
    plot.edge_alpha = opts.edge_alpha;
    plot.line_width = opts.line_width;
    plot.label = opts.label;
    plot.visible = opts.visible;
    Ok(plot)
}

fn apply_positional_data(opts: &mut PatchOptions, args: &mut Vec<Value>) -> BuiltinResult<()> {
    if args.len() < 2 {
        apply_property_pairs(opts, args)?;
        args.clear();
        return Ok(());
    }
    opts.x_data = Some(tensor_from_value(args.remove(0))?);
    opts.y_data = Some(tensor_from_value(args.remove(0))?);

    if args.first().is_none_or(is_property_name) {
        return Ok(());
    }

    if args.len() >= 2 && !is_property_name(&args[1]) {
        opts.z_data = Some(tensor_from_value(args.remove(0))?);
        apply_color_argument(opts, &args.remove(0));
    } else {
        apply_color_argument(opts, &args.remove(0));
    }
    Ok(())
}

fn apply_struct_options(opts: &mut PatchOptions, st: &StructValue) -> BuiltinResult<()> {
    for (key, value) in &st.fields {
        apply_property(opts, key, value)?;
    }
    Ok(())
}

fn apply_property_pairs(opts: &mut PatchOptions, args: &[Value]) -> BuiltinResult<()> {
    if args.is_empty() {
        return Ok(());
    }
    if !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "patch: property/value arguments must come in pairs",
        ));
    }
    for pair in args.chunks_exact(2) {
        let key = value_as_string(&pair[0])
            .ok_or_else(|| plotting_error(BUILTIN_NAME, "patch: property names must be strings"))?;
        apply_property(opts, &key, &pair[1])?;
    }
    Ok(())
}

fn apply_property(opts: &mut PatchOptions, key: &str, value: &Value) -> BuiltinResult<()> {
    match key.trim().to_ascii_lowercase().as_str() {
        "xdata" => opts.x_data = Some(tensor_from_value(value.clone())?),
        "ydata" => opts.y_data = Some(tensor_from_value(value.clone())?),
        "zdata" => opts.z_data = Some(tensor_from_value(value.clone())?),
        "faces" => opts.faces = Some(tensor_from_value(value.clone())?),
        "vertices" => opts.vertices = Some(tensor_from_value(value.clone())?),
        "facecolor" | "color" => apply_face_color(opts, value)?,
        "edgecolor" => apply_edge_color(opts, value)?,
        "facealpha" => {
            opts.face_alpha = value_as_f64(value)
                .ok_or_else(|| plotting_error(BUILTIN_NAME, "patch: FaceAlpha must be numeric"))?
                .clamp(0.0, 1.0) as f32;
        }
        "edgealpha" => {
            opts.edge_alpha = value_as_f64(value)
                .ok_or_else(|| plotting_error(BUILTIN_NAME, "patch: EdgeAlpha must be numeric"))?
                .clamp(0.0, 1.0) as f32;
        }
        "linewidth" => {
            opts.line_width = value_as_f64(value)
                .ok_or_else(|| plotting_error(BUILTIN_NAME, "patch: LineWidth must be numeric"))?
                .max(0.0) as f32;
        }
        "displayname" => opts.label = value_as_string(value),
        "visible" => {
            opts.visible = value_as_bool(value)
                .ok_or_else(|| plotting_error(BUILTIN_NAME, "patch: Visible must be on/off"))?;
        }
        _ => {}
    }
    Ok(())
}

fn tensor_from_value(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gather_tensor_from_gpu(handle, BUILTIN_NAME),
        other => Tensor::try_from(&other)
            .map_err(|err| plotting_error(BUILTIN_NAME, format!("patch: {err}"))),
    }
}

fn apply_color_argument(opts: &mut PatchOptions, value: &Value) {
    if let Ok(color) = parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value) {
        opts.face_color = color;
        opts.face_color_mode = PatchFaceColorMode::Color;
    }
}

fn apply_face_color(opts: &mut PatchOptions, value: &Value) -> BuiltinResult<()> {
    if let Some(text) = value_as_string(value) {
        match text.trim().to_ascii_lowercase().as_str() {
            "none" => {
                opts.face_color_mode = PatchFaceColorMode::None;
                return Ok(());
            }
            "flat" | "interp" => {
                opts.face_color_mode = PatchFaceColorMode::Flat;
                return Ok(());
            }
            _ => {}
        }
    }
    opts.face_color = parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value)?;
    opts.face_color_mode = PatchFaceColorMode::Color;
    Ok(())
}

fn apply_edge_color(opts: &mut PatchOptions, value: &Value) -> BuiltinResult<()> {
    if let Some(text) = value_as_string(value) {
        if text.trim().eq_ignore_ascii_case("none") {
            opts.edge_color_mode = PatchEdgeColorMode::None;
            return Ok(());
        }
    }
    opts.edge_color = parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value)?;
    opts.edge_color_mode = PatchEdgeColorMode::Color;
    Ok(())
}

fn vertices_from_tensor(tensor: &Tensor) -> BuiltinResult<Vec<Vec3>> {
    if tensor.cols != 2 && tensor.cols != 3 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "patch: Vertices must be an N-by-2 or N-by-3 matrix",
        ));
    }
    let mut out = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let x = tensor.data[row];
        let y = tensor.data[row + tensor.rows];
        let z = if tensor.cols >= 3 {
            tensor.data[row + 2 * tensor.rows]
        } else {
            0.0
        };
        out.push(Vec3::new(x as f32, y as f32, z as f32));
    }
    Ok(out)
}

fn faces_from_tensor(tensor: &Tensor) -> BuiltinResult<Vec<Vec<usize>>> {
    if tensor.rows == 0 || tensor.cols == 0 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "patch: Faces must not be empty",
        ));
    }
    let mut faces = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let mut face = Vec::new();
        for col in 0..tensor.cols {
            let value = tensor.data[row + col * tensor.rows];
            if value.is_nan() {
                continue;
            }
            if value < 1.0 || value.fract() != 0.0 {
                return Err(plotting_error(
                    BUILTIN_NAME,
                    "patch: Faces must contain positive integer vertex indices",
                ));
            }
            face.push(value as usize - 1);
        }
        if face.len() >= 3 {
            faces.push(face);
        }
    }
    Ok(faces)
}

fn vertices_faces_from_xyz(opts: &PatchOptions) -> BuiltinResult<(Vec<Vec3>, Vec<Vec<usize>>)> {
    let x = opts
        .x_data
        .as_ref()
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "patch: missing XData"))?;
    let y = opts
        .y_data
        .as_ref()
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "patch: missing YData"))?;
    if x.rows != y.rows || x.cols != y.cols {
        return Err(plotting_error(
            BUILTIN_NAME,
            "patch: XData and YData must have the same size",
        ));
    }
    if let Some(z) = &opts.z_data {
        if z.rows != x.rows || z.cols != x.cols {
            return Err(plotting_error(
                BUILTIN_NAME,
                "patch: ZData must have the same size as XData and YData",
            ));
        }
    }
    if is_vector_tensor(x) && is_vector_tensor(y) {
        let x_values = x.data.clone();
        let y_values = y.data.clone();
        let z_values = opts.z_data.as_ref().map(|z| z.data.clone());
        if x_values.len() != y_values.len()
            || z_values
                .as_ref()
                .map(|z| z.len() != x_values.len())
                .unwrap_or(false)
        {
            return Err(plotting_error(
                BUILTIN_NAME,
                "patch: vector XData, YData, and ZData must have the same length",
            ));
        }
        let mut vertices = Vec::new();
        let mut face = Vec::new();
        for idx in 0..x_values.len() {
            let xv = x_values[idx];
            let yv = y_values[idx];
            let zv = z_values.as_ref().map(|z| z[idx]).unwrap_or(0.0);
            if xv.is_nan() || yv.is_nan() || zv.is_nan() {
                continue;
            }
            face.push(vertices.len());
            vertices.push(Vec3::new(xv as f32, yv as f32, zv as f32));
        }
        return Ok((vertices, vec![face]));
    }
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for col in 0..x.cols {
        let mut face = Vec::new();
        for row in 0..x.rows {
            let idx = row + col * x.rows;
            let xv = x.data[idx];
            let yv = y.data[idx];
            let zv = opts.z_data.as_ref().map(|z| z.data[idx]).unwrap_or(0.0);
            if xv.is_nan() || yv.is_nan() || zv.is_nan() {
                continue;
            }
            face.push(vertices.len());
            vertices.push(Vec3::new(xv as f32, yv as f32, zv as f32));
        }
        if face.len() >= 3 {
            faces.push(face);
        }
    }
    Ok((vertices, faces))
}

fn is_vector_tensor(tensor: &Tensor) -> bool {
    tensor.rows == 1 || tensor.cols == 1
}

fn is_property_name(value: &Value) -> bool {
    value_as_string(value)
        .map(|name| {
            matches!(
                name.trim().to_ascii_lowercase().as_str(),
                "xdata"
                    | "ydata"
                    | "zdata"
                    | "faces"
                    | "vertices"
                    | "facecolor"
                    | "edgecolor"
                    | "facealpha"
                    | "edgealpha"
                    | "linewidth"
                    | "displayname"
                    | "visible"
                    | "color"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::NumericDType;

    fn tensor(rows: usize, cols: usize, data: &[f64]) -> Value {
        Value::Tensor(Tensor {
            rows,
            cols,
            shape: vec![rows, cols],
            data: data.to_vec(),
            dtype: NumericDType::F64,
        })
    }

    #[test]
    fn patch_xyc_vector_builds_single_polygon() {
        let plot = parse_patch_plot(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            Value::String("r".into()),
        ])
        .unwrap();
        assert_eq!(plot.faces.len(), 1);
        assert_eq!(plot.vertices.len(), 3);
        assert_eq!(plot.face_color, Vec4::new(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn patch_matrix_columns_build_multiple_polygons() {
        let plot = parse_patch_plot(vec![
            tensor(4, 2, &[0.0, 1.0, 1.0, 0.0, 2.0, 3.0, 3.0, 2.0]),
            tensor(4, 2, &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
            Value::String("g".into()),
        ])
        .unwrap();
        assert_eq!(plot.faces.len(), 2);
        assert_eq!(plot.vertices.len(), 8);
    }

    #[test]
    fn patch_xy_accepts_trailing_name_value_pairs() {
        let plot = parse_patch_plot(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            Value::String("FaceColor".into()),
            Value::String("r".into()),
            Value::String("EdgeColor".into()),
            Value::String("none".into()),
        ])
        .unwrap();
        assert_eq!(plot.face_color, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(plot.edge_color_mode, PatchEdgeColorMode::None);
    }

    #[test]
    fn patch_faces_vertices_uses_one_based_faces() {
        let plot = parse_patch_plot(vec![
            Value::String("Faces".into()),
            tensor(1, 3, &[1.0, 2.0, 3.0]),
            Value::String("Vertices".into()),
            tensor(3, 2, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            Value::String("EdgeColor".into()),
            Value::String("none".into()),
        ])
        .unwrap();
        assert_eq!(plot.faces, vec![vec![0, 1, 2]]);
        assert_eq!(plot.edge_color_mode, PatchEdgeColorMode::None);
    }

    #[test]
    fn patch_registers_as_dispatch_builtin_and_returns_handle() {
        unsafe {
            std::env::set_var("RUNMAT_DISABLE_INTERACTIVE_PLOTS", "1");
        }
        let handle = crate::call_builtin(
            "patch",
            &[
                tensor(3, 1, &[0.0, 1.0, 0.0]),
                tensor(3, 1, &[0.0, 0.0, 1.0]),
                Value::String("b".into()),
            ],
        )
        .expect("patch builtin should dispatch");
        let Value::Num(handle) = handle else {
            panic!("expected numeric graphics handle");
        };
        let ty = crate::call_builtin("get", &[Value::Num(handle), Value::String("Type".into())])
            .expect("get patch type");
        assert_eq!(ty, Value::String("patch".into()));
    }
}
