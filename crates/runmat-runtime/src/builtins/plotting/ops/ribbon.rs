//! MATLAB-compatible `ribbon` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, value_as_string, SurfaceStyle, SurfaceStyleDefaults};

const BUILTIN_NAME: &str = "ribbon";
const DEFAULT_RIBBON_WIDTH: f64 = 0.75;

const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Surface handle or row vector of surface handles for the rendered ribbons.",
}];

const INPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Ribbon heights. Each matrix column becomes one ribbon.",
}];

const INPUT_Y_WIDTH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ribbon heights. Each matrix column becomes one ribbon.",
    },
    BuiltinParamDescriptor {
        name: "width",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0.75"),
        description: "Ribbon width around each column center.",
    },
];

const INPUT_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates for the rows of Y.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ribbon heights. Each matrix column becomes one ribbon.",
    },
];

const INPUT_X_Y_WIDTH: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates for the rows of Y.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ribbon heights. Each matrix column becomes one ribbon.",
    },
    BuiltinParamDescriptor {
        name: "width",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0.75"),
        description: "Ribbon width around each column center.",
    },
];

const INPUT_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional axes target and surface style name/value pairs.",
}];

const RIBBON_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = ribbon(Y)",
        inputs: &INPUT_Y,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(Y, width)",
        inputs: &INPUT_Y_WIDTH,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(X, Y)",
        inputs: &INPUT_X_Y,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(X, Y, width)",
        inputs: &INPUT_X_Y_WIDTH,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(___, Name, Value, ...)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(ax, ___)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
];

const RIBBON_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RIBBON.INVALID_ARGUMENT",
    identifier: Some("RunMat:ribbon:InvalidArgument"),
    when: "Ribbon data, x coordinates, width, axes target, or style arguments are invalid.",
    message: "ribbon: invalid argument",
};

const RIBBON_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RIBBON.INTERNAL",
    identifier: Some("RunMat:ribbon:Internal"),
    when: "Internal ribbon surface construction or rendering fails.",
    message: "ribbon: internal operation failed",
};

const RIBBON_ERRORS: [BuiltinErrorDescriptor; 2] =
    [RIBBON_ERROR_INVALID_ARGUMENT, RIBBON_ERROR_INTERNAL];

pub const RIBBON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RIBBON_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RIBBON_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::ribbon")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ribbon",
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
    notes: "ribbon lowers input columns to surface strips and currently gathers any gpuArray input before constructing the ribbon coordinate grids.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::ribbon")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ribbon",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "ribbon performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "ribbon",
    category = "plotting",
    summary = "Create 3-D ribbon plots from columns of data.",
    keywords = "ribbon,plotting,3d,surface",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::ribbon::RIBBON_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::ribbon"
)]
pub fn ribbon_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (axes_target, args) =
        split_leading_axes_handle(args, BUILTIN_NAME).map_err(map_invalid_argument)?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_invalid_argument)?;

    let parsed = parse_ribbon_args(args)?;
    let mut surfaces = Vec::with_capacity(parsed.y.cols);
    for col in 0..parsed.y.cols {
        let mut surface = build_ribbon_surface(&parsed.x, &parsed.y, col, parsed.width)?;
        parsed.style.apply_to_plot(&mut surface);
        surfaces.push(surface);
    }

    render_ribbons(surfaces).map_err(map_internal)
}

struct RibbonData {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

struct ParsedRibbon {
    x: Vec<f64>,
    y: RibbonData,
    width: f64,
    style: SurfaceStyle,
}

fn parse_ribbon_args(args: Vec<Value>) -> BuiltinResult<ParsedRibbon> {
    if args.is_empty() {
        return Err(invalid_argument("expected Y data"));
    }
    let property_start = find_property_start(&args);
    let (positional, rest) = args.split_at(property_start);
    if positional.is_empty() || positional.len() > 3 {
        return Err(invalid_argument(
            "expected ribbon(Y), ribbon(Y,width), ribbon(X,Y), or ribbon(X,Y,width)",
        ));
    }

    let (x_value, y_value, width_value) = match positional.len() {
        1 => (None, &positional[0], None),
        2 if scalar_f64(&positional[1]).is_some() => (None, &positional[0], Some(&positional[1])),
        2 => (Some(&positional[0]), &positional[1], None),
        3 => (Some(&positional[0]), &positional[1], Some(&positional[2])),
        _ => unreachable!(),
    };

    let y = ribbon_data_from_value(y_value)?;
    let x = if let Some(x_value) = x_value {
        let x = numeric_vector(x_value, "X")?;
        if x.len() != y.rows {
            return Err(invalid_argument(format!(
                "X length ({}) must match the number of Y rows ({})",
                x.len(),
                y.rows
            )));
        }
        x
    } else {
        (1..=y.rows).map(|idx| idx as f64).collect()
    };
    if x.iter().any(|value| !value.is_finite()) {
        return Err(invalid_argument("X coordinates must be finite"));
    }

    let width = width_value
        .map(|value| {
            scalar_f64(value).ok_or_else(|| invalid_argument("width must be a numeric scalar"))
        })
        .transpose()?
        .unwrap_or(DEFAULT_RIBBON_WIDTH);
    if !width.is_finite() || width <= 0.0 {
        return Err(invalid_argument("width must be a positive finite scalar"));
    }

    let style = parse_surface_style_args(
        BUILTIN_NAME,
        rest,
        SurfaceStyleDefaults::new(
            ColorMap::Parula,
            ShadingMode::Smooth,
            false,
            1.0,
            false,
            true,
        ),
    )
    .map_err(map_invalid_argument)?;

    Ok(ParsedRibbon { x, y, width, style })
}

fn ribbon_data_from_value(value: &Value) -> BuiltinResult<RibbonData> {
    let tensor = Tensor::try_from(value)
        .map_err(|err| invalid_argument(format!("Y must be numeric: {err}")))?;
    if tensor.data.is_empty() {
        return Err(invalid_argument("Y must be non-empty"));
    }
    if tensor.shape.len() > 2 {
        return Err(invalid_argument("Y must be a vector or 2-D matrix"));
    }
    if tensor.shape.len() == 1 || tensor.rows == 1 || tensor.cols == 1 {
        let data = tensor_utils::tensor_into_values_f64(tensor);
        return Ok(RibbonData {
            rows: data.len(),
            cols: 1,
            data,
        });
    }
    let rows = tensor.rows;
    let cols = tensor.cols;
    let data = tensor_utils::tensor_into_values_f64(tensor);
    Ok(RibbonData { rows, cols, data })
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    if let Some(value) = scalar_f64(value) {
        return Ok(vec![value]);
    }
    let tensor =
        Tensor::try_from(value).map_err(|err| invalid_argument(format!("{name}: {err}")))?;
    if tensor.shape.len() > 2 || (tensor.rows > 1 && tensor.cols > 1) {
        return Err(invalid_argument(format!("{name} must be a vector")));
    }
    Ok(tensor_utils::tensor_into_values_f64(tensor))
}

fn scalar_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn find_property_start(args: &[Value]) -> usize {
    args.iter()
        .position(|value| value_as_string(value).is_some())
        .unwrap_or(args.len())
}

fn build_ribbon_surface(
    x: &[f64],
    y: &RibbonData,
    col: usize,
    width: f64,
) -> BuiltinResult<SurfacePlot> {
    let center = col as f64 + 1.0;
    let lower = center - width / 2.0;
    let upper = center + width / 2.0;
    let heights = (0..y.rows)
        .map(|row| y.data[row + col * y.rows])
        .collect::<Vec<_>>();
    let x_grid = vec![x.to_vec(), x.to_vec()];
    let y_grid = vec![vec![lower; y.rows], vec![upper; y.rows]];
    let z_grid = vec![heights.clone(), heights];
    SurfacePlot::from_coordinate_grids(x_grid, y_grid, z_grid)
        .map(|surface| {
            surface
                .with_colormap(ColorMap::Parula)
                .with_shading(ShadingMode::Smooth)
        })
        .map_err(|err| invalid_argument(format!("failed to build ribbon surface: {err}")))
}

fn render_ribbons(surfaces: Vec<SurfacePlot>) -> BuiltinResult<Value> {
    let mut surfaces = Some(surfaces);
    let plot_indices_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Ribbon Plot",
            x_label: "X",
            y_label: "Ribbon",
            axis_equal: false,
            ..Default::default()
        },
        move |figure, axes| {
            let surfaces = surfaces.take().expect("ribbon surfaces consumed once");
            let mut plot_indices = Vec::with_capacity(surfaces.len());
            for surface in surfaces {
                let plot_index = figure.add_surface_plot_on_axes(surface, axes);
                plot_indices.push((axes, plot_index));
            }
            figure.set_axes_view(axes, -37.5, 30.0);
            figure.set_axes_zlabel(axes, "Y");
            *plot_indices_slot.borrow_mut() = plot_indices;
            Ok(())
        },
    );

    let plot_indices = plot_indices_out.borrow().clone();
    if plot_indices.is_empty() {
        return render_result.map(|_| Value::Num(f64::NAN));
    }
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if !lower.contains("plotting is unavailable") && !lower.contains("non-main thread") {
            return Err(err);
        }
    }
    let handles = plot_indices
        .into_iter()
        .map(|(axes, plot_index)| {
            crate::builtins::plotting::state::register_surface_handle(
                figure_handle,
                axes,
                plot_index,
            )
        })
        .collect::<Vec<_>>();
    Ok(handles_value(handles))
}

fn handles_value(handles: Vec<f64>) -> Value {
    if handles.len() == 1 {
        Value::Num(handles[0])
    } else {
        let len = handles.len();
        Value::Tensor(Tensor::new_2d(handles, 1, len).expect("valid handle vector"))
    }
}

fn error_with_detail(
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

fn map_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    error_with_detail(&RIBBON_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    error_with_detail(&RIBBON_ERROR_INTERNAL, err.message)
}

fn invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    error_with_detail(&RIBBON_ERROR_INVALID_ARGUMENT, detail)
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
    use runmat_builtins::{IntegerStorage, NumericDType};
    use runmat_plot::plots::PlotElement;

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor {
            data,
            rows,
            cols,
            shape: vec![rows, cols],
            integer_data: None,
            dtype: NumericDType::F64,
        })
    }

    fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let mut tensor = Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor");
        tensor.data.fill(f64::NAN);
        Value::Tensor(tensor)
    }

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn ribbon_builds_one_surface_per_matrix_column() {
        let _guard = setup();
        let handles =
            ribbon_builtin(vec![tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2)]).expect("ribbon");
        let Value::Tensor(handles) = handles else {
            panic!("expected handle vector");
        };
        assert_eq!(handles.data.len(), 2);

        let figure = clone_figure(current_figure_handle()).expect("figure");
        assert_eq!(figure.plots().count(), 2);
        let PlotElement::Surface(first) = figure.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(first.x_grid.as_ref().unwrap()[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(first.y_grid.as_ref().unwrap()[0], vec![0.625, 0.625, 0.625]);
        assert_eq!(first.y_grid.as_ref().unwrap()[1], vec![1.375, 1.375, 1.375]);
        assert_eq!(first.z_data.as_ref().unwrap()[0], vec![1.0, 2.0, 3.0]);

        let ty = get_builtin(vec![
            Value::Num(handles.data[0]),
            Value::String("Type".into()),
        ])
        .expect("get type");
        assert_eq!(ty, Value::String("surface".into()));
    }

    #[test]
    fn ribbon_supports_x_width_axes_target_and_surface_style() {
        let _guard = setup();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let handle = ribbon_builtin(vec![
            Value::Num(ax),
            Value::Tensor(Tensor::new_2d(vec![10.0, 20.0, 30.0], 1, 3).unwrap()),
            tensor(vec![2.0, 3.0, 4.0], 3, 1),
            Value::Num(0.5),
            Value::String("FaceAlpha".into()),
            Value::Num(0.25),
            Value::String("DisplayName".into()),
            Value::String("band".into()),
        ])
        .expect("ribbon");
        let Value::Num(handle) = handle else {
            panic!("expected scalar handle");
        };
        let figure = clone_figure(current_figure_handle()).expect("figure");
        assert_eq!(figure.plot_axes_indices(), vec![1]);
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.x_grid.as_ref().unwrap()[0], vec![10.0, 20.0, 30.0]);
        assert_eq!(surface.y_grid.as_ref().unwrap()[0], vec![0.75, 0.75, 0.75]);
        assert_eq!(surface.y_grid.as_ref().unwrap()[1], vec![1.25, 1.25, 1.25]);
        assert_eq!(surface.alpha, 0.25);
        assert_eq!(surface.label.as_deref(), Some("band"));
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("XData".into())]).unwrap(),
            tensor(vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0], 3, 2)
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("YData".into())]).unwrap(),
            tensor(vec![0.75, 0.75, 0.75, 1.25, 1.25, 1.25], 3, 2)
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("ZData".into())]).unwrap(),
            tensor(vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0], 3, 2)
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayName".into())
            ])
            .unwrap(),
            Value::String("band".into())
        );
    }

    #[test]
    fn ribbon_reads_typed_integer_x_y_and_width_storage_exactly() {
        let _guard = setup();
        let handle = ribbon_builtin(vec![
            poisoned_int_tensor(IntegerStorage::U16(vec![10, 20, 30]), 1, 3),
            poisoned_int_tensor(IntegerStorage::I16(vec![2, 3, 4]), 3, 1),
            poisoned_int_tensor(IntegerStorage::U8(vec![2]), 1, 1),
        ])
        .expect("ribbon");
        let Value::Num(handle) = handle else {
            panic!("expected scalar handle");
        };

        let figure = clone_figure(current_figure_handle()).expect("figure");
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.x_grid.as_ref().unwrap()[0], vec![10.0, 20.0, 30.0]);
        assert_eq!(surface.y_grid.as_ref().unwrap()[0], vec![0.0, 0.0, 0.0]);
        assert_eq!(surface.y_grid.as_ref().unwrap()[1], vec![2.0, 2.0, 2.0]);
        assert_eq!(surface.z_data.as_ref().unwrap()[0], vec![2.0, 3.0, 4.0]);
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("ZData".into())]).unwrap(),
            tensor(vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0], 3, 2)
        );
    }

    #[test]
    fn ribbon_handles_row_vectors_nan_and_empty_default_display_name() {
        let _guard = setup();
        let handle = ribbon_builtin(vec![Value::Tensor(
            Tensor::new_2d(vec![1.0, f64::NAN, 3.0], 1, 3).unwrap(),
        )])
        .expect("ribbon row vector");
        let Value::Num(handle) = handle else {
            panic!("expected scalar handle");
        };

        let figure = clone_figure(current_figure_handle()).expect("figure");
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert!(surface.z_data.as_ref().unwrap()[0][1].is_nan());
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayName".into())
            ])
            .unwrap(),
            Value::String(String::new())
        );
    }

    #[test]
    fn ribbon_surface_handle_supports_settable_surface_properties() {
        let _guard = setup();
        let handle = ribbon_builtin(vec![tensor(vec![1.0, 2.0, 3.0], 3, 1)]).expect("ribbon");
        let Value::Num(handle) = handle else {
            panic!("expected scalar handle");
        };

        set_builtin(vec![
            Value::Num(handle),
            Value::String("FaceAlpha".into()),
            Value::Num(0.4),
            Value::String("DisplayName".into()),
            Value::String("after".into()),
            Value::String("Visible".into()),
            Value::String("off".into()),
            Value::String("FaceColor".into()),
            Value::String("red".into()),
            Value::String("EdgeColor".into()),
            Value::String("flat".into()),
        ])
        .expect("set surface props");

        let alpha =
            get_builtin(vec![Value::Num(handle), Value::String("FaceAlpha".into())]).unwrap();
        let Value::Num(alpha) = alpha else {
            panic!("expected numeric alpha");
        };
        assert!((alpha - 0.4).abs() < 1e-6);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayName".into())
            ])
            .unwrap(),
            Value::String("after".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Visible".into())]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("FaceColor".into())]).unwrap(),
            Value::String("r".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("EdgeColor".into())]).unwrap(),
            Value::String("flat".into())
        );

        let err = set_builtin(vec![
            Value::Num(handle),
            Value::String("LineWidth".into()),
            Value::Num(2.0),
        ])
        .expect_err("unsupported surface set should fail");
        assert!(err.message.contains("unsupported surface property"));
    }

    #[test]
    fn ribbon_rejects_invalid_width_and_x_length() {
        let err = ribbon_builtin(vec![tensor(vec![1.0, 2.0], 2, 1), Value::Num(0.0)])
            .expect_err("zero width should fail");
        assert_eq!(err.identifier(), Some("RunMat:ribbon:InvalidArgument"));

        let err = ribbon_builtin(vec![
            Value::Tensor(Tensor::new_2d(vec![1.0, 2.0], 1, 2).unwrap()),
            tensor(vec![1.0, 2.0, 3.0], 3, 1),
        ])
        .expect_err("x length mismatch should fail");
        assert_eq!(err.identifier(), Some("RunMat:ribbon:InvalidArgument"));
    }
}
