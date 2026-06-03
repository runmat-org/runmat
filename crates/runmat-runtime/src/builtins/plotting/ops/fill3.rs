//! MATLAB-compatible `fill3` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "fill3";

const FILL3_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle or row-vector of handles for created patch objects.",
}];

const FILL3_INPUTS_X_Y_Z_C: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Polygon X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Polygon Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Polygon Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Color specification.",
    },
];

const FILL3_INPUTS_MULTI_GROUP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "groups",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One or more `(X, Y, Z, C)` polygon groups.",
}];

const FILL3_INPUTS_MULTI_GROUP_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "groups",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "One or more `(X, Y, Z, C)` polygon groups.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Trailing patch name/value property pairs.",
    },
];

const FILL3_INPUTS_AX_X_Y_Z_C: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Polygon X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Polygon Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Polygon Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Color specification.",
    },
];

const FILL3_INPUTS_AX_MULTI_GROUP: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "groups",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "One or more `(X, Y, Z, C)` polygon groups.",
    },
];

const FILL3_INPUTS_AX_MULTI_GROUP_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "groups",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "One or more `(X, Y, Z, C)` polygon groups.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Trailing patch name/value property pairs.",
    },
];

const FILL3_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = fill3(X, Y, Z, C)",
        inputs: &FILL3_INPUTS_X_Y_Z_C,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(X1, Y1, Z1, C1, ..., Xn, Yn, Zn, Cn)",
        inputs: &FILL3_INPUTS_MULTI_GROUP,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(X1, Y1, Z1, C1, ..., Name, Value, ...)",
        inputs: &FILL3_INPUTS_MULTI_GROUP_PROPS,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(ax, X, Y, Z, C)",
        inputs: &FILL3_INPUTS_AX_X_Y_Z_C,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(ax, X1, Y1, Z1, C1, ..., Xn, Yn, Zn, Cn)",
        inputs: &FILL3_INPUTS_AX_MULTI_GROUP,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(ax, X1, Y1, Z1, C1, ..., Name, Value, ...)",
        inputs: &FILL3_INPUTS_AX_MULTI_GROUP_PROPS,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
];

pub const FILL3_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILL3.INVALID_ARGUMENT",
    identifier: Some("RunMat:fill3:InvalidArgument"),
    when: "Input groups, axes targeting, or trailing property pairs are invalid.",
    message: "fill3: invalid argument",
};

pub const FILL3_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILL3.INTERNAL",
    identifier: Some("RunMat:fill3:Internal"),
    when: "Internal patch construction or rendering fails unexpectedly.",
    message: "fill3: internal operation failed",
};

const FILL3_ERRORS: [BuiltinErrorDescriptor; 2] =
    [FILL3_ERROR_INVALID_ARGUMENT, FILL3_ERROR_INTERNAL];

pub const FILL3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILL3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILL3_ERRORS,
};

fn fill3_error_with_detail(
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

fn map_fill3_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fill3_error_with_detail(&FILL3_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_fill3_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fill3_error_with_detail(&FILL3_ERROR_INTERNAL, err.message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::fill3")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fill3",
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
    notes: "fill3 is a plotting sink. It lowers MATLAB fill3 argument groups to patch plots and renders through the shared patch machinery.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::fill3")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fill3",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fill3 performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "fill3",
    category = "plotting",
    summary = "Create 3-D filled polygon patches.",
    keywords = "fill3,patch,plotting,polygon,3d",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::fill3::FILL3_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::fill3"
)]
pub fn fill3_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (axes_target, args) =
        split_leading_axes_handle(args, BUILTIN_NAME).map_err(map_fill3_invalid_argument)?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_fill3_invalid_argument)?;

    let patch_arg_groups =
        parse_fill3_patch_arg_groups(args).map_err(map_fill3_invalid_argument)?;
    let mut plots = patch_arg_groups
        .into_iter()
        .map(|args| {
            let mut plot =
                super::patch::parse_patch_plot(args).map_err(map_fill3_invalid_argument)?;
            plot.set_force_3d(true);
            Ok(plot)
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;

    let mut plots_opt = Some(std::mem::take(&mut plots));
    let plot_indices_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "3-D Filled Polygons",
            x_label: "X",
            y_label: "Y",
            axis_equal: true,
            ..Default::default()
        },
        move |figure, axes| {
            let plots = plots_opt.take().expect("fill3 plots consumed once");
            let mut plot_indices = Vec::with_capacity(plots.len());
            for plot in plots {
                let plot_index = figure.add_patch_plot_on_axes(plot, axes);
                plot_indices.push((axes, plot_index));
            }
            figure.set_axes_view(axes, -37.5, 30.0);
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
            return Err(map_fill3_internal(err));
        }
    }
    let handles = plot_indices
        .into_iter()
        .map(|(axes, plot_index)| {
            crate::builtins::plotting::state::register_patch_handle(figure_handle, axes, plot_index)
        })
        .collect::<Vec<_>>();
    Ok(handles_value(handles))
}

fn parse_fill3_patch_arg_groups(args: Vec<Value>) -> crate::BuiltinResult<Vec<Vec<Value>>> {
    if args.len() < 4 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "fill3: expected X, Y, Z, and C data",
        ));
    }

    let property_start = find_trailing_property_start(&args);
    let (positional, properties) = args.split_at(property_start);
    if positional.len() < 4 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "fill3: expected X, Y, Z, and C data",
        ));
    }

    let mut groups = Vec::new();
    let mut offset = 0;
    while offset < positional.len() {
        let remaining = positional.len() - offset;
        if remaining < 4 {
            return Err(plotting_error(
                BUILTIN_NAME,
                "fill3: each polygon group must include X, Y, Z, and C data",
            ));
        }

        let mut group = vec![
            Value::String("XData".into()),
            positional[offset].clone(),
            Value::String("YData".into()),
            positional[offset + 1].clone(),
            Value::String("ZData".into()),
            positional[offset + 2].clone(),
            Value::String("FaceColor".into()),
            positional[offset + 3].clone(),
        ];
        offset += 4;

        group.extend_from_slice(properties);
        groups.push(group);
    }

    Ok(groups)
}

fn find_trailing_property_start(args: &[Value]) -> usize {
    for idx in (4..=args.len()).step_by(4) {
        if args[idx..].iter().step_by(2).all(is_string_like) {
            return idx;
        }
    }
    args.len()
}

fn is_string_like(value: &Value) -> bool {
    matches!(value, Value::CharArray(_) | Value::String(_))
}

fn handles_value(handles: Vec<f64>) -> Value {
    if handles.len() == 1 {
        Value::Num(handles[0])
    } else {
        let len = handles.len();
        Value::Tensor(Tensor::new_2d(handles, 1, len).expect("valid handle vector"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use glam::Vec4;
    use runmat_builtins::NumericDType;
    use runmat_plot::plots::figure::PlotElement;
    use runmat_plot::plots::{PatchEdgeColorMode, PatchFaceColorMode};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        super::super::state::reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

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
    fn fill3_positional_color_string_builds_patch() {
        let _guard = setup();
        let handle = fill3_builtin(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            tensor(3, 1, &[0.0, 0.5, 0.0]),
            Value::String("r".into()),
        ])
        .unwrap();

        assert!(matches!(handle, Value::Num(_)));
        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.len(), 1);
        let meta = figure.axes_metadata(0).unwrap();
        assert_eq!(meta.view_azimuth_deg, Some(-37.5));
        assert_eq!(meta.view_elevation_deg, Some(30.0));
        let Some(PlotElement::Patch(patch)) = figure.plots().next() else {
            panic!("expected patch plot");
        };
        assert!(patch.force_3d());
        assert_eq!(patch.faces().len(), 1);
        assert_eq!(patch.vertices()[1].z, 0.5);
        assert_eq!(patch.face_color(), Vec4::new(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn fill3_multiple_polygon_groups_return_handle_vector() {
        let _guard = setup();
        let handles = fill3_builtin(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            tensor(3, 1, &[0.0, 0.0, 0.0]),
            Value::String("g".into()),
            tensor(4, 1, &[2.0, 3.0, 3.0, 2.0]),
            tensor(4, 1, &[0.0, 0.0, 1.0, 1.0]),
            tensor(4, 1, &[1.0, 1.0, 1.0, 1.0]),
            tensor(1, 3, &[0.0, 0.0, 1.0]),
        ])
        .unwrap();

        let Value::Tensor(handles) = handles else {
            panic!("expected vector of handles");
        };
        assert_eq!(handles.data.len(), 2);
        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.len(), 2);
        let patches = figure.plots().collect::<Vec<_>>();
        let PlotElement::Patch(first) = patches[0] else {
            panic!("expected first patch plot");
        };
        assert_eq!(first.vertices()[0].z, 0.0);
        assert_eq!(first.face_color(), Vec4::new(0.0, 1.0, 0.0, 1.0));
        let PlotElement::Patch(second) = patches[1] else {
            panic!("expected second patch plot");
        };
        assert_eq!(second.vertices()[0].z, 1.0);
        assert_eq!(second.face_color(), Vec4::new(0.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn fill3_trailing_patch_properties_apply_to_each_group() {
        let _guard = setup();
        let handle = fill3_builtin(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            tensor(3, 1, &[0.25, 0.5, 0.75]),
            tensor(1, 3, &[0.25, 0.5, 0.75]),
            Value::String("EdgeColor".into()),
            Value::String("none".into()),
            Value::String("FaceAlpha".into()),
            Value::Num(0.5),
        ])
        .unwrap();

        let Value::Num(handle) = handle else {
            panic!("expected scalar handle");
        };
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap(),
            Value::String("patch".into())
        );
        let figure = clone_figure(current_figure_handle()).unwrap();
        let Some(PlotElement::Patch(patch)) = figure.plots().next() else {
            panic!("expected patch plot");
        };
        assert_eq!(patch.face_color(), Vec4::new(0.25, 0.5, 0.75, 1.0));
        assert_eq!(patch.edge_color_mode(), PatchEdgeColorMode::None);
        assert_eq!(patch.face_color_mode(), PatchFaceColorMode::Color);
        assert_eq!(patch.face_alpha(), 0.5);
    }

    #[test]
    fn fill3_unknown_trailing_string_keys_are_property_pairs() {
        let args = vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            tensor(3, 1, &[0.25, 0.5, 0.75]),
            tensor(1, 3, &[0.25, 0.5, 0.75]),
            Value::String("FuturePatchProperty".into()),
            Value::Num(1.0),
            Value::CharArray(runmat_builtins::CharArray::new_row("AnotherFutureProperty")),
            Value::String("value".into()),
        ];

        assert_eq!(find_trailing_property_start(&args), 4);
    }

    #[test]
    fn fill3_string_color_before_dangling_property_reports_pair_error() {
        let err = fill3_builtin(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            tensor(3, 1, &[0.25, 0.5, 0.75]),
            Value::String("r".into()),
            Value::String("EdgeColor".into()),
        ])
        .unwrap_err()
        .to_string();

        assert!(err.contains("property/value arguments must come in pairs"));
    }

    #[test]
    fn fill3_descriptor_signatures_cover_supported_forms() {
        let labels: Vec<&str> = FILL3_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = fill3(X, Y, Z, C)"));
        assert!(labels.contains(&"h = fill3(X1, Y1, Z1, C1, ..., Xn, Yn, Zn, Cn)"));
        assert!(labels.contains(&"h = fill3(ax, X, Y, Z, C)"));
        assert!(labels.contains(&"h = fill3(ax, X1, Y1, Z1, C1, ..., Name, Value, ...)"));
    }

    #[test]
    fn fill3_missing_args_uses_stable_identifier() {
        let _guard = setup();
        let err = fill3_builtin(vec![]).expect_err("missing args should fail");
        assert_eq!(err.identifier(), FILL3_ERROR_INVALID_ARGUMENT.identifier);
    }
}
