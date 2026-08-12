use std::sync::Arc;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};

use super::op_common::surface_inputs::{
    axis_sources_to_host, image_axis_sources_from_xy_values, parse_image_call_args,
};
use super::state::{color_limits_snapshot, render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "imagesc";

const IMAGESC_FOUR_CHANNEL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "imagesc-four-channel-cdata",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "imagesc with M-by-N-by-4 CData is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ImagescFourChannelCDataExtension"),
};

pub const IMAGESC_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [IMAGESC_FOUR_CHANNEL_EXTENSION];

const IMAGESC_INTEGER_CDATA: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "CData",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scaled indexed and truecolor CData accept every built-in integer class and remain authoritative on the image object.",
    }];
pub const IMAGESC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = imagesc(integer_CData, ...)",
        inputs: &IMAGESC_INTEGER_CDATA,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact CData and scaled-mapping provenance are retained while colormap normalization and rendering cross an explicit client boundary.",
    }];

const IMAGESC_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered scaled image object.",
}];

const IMAGESC_INPUTS_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed image matrix.",
}];

const IMAGESC_INPUTS_X_Y_C: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates or extent vector.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates or extent vector.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image matrix.",
    },
];

const IMAGESC_INPUTS_C_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image matrix.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value surface style options.",
    },
];

const IMAGESC_INPUTS_X_Y_C_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates or extent vector.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates or extent vector.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image matrix.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value surface style options.",
    },
];

const IMAGESC_INPUTS_C_CLIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image matrix.",
    },
    BuiltinParamDescriptor {
        name: "clims",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element color limits.",
    },
];

const IMAGESC_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = imagesc(C)",
        inputs: &IMAGESC_INPUTS_C,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imagesc(X, Y, C)",
        inputs: &IMAGESC_INPUTS_X_Y_C,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imagesc(C, clims)",
        inputs: &IMAGESC_INPUTS_C_CLIMS,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imagesc(C, Name, Value, ...)",
        inputs: &IMAGESC_INPUTS_C_PROPS,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imagesc(X, Y, C, Name, Value, ...)",
        inputs: &IMAGESC_INPUTS_X_Y_C_PROPS,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
];

pub const IMAGESC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMAGESC.INVALID_ARGUMENT",
    identifier: Some("RunMat:imagesc:InvalidArgument"),
    when: "Image data, axis inputs, or name/value style arguments are invalid.",
    message: "imagesc: invalid argument",
};

pub const IMAGESC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMAGESC.INTERNAL",
    identifier: Some("RunMat:imagesc:Internal"),
    when: "Internal image/surface construction or rendering fails unexpectedly.",
    message: "imagesc: internal operation failed",
};

const IMAGESC_ERRORS: [BuiltinErrorDescriptor; 2] =
    [IMAGESC_ERROR_INVALID_ARGUMENT, IMAGESC_ERROR_INTERNAL];

pub const IMAGESC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMAGESC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMAGESC_ERRORS,
};

fn imagesc_error_with_detail(
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

fn map_imagesc_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    imagesc_error_with_detail(&IMAGESC_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_imagesc_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    imagesc_error_with_detail(&IMAGESC_ERROR_INTERNAL, err.message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::imagesc")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imagesc",
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
    notes: "imagesc is a plotting sink; GPU inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::imagesc")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imagesc",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "imagesc terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "imagesc",
    category = "plotting",
    summary = "Display scaled matrix images.",
    keywords = "imagesc,plotting,image,colormap",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::imagesc::IMAGESC_DESCRIPTOR),
    extensions(crate::builtins::plotting::imagesc::IMAGESC_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::imagesc::IMAGESC_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::imagesc"
)]
pub async fn imagesc_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (x, y, c, mut rest) =
        parse_image_call_args(args, BUILTIN_NAME).map_err(map_imagesc_invalid_argument)?;
    let (c_data_mapping, parsed_rest) =
        super::image::extract_cdata_mapping(rest, "scaled", BUILTIN_NAME)
            .map_err(map_imagesc_invalid_argument)?;
    rest = parsed_rest;
    if super::image::image_channel_count(&c) == Some(4) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMAGESC_FOUR_CHANNEL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let explicit_color_limits = if rest.len() == 1 {
        parse_imagesc_color_limits(&rest[0])?
    } else {
        None
    };
    if explicit_color_limits.is_some() {
        rest.clear();
    }
    let (rows, cols, kind, retained_c_data) = super::image::classify_image_input(&c, BUILTIN_NAME)
        .await
        .map_err(map_imagesc_invalid_argument)?;
    let (x_axis, y_axis) = image_axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME)
        .await
        .map_err(map_imagesc_invalid_argument)?;

    let defaults =
        SurfaceStyleDefaults::new(ColorMap::Parula, ShadingMode::None, false, 1.0, true, false);
    let style = Arc::new(
        parse_surface_style_args(BUILTIN_NAME, &rest, defaults)
            .map_err(map_imagesc_invalid_argument)?,
    );
    let color_limits = explicit_color_limits.or_else(color_limits_snapshot);

    let mut surface = match kind {
        super::image::ImageInputKind::TrueColorHost(tensor) => {
            let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
                .await
                .map_err(map_imagesc_invalid_argument)?;
            super::image::build_truecolor_image_surface(tensor, x_host, y_host)
                .map_err(map_imagesc_invalid_argument)?
        }
        super::image::ImageInputKind::TrueColorGpu(handle, channels) => {
            super::image::build_truecolor_image_surface_gpu(
                &handle, &x_axis, &y_axis, rows, cols, channels,
            )
            .map_err(map_imagesc_invalid_argument)?
        }
        super::image::ImageInputKind::Indexed(c_input) => {
            super::image::build_indexed_image_surface(
                &c_input,
                &x_axis,
                &y_axis,
                style.colormap.clone(),
                color_limits,
                &c_data_mapping,
            )
            .await
            .map_err(map_imagesc_invalid_argument)?
        }
    };

    surface = surface.with_flatten_z(true).with_image_mode(true);
    if c_data_mapping != "direct" && color_limits.is_some() {
        surface = surface.with_color_limits(color_limits);
    }
    surface.colormap = style.colormap.clone();
    let mut surface = Some(surface);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let opts = PlotRenderOptions {
        title: "Image",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        if explicit_color_limits.is_some() {
            figure.set_axes_color_limits(axes, explicit_color_limits);
        }
        let surface = surface.take().expect("imagesc plot consumed once");
        let plot_index = figure.add_surface_plot_on_axes(surface, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_image_handle(
        figure_handle,
        axes,
        plot_index,
        Some(retained_c_data),
        &c_data_mapping,
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_imagesc_internal(err));
    }
    Ok(handle)
}

fn parse_imagesc_color_limits(value: &Value) -> crate::BuiltinResult<Option<(f64, f64)>> {
    if matches!(value, Value::String(_) | Value::CharArray(_)) {
        return Ok(None);
    }
    let tensor = Tensor::try_from(value).map_err(|_| {
        imagesc_error_with_detail(
            &IMAGESC_ERROR_INVALID_ARGUMENT,
            "imagesc: color limits must be a two-element numeric vector",
        )
    })?;
    if tensor.shape.iter().copied().product::<usize>() != 2 {
        return Ok(None);
    }
    let values = tensor.materialize_f64();
    let (lo, hi) = (values[0], values[1]);
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return Err(imagesc_error_with_detail(
            &IMAGESC_ERROR_INVALID_ARGUMENT,
            "imagesc: color limits must be finite and strictly increasing",
        ));
    }
    Ok(Some((lo, hi)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::Tensor;
    use runmat_plot::plots::PlotElement;

    fn grid_tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor::new(data, vec![rows, cols]).expect("imagesc test grid")
    }

    #[test]
    fn imagesc_four_channel_cdata_is_strictly_extension_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let rgba = Value::Tensor(Tensor::new(vec![0.0; 4], vec![1, 1, 4]).expect("RGBA"));
        let error = futures::executor::block_on(imagesc_builtin(vec![rgba]))
            .expect_err("four-channel imagesc should be gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ImagescFourChannelCDataExtension")
        );
    }

    #[test]
    fn imagesc_z_only_shorthand_builds_flattened_surface() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(imagesc_builtin(vec![Value::Tensor(grid_tensor(
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            2,
        ))]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let plot = fig.plots().next().unwrap();
        let PlotElement::Surface(surface) = plot else {
            panic!("expected surface");
        };
        assert!(surface.flatten_z);
        assert!(surface.image_mode);
        assert_eq!(surface.x_data, vec![1.0, 2.0]);
        assert_eq!(surface.y_data, vec![1.0, 2.0]);
    }

    #[test]
    fn imagesc_cdata_property_form_is_not_misparsed_as_explicit_axes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let result = futures::executor::block_on(imagesc_builtin(vec![
            Value::Tensor(grid_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
            Value::String("Alpha".into()),
            Value::Num(0.5),
        ]));
        assert!(
            result.is_ok(),
            "CData property form should parse: {result:?}"
        );
    }

    #[test]
    fn imagesc_constructor_accepts_direct_cdata_mapping() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(imagesc_builtin(vec![
            Value::Tensor(grid_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
            Value::String("CDataMapping".into()),
            Value::String("direct".into()),
        ]))
        .expect("imagesc constructor CDataMapping");
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("CDataMapping".into())
            ])
            .unwrap(),
            Value::String("direct".into())
        );
    }

    #[test]
    fn imagesc_direct_mapping_ignores_existing_clim_for_integer_and_float_cdata() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        crate::builtins::plotting::state::set_color_limits_runtime(Some((20.0, 30.0)));

        futures::executor::block_on(imagesc_builtin(vec![
            Value::Tensor(
                Tensor::new_integer(runmat_builtins::IntegerStorage::U8(vec![0, 1]), vec![1, 2])
                    .unwrap(),
            ),
            Value::String("CDataMapping".into()),
            Value::String("direct".into()),
        ]))
        .expect("integer direct imagesc");
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(integer_surface) = figure.plots().next().unwrap() else {
            panic!("expected integer image surface");
        };
        assert_eq!(integer_surface.color_limits, Some((0.0, 255.0)));

        let _ = clear_figure(None);
        crate::builtins::plotting::state::set_color_limits_runtime(Some((20.0, 30.0)));
        futures::executor::block_on(imagesc_builtin(vec![
            Value::Tensor(grid_tensor(vec![1.0, 2.0], 1, 2)),
            Value::String("CDataMapping".into()),
            Value::String("direct".into()),
        ]))
        .expect("floating direct imagesc");
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(float_surface) = figure.plots().next().unwrap() else {
            panic!("expected floating image surface");
        };
        assert_eq!(float_surface.color_limits, Some((1.0, 256.0)));
    }

    #[test]
    fn imagesc_two_argument_color_limits_update_axes_and_surface() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        futures::executor::block_on(imagesc_builtin(vec![
            Value::Tensor(grid_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
            Value::Tensor(Tensor::new(vec![0.0, 5.0], vec![1, 2]).unwrap()),
        ]))
        .expect("imagesc(C,clims)");
        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            figure.axes_metadata(0).unwrap().color_limits,
            Some((0.0, 5.0))
        );
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected surface")
        };
        assert_eq!(surface.color_limits, Some((0.0, 5.0)));
    }

    #[test]
    fn imagesc_integer_truecolor_uses_truecolor_grid_and_retains_cdata() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let source = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U8(vec![255, 0, 0, 255, 0, 0]),
            vec![1, 2, 3],
        )
        .unwrap();
        let handle =
            futures::executor::block_on(imagesc_builtin(vec![Value::Tensor(source.clone())]))
                .expect("integer truecolor imagesc");
        let resolved =
            crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), "get")
                .unwrap();
        let retained =
            crate::builtins::plotting::properties::get_properties(resolved, Some("CData"), "get")
                .unwrap();
        assert_eq!(Tensor::try_from(&retained).unwrap(), source);
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected surface")
        };
        assert!(surface.color_grid.is_some());
    }

    #[test]
    fn imagesc_applies_explicit_axes_and_color_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        crate::builtins::plotting::state::set_color_limits_runtime(Some((0.0, 10.0)));

        let _ = futures::executor::block_on(imagesc_builtin(vec![
            Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2]).expect("imagesc x extent")),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).expect("imagesc y extent")),
            Value::Tensor(grid_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert_eq!(surface.x_data, vec![10.0, 20.0]);
        assert_eq!(surface.y_data, vec![1.0, 2.0]);
        assert_eq!(surface.color_limits, Some((0.0, 10.0)));
    }

    #[test]
    fn imagesc_accepts_two_element_extent_vectors() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = futures::executor::block_on(imagesc_builtin(vec![
            Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2]).expect("imagesc x extent")),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![2]).expect("imagesc y extent")),
            Value::Tensor(grid_tensor((1..=12).map(|v| v as f64).collect(), 3, 4)),
        ]))
        .expect("imagesc with extent vectors should succeed");
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface")
        };
        assert_eq!(
            surface.x_data,
            vec![10.0, 13.333333333333332, 16.666666666666664, 20.0]
        );
        assert_eq!(surface.y_data, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn imagesc_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = IMAGESC_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = imagesc(C)"));
        assert!(labels.contains(&"h = imagesc(X, Y, C)"));
        assert!(labels.contains(&"h = imagesc(X, Y, C, Name, Value, ...)"));
    }

    #[test]
    fn imagesc_missing_input_uses_stable_identifier() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let err = futures::executor::block_on(imagesc_builtin(vec![]))
            .expect_err("missing args should fail");
        assert_eq!(err.identifier(), IMAGESC_ERROR_INVALID_ARGUMENT.identifier);
    }
}
