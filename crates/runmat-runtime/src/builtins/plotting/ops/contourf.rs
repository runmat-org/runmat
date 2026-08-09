//! MATLAB-compatible `contourf` builtin (filled contour plot).

use log::warn;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::ColorMap;

use super::common::tensor_to_surface_grid_matlab_xy;
use super::contour::{
    build_contour_fill_gpu_plot, build_contour_fill_plot, build_contour_gpu_plot,
    build_contour_plot, parse_contour_args, ContourArgs, ContourLineColor,
    CONTOUR_INTEGER_LINE_COLOR_EXTENSION,
};
use super::op_common::axes_target::{apply_axes_target, split_leading_axes_handle};
use super::state::{render_active_plot, PlotRenderOptions};
use crate::build_runtime_error;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const BUILTIN_NAME: &str = "contourf";

const CONTOURF_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to filled contour plot.",
}];

const CONTOURF_INPUTS_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Contour height grid.",
}];

const CONTOURF_INPUTS_Z_LEVEL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour height grid.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour level count/value vector.",
    },
];

const CONTOURF_INPUTS_Z_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour height grid.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value contour options.",
    },
];

const CONTOURF_INPUTS_Z_LEVEL_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour height grid.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour level count/value vector.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value contour options.",
    },
];

const CONTOURF_INPUTS_X_Y_Z: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate vector matching Z columns or coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate vector matching Z rows or coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour height grid.",
    },
];

const CONTOURF_INPUTS_X_Y_Z_LEVEL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate vector matching Z columns or coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate vector matching Z rows or coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour height grid.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour level count/value vector.",
    },
];

const CONTOURF_INPUTS_X_Y_Z_LEVEL_PROPS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate vector matching Z columns or coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate vector matching Z rows or coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour height grid.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Contour level count/value vector.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value contour options.",
    },
];

const CONTOURF_INPUTS_TARGET_ARGS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes object.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Any supported filled-contour data, level, line-specification, and property arguments.",
    },
];

const CONTOURF_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = contourf(Z)",
        inputs: &CONTOURF_INPUTS_Z,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(Z, V)",
        inputs: &CONTOURF_INPUTS_Z_LEVEL,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(Z, Name, Value, ...)",
        inputs: &CONTOURF_INPUTS_Z_PROPS,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(Z, V, Name, Value, ...)",
        inputs: &CONTOURF_INPUTS_Z_LEVEL_PROPS,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(X, Y, Z)",
        inputs: &CONTOURF_INPUTS_X_Y_Z,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(X, Y, Z, V)",
        inputs: &CONTOURF_INPUTS_X_Y_Z_LEVEL,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(X, Y, Z, V, Name, Value, ...)",
        inputs: &CONTOURF_INPUTS_X_Y_Z_LEVEL_PROPS,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(ax, ...)",
        inputs: &CONTOURF_INPUTS_TARGET_ARGS,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
];

pub const CONTOURF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTOURF.INVALID_ARGUMENT",
    identifier: Some("RunMat:contourf:InvalidArgument"),
    when: "Contour input arrays, level arguments, or name/value options are invalid.",
    message: "contourf: invalid argument",
};

pub const CONTOURF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTOURF.INTERNAL",
    identifier: Some("RunMat:contourf:Internal"),
    when: "Internal filled-contour render preparation fails unexpectedly.",
    message: "contourf: internal operation failed",
};

const CONTOURF_ERRORS: [BuiltinErrorDescriptor; 2] =
    [CONTOURF_ERROR_INVALID_ARGUMENT, CONTOURF_ERROR_INTERNAL];

pub const CONTOURF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTOURF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTOURF_ERRORS,
};

pub const CONTOURF_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [CONTOUR_INTEGER_LINE_COLOR_EXTENSION];

const fn documented_integer_input(
    name: &'static str,
    notes: &'static str,
) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes,
    }
}

const INTEGER_Z: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "Z",
    "The documented height matrix accepts every built-in integer class and must contain at least two exact distinct values.",
)];
const INTEGER_Z_N: [BuiltinIntegerInputCapability; 2] = [
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
    documented_integer_input("N", "A scalar integer selects a positive bounded contour-level count through exact structural validation."),
];
const INTEGER_Z_V: [BuiltinIntegerInputCapability; 2] = [
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
    documented_integer_input("V", "The documented integer level vector is checked for exact monotonic order before graphics conversion."),
];
const INTEGER_X_Y_Z: [BuiltinIntegerInputCapability; 3] = [
    documented_integer_input("X", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly for shape, orientation, and strict monotonicity."),
    documented_integer_input("Y", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly for shape, orientation, and strict monotonicity."),
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
];
const INTEGER_X_Y_Z_N: [BuiltinIntegerInputCapability; 4] = [
    documented_integer_input("X", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly before graphics conversion."),
    documented_integer_input("Y", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly before graphics conversion."),
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
    documented_integer_input("N", "A scalar integer selects a positive bounded contour-level count through exact structural validation."),
];
const INTEGER_X_Y_Z_V: [BuiltinIntegerInputCapability; 4] = [
    documented_integer_input("X", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly before graphics conversion."),
    documented_integer_input("Y", "The documented coordinate vector or matrix accepts every built-in integer class and is validated exactly before graphics conversion."),
    documented_integer_input("Z", "The documented height matrix accepts every built-in integer class and is validated exactly before filled-contour geometry."),
    documented_integer_input("V", "The documented integer level vector is checked for exact monotonic order before graphics conversion."),
];
const INTEGER_LEVEL_LIST: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "LevelList",
    "The documented property accepts every built-in integer class as an exact monotonically increasing level vector.",
)];
const INTEGER_LEVEL_STEP: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "LevelStep",
    "The documented property accepts every built-in integer class as a positive scalar spacing.",
)];
const INTEGER_LINE_WIDTH: [BuiltinIntegerInputCapability; 1] = [documented_integer_input(
    "LineWidth",
    "A positive scalar integer line width crosses the explicit graphics-property boundary after scalar validation.",
)];
const INTEGER_LINE_COLOR: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "LineColor",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "RunMat mode admits a typed-integer RGB triplet only when every exact component is 0 or 1.",
}];

pub const CONTOURF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 10] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_Z)", inputs: &INTEGER_Z, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Authoritative integer heights are validated exactly and then cross the client filled-contour boundary; resident integer grids gather through their owning provider instead of entering the floating WGPU path." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_Z, integer_N)", inputs: &INTEGER_Z_N, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Z crosses the graphics boundary after exact validation while N remains exact through positive allocation-bound validation." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_Z, integer_V)", inputs: &INTEGER_Z_V, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer level order and height distinctness are established before client graphics conversion." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_X, integer_Y, integer_Z)", inputs: &INTEGER_X_Y_Z, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Coordinate shape, matrix orientation, monotonicity, and height distinctness are checked from authoritative integer storage before filled-contour conversion." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_X, integer_Y, integer_Z, integer_N)", inputs: &INTEGER_X_Y_Z_N, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Coordinates and heights cross the graphics boundary after exact validation; N is a bounded structural count." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(integer_X, integer_Y, integer_Z, integer_V)", inputs: &INTEGER_X_Y_Z_V, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Coordinate and level order are exact before the explicit client graphics conversion; resident integer inputs gather." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(..., 'LevelList', integer_levels)", inputs: &INTEGER_LEVEL_LIST, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "LevelList is ordered from authoritative integer storage and then crosses the floating contour-level boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(..., 'LevelStep', integer_step)", inputs: &INTEGER_LEVEL_STEP, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "LevelStep is validated as one positive scalar and then crosses the f32 graphics-property boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(..., 'LineWidth', integer_width)", inputs: &INTEGER_LINE_WIDTH, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "LineWidth is validated as one positive scalar and then crosses the f32 graphics-property boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = contourf(..., 'LineColor', integer_rgb)", inputs: &INTEGER_LINE_COLOR, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The shared RunMat-only integer RGB gate precedes owning-provider gather; exact 0/1 validation follows gather and precedes f32 conversion." },
];

fn contourf_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_contourf_invalid_argument(err: crate::RuntimeError) -> crate::RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    contourf_error_with_detail(&CONTOURF_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_contourf_internal(err: crate::RuntimeError) -> crate::RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    contourf_error_with_detail(&CONTOURF_ERROR_INTERNAL, err.message)
}

#[runtime_builtin(
    name = "contourf",
    category = "plotting",
    summary = "Create filled contour plots.",
    keywords = "contourf,plotting,filled,contour",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::contourf::CONTOURF_DESCRIPTOR),
    extensions(crate::builtins::plotting::contourf::CONTOURF_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::contourf::CONTOURF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::contourf"
)]
pub fn contourf_builtin(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<f64> {
    let mut all = Vec::with_capacity(rest.len() + 1);
    all.push(first);
    all.extend(rest);
    let (axes_target, mut all) = if all.first().is_some_and(is_typed_integer_value) {
        (None, all)
    } else {
        split_leading_axes_handle(all, BUILTIN_NAME).map_err(map_contourf_invalid_argument)?
    };
    let first = all.first().cloned().ok_or_else(|| {
        contourf_error_with_detail(&CONTOURF_ERROR_INVALID_ARGUMENT, "expected Z input")
    })?;
    let rest = all.drain(1..).collect();
    let mut args =
        Some(parse_contour_args(BUILTIN_NAME, first, rest).map_err(map_contourf_invalid_argument)?);
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_contourf_invalid_argument)?;
    let opts = PlotRenderOptions {
        title: "Filled Contour Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let before = figure.plots().count();
        let ContourArgs {
            name,
            x_axis,
            y_axis,
            z_input,
            level_spec,
            line_color,
            line_width,
        } = args.take().expect("contourf args consumed once");
        let color_map = ColorMap::Parula;
        let base_z = 0.0;

        if let Some(handle) = z_input.gpu_handle() {
            match build_contour_fill_gpu_plot(
                name,
                &x_axis,
                &y_axis,
                handle,
                color_map.clone(),
                base_z,
                &level_spec,
            ) {
                Ok(fill_plot) => {
                    figure.add_contour_fill_plot_on_axes(fill_plot, axes);
                    *plot_index_slot.borrow_mut() = Some((axes, before));
                    if !matches!(line_color, ContourLineColor::None) {
                        match build_contour_gpu_plot(
                            name,
                            &x_axis,
                            &y_axis,
                            handle,
                            color_map.clone(),
                            base_z,
                            &level_spec,
                            &line_color,
                        ) {
                            Ok(contours) => {
                                figure.add_contour_plot_on_axes(
                                    contours.with_line_width(line_width),
                                    axes,
                                );
                            }
                            Err(err) => {
                                warn!("contourf contour overlay unavailable: {err}");
                            }
                        }
                    }
                    return Ok(());
                }
                Err(err) => {
                    warn!("contourf GPU path unavailable: {err}");
                }
            }
        }

        let grid = tensor_to_surface_grid_matlab_xy(
            z_input
                .into_tensor(name)
                .map_err(map_contourf_invalid_argument)?,
            y_axis.len(),
            x_axis.len(),
            name,
        )
        .map_err(map_contourf_invalid_argument)?;
        let fill_plot = build_contour_fill_plot(
            name,
            &x_axis,
            &y_axis,
            &grid,
            color_map.clone(),
            base_z,
            &level_spec,
        )
        .map_err(map_contourf_invalid_argument)?;
        figure.add_contour_fill_plot_on_axes(fill_plot, axes);
        *plot_index_slot.borrow_mut() = Some((axes, before));
        if !matches!(line_color, ContourLineColor::None) {
            match build_contour_plot(
                name,
                &x_axis,
                &y_axis,
                &grid,
                color_map,
                base_z,
                &level_spec,
                &line_color,
            ) {
                Ok(contours) => {
                    figure.add_contour_plot_on_axes(contours.with_line_width(line_width), axes);
                }
                Err(err) => {
                    warn!("contourf overlay contour unavailable: {err}");
                }
            }
        }
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_contour_fill_handle(
        figure_handle,
        axes,
        plot_index,
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_contourf_internal(err));
    }
    Ok(handle)
}

fn is_typed_integer_value(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::lock_plot_test_context;
    use crate::builtins::plotting::state::{clone_figure, current_figure_handle, reset_plot_state};
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use runmat_builtins::{IntegerStorage, Tensor};
    use runmat_builtins::{ResolveContext, Type};
    use runmat_plot::plots::figure::PlotElement;

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("contourf test vector")
    }

    fn matrix_from(data: &[f64], rows: usize, cols: usize) -> Tensor {
        Tensor::new(data.to_vec(), vec![rows, cols]).expect("contourf test matrix")
    }

    fn all_integer_z_storages() -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![0, 1, 2, 3]),
            IntegerStorage::I16(vec![0, 1, 2, 3]),
            IntegerStorage::I32(vec![0, 1, 2, 3]),
            IntegerStorage::I64(vec![0, 1, 2, 3]),
            IntegerStorage::U8(vec![0, 1, 2, 3]),
            IntegerStorage::U16(vec![0, 1, 2, 3]),
            IntegerStorage::U32(vec![0, 1, 2, 3]),
            IntegerStorage::U64(vec![0, 1, 2, 3]),
        ]
    }

    fn all_integer_level_storages() -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![0, 1, 2]),
            IntegerStorage::I16(vec![0, 1, 2]),
            IntegerStorage::I32(vec![0, 1, 2]),
            IntegerStorage::I64(vec![0, 1, 2]),
            IntegerStorage::U8(vec![0, 1, 2]),
            IntegerStorage::U16(vec![0, 1, 2]),
            IntegerStorage::U32(vec![0, 1, 2]),
            IntegerStorage::U64(vec![0, 1, 2]),
        ]
    }

    fn all_integer_xyz_storages() -> [(IntegerStorage, IntegerStorage, IntegerStorage); 8] {
        [
            (
                IntegerStorage::I8(vec![10, 20, 30]),
                IntegerStorage::I8(vec![1, 2]),
                IntegerStorage::I8(vec![0, 1, 1, 2, 2, 3]),
            ),
            (
                IntegerStorage::I16(vec![10, 20, 30]),
                IntegerStorage::I16(vec![1, 2]),
                IntegerStorage::I16(vec![0, 1, 1, 2, 2, 3]),
            ),
            (
                IntegerStorage::I32(vec![10, 20, 30]),
                IntegerStorage::I32(vec![1, 2]),
                IntegerStorage::I32(vec![0, 1, 1, 2, 2, 3]),
            ),
            (
                IntegerStorage::I64(vec![10, 20, 30]),
                IntegerStorage::I64(vec![1, 2]),
                IntegerStorage::I64(vec![0, 1, 1, 2, 2, 3]),
            ),
            (
                IntegerStorage::U8(vec![10, 20, 30]),
                IntegerStorage::U8(vec![1, 2]),
                IntegerStorage::U8(vec![0, 1, 1, 2, 2, 3]),
            ),
            (
                IntegerStorage::U16(vec![10, 20, 30]),
                IntegerStorage::U16(vec![1, 2]),
                IntegerStorage::U16(vec![0, 1, 1, 2, 2, 3]),
            ),
            (
                IntegerStorage::U32(vec![10, 20, 30]),
                IntegerStorage::U32(vec![1, 2]),
                IntegerStorage::U32(vec![0, 1, 1, 2, 2, 3]),
            ),
            (
                IntegerStorage::U64(vec![10, 20, 30]),
                IntegerStorage::U64(vec![1, 2]),
                IntegerStorage::U64(vec![0, 1, 1, 2, 2, 3]),
            ),
        ]
    }

    fn assert_flat_finite_triangles(vertices: &[runmat_plot::core::Vertex]) {
        assert!(!vertices.is_empty());
        assert_eq!(vertices.len() % 3, 0);
        for tri in vertices.chunks_exact(3) {
            for vertex in tri {
                assert!(vertex.position[0].is_finite());
                assert!(vertex.position[1].is_finite());
                assert!(vertex.position[2].is_finite());
            }
            assert_eq!(tri[0].color, tri[1].color);
            assert_eq!(tri[1].color, tri[2].color);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contourf_requires_matching_grid() {
        setup_plot_tests();
        let res = contourf_builtin(
            Value::Tensor(tensor_from(&[0.0])),
            vec![
                Value::Tensor(tensor_from(&[0.0, 1.0])),
                Value::Tensor(tensor_from(&[0.0, 1.0])),
            ],
        );
        assert!(res.is_err());
    }

    #[test]
    fn contourf_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }

    #[test]
    fn contourf_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CONTOURF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = contourf(Z)"));
        assert!(labels.contains(&"h = contourf(Z, V)"));
        assert!(labels.contains(&"h = contourf(X, Y, Z)"));
        assert!(labels.contains(&"h = contourf(X, Y, Z, V, Name, Value, ...)"));
        assert!(labels.contains(&"h = contourf(ax, ...)"));
    }

    #[test]
    fn contourf_integer_capabilities_cover_all_documented_roles_and_classes() {
        assert_eq!(CONTOURF_INTEGER_CAPABILITIES.len(), 10);
        assert!(CONTOURF_INTEGER_CAPABILITIES
            .iter()
            .flat_map(|capability| capability.inputs)
            .all(|input| input.classes.len() == 8));
        assert_eq!(CONTOURF_EXTENSIONS[0].id, "contour-integer-line-color");
    }

    #[test]
    fn contourf_accepts_all_integer_height_and_level_classes_exactly() {
        for storage in all_integer_z_storages() {
            let expected = storage.clone();
            let args = parse_contour_args(
                BUILTIN_NAME,
                Value::Tensor(Tensor::new_integer(storage, vec![2, 2]).expect("integer Z")),
                Vec::new(),
            )
            .expect("integer contourf Z");
            let super::super::common::SurfaceDataInput::Host(z) = args.z_input else {
                panic!("host integer Z must stay authoritative on the client");
            };
            assert_eq!(z.integer_storage(), Some(&expected));
        }
        for storage in all_integer_level_storages() {
            let args = parse_contour_args(
                BUILTIN_NAME,
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![0, 1, 2, 3]), vec![2, 2])
                        .expect("Z"),
                ),
                vec![Value::Tensor(
                    Tensor::new_integer(storage, vec![1, 3]).expect("levels"),
                )],
            )
            .expect("integer contourf levels");
            assert!(
                matches!(args.level_spec, super::super::contour::ContourLevelSpec::Values(values) if values == vec![0.0, 1.0, 2.0])
            );
        }
    }

    #[test]
    fn contourf_integrates_all_integer_xyz_classes_with_vector_geometry() {
        let _guard = lock_plot_test_context();
        setup_plot_tests();
        reset_plot_state();
        for (x, y, z) in all_integer_xyz_storages() {
            let x = Value::Tensor(Tensor::new_integer(x, vec![1, 3]).expect("integer X"));
            let y = Value::Tensor(Tensor::new_integer(y, vec![2, 1]).expect("integer Y"));
            let z = Value::Tensor(Tensor::new_integer(z, vec![2, 3]).expect("integer Z"));
            let handle = contourf_builtin(x, vec![y, z]).expect("integer X/Y/Z contourf");
            assert!(handle.is_finite());
        }
    }

    #[test]
    fn contourf_integrates_transposed_integer_coordinate_matrices() {
        let _guard = lock_plot_test_context();
        setup_plot_tests();
        reset_plot_state();
        let x = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 1, 2, 1, 2]), vec![2, 3])
                .expect("transposed X"),
        );
        let y = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![10, 10, 20, 20, 30, 30]),
                vec![2, 3],
            )
            .expect("transposed Y"),
        );
        let z = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![0, 1, 2, 3, 4, 5]), vec![2, 3])
                .expect("transposed Z"),
        );
        let handle = contourf_builtin(x, vec![y, z]).expect("transposed integer contourf");
        assert!(handle.is_finite());
    }

    #[test]
    fn contourf_integrates_resident_integer_z_through_owning_provider() {
        let _guard = lock_plot_test_context();
        setup_plot_tests();
        reset_plot_state();
        test_support::with_test_provider(|provider| {
            let z = Tensor::new_integer(IntegerStorage::I64(vec![0, 1, 2, 3]), vec![2, 2])
                .expect("resident integer Z");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &z)
                .expect("integer upload");
            assert!(contourf_builtin(Value::GpuTensor(handle), Vec::new()).is_ok());
        });
    }

    #[test]
    fn contourf_integrates_integer_rgb_gate_and_line_width_output() {
        let _guard = lock_plot_test_context();
        setup_plot_tests();
        reset_plot_state();
        let z = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![0, 1, 2, 3]), vec![2, 2])
                .expect("integer Z"),
        );
        let rgb = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![1, 0, 1]), vec![1, 3])
                .expect("integer RGB"),
        );

        let compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = contourf_builtin(z.clone(), vec![Value::from("LineColor"), rgb.clone()])
            .expect_err("integer RGB extension must be gated");
        assert_eq!(
            error.identifier(),
            super::super::contour::CONTOUR_INTEGER_LINE_COLOR_EXTENSION.error_identifier
        );
        drop(compatibility);

        let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = contourf_builtin(
            z,
            vec![
                Value::from("LineColor"),
                rgb,
                Value::from("LineWidth"),
                Value::Int(runmat_builtins::IntValue::U16(3)),
            ],
        )
        .expect("integer RGB and LineWidth contourf");
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::from("Type")]).unwrap(),
            Value::from("contour")
        );
        let figure = clone_figure(current_figure_handle()).expect("current figure");
        let overlay = figure
            .plots()
            .filter_map(|plot| match plot {
                PlotElement::Contour(contour) => Some(contour),
                _ => None,
            })
            .last()
            .expect("contourf line overlay");
        assert_eq!(overlay.line_width, 3.0);
    }

    #[test]
    fn contourf_accepts_target_axes_and_does_not_alias_integer_z_as_a_handle() {
        setup_plot_tests();
        let ax = crate::builtins::plotting::axes::axes_builtin(Vec::new()).expect("axes");
        let z = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I8(vec![0, 1, 2, 3]), vec![2, 2])
                .expect("integer Z"),
        );
        assert!(contourf_builtin(Value::Num(ax), vec![z.clone()]).is_ok());
        assert!(contourf_builtin(z, Vec::new()).is_ok());
    }

    #[test]
    fn contourf_invalid_grid_uses_stable_identifier() {
        setup_plot_tests();
        let err = contourf_builtin(Value::Num(0.0), Vec::new()).expect_err("invalid z");
        assert_eq!(err.identifier(), CONTOURF_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn contourf_returns_handle() {
        setup_plot_tests();
        let handle = contourf_builtin(
            Value::Tensor(
                Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).expect("contourf surface"),
            ),
            Vec::new(),
        )
        .expect("contourf should return handle");
        assert!(handle.is_finite());
    }

    #[test]
    fn contourf_accepts_explicit_axes_and_scalar_level_count() {
        setup_plot_tests();
        let handle = contourf_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            vec![
                Value::Tensor(tensor_from(&[0.0, 1.0])),
                Value::Tensor(
                    Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).expect("contourf surface"),
                ),
                Value::Num(12.0),
            ],
        )
        .expect("contourf should accept scalar level counts with explicit axes");
        assert!(handle.is_finite());
    }

    #[test]
    fn contourf_accepts_non_square_meshgrid_axes() {
        setup_plot_tests();
        let handle = contourf_builtin(
            Value::Tensor(matrix_from(&[10.0, 10.0, 20.0, 20.0, 30.0, 30.0], 2, 3)),
            vec![
                Value::Tensor(matrix_from(&[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], 2, 3)),
                Value::Tensor(matrix_from(&[0.0, 1.0, 1.0, 0.0, 2.0, 3.0], 2, 3)),
                Value::Num(12.0),
            ],
        )
        .expect("contourf should accept non-square meshgrid axes");
        assert!(handle.is_finite());
    }

    #[test]
    fn contourf_fill_cells_use_flat_band_colors() {
        let grid = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let plot = build_contour_fill_plot(
            "contourf",
            &[0.0, 1.0],
            &[0.0, 1.0],
            &grid,
            ColorMap::Parula,
            0.0,
            &super::super::contour::ContourLevelSpec::Count(4),
        )
        .expect("filled contour plot");
        let mut plot = plot;
        let render = plot.render_data();
        assert_flat_finite_triangles(&render.vertices);
    }

    #[test]
    fn contourf_nonuniform_axes_fixture_emits_flat_finite_triangles() {
        let grid = vec![
            vec![0.0, 0.2, 0.8, 1.2],
            vec![-0.3, 0.1, 0.6, 1.0],
            vec![-0.7, -0.2, 0.3, 0.9],
            vec![-0.9, -0.4, 0.0, 0.5],
        ];
        let plot = build_contour_fill_plot(
            "contourf",
            &[-3.0, -1.0, 0.5, 2.0],
            &[-2.0, -0.25, 1.5, 3.0],
            &grid,
            ColorMap::Parula,
            0.0,
            &super::super::contour::ContourLevelSpec::Values(vec![-0.5, 0.0, 0.5, 1.0]),
        )
        .expect("filled contour plot");
        let mut plot = plot;
        let render = plot.render_data();
        assert_flat_finite_triangles(&render.vertices);
    }

    #[test]
    fn contourf_saddle_fixture_emits_flat_finite_triangles() {
        let grid = vec![
            vec![1.0, -1.0, 1.0],
            vec![-1.0, 1.0, -1.0],
            vec![1.0, -1.0, 1.0],
        ];
        let plot = build_contour_fill_plot(
            "contourf",
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 2.0],
            &grid,
            ColorMap::Parula,
            0.0,
            &super::super::contour::ContourLevelSpec::Values(vec![-0.5, 0.0, 0.5]),
        )
        .expect("filled contour plot");
        let mut plot = plot;
        let render = plot.render_data();
        assert_flat_finite_triangles(&render.vertices);
    }
}
