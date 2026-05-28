//! MATLAB-compatible `contourf` builtin (filled contour plot).

use log::warn;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::ColorMap;

use super::common::tensor_to_surface_grid;
use super::contour::{
    build_contour_fill_gpu_plot, build_contour_fill_plot, build_contour_gpu_plot,
    build_contour_plot, parse_contour_args, ContourArgs, ContourLineColor,
};
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
        description: "X axis vector/meshgrid matrix matching Z rows.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y axis vector/meshgrid matrix matching Z columns.",
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
        description: "X axis vector/meshgrid matrix matching Z rows.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y axis vector/meshgrid matrix matching Z columns.",
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
        description: "X axis vector/meshgrid matrix matching Z rows.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y axis vector/meshgrid matrix matching Z columns.",
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

const CONTOURF_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
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
    summary = "Render MATLAB-compatible filled contour plots.",
    keywords = "contourf,plotting,filled,contour",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::contourf::CONTOURF_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::contourf"
)]
pub fn contourf_builtin(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<f64> {
    let mut args =
        Some(parse_contour_args("contourf", first, rest).map_err(map_contourf_invalid_argument)?);
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
            line_width: _,
        } = args.take().expect("contourf args consumed once");
        let color_map = ColorMap::Parula;
        let base_z = 0.0;

        if let Some(handle) = z_input.gpu_handle() {
            match build_contour_fill_gpu_plot(
                name,
                &x_axis,
                &y_axis,
                handle,
                color_map,
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
                            color_map,
                            base_z,
                            &level_spec,
                            &line_color,
                        ) {
                            Ok(contours) => {
                                figure.add_contour_plot_on_axes(contours, axes);
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

        let grid = tensor_to_surface_grid(
            z_input
                .into_tensor(name)
                .map_err(map_contourf_invalid_argument)?,
            x_axis.len(),
            y_axis.len(),
            name,
        )
        .map_err(map_contourf_invalid_argument)?;
        let fill_plot = build_contour_fill_plot(
            name,
            &x_axis,
            &y_axis,
            &grid,
            color_map,
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
                    figure.add_contour_plot_on_axes(contours, axes);
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use runmat_builtins::Tensor;
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
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
            Value::Tensor(Tensor {
                data: vec![0.0, 1.0, 1.0, 0.0],
                shape: vec![2, 2],
                rows: 2,
                cols: 2,
                dtype: runmat_builtins::NumericDType::F64,
            }),
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
                Value::Tensor(Tensor {
                    data: vec![0.0, 1.0, 1.0, 0.0],
                    shape: vec![2, 2],
                    rows: 2,
                    cols: 2,
                    dtype: runmat_builtins::NumericDType::F64,
                }),
                Value::Num(12.0),
            ],
        )
        .expect("contourf should accept scalar level counts with explicit axes");
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
