//! MATLAB-compatible `contourf` builtin (filled contour plot).
use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{CONTOURF_ERROR_INTERNAL, CONTOURF_ERROR_INVALID_ARGUMENT};

use runmat_macros::runtime_builtin;
use runmat_plot::plots::ColorMap;
use runmat_value::Value;

use super::common::tensor_to_surface_grid_matlab_xy;
use super::contour::{
    build_contour_fill_gpu_plot, build_contour_fill_plot, build_contour_plot, parse_contour_args,
    ContourArgs, ContourLineColor,
};
use super::op_common::axes_target::{apply_axes_target, split_leading_axes_arguments};
use super::state::{render_active_plot, PlotRenderOptions};
use crate::build_runtime_error;

const BUILTIN_NAME: &str = "contourf";

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
    let mapped = contourf_error_with_detail(&CONTOURF_ERROR_INVALID_ARGUMENT, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_contourf_internal(err: crate::RuntimeError) -> crate::RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = contourf_error_with_detail(&CONTOURF_ERROR_INTERNAL, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "contourf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::contourf"
)]
pub fn contourf_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "contourf: expected at most one output",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "contourf")?;
    let _arguments = args.fork();
    work.finish(contourf_impl(args, &work))
}

fn contourf_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let all = args;
    let (axes_target, all) = if all.first().is_some_and(is_typed_integer_value) {
        (None, all)
    } else {
        split_leading_axes_arguments(all, BUILTIN_NAME).map_err(map_contourf_invalid_argument)?
    };
    let first = all.first().cloned().ok_or_else(|| {
        contourf_error_with_detail(&CONTOURF_ERROR_INVALID_ARGUMENT, "expected Z input")
    })?;
    let rest = all.view().range(1..)?;
    let mut args =
        Some(parse_contour_args(BUILTIN_NAME, first, rest).map_err(map_contourf_invalid_argument)?);
    work.check()?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_contourf_invalid_argument)?;
    let opts = PlotRenderOptions {
        // MATLAB leaves the data aspect free; `axis equal` sets it.
        axis_equal: false,
        ..Default::default()
    };
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
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

        let x_source = super::op_common::surface_inputs::AxisSource::Host(x_axis);
        let y_source = super::op_common::surface_inputs::AxisSource::Host(y_axis);
        let super::op_common::surface_inputs::AxisSource::Host(x_axis) = &x_source else {
            unreachable!()
        };
        let super::op_common::surface_inputs::AxisSource::Host(y_axis) = &y_source else {
            unreachable!()
        };
        if let Some(handle) = z_input.gpu_handle() {
            if let Some(mut prepared) = futures::executor::block_on(
                super::op_common::surface_inputs::PreparedSurfaceGpu::prepare(
                    name, &x_source, &y_source, handle,
                ),
            )? {
                let fill_plot = build_contour_fill_gpu_plot(
                    name,
                    x_axis,
                    y_axis,
                    handle,
                    &mut prepared,
                    color_map.clone(),
                    base_z,
                    &level_spec,
                )?;
                let contours = if matches!(line_color, ContourLineColor::None) {
                    None
                } else {
                    Some(
                        super::contour::build_contour_gpu_plot_prepared(
                            name,
                            &x_source,
                            &y_source,
                            handle,
                            &mut prepared,
                            color_map.clone(),
                            base_z,
                            &level_spec,
                            &line_color,
                            super::contour::ContourZMode::Base,
                        )?
                        .with_line_width(line_width),
                    )
                };
                // Publish the requested composite only after every required
                // stage succeeds; never silently omit an overlay.
                figure.add_contour_fill_plot_on_axes(fill_plot, axes);
                *plot_index_slot.borrow_mut() = Some((axes, before));
                if let Some(contours) = contours {
                    figure.add_contour_plot_on_axes(contours, axes);
                }
                return Ok(());
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
            x_axis,
            y_axis,
            &grid,
            color_map.clone(),
            base_z,
            &level_spec,
        )
        .map_err(map_contourf_invalid_argument)?;
        let contours = if matches!(line_color, ContourLineColor::None) {
            None
        } else {
            Some(
                build_contour_plot(
                    name,
                    x_axis,
                    y_axis,
                    &grid,
                    color_map,
                    base_z,
                    &level_spec,
                    &line_color,
                )
                .map_err(map_contourf_invalid_argument)?
                .with_line_width(line_width),
            )
        };
        figure.add_contour_fill_plot_on_axes(fill_plot, axes);
        *plot_index_slot.borrow_mut() = Some((axes, before));
        if let Some(contours) = contours {
            figure.add_contour_plot_on_axes(contours, axes);
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
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
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
    use runmat_builtins::{
        CONTOURF_DESCRIPTOR, CONTOURF_EXTENSIONS, CONTOURF_INTEGER_CAPABILITIES,
    };
    fn parse_contour_args(
        name: &'static str,
        first: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<super::super::contour::ContourArgs> {
        super::super::contour::parse_contour_args(
            name,
            first,
            crate::call::FunctionArgumentView::from_slice(&rest),
        )
    }

    fn contourf_builtin(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<f64> {
        let args = std::iter::once(first).chain(rest).collect();
        super::contourf_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
    }

    use crate::builtins::common::test_support;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::lock_plot_test_context;
    use crate::builtins::plotting::state::{clone_figure, current_figure_handle, reset_plot_state};
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use runmat_plot::plots::figure::PlotElement;
    use runmat_value::{IntegerStorage, Tensor};

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
        let _guard = lock_plot_test_context();
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
    fn contourf_catalog_infers_host_double_handle() {
        use runmat_types::*;
        let numeric = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        let input = ValueFact::proven(
            numeric.clone(),
            ShapeFact::from(vec![Some(3), Some(2)]),
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![input; 1],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result =
            runmat_builtins::infer_catalog_call(&runmat_builtins::CONTOURF_CATALOG_ENTRY, &request);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 1);
        let output = &result.outputs[0];
        assert_eq!(output.kind, numeric);
        assert_eq!(output.shape, ShapeFact::Scalar);
        assert_eq!(output.storage, StorageFact::Dense);
        assert_eq!(output.residency, ResidencyFact::Host);
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
                Value::Int(runmat_value::IntValue::U16(3)),
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
        let _guard = lock_plot_test_context();
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
        let _guard = lock_plot_test_context();
        setup_plot_tests();
        let err = contourf_builtin(Value::Num(0.0), Vec::new()).expect_err("invalid z");
        assert_eq!(err.identifier(), CONTOURF_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn contourf_returns_handle() {
        let _guard = lock_plot_test_context();
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
        let _guard = lock_plot_test_context();
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
        let _guard = lock_plot_test_context();
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
