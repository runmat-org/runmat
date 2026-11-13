//! MATLAB-compatible `fill3` builtin.
use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{
    FILL3_ERROR_INTERNAL, FILL3_ERROR_INVALID_ARGUMENT,
    FILL3_INTEGER_AXES_EXTENSION as INTEGER_AXES_EXTENSION,
};

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::op_common::{apply_axes_target, split_leading_axes_arguments};
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "fill3";

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
    let mapped = fill3_error_with_detail(&FILL3_ERROR_INVALID_ARGUMENT, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_fill3_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mapped = fill3_error_with_detail(&FILL3_ERROR_INTERNAL, err.message.clone());
    let mut builder = crate::build_runtime_error(mapped.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = mapped.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::fill3"
)]
pub fn fill3_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "fill3: expected at most one output",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "fill3")?;
    let _arguments = args.fork();
    work.finish(fill3_impl(args, &work))
}

fn fill3_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<Value> {
    gate_typed_integer_axes_alias(args.view())?;
    let (axes_target, args) =
        split_leading_axes_arguments(args, BUILTIN_NAME).map_err(map_fill3_invalid_argument)?;

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
    work.check()?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_fill3_invalid_argument)?;

    let mut plots_opt = Some(std::mem::take(&mut plots));
    let plot_indices_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            // MATLAB leaves the data aspect free; `axis equal` sets it.
            axis_equal: false,
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
        if !crate::builtins::plotting::state::presentation_unavailable(&err) {
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

fn parse_fill3_patch_arg_groups(
    arguments: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Vec<Vec<Value>>> {
    let args = arguments.view();
    if args.len() < 4 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "fill3: expected X, Y, Z, and C data",
        ));
    }

    let property_start = find_trailing_property_start(args);
    let positional = args.range(..property_start)?;
    let properties = args.range(property_start..)?;
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

        let color_name = if super::patch::is_color_literal(&positional[offset + 3])? {
            "FaceColor"
        } else {
            "CData"
        };
        let columns = super::patch::split_coordinate_group_columns(
            &[
                positional[offset].clone(),
                positional[offset + 1].clone(),
                positional[offset + 2].clone(),
            ],
            &positional[offset + 3],
            BUILTIN_NAME,
        )?;
        offset += 4;
        for (coordinates, color) in columns {
            let mut group = vec![
                Value::String("XData".into()),
                coordinates[0].clone(),
                Value::String("YData".into()),
                coordinates[1].clone(),
                Value::String("ZData".into()),
                coordinates[2].clone(),
                Value::String(color_name.into()),
                color,
            ];
            group.extend(properties.iter().cloned());
            groups.push(group);
        }
    }

    Ok(groups)
}

fn gate_typed_integer_axes_alias(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<()> {
    let Some(first) = args.first() else {
        return Ok(());
    };
    let typed_integer = matches!(first, Value::Int(_))
        || matches!(first, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(first, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if typed_integer && super::properties::resolve_plot_handle(first, BUILTIN_NAME).is_ok() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_AXES_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn find_trailing_property_start(args: crate::call::FunctionArgumentView<'_>) -> usize {
    for idx in (4..=args.len()).step_by(4) {
        if args
            .range(idx..)
            .expect("bounded property suffix")
            .iter()
            .step_by(2)
            .all(is_string_like)
        {
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
    use runmat_builtins::FILL3_DESCRIPTOR;
    fn fill3_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        super::fill3_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
    }

    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use glam::Vec4;
    use runmat_plot::plots::figure::PlotElement;
    use runmat_plot::plots::{PatchEdgeColorMode, PatchFaceColorMode};
    use runmat_value::IntegerStorage;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        super::super::state::reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(rows: usize, cols: usize, data: &[f64]) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![rows, cols]).expect("fill3 test tensor"))
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("fill3 integer tensor"))
    }

    #[test]
    fn fill3_preserves_wide_integer_coordinate_and_cdata_properties() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let handle = fill3_builtin(vec![
            integer_tensor(IntegerStorage::U64(vec![wide, 1, 0]), vec![1, 3]),
            integer_tensor(IntegerStorage::U64(vec![0, 0, 1]), vec![1, 3]),
            integer_tensor(IntegerStorage::I64(vec![0, 1, 2]), vec![1, 3]),
            integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
        ])
        .expect("integer fill3");
        let Value::Num(handle) = handle else {
            panic!("expected scalar Patch handle");
        };
        let Value::Tensor(x_data) =
            get_builtin(vec![Value::Num(handle), Value::from("XData")]).unwrap()
        else {
            panic!("expected XData tensor");
        };
        assert_eq!(x_data.shape(), [1, 3]);
        assert_eq!(
            x_data.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide, 1, 0]))
        );
        let Value::Tensor(c_data) =
            get_builtin(vec![Value::Num(handle), Value::from("CData")]).unwrap()
        else {
            panic!("expected CData tensor");
        };
        assert_eq!(
            c_data.integer_storage(),
            Some(&IntegerStorage::U16(vec![1]))
        );
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
        assert_eq!(handles.materialize_f64().len(), 2);
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
    fn fill3_matrix_columns_create_distinct_patches_with_exact_properties() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let handles = fill3_builtin(vec![
            integer_tensor(
                IntegerStorage::U64(vec![wide, 2, 3, wide + 2, 5, 6]),
                vec![3, 2],
            ),
            integer_tensor(IntegerStorage::I16(vec![0, 0, 1, 2, 2, 3]), vec![3, 2]),
            integer_tensor(IntegerStorage::I32(vec![7, 8, 9, 10, 11, 12]), vec![3, 2]),
            integer_tensor(
                IntegerStorage::U16(vec![10, 20, 30, 40, 50, 60]),
                vec![3, 2],
            ),
        ])
        .expect("matrix fill3");
        let Value::Tensor(handles) = handles else {
            panic!("expected patch handle vector");
        };
        let handles = handles.materialize_f64();
        assert_eq!(handles.len(), 2);
        assert_eq!(clone_figure(current_figure_handle()).unwrap().len(), 2);

        for (index, expected) in [
            [
                IntegerStorage::U64(vec![wide, 2, 3]),
                IntegerStorage::I16(vec![0, 0, 1]),
                IntegerStorage::I32(vec![7, 8, 9]),
                IntegerStorage::U16(vec![10, 20, 30]),
            ],
            [
                IntegerStorage::U64(vec![wide + 2, 5, 6]),
                IntegerStorage::I16(vec![2, 2, 3]),
                IntegerStorage::I32(vec![10, 11, 12]),
                IntegerStorage::U16(vec![40, 50, 60]),
            ],
        ]
        .into_iter()
        .enumerate()
        {
            for (property, expected) in ["XData", "YData", "ZData", "CData"]
                .into_iter()
                .zip(expected)
            {
                let Value::Tensor(data) =
                    get_builtin(vec![Value::Num(handles[index]), Value::from(property)])
                        .expect("exact Patch property")
                else {
                    panic!("expected typed {property}");
                };
                assert_eq!(data.shape(), [3, 1]);
                assert_eq!(data.integer_storage(), Some(&expected));
            }
        }
    }

    #[test]
    fn fill3_accepts_equal_length_vectors_with_different_orientations() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let handle = fill3_builtin(vec![
            integer_tensor(IntegerStorage::U64(vec![wide, 2, 3]), vec![1, 3]),
            integer_tensor(IntegerStorage::I16(vec![0, 0, 1]), vec![3, 1]),
            integer_tensor(IntegerStorage::I32(vec![7, 8, 9]), vec![1, 3]),
            integer_tensor(IntegerStorage::U16(vec![10]), vec![1, 1]),
        ])
        .expect("oriented vector fill3");
        let Value::Num(handle) = handle else {
            panic!("expected scalar Patch handle");
        };
        for (property, shape, expected) in [
            ("XData", vec![1, 3], IntegerStorage::U64(vec![wide, 2, 3])),
            ("YData", vec![3, 1], IntegerStorage::I16(vec![0, 0, 1])),
            ("ZData", vec![1, 3], IntegerStorage::I32(vec![7, 8, 9])),
        ] {
            let Value::Tensor(data) =
                get_builtin(vec![Value::Num(handle), Value::from(property)]).unwrap()
            else {
                panic!("expected typed {property}");
            };
            assert_eq!(data.shape(), shape);
            assert_eq!(data.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn fill3_broadcasts_shared_wide_integer_vector_across_matrix_columns() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let handles = fill3_builtin(vec![
            integer_tensor(IntegerStorage::U64(vec![wide, 2, 3]), vec![3, 1]),
            integer_tensor(IntegerStorage::I16(vec![0, 0, 1, 2, 2, 3]), vec![3, 2]),
            integer_tensor(IntegerStorage::I32(vec![7, 8, 9, 10, 11, 12]), vec![3, 2]),
            integer_tensor(
                IntegerStorage::U16(vec![10, 20, 30, 40, 50, 60]),
                vec![3, 2],
            ),
        ])
        .expect("shared-vector matrix fill3");
        let Value::Tensor(handles) = handles else {
            panic!("expected Patch handle vector");
        };
        let handles = handles.materialize_f64();
        assert_eq!(handles.len(), 2);
        for (index, expected_y, expected_z) in [
            (
                0,
                IntegerStorage::I16(vec![0, 0, 1]),
                IntegerStorage::I32(vec![7, 8, 9]),
            ),
            (
                1,
                IntegerStorage::I16(vec![2, 2, 3]),
                IntegerStorage::I32(vec![10, 11, 12]),
            ),
        ] {
            let Value::Tensor(x_data) =
                get_builtin(vec![Value::Num(handles[index]), Value::from("XData")]).unwrap()
            else {
                panic!("expected XData");
            };
            assert_eq!(x_data.shape(), [3, 1]);
            assert_eq!(
                x_data.integer_storage(),
                Some(&IntegerStorage::U64(vec![wide, 2, 3]))
            );
            for (property, expected) in [("YData", expected_y), ("ZData", expected_z)] {
                let Value::Tensor(data) =
                    get_builtin(vec![Value::Num(handles[index]), Value::from(property)]).unwrap()
                else {
                    panic!("expected {property}");
                };
                assert_eq!(data.shape(), [3, 1]);
                assert_eq!(data.integer_storage(), Some(&expected));
            }
        }
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
            Value::CharArray(runmat_value::CharArray::new_row("AnotherFutureProperty")),
            Value::String("value".into()),
        ];

        assert_eq!(
            find_trailing_property_start(crate::call::FunctionArgumentView::from_slice(&args)),
            4
        );
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
