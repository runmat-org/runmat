//! MATLAB-compatible `fill` builtin for filled 2-D patch graphics.
pub use runmat_builtins::{
    FILL_INTEGER_AXES_EXTENSION as INTEGER_AXES_EXTENSION, FILL_INVALID as INVALID,
};

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use crate::{build_runtime_error, RuntimeError};

use super::op_common::{apply_axes_target, split_leading_axes_arguments};
use super::state::{render_active_plot, PlotRenderOptions};

const BUILTIN_NAME: &str = "fill";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::fill")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fill",
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
    notes: "fill gathers resident patch data exactly and renders on the client.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::fill")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fill",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fill is a plotting sink and fusion barrier.",
};

#[runtime_builtin(
    name = "fill",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::fill"
)]
pub fn fill_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "fill: expected at most one output",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "fill")?;
    let _arguments = args.fork();
    work.finish(fill_impl(args, &work))
}

fn fill_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<Value> {
    gate_typed_integer_axes_alias(args.view())?;
    let (axes_target, args) = split_leading_axes_arguments(args, BUILTIN_NAME)?;
    let groups = parse_groups(args)?;
    let mut plots = groups
        .into_iter()
        .map(super::patch::parse_patch_plot)
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    work.check()?;
    apply_axes_target(axes_target, BUILTIN_NAME)?;

    let mut plots_slot = Some(std::mem::take(&mut plots));
    let indices_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let indices_slot = std::rc::Rc::clone(&indices_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions::default(),
        move |figure, axes| {
            for plot in plots_slot.take().expect("fill plots consumed once") {
                let plot_index = figure.add_patch_plot_on_axes(plot, axes);
                indices_slot.borrow_mut().push((axes, plot_index));
            }
            Ok(())
        },
    );
    let indices = indices_out.borrow().clone();
    if indices.is_empty() {
        return render_result.map(|_| Value::Num(f64::NAN));
    }
    if let Err(err) = render_result {
        if !crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Err(err);
        }
    }
    let handles = indices
        .into_iter()
        .map(|(axes, plot_index)| {
            crate::builtins::plotting::state::register_patch_handle(figure_handle, axes, plot_index)
        })
        .collect();
    Ok(handles_value(handles))
}

fn parse_groups(arguments: crate::call::BuiltinArguments) -> crate::BuiltinResult<Vec<Vec<Value>>> {
    let args = arguments.view();
    if args.len() < 3 {
        return Err(error("expected X, Y, and C data"));
    }
    let property_start = (3..=args.len())
        .step_by(3)
        .find(|&idx| {
            args.range(idx..)
                .expect("bounded property suffix")
                .iter()
                .step_by(2)
                .all(is_string_like)
        })
        .unwrap_or(args.len());
    let positional = args.range(..property_start)?;
    let properties = args.range(property_start..)?;
    if positional.len() % 3 != 0 {
        return Err(error("each polygon group must contain X, Y, and C"));
    }
    let mut groups = Vec::new();
    for group in positional.chunks_exact(3) {
        for (coordinates, color) in super::patch::split_coordinate_group_columns(
            &[group[0].clone(), group[1].clone()],
            &group[2],
            BUILTIN_NAME,
        )? {
            // Classify the retained, gathered column rather than the original
            // argument. RGB literals share patch/fill3 admission; indexed
            // columns and scalar colors retain their CData representation.
            let color_property =
                if is_string_like(&color) || super::patch::is_color_literal(&color)? {
                    "FaceColor"
                } else {
                    "CData"
                };
            let mut patch_args = vec![
                Value::String("XData".into()),
                coordinates[0].clone(),
                Value::String("YData".into()),
                coordinates[1].clone(),
                Value::String(color_property.into()),
                color,
            ];
            patch_args.extend(properties.iter().cloned());
            groups.push(patch_args);
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

fn is_string_like(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::CharArray(_))
}

fn handles_value(handles: Vec<f64>) -> Value {
    if handles.len() == 1 {
        Value::Num(handles[0])
    } else {
        let len = handles.len();
        Value::Tensor(Tensor::new(handles, vec![1, len]).expect("valid handle vector"))
    }
}

fn error(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("fill: {}", detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(
            INVALID
                .identifier
                .expect("fill invalid-argument descriptor identifier"),
        )
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{
        BuiltinIntegerInputAvailability, FILL_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES,
    };
    fn fill_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        super::fill_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
    }

    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_value::{IntValue, IntegerStorage};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer patch data"))
    }

    #[test]
    fn fill_rgb_literal_and_indexed_colors_keep_distinct_patch_sources() {
        let parse = |color| {
            let args = crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]).unwrap()),
                color,
            ]);
            let mut groups = parse_groups(args).unwrap();
            assert_eq!(groups.len(), 1);
            crate::builtins::plotting::patch::parse_patch_plot(groups.pop().unwrap()).unwrap()
        };
        let rgb = parse(Value::Tensor(
            Tensor::new(vec![0.3, 0.6, 0.9], vec![1, 3]).unwrap(),
        ));
        assert_eq!(rgb.face_color(), glam::Vec4::new(0.3, 0.6, 0.9, 1.0));
        assert_eq!(
            rgb.face_color_mode(),
            runmat_plot::plots::PatchFaceColorMode::Color
        );
        assert!(rgb.source_data().3.is_none());

        let indexed = parse(integer_tensor(
            IntegerStorage::U16(vec![1, 2, 3, 4]),
            vec![4, 1],
        ));
        assert_eq!(
            indexed.color_source_kind(),
            runmat_plot::plots::PatchColorSourceKind::CoordinateGrid
        );
        let colors = indexed.source_data().3.unwrap();
        assert_eq!(colors.shape(), [4, 1]);
        assert_eq!(
            colors.storage(),
            &runmat_value::NumericStorage::U16(vec![1, 2, 3, 4])
        );

        let scalar = parse(Value::Int(IntValue::U8(2)));
        assert_eq!(
            scalar.color_source_kind(),
            runmat_plot::plots::PatchColorSourceKind::FaceVertex
        );
        assert_eq!(
            scalar.source_data().3.unwrap().storage(),
            &runmat_value::NumericStorage::U8(vec![2])
        );
    }

    #[test]
    fn fill_preserves_wide_integer_xdata_and_shape() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let handle = fill_builtin(vec![
            integer_tensor(IntegerStorage::U64(vec![wide, 1, 0]), vec![1, 3]),
            integer_tensor(IntegerStorage::U64(vec![0, 0, 1]), vec![1, 3]),
            Value::Int(IntValue::U8(1)),
        ])
        .expect("integer fill plot");
        let Value::Num(handle) = handle else {
            panic!("expected scalar Patch handle");
        };
        let x_data =
            get_builtin(vec![Value::Num(handle), Value::from("XData")]).expect("Patch XData");
        let Value::Tensor(x_data) = x_data else {
            panic!("expected typed XData tensor");
        };
        assert_eq!(x_data.shape(), [1, 3]);
        assert_eq!(
            x_data.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide, 1, 0]))
        );
    }

    #[test]
    fn fill_declares_all_integer_coordinate_color_capabilities() {
        assert_eq!(INTEGER_CAPABILITIES.len(), 4);
        for capability in &INTEGER_CAPABILITIES[..3] {
            assert_eq!(capability.inputs[0].classes.len(), 8);
            assert_eq!(
                capability.inputs[0].availability,
                BuiltinIntegerInputAvailability::Documented
            );
        }
    }

    #[test]
    fn fill_matrix_columns_create_distinct_patches_with_exact_properties() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let handles = fill_builtin(vec![
            integer_tensor(
                IntegerStorage::U64(vec![wide, 2, 3, wide + 2, 5, 6]),
                vec![3, 2],
            ),
            integer_tensor(IntegerStorage::I16(vec![0, 0, 1, 2, 2, 3]), vec![3, 2]),
            integer_tensor(
                IntegerStorage::U16(vec![10, 20, 30, 40, 50, 60]),
                vec![3, 2],
            ),
        ])
        .expect("matrix fill");
        let Value::Tensor(handles) = handles else {
            panic!("expected patch handle vector");
        };
        let handles = handles.materialize_f64();
        assert_eq!(handles.len(), 2);
        assert_eq!(clone_figure(current_figure_handle()).unwrap().len(), 2);

        for (index, expected_x, expected_y, expected_c) in [
            (
                0,
                IntegerStorage::U64(vec![wide, 2, 3]),
                IntegerStorage::I16(vec![0, 0, 1]),
                IntegerStorage::U16(vec![10, 20, 30]),
            ),
            (
                1,
                IntegerStorage::U64(vec![wide + 2, 5, 6]),
                IntegerStorage::I16(vec![2, 2, 3]),
                IntegerStorage::U16(vec![40, 50, 60]),
            ),
        ] {
            for (property, expected) in [
                ("XData", expected_x),
                ("YData", expected_y),
                ("CData", expected_c),
            ] {
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
    fn fill_accepts_equal_length_vectors_with_different_orientations() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let handle = fill_builtin(vec![
            integer_tensor(IntegerStorage::U64(vec![wide, 2, 3]), vec![1, 3]),
            integer_tensor(IntegerStorage::I16(vec![0, 0, 1]), vec![3, 1]),
            Value::Int(IntValue::U8(1)),
        ])
        .expect("oriented vector fill");
        let Value::Num(handle) = handle else {
            panic!("expected scalar Patch handle");
        };
        for (property, shape, expected) in [
            ("XData", vec![1, 3], IntegerStorage::U64(vec![wide, 2, 3])),
            ("YData", vec![3, 1], IntegerStorage::I16(vec![0, 0, 1])),
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
    fn fill_broadcasts_shared_wide_integer_vector_across_matrix_columns() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let handles = fill_builtin(vec![
            integer_tensor(IntegerStorage::U64(vec![wide, 2, 3]), vec![1, 3]),
            integer_tensor(IntegerStorage::I16(vec![0, 0, 1, 2, 2, 3]), vec![3, 2]),
            integer_tensor(
                IntegerStorage::U16(vec![10, 20, 30, 40, 50, 60]),
                vec![3, 2],
            ),
        ])
        .expect("shared-vector matrix fill");
        let Value::Tensor(handles) = handles else {
            panic!("expected Patch handle vector");
        };
        let handles = handles.materialize_f64();
        assert_eq!(handles.len(), 2);
        for (index, expected_y) in [
            IntegerStorage::I16(vec![0, 0, 1]),
            IntegerStorage::I16(vec![2, 2, 3]),
        ]
        .into_iter()
        .enumerate()
        {
            let Value::Tensor(x_data) =
                get_builtin(vec![Value::Num(handles[index]), Value::from("XData")]).unwrap()
            else {
                panic!("expected XData");
            };
            assert_eq!(x_data.shape(), [1, 3]);
            assert_eq!(
                x_data.integer_storage(),
                Some(&IntegerStorage::U64(vec![wide, 2, 3]))
            );
            let Value::Tensor(y_data) =
                get_builtin(vec![Value::Num(handles[index]), Value::from("YData")]).unwrap()
            else {
                panic!("expected YData");
            };
            assert_eq!(y_data.shape(), [3, 1]);
            assert_eq!(y_data.integer_storage(), Some(&expected_y));
        }
    }
}
