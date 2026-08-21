//! MATLAB-compatible `fill` builtin for filled 2-D patch graphics.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::state::{render_active_plot, PlotRenderOptions};

const BUILTIN_NAME: &str = "fill";

const INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fill-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fill with a typed-integer numeric axes alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FillIntegerAxesHandleExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [INTEGER_AXES_EXTENSION];

macro_rules! input_capability {
    ($name:literal) => {
        [BuiltinIntegerInputCapability {
            name: $name,
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::Documented,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "MATLAB documents every built-in integer class; exact class and shape remain authoritative on the Patch object before renderer conversion.",
        }]
    };
}
const INTEGER_X: [BuiltinIntegerInputCapability; 1] = input_capability!("X");
const INTEGER_Y: [BuiltinIntegerInputCapability; 1] = input_capability!("Y");
const INTEGER_C: [BuiltinIntegerInputCapability; 1] = input_capability!("C");
const INTEGER_AX: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ax",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "MATLAB axes are graphics objects; typed-integer numeric aliases are a gated RunMat representation extension.",
}];

const fn capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Patch data remains authoritative; client rendering is the explicit floating-point boundary.",
    }
}

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    capability("p = fill(integer_X, Y, C, ...)", &INTEGER_X),
    capability("p = fill(X, integer_Y, C, ...)", &INTEGER_Y),
    capability("p = fill(X, Y, integer_C, ...)", &INTEGER_C),
    BuiltinIntegerCapabilityDescriptor {
        form: "p = fill(integer_ax, X, Y, C, ...)",
        inputs: &INTEGER_AX,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility extension gate runs before axes selection or input gathering.",
    },
];

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Patch handle or row vector of patch handles.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One or more X,Y,C groups followed by Patch name/value properties.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = fill(X, Y, C, ..., Name, Value)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILL.INVALID_ARGUMENT",
    identifier: Some("RunMat:fill:InvalidArgument"),
    when: "Coordinate/color groups, axes targeting, or properties are invalid.",
    message: "fill: invalid argument",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [INVALID];
pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

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
    category = "plotting",
    summary = "Create filled 2-D patch graphics.",
    keywords = "fill,patch,polygon,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::fill::DESCRIPTOR),
    extensions(crate::builtins::plotting::fill::EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::fill::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::fill"
)]
pub fn fill_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    gate_typed_integer_axes_alias(&args)?;
    let (axes_target, args) = split_leading_axes_handle(args, BUILTIN_NAME)?;
    let groups = parse_groups(args)?;
    let mut plots = groups
        .into_iter()
        .map(super::patch::parse_patch_plot)
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    apply_axes_target(axes_target, BUILTIN_NAME)?;

    let mut plots_slot = Some(std::mem::take(&mut plots));
    let indices_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let indices_slot = std::rc::Rc::clone(&indices_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Filled Polygons",
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
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
        let lower = err.to_string().to_ascii_lowercase();
        if !lower.contains("plotting is unavailable") && !lower.contains("non-main thread") {
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

fn parse_groups(args: Vec<Value>) -> crate::BuiltinResult<Vec<Vec<Value>>> {
    if args.len() < 3 {
        return Err(error("expected X, Y, and C data"));
    }
    let property_start = (3..=args.len())
        .step_by(3)
        .find(|&idx| args[idx..].iter().step_by(2).all(is_string_like))
        .unwrap_or(args.len());
    let (positional, properties) = args.split_at(property_start);
    if positional.len() % 3 != 0 {
        return Err(error("each polygon group must contain X, Y, and C"));
    }
    let mut groups = Vec::new();
    for group in positional.chunks_exact(3) {
        let color_property = if is_string_like(&group[2]) {
            "FaceColor"
        } else {
            "CData"
        };
        for (coordinates, color) in
            super::patch::split_coordinate_group_columns(&group[..2], &group[2], BUILTIN_NAME)?
        {
            let mut patch_args = vec![
                Value::String("XData".into()),
                coordinates[0].clone(),
                Value::String("YData".into()),
                coordinates[1].clone(),
                Value::String(color_property.into()),
                color,
            ];
            patch_args.extend_from_slice(properties);
            groups.push(patch_args);
        }
    }
    Ok(groups)
}

fn gate_typed_integer_axes_alias(args: &[Value]) -> crate::BuiltinResult<()> {
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
        assert_eq!(x_data.shape, vec![1, 3]);
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
                assert_eq!(data.shape, vec![3, 1]);
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
            assert_eq!(data.shape, shape);
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
            assert_eq!(x_data.shape, vec![1, 3]);
            assert_eq!(
                x_data.integer_storage(),
                Some(&IntegerStorage::U64(vec![wide, 2, 3]))
            );
            let Value::Tensor(y_data) =
                get_builtin(vec![Value::Num(handles[index]), Value::from("YData")]).unwrap()
            else {
                panic!("expected YData");
            };
            assert_eq!(y_data.shape, vec![3, 1]);
            assert_eq!(y_data.integer_storage(), Some(&expected_y));
        }
    }
}
