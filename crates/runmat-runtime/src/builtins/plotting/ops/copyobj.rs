//! MATLAB-compatible `copyobj` graphics-object copying builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    copy_plot_child_to_parent, validate_plot_child_copy_source, CopyParentTarget, FigureError,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::handles::numeric_handle_from_integer;
use crate::builtins::plotting::type_resolvers::handle_array_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "copyobj";

const COPYOBJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "hnew",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Copied graphics object handle or handle array.",
}];

const COPYOBJ_INPUTS_SOURCE_PARENT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Graphics object handle or handle array to copy.",
    },
    BuiltinParamDescriptor {
        name: "parent",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination axes handle compatible with the plot child.",
    },
];

const COPYOBJ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "hnew = copyobj(h, parent)",
    inputs: &COPYOBJ_INPUTS_SOURCE_PARENT,
    outputs: &COPYOBJ_OUTPUT,
}];

const COPYOBJ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COPYOBJ.INVALID_ARGUMENT",
    identifier: Some("RunMat:copyobj:InvalidArgument"),
    when: "Argument count, source handle, or target parent handle is invalid or unsupported.",
    message: "copyobj: invalid argument",
};

const COPYOBJ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COPYOBJ.INTERNAL",
    identifier: Some("RunMat:copyobj:Internal"),
    when: "Internal graphics state lookup or copy registration fails.",
    message: "copyobj: internal operation failed",
};

const COPYOBJ_ERRORS: [BuiltinErrorDescriptor; 2] =
    [COPYOBJ_ERROR_INVALID_ARGUMENT, COPYOBJ_ERROR_INTERNAL];

pub const COPYOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COPYOBJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COPYOBJ_ERRORS,
};

pub const COPYOBJ_INTEGER_HANDLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "copyobj-integer-handle-aliases",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "copyobj with typed-integer numeric graphics-handle aliases is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CopyobjIntegerHandleExtension"),
    };

pub const COPYOBJ_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [COPYOBJ_INTEGER_HANDLE_EXTENSION];

const COPYOBJ_INTEGER_SOURCE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "h",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode treats exactly representable nonnegative host integers as aliases for its numeric graphics-handle representation; the public contract accepts graphics objects, not integer arrays.",
    }];
const COPYOBJ_INTEGER_PARENT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "parent",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode treats exactly representable nonnegative host integers as aliases for numeric axes handles; resident numeric arrays are not graphics objects.",
    }];
const COPYOBJ_RESIDENT_INTEGER: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident h or parent",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident integer arrays are numeric data rather than graphics objects and reject without provider gather.",
    }];

pub const COPYOBJ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "hnew = copyobj(integer_h, parent)", inputs: &COPYOBJ_INTEGER_SOURCE, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "A separately gated RunMat alias reads authoritative host integer storage, rejects values that cannot be represented exactly by RunMat's f64 graphics registry, validates every source before mutation, and returns fresh host graphics handles." },
    BuiltinIntegerCapabilityDescriptor { form: "hnew = copyobj(h, integer_parent)", inputs: &COPYOBJ_INTEGER_PARENT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "A separately gated RunMat alias resolves authoritative host integers as axes parents before any copy is made; output handles follow the controlling source/parent shape." },
    BuiltinIntegerCapabilityDescriptor { form: "copyobj(resident_integer_handle_alias, ...)", inputs: &COPYOBJ_RESIDENT_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "GPU-resident numeric arrays are not graphics-object handles; copyobj performs no provider dispatch or implicit gather." },
];

fn copyobj_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()));
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "copyobj",
    category = "plotting",
    summary = "Copy plot-child graphics objects to a compatible axes parent.",
    keywords = "copyobj,copy,graphics,handle,axes,figure,plotting",
    suppress_auto_output = true,
    type_resolver(handle_array_type),
    descriptor(crate::builtins::plotting::copyobj::COPYOBJ_DESCRIPTOR),
    extensions(crate::builtins::plotting::copyobj::COPYOBJ_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::copyobj::COPYOBJ_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::copyobj"
)]
pub fn copyobj_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let mut iter = args.into_iter();
    let source = iter.next().ok_or_else(|| {
        copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "expected source graphics handle",
        )
    })?;
    let parent = iter.next().ok_or_else(|| {
        copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "expected destination parent handle",
        )
    })?;
    if iter.next().is_some() {
        return Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "expected exactly two arguments",
        ));
    }

    if is_host_integer_handle_alias(&source) || is_host_integer_handle_alias(&parent) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COPYOBJ_INTEGER_HANDLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    let sources = handle_list(&source, "source")?;
    let parents = parent_target_list(&parent)?;
    let plan = copy_plan(&sources, &parents)?;

    for &(source_handle, _) in &plan.pairs {
        validate_copy_source(source_handle)?;
    }

    let mut copied = Vec::with_capacity(plan.pairs.len());
    for (source_handle, target) in plan.pairs {
        copied.push(copy_plot_child(source_handle, target)?);
    }

    if plan.scalar_output {
        Ok(Value::Num(copied[0]))
    } else {
        Ok(Value::Tensor(Tensor::new_with_dtype(
            copied,
            plan.output_shape,
            NumericDType::F64,
        )?))
    }
}

#[derive(Clone, Debug)]
struct HandleList {
    handles: Vec<f64>,
    shape: Vec<usize>,
    is_scalar: bool,
}

#[derive(Clone, Debug)]
struct ParentTargetList {
    targets: Vec<CopyParentTarget>,
    shape: Vec<usize>,
    is_scalar: bool,
}

#[derive(Clone, Debug)]
struct CopyPlan {
    pairs: Vec<(f64, CopyParentTarget)>,
    output_shape: Vec<usize>,
    scalar_output: bool,
}

fn copy_plan(sources: &HandleList, parents: &ParentTargetList) -> BuiltinResult<CopyPlan> {
    if sources.handles.is_empty() {
        return Ok(CopyPlan {
            pairs: Vec::new(),
            output_shape: sources.shape.clone(),
            scalar_output: false,
        });
    }
    if parents.targets.is_empty() {
        return Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "parent handle array must not be empty",
        ));
    }

    let (count, output_shape) = if parents.is_scalar {
        (sources.handles.len(), sources.shape.clone())
    } else if sources.is_scalar {
        (parents.targets.len(), parents.shape.clone())
    } else if sources.handles.len() == parents.targets.len() {
        (sources.handles.len(), sources.shape.clone())
    } else {
        return Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "source and parent arrays must be scalar-expandable or have matching element counts",
        ));
    };

    let mut pairs = Vec::with_capacity(count);
    for idx in 0..count {
        let source = if sources.is_scalar {
            sources.handles[0]
        } else {
            sources.handles[idx]
        };
        let target = if parents.is_scalar {
            parents.targets[0]
        } else {
            parents.targets[idx]
        };
        pairs.push((source, target));
    }

    Ok(CopyPlan {
        pairs,
        output_shape,
        scalar_output: sources.is_scalar && parents.is_scalar,
    })
}

fn validate_copy_source(source_handle: f64) -> BuiltinResult<()> {
    match resolve_plot_handle(&Value::Num(source_handle), BUILTIN_NAME)
        .map_err(|err| copyobj_error(&COPYOBJ_ERROR_INVALID_ARGUMENT, err.message))?
    {
        PlotHandle::PlotChild(_, state)
            if matches!(
                *state,
                crate::builtins::plotting::state::PlotChildHandleState::TextScatter(_)
                    | crate::builtins::plotting::state::PlotChildHandleState::StackedPlot(_)
            ) =>
        {
            Err(copyobj_error(
                &COPYOBJ_ERROR_INVALID_ARGUMENT,
                "Composite chart handles cannot be copied in this release",
            ))
        }
        PlotHandle::PlotChild(_, _) => validate_plot_child_copy_source(source_handle)
            .map_err(|err| map_copy_error(&COPYOBJ_ERROR_INVALID_ARGUMENT, err)),
        _ => Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "only plot-child graphics objects can be copied in this release",
        )),
    }
}

fn copy_plot_child(source_handle: f64, target: CopyParentTarget) -> BuiltinResult<f64> {
    match resolve_plot_handle(&Value::Num(source_handle), BUILTIN_NAME)
        .map_err(|err| copyobj_error(&COPYOBJ_ERROR_INVALID_ARGUMENT, err.message))?
    {
        PlotHandle::PlotChild(_, state)
            if matches!(
                *state,
                crate::builtins::plotting::state::PlotChildHandleState::TextScatter(_)
                    | crate::builtins::plotting::state::PlotChildHandleState::StackedPlot(_)
            ) =>
        {
            Err(copyobj_error(
                &COPYOBJ_ERROR_INVALID_ARGUMENT,
                "Composite chart handles cannot be copied in this release",
            ))
        }
        PlotHandle::PlotChild(_, _) => copy_plot_child_to_parent(source_handle, target)
            .map_err(|err| map_copy_error(&COPYOBJ_ERROR_INTERNAL, err)),
        _ => Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "only plot-child graphics objects can be copied in this release",
        )),
    }
}

fn handle_list(value: &Value, role: &'static str) -> BuiltinResult<HandleList> {
    match value {
        Value::Tensor(tensor) => {
            let handles = if let Some(storage) = tensor.integer_storage() {
                storage
                    .exact_values()
                    .iter()
                    .map(|value| exact_numeric_handle(value, role))
                    .collect::<BuiltinResult<Vec<_>>>()?
            } else {
                tensor_utils::tensor_values_f64(tensor)
            };
            Ok(HandleList {
                handles,
                shape: tensor.shape.clone(),
                is_scalar: tensor_utils::tensor_element_len(tensor) == 1,
            })
        }
        value => Ok(HandleList {
            handles: vec![handle_scalar(value, role)?],
            shape: vec![1, 1],
            is_scalar: true,
        }),
    }
}

fn parent_target_list(value: &Value) -> BuiltinResult<ParentTargetList> {
    let handles = handle_list(value, "parent")?;
    let mut targets = Vec::with_capacity(handles.handles.len());
    for handle in handles.handles {
        targets.push(copy_parent_target(handle)?);
    }
    Ok(ParentTargetList {
        targets,
        shape: handles.shape,
        is_scalar: handles.is_scalar,
    })
}

fn copy_parent_target(handle: f64) -> BuiltinResult<CopyParentTarget> {
    let value = Value::Num(handle);
    match resolve_plot_handle(&value, BUILTIN_NAME)
        .map_err(|err| copyobj_error(&COPYOBJ_ERROR_INVALID_ARGUMENT, err.message))?
    {
        PlotHandle::Axes(handle, axes_index) => Ok(CopyParentTarget::Axes(handle, axes_index)),
        PlotHandle::Figure(_)
        | PlotHandle::PlotChild(_, _)
        | PlotHandle::Root
        | PlotHandle::Ruler(_, _, _)
        | PlotHandle::Text(_, _, _)
        | PlotHandle::Legend(_, _) => Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "plot-child parent must be an axes handle",
        )),
    }
}

fn handle_scalar(value: &Value, role: &'static str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => exact_numeric_handle(value, role),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                exact_numeric_handle(
                    &storage
                        .value_at(0)
                        .expect("scalar integer storage has one value"),
                    role,
                )
            } else {
                Ok(tensor_utils::tensor_value_f64(tensor, 0))
            }
        }
        _ => Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            format!("expected {role} graphics handle or numeric handle array"),
        )),
    }
}

fn exact_numeric_handle(value: &IntValue, role: &'static str) -> BuiltinResult<f64> {
    numeric_handle_from_integer(value).ok_or_else(|| {
        copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            format!("{role} integer handle must be nonnegative and exactly representable"),
        )
    })
}

fn is_host_integer_handle_alias(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn map_copy_error(error: &'static BuiltinErrorDescriptor, source: FigureError) -> RuntimeError {
    copyobj_error(error, source.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::line::line_builtin;
    use crate::builtins::plotting::lock_plot_test_context;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::{clone_figure, current_figure_handle, reset_plot_state};
    use crate::builtins::plotting::subplot::subplot_builtin;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage};

    fn tensor(data: &[f64]) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![1, data.len()]).unwrap())
    }

    fn value_string(value: Value) -> String {
        match value {
            Value::String(value) => value,
            other => panic!("expected string, got {other:?}"),
        }
    }

    fn value_num(value: Value) -> f64 {
        match value {
            Value::Num(value) => value,
            other => panic!("expected number, got {other:?}"),
        }
    }

    fn line_handle(y: &[f64]) -> f64 {
        let x: Vec<f64> = (1..=y.len()).map(|index| index as f64).collect();
        value_num(
            block_on(line_builtin(vec![
                Value::String("XData".into()),
                tensor(&x),
                Value::String("YData".into()),
                tensor(y),
            ]))
            .unwrap(),
        )
    }

    #[test]
    fn copyobj_handle_scalar_reads_typed_integer_storage_exactly() {
        let tensor =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U32(vec![7]), vec![1, 1]).unwrap();

        assert_eq!(
            handle_scalar(&Value::Tensor(tensor), "source").unwrap(),
            7.0
        );
    }

    #[test]
    fn copyobj_handle_list_reads_typed_integer_storage_exactly() {
        let tensor =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U32(vec![3, 5]), vec![1, 2])
                .unwrap();

        let list = handle_list(&Value::Tensor(tensor), "source").expect("handle list");

        assert_eq!(list.handles, vec![3.0, 5.0]);
        assert_eq!(list.shape, vec![1, 2]);
        assert!(!list.is_scalar);
    }

    #[test]
    fn copyobj_integer_capabilities_cover_alias_and_resident_policies() {
        assert_eq!(COPYOBJ_INTEGER_CAPABILITIES.len(), 3);
        for capability in &COPYOBJ_INTEGER_CAPABILITIES[..2] {
            let input = &capability.inputs[0];
            assert_eq!(input.classes.len(), 8);
            assert_eq!(
                input.availability,
                BuiltinIntegerInputAvailability::RunMatOnly
            );
            assert_eq!(
                input.scalar_double,
                BuiltinIntegerScalarDoubleRule::NotApplicable
            );
            assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
        }
        let resident = &COPYOBJ_INTEGER_CAPABILITIES[2];
        assert!(resident.inputs[0].classes.is_empty());
        assert_eq!(
            resident.inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
        assert_eq!(
            resident.inputs[0].scalar_double,
            BuiltinIntegerScalarDoubleRule::NotApplicable
        );
        assert_eq!(resident.backend, BuiltinIntegerBackendRule::GpuRestricted);
        assert_eq!(
            COPYOBJ_INTEGER_HANDLE_EXTENSION.id,
            "copyobj-integer-handle-aliases"
        );
    }

    #[test]
    fn copyobj_reads_all_integer_handle_classes_from_authoritative_storage() {
        let storages = [
            IntegerStorage::I8(vec![3, 5]),
            IntegerStorage::I16(vec![3, 5]),
            IntegerStorage::I32(vec![3, 5]),
            IntegerStorage::I64(vec![3, 5]),
            IntegerStorage::U8(vec![3, 5]),
            IntegerStorage::U16(vec![3, 5]),
            IntegerStorage::U32(vec![3, 5]),
            IntegerStorage::U64(vec![3, 5]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer handles");
            let list = handle_list(&Value::Tensor(tensor), "source").expect("handle aliases");
            assert_eq!(list.handles, vec![3.0, 5.0]);
        }
        assert!(exact_numeric_handle(&IntValue::U64((1_u64 << 53) + 1), "source").is_err());
        assert_eq!(
            exact_numeric_handle(&IntValue::U64(1_u64 << 54), "source").unwrap(),
            (1_u64 << 54) as f64
        );
        assert!(exact_numeric_handle(&IntValue::U64(u64::MAX), "source").is_err());
        assert!(exact_numeric_handle(&IntValue::I64(-1), "source").is_err());
    }

    #[test]
    fn copyobj_aligned_wide_integer_alias_reaches_ordinary_handle_lookup() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = copyobj_builtin(vec![
            Value::Int(IntValue::U64(1_u64 << 54)),
            Value::Num(1.0),
        ])
        .expect_err("wide exact alias is not a registered graphics handle");
        assert_eq!(
            error.identifier(),
            COPYOBJ_ERROR_INVALID_ARGUMENT.identifier
        );
        assert!(!error.message.contains("exactly representable"));
    }

    #[test]
    fn copyobj_validates_call_shape_before_integer_extension_gate() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = copyobj_builtin(vec![
            Value::Int(IntValue::U8(1)),
            Value::Num(1.0),
            Value::Num(1.0),
        ])
        .expect_err("malformed call must fail before extension policy");
        assert_eq!(
            error.identifier(),
            COPYOBJ_ERROR_INVALID_ARGUMENT.identifier
        );
    }

    #[test]
    fn copyobj_integer_handle_aliases_are_compatibility_gated() {
        let _guard = lock_plot_test_context();
        reset_plot_state();
        let source = line_handle(&[1.0, 2.0]);
        let target_axes =
            subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let integer_source = Value::Int(IntValue::U64(source as u64));

        let compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = copyobj_builtin(vec![integer_source.clone(), Value::Num(target_axes)])
            .expect_err("integer handle alias must be gated");
        assert_eq!(
            error.identifier(),
            COPYOBJ_INTEGER_HANDLE_EXTENSION.error_identifier
        );
        drop(compat);

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(copyobj_builtin(vec![integer_source, Value::Num(target_axes)]).is_ok());
    }

    #[test]
    fn copyobj_rejects_all_resident_integer_classes_without_provider_dispatch() {
        for (offset, element_type) in [
            (0, runmat_accelerate_api::IntegerElementType::I8),
            (1, runmat_accelerate_api::IntegerElementType::I16),
            (2, runmat_accelerate_api::IntegerElementType::I32),
            (3, runmat_accelerate_api::IntegerElementType::I64),
            (4, runmat_accelerate_api::IntegerElementType::U8),
            (5, runmat_accelerate_api::IntegerElementType::U16),
            (6, runmat_accelerate_api::IntegerElementType::U32),
            (7, runmat_accelerate_api::IntegerElementType::U64),
        ] {
            let handle = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: u32::MAX,
                buffer_id: u64::MAX - 510 - offset,
                descriptor: Default::default(),
            };
            runmat_accelerate_api::set_handle_integer_type(&handle, element_type);
            let result = copyobj_builtin(vec![Value::GpuTensor(handle.clone()), Value::Num(1.0)]);
            runmat_accelerate_api::clear_handle_integer_type(&handle);
            let error = result.expect_err("resident numeric handles must reject");
            assert_eq!(
                error.identifier(),
                COPYOBJ_ERROR_INVALID_ARGUMENT.identifier
            );
        }
    }

    #[test]
    fn copyobj_rejects_resident_integer_parent_without_provider_dispatch() {
        let _guard = lock_plot_test_context();
        reset_plot_state();
        let source = line_handle(&[1.0, 2.0]);
        for (offset, element_type) in [
            (0, runmat_accelerate_api::IntegerElementType::I8),
            (1, runmat_accelerate_api::IntegerElementType::I16),
            (2, runmat_accelerate_api::IntegerElementType::I32),
            (3, runmat_accelerate_api::IntegerElementType::I64),
            (4, runmat_accelerate_api::IntegerElementType::U8),
            (5, runmat_accelerate_api::IntegerElementType::U16),
            (6, runmat_accelerate_api::IntegerElementType::U32),
            (7, runmat_accelerate_api::IntegerElementType::U64),
        ] {
            let handle = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: u32::MAX,
                buffer_id: u64::MAX - 610 - offset,
                descriptor: Default::default(),
            };
            runmat_accelerate_api::set_handle_integer_type(&handle, element_type);
            let result =
                copyobj_builtin(vec![Value::Num(source), Value::GpuTensor(handle.clone())]);
            runmat_accelerate_api::clear_handle_integer_type(&handle);
            let error = result.expect_err("resident integer parent must reject");
            assert_eq!(
                error.identifier(),
                COPYOBJ_ERROR_INVALID_ARGUMENT.identifier
            );
        }
    }

    #[test]
    fn copyobj_copies_line_to_axes_parent() {
        let _guard = lock_plot_test_context();
        reset_plot_state();

        let source = value_num(
            block_on(line_builtin(vec![
                Value::String("XData".into()),
                tensor(&[1.0, 2.0, 3.0]),
                Value::String("YData".into()),
                tensor(&[2.0, 4.0, 8.0]),
                Value::String("DisplayName".into()),
                Value::String("source".into()),
            ]))
            .unwrap(),
        );
        let target_axes =
            subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();

        let copied = value_num(
            copyobj_builtin(vec![Value::Num(source), Value::Num(target_axes)])
                .expect("copy line to axes"),
        );

        assert_ne!(copied, source);
        assert_eq!(
            value_string(
                get_builtin(vec![Value::Num(copied), Value::String("Type".into())]).unwrap()
            ),
            "line"
        );
        assert_eq!(
            value_num(
                get_builtin(vec![Value::Num(copied), Value::String("Parent".into())]).unwrap()
            ),
            target_axes
        );
        assert_eq!(
            value_string(
                get_builtin(vec![
                    Value::Num(copied),
                    Value::String("DisplayName".into())
                ])
                .unwrap()
            ),
            "source"
        );
        set_builtin(vec![
            Value::Num(copied),
            Value::String("DisplayName".into()),
            Value::String("copy".into()),
        ])
        .unwrap();
        assert_eq!(
            value_string(
                get_builtin(vec![
                    Value::Num(source),
                    Value::String("DisplayName".into())
                ])
                .unwrap()
            ),
            "source"
        );
    }

    #[test]
    fn copyobj_preserves_handle_array_shape() {
        let _guard = lock_plot_test_context();
        reset_plot_state();

        let first = line_handle(&[1.0, 2.0]);
        let second = line_handle(&[3.0, 4.0]);
        let target_axes =
            subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let sources = Tensor::new(vec![first, second], vec![1, 2]).unwrap();

        let copied = copyobj_builtin(vec![Value::Tensor(sources), Value::Num(target_axes)])
            .expect("copy handle array");
        let Value::Tensor(copied) = copied else {
            panic!("expected tensor result");
        };
        assert_eq!(copied.shape, vec![1, 2]);
        assert_eq!(copied.materialize_f64().len(), 2);
        assert!(copied
            .materialize_f64()
            .iter()
            .all(|&handle| handle != first && handle != second));
        for handle in copied.materialize_f64() {
            assert_eq!(
                value_num(
                    get_builtin(vec![Value::Num(handle), Value::String("Parent".into())]).unwrap()
                ),
                target_axes
            );
        }
    }

    #[test]
    fn copyobj_copies_scalar_source_to_parent_array() {
        let _guard = lock_plot_test_context();
        reset_plot_state();

        let source = line_handle(&[1.0, 2.0, 3.0]);
        let first_axes =
            subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(1.0)).unwrap();
        let second_axes =
            subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let parents = Tensor::new(vec![first_axes, second_axes], vec![1, 2]).unwrap();

        let copied = copyobj_builtin(vec![Value::Num(source), Value::Tensor(parents)])
            .expect("copy to parent array");
        let Value::Tensor(copied) = copied else {
            panic!("expected tensor result");
        };

        assert_eq!(copied.shape, vec![1, 2]);
        assert_eq!(
            value_num(
                get_builtin(vec![
                    Value::Num(copied.materialize_f64()[0]),
                    Value::String("Parent".into())
                ])
                .unwrap()
            ),
            first_axes
        );
        assert_eq!(
            value_num(
                get_builtin(vec![
                    Value::Num(copied.materialize_f64()[1]),
                    Value::String("Parent".into())
                ])
                .unwrap()
            ),
            second_axes
        );
    }

    #[test]
    fn copyobj_requires_parent_and_does_not_partially_copy_invalid_arrays() {
        let _guard = lock_plot_test_context();
        reset_plot_state();

        let source = line_handle(&[1.0, 2.0]);
        let target_axes =
            subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();

        let err = copyobj_builtin(vec![Value::Num(source)]).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:copyobj:InvalidArgument")
        );

        let before_len = clone_figure(current_figure_handle()).unwrap().len();
        let sources = Tensor::new(vec![source, f64::NAN], vec![1, 2]).unwrap();
        let err = copyobj_builtin(vec![Value::Tensor(sources), Value::Num(target_axes)])
            .expect_err("invalid source array should fail");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:copyobj:InvalidArgument")
        );
        let after_len = clone_figure(current_figure_handle()).unwrap().len();
        assert_eq!(after_len, before_len);
    }

    #[test]
    fn copyobj_rejects_figure_parent_for_plot_child() {
        let _guard = lock_plot_test_context();
        reset_plot_state();

        let source = line_handle(&[1.0, 2.0]);
        let figure = current_figure_handle().as_u32() as f64;
        let err = copyobj_builtin(vec![Value::Num(source), Value::Num(figure)]).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:copyobj:InvalidArgument")
        );
    }
}
