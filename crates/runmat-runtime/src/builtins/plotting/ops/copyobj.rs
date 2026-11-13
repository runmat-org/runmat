//! MATLAB-compatible `copyobj` graphics-object copying builtin.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    COPYOBJ_DESCRIPTOR, COPYOBJ_ERROR_INTERNAL, COPYOBJ_ERROR_INVALID_ARGUMENT,
    COPYOBJ_ERROR_TOO_MANY_OUTPUTS, COPYOBJ_INTEGER_HANDLE_EXTENSION,
};

use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, Tensor, Value};

use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    copy_plot_child_to_parent, validate_plot_child_copy_source, CopyParentTarget, FigureError,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::handles::numeric_handle_from_integer;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "copyobj";

fn copyobj_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "copyobj",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::copyobj"
)]
pub fn copyobj_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let _scope = runtime.as_ref().map(|runtime| runtime.enter());
    crate::output_count::validate_fixed_outputs(
        &COPYOBJ_DESCRIPTOR,
        &COPYOBJ_ERROR_TOO_MANY_OUTPUTS,
        "copyobj",
    )
    .map_err(super::coordinate_admission::terminal)?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "copyobj")?;
    work.finish(copyobj_impl(arguments.view(), &work))
}

fn copyobj_impl(
    args: crate::call::FunctionArgumentView<'_>,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> BuiltinResult<Value> {
    let mut iter = args.iter();
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

    if is_host_integer_handle_alias(source) || is_host_integer_handle_alias(parent) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COPYOBJ_INTEGER_HANDLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    let sources = handle_list(source, "source")?;
    let parents = parent_target_list(parent)?;
    let plan = copy_plan(&sources, &parents)?;

    for &(source_handle, _) in &plan.pairs {
        validate_copy_source(source_handle)?;
    }

    work.check()?;
    let mut copied = Vec::with_capacity(plan.pairs.len());
    for (source_handle, target) in plan.pairs {
        work.check()?;
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
        let CopyParentTarget::Axes(figure, axes) = target;
        if pairs.iter().any(|&(prior, prior_target)| {
            let CopyParentTarget::Axes(prior_figure, prior_axes) = prior_target;
            prior == source && prior_figure == figure && prior_axes == axes
        }) {
            return Err(copyobj_error(
                &COPYOBJ_ERROR_INVALID_ARGUMENT,
                "the same object cannot be copied twice to the same parent in one request",
            ));
        }
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
                shape: tensor.shape().to_vec(),
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
    let mut builder = build_runtime_error(format!("{}: {}", error.message, source))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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
    use runmat_builtins::{
        BuiltinIntegerBackendRule, BuiltinIntegerInputAvailability, BuiltinIntegerScalarDoubleRule,
    };
    use runmat_builtins::{
        COPYOBJ_ERROR_INVALID_ARGUMENT, COPYOBJ_INTEGER_CAPABILITIES,
        COPYOBJ_INTEGER_HANDLE_EXTENSION,
    };
    use runmat_value::{IntValue, IntegerStorage};

    fn tensor(data: &[f64]) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![1, data.len()]).unwrap())
    }

    fn value_string(value: Value) -> String {
        match value {
            Value::String(runmat_value::StringElement::Text(value)) => value.to_string(),
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
            block_on(line_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("XData".into()),
                    tensor(&x),
                    Value::String("YData".into()),
                    tensor(y),
                ]),
            ))
            .unwrap(),
        )
    }

    #[test]
    fn copyobj_handle_scalar_reads_typed_integer_storage_exactly() {
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::U32(vec![7]), vec![1, 1]).unwrap();

        assert_eq!(
            handle_scalar(&Value::Tensor(tensor), "source").unwrap(),
            7.0
        );
    }

    #[test]
    fn copyobj_handle_list_reads_typed_integer_storage_exactly() {
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::U32(vec![3, 5]), vec![1, 2]).unwrap();

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
        let error = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::U64(1_u64 << 54)),
                Value::Num(1.0),
            ]),
        )
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
        let error = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::U8(1)),
                Value::Num(1.0),
                Value::Num(1.0),
            ]),
        )
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
        let error = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                integer_source.clone(),
                Value::Num(target_axes),
            ]),
        )
        .expect_err("integer handle alias must be gated");
        assert_eq!(
            error.identifier(),
            COPYOBJ_INTEGER_HANDLE_EXTENSION.error_identifier
        );
        drop(compat);

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                integer_source,
                Value::Num(target_axes)
            ])
        )
        .is_ok());
    }

    #[test]
    fn copyobj_rejects_all_resident_integer_classes_without_provider_dispatch() {
        for (_offset, element_type) in [
            (0, runmat_accelerate_api::IntegerElementType::I8),
            (1, runmat_accelerate_api::IntegerElementType::I16),
            (2, runmat_accelerate_api::IntegerElementType::I32),
            (3, runmat_accelerate_api::IntegerElementType::I64),
            (4, runmat_accelerate_api::IntegerElementType::U8),
            (5, runmat_accelerate_api::IntegerElementType::U16),
            (6, runmat_accelerate_api::IntegerElementType::U32),
            (7, runmat_accelerate_api::IntegerElementType::U64),
        ] {
            let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    element_type.into(),
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap()
                .handle;
            let result = copyobj_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::GpuTensor(handle.clone()),
                    Value::Num(1.0),
                ]),
            );
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
        for (_offset, element_type) in [
            (0, runmat_accelerate_api::IntegerElementType::I8),
            (1, runmat_accelerate_api::IntegerElementType::I16),
            (2, runmat_accelerate_api::IntegerElementType::I32),
            (3, runmat_accelerate_api::IntegerElementType::I64),
            (4, runmat_accelerate_api::IntegerElementType::U8),
            (5, runmat_accelerate_api::IntegerElementType::U16),
            (6, runmat_accelerate_api::IntegerElementType::U32),
            (7, runmat_accelerate_api::IntegerElementType::U64),
        ] {
            let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    element_type.into(),
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap()
                .handle;
            let result = copyobj_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(source),
                    Value::GpuTensor(handle.clone()),
                ]),
            );
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
            block_on(line_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("XData".into()),
                    tensor(&[1.0, 2.0, 3.0]),
                    Value::String("YData".into()),
                    tensor(&[2.0, 4.0, 8.0]),
                    Value::String("DisplayName".into()),
                    Value::String("source".into()),
                ]),
            ))
            .unwrap(),
        );
        let target_axes =
            subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();

        let copied = value_num(
            copyobj_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(source),
                    Value::Num(target_axes),
                ]),
            )
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

        let copied = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(sources),
                Value::Num(target_axes),
            ]),
        )
        .expect("copy handle array");
        let Value::Tensor(copied) = copied else {
            panic!("expected tensor result");
        };
        assert_eq!(copied.shape(), [1, 2]);
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
        // Public vectors pair by length, not by stored row/column orientation.
        let row_sources = Tensor::new(vec![first, second], vec![1, 2]).unwrap();
        let column_parents = Tensor::new(vec![target_axes, target_axes], vec![2, 1]).unwrap();
        let paired = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(row_sources),
                Value::Tensor(column_parents),
            ]),
        )
        .unwrap();
        let Value::Tensor(paired) = paired else {
            panic!("expected paired handles");
        };
        assert_eq!(paired.shape(), [1, 2]);
        let children = || {
            let Value::Tensor(handles) = get_builtin(vec![
                Value::Num(target_axes),
                Value::String("Children".into()),
            ])
            .unwrap() else {
                panic!("expected children");
            };
            handles.len()
        };
        let before = children();
        let duplicate = Tensor::new(vec![first, first], vec![1, 2]).unwrap();
        let error = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(duplicate),
                Value::Num(target_axes),
            ]),
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:copyobj:InvalidArgument"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(children(), before);
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

        let copied = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(source),
                Value::Tensor(parents),
            ]),
        )
        .expect("copy to parent array");
        let Value::Tensor(copied) = copied else {
            panic!("expected tensor result");
        };

        assert_eq!(copied.shape(), [1, 2]);
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

        let err = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(source)]),
        )
        .unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:copyobj:InvalidArgument")
        );

        let before_len = clone_figure(current_figure_handle()).unwrap().len();
        let sources = Tensor::new(vec![source, f64::NAN], vec![1, 2]).unwrap();
        let err = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(sources),
                Value::Num(target_axes),
            ]),
        )
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
        let err = copyobj_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(source),
                Value::Num(figure),
            ]),
        )
        .unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:copyobj:InvalidArgument")
        );
    }
}
