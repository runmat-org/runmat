//! MATLAB-compatible `copyobj` graphics-object copying builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    copy_plot_child_to_parent, validate_plot_child_copy_source, CopyParentTarget, FigureError,
};
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
        description: "Destination axes or figure handle.",
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
    summary = "Copy graphics objects to an axes or figure parent.",
    keywords = "copyobj,copy,graphics,handle,axes,figure,plotting",
    suppress_auto_output = true,
    type_resolver(handle_array_type),
    descriptor(crate::builtins::plotting::copyobj::COPYOBJ_DESCRIPTOR),
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
        PlotHandle::PlotChild(_) => validate_plot_child_copy_source(source_handle)
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
        PlotHandle::PlotChild(_) => copy_plot_child_to_parent(source_handle, target)
            .map_err(|err| map_copy_error(&COPYOBJ_ERROR_INTERNAL, err)),
        _ => Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "only plot-child graphics objects can be copied in this release",
        )),
    }
}

fn handle_list(value: &Value, role: &'static str) -> BuiltinResult<HandleList> {
    match value {
        Value::Tensor(tensor) => Ok(HandleList {
            handles: tensor.data.clone(),
            shape: tensor.shape.clone(),
            is_scalar: tensor.data.len() == 1,
        }),
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
        PlotHandle::Figure(handle) => Ok(CopyParentTarget::Figure(handle)),
        PlotHandle::Axes(handle, axes_index) => Ok(CopyParentTarget::Axes(handle, axes_index)),
        PlotHandle::PlotChild(_)
        | PlotHandle::Root
        | PlotHandle::Ruler(_, _, _)
        | PlotHandle::Text(_, _, _)
        | PlotHandle::Legend(_, _) => Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            "parent must be an axes or figure handle",
        )),
    }
}

fn handle_scalar(value: &Value, role: &'static str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        _ => Err(copyobj_error(
            &COPYOBJ_ERROR_INVALID_ARGUMENT,
            format!("expected {role} graphics handle or numeric handle array"),
        )),
    }
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
        assert_eq!(copied.data.len(), 2);
        assert!(copied
            .data
            .iter()
            .all(|&handle| handle != first && handle != second));
        for handle in copied.data {
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
                    Value::Num(copied.data[0]),
                    Value::String("Parent".into())
                ])
                .unwrap()
            ),
            first_axes
        );
        assert_eq!(
            value_num(
                get_builtin(vec![
                    Value::Num(copied.data[1]),
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
    fn copyobj_rejects_non_axes_parent() {
        let _guard = lock_plot_test_context();
        reset_plot_state();

        let source = line_handle(&[1.0, 2.0]);
        let err = copyobj_builtin(vec![Value::Num(source), Value::Num(source)]).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:copyobj:InvalidArgument")
        );
    }
}
