//! MATLAB-compatible `axes` builtin.

use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::plotting_error;
use super::properties::{map_figure_error, resolve_plot_handle, set_properties, PlotHandle};
use super::state::{
    create_axes_for_figure, encode_axes_handle, select_axes_for_figure, FigureHandle,
};
use super::style::value_as_string;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const BUILTIN_NAME: &str = "axes";

const AXES_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes handle.",
}];

const AXES_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const AXES_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Existing axes handle to make current, or figure handle to parent new axes.",
}];

const AXES_INPUTS_PAIRS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "properties",
    ty: BuiltinParamType::PropertyName,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Axes property/value pairs. The construction-time Parent property may target a figure.",
}];

const AXES_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "ax = axes()",
        inputs: &AXES_INPUTS_NONE,
        outputs: &AXES_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ax = axes(target)",
        inputs: &AXES_INPUTS_HANDLE,
        outputs: &AXES_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ax = axes(property, value, ...)",
        inputs: &AXES_INPUTS_PAIRS,
        outputs: &AXES_OUTPUT_HANDLE,
    },
];

const AXES_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AXES.INVALID_ARGUMENT",
    identifier: Some("RunMat:axes:InvalidArgument"),
    when: "Unsupported handle or property/value arguments are provided.",
    message: "axes: invalid argument",
};

const AXES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AXES.INTERNAL",
    identifier: Some("RunMat:axes:Internal"),
    when: "Internal axes state operation fails.",
    message: "axes: internal operation failed",
};

const AXES_ERRORS: [BuiltinErrorDescriptor; 2] = [AXES_ERROR_INVALID_ARGUMENT, AXES_ERROR_INTERNAL];

pub const AXES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AXES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &AXES_ERRORS,
};

#[runtime_builtin(
    name = "axes",
    category = "plotting",
    summary = "Create axes or make existing axes current.",
    keywords = "axes,graphics,plotting,current axes",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::axes::AXES_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::axes"
)]
pub fn axes_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target, property_args) = split_axes_target(args)?;
    match target {
        Some(AxesTarget::ExistingAxes(handle, axes_index)) => {
            select_axes_for_figure(handle, axes_index)
                .map_err(|err| map_figure_error(BUILTIN_NAME, err))?;
            if !property_args.is_empty() {
                set_properties(
                    PlotHandle::Axes(handle, axes_index),
                    &property_args,
                    BUILTIN_NAME,
                )?;
            }
            Ok(encode_axes_handle(handle, axes_index))
        }
        Some(AxesTarget::ParentFigure(handle)) => {
            create_and_configure_axes(Some(handle), property_args)
        }
        None => {
            let (parent, filtered) = extract_parent_property(property_args)?;
            create_and_configure_axes(parent, filtered)
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AxesTarget {
    ExistingAxes(FigureHandle, usize),
    ParentFigure(FigureHandle),
}

fn split_axes_target(args: Vec<Value>) -> crate::BuiltinResult<(Option<AxesTarget>, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if starts_property_pairs(&first) {
        let mut property_args = vec![first];
        property_args.extend(iter);
        return Ok((None, property_args));
    }
    match resolve_plot_handle(&first, BUILTIN_NAME) {
        Ok(PlotHandle::Axes(handle, axes_index)) => Ok((
            Some(AxesTarget::ExistingAxes(handle, axes_index)),
            iter.collect(),
        )),
        Ok(PlotHandle::Figure(handle)) => {
            Ok((Some(AxesTarget::ParentFigure(handle)), iter.collect()))
        }
        Ok(_) => Err(plotting_error(
            BUILTIN_NAME,
            "axes: target must be an axes or figure handle",
        )),
        Err(err) if iter.len() == 0 => Err(err),
        Err(_) => {
            let mut property_args = vec![first];
            property_args.extend(iter);
            Ok((None, property_args))
        }
    }
}

fn create_and_configure_axes(
    parent: Option<FigureHandle>,
    property_args: Vec<Value>,
) -> crate::BuiltinResult<f64> {
    if !property_args.len().is_multiple_of(2) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "axes: property/value arguments must come in pairs",
        ));
    }
    let (handle, axes_index) =
        create_axes_for_figure(parent).map_err(|err| map_figure_error(BUILTIN_NAME, err))?;
    if !property_args.is_empty() {
        set_properties(
            PlotHandle::Axes(handle, axes_index),
            &property_args,
            BUILTIN_NAME,
        )?;
    }
    Ok(encode_axes_handle(handle, axes_index))
}

fn extract_parent_property(
    args: Vec<Value>,
) -> crate::BuiltinResult<(Option<FigureHandle>, Vec<Value>)> {
    if !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "axes: property/value arguments must come in pairs",
        ));
    }
    let mut parent = None;
    let mut filtered = Vec::with_capacity(args.len());
    for pair in args.chunks_exact(2) {
        if property_name_is(&pair[0], "parent") {
            parent = Some(parent_figure_from_value(&pair[1])?);
        } else {
            filtered.push(pair[0].clone());
            filtered.push(pair[1].clone());
        }
    }
    Ok((parent, filtered))
}

fn parent_figure_from_value(value: &Value) -> crate::BuiltinResult<FigureHandle> {
    match resolve_plot_handle(value, BUILTIN_NAME)? {
        PlotHandle::Figure(handle) => Ok(handle),
        _ => Err(plotting_error(
            BUILTIN_NAME,
            "axes: Parent must be a figure handle",
        )),
    }
}

fn starts_property_pairs(value: &Value) -> bool {
    value_as_string(value).is_some()
}

fn property_name_is(value: &Value, expected: &str) -> bool {
    value_as_string(value)
        .map(|name| name.trim().eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{current_figure_handle, reset_plot_state};

    #[test]
    fn axes_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = AXES_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"ax = axes()"));
        assert!(labels.contains(&"ax = axes(target)"));
        assert!(labels.contains(&"ax = axes(property, value, ...)"));
    }

    #[test]
    fn axes_creates_handle_and_sets_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();

        let handle = axes_builtin(vec![
            Value::String("Position".into()),
            Value::Tensor(
                runmat_builtins::Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4]).unwrap(),
            ),
            Value::String("Units".into()),
            Value::String("normalized".into()),
        ])
        .expect("axes");
        let props = get_builtin(vec![Value::Num(handle)]).expect("get axes");
        assert!(matches!(props, Value::Struct(_)));
    }

    #[test]
    fn axes_selects_existing_axes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();

        let first = axes_builtin(Vec::new()).expect("first axes");
        let second = axes_builtin(Vec::new()).expect("second axes");
        assert_ne!(first, second);
        let selected = axes_builtin(vec![Value::Num(first)]).expect("select first");
        assert_eq!(selected, first);
    }

    #[test]
    fn axes_parent_figure_selects_target_figure() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();

        let fig = crate::builtins::plotting::figure::figure_builtin(vec![Value::Num(9.0)])
            .expect("figure");
        let ax = axes_builtin(vec![Value::String("Parent".into()), Value::Num(fig)])
            .expect("axes parent");
        assert!(ax > fig);
        assert_eq!(current_figure_handle().as_u32(), 9);
    }
}
