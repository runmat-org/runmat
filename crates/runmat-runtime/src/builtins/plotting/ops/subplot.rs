//! MATLAB-compatible `subplot` builtin for selecting axes within a figure.

use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::op_common::cmd_parsing::scalar_from_value;
use super::state::{configure_subplot, current_axes_state, encode_axes_handle, FigureError};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "subplot";

const SUBPLOT_OUTPUT_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Encoded handle for the selected subplot axes.",
}];

const SUBPLOT_INPUTS_GRID: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "rows",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of subplot rows.",
    },
    BuiltinParamDescriptor {
        name: "cols",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of subplot columns.",
    },
    BuiltinParamDescriptor {
        name: "position",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "1-based subplot position in row-major order.",
    },
];

const SUBPLOT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ax = subplot(rows, cols, position)",
    inputs: &SUBPLOT_INPUTS_GRID,
    outputs: &SUBPLOT_OUTPUT_AX,
}];

const SUBPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:subplot:InvalidArgument"),
    when: "Rows/cols/position are unsupported, non-scalar, non-finite, or non-positive.",
    message: "subplot: invalid argument",
};

const SUBPLOT_ERROR_INDEX_OUT_OF_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBPLOT.INDEX_OUT_OF_RANGE",
    identifier: Some("RunMat:subplot:IndexOutOfRange"),
    when: "Position is outside the configured subplot grid.",
    message: "subplot: subplot index is out of range for the grid",
};

const SUBPLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBPLOT.INTERNAL",
    identifier: Some("RunMat:subplot:Internal"),
    when: "Internal subplot state operation fails.",
    message: "subplot: internal operation failed",
};

const SUBPLOT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SUBPLOT_ERROR_INVALID_ARGUMENT,
    SUBPLOT_ERROR_INDEX_OUT_OF_RANGE,
    SUBPLOT_ERROR_INTERNAL,
];

pub const SUBPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBPLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUBPLOT_ERRORS,
};

fn subplot_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    subplot_error_with_message(error.message, error)
}

fn subplot_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_subplot_figure_error(err: FigureError) -> RuntimeError {
    match err {
        FigureError::InvalidSubplotIndex { .. } => subplot_error(&SUBPLOT_ERROR_INDEX_OUT_OF_RANGE),
        FigureError::InvalidSubplotGrid { .. } => subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT),
        other => subplot_error_with_message(
            format!("{}: {}", SUBPLOT_ERROR_INTERNAL.message, other),
            &SUBPLOT_ERROR_INTERNAL,
        ),
    }
}

#[runtime_builtin(
    name = "subplot",
    category = "plotting",
    summary = "Select or create an axes at a subplot grid location.",
    keywords = "subplot,axes,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::subplot::SUBPLOT_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::subplot"
)]
pub fn subplot_builtin(rows: Value, cols: Value, position: Value) -> crate::BuiltinResult<f64> {
    let m = scalar_from_value(&rows, BUILTIN_NAME)
        .map_err(|_| subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT))?;
    let n = scalar_from_value(&cols, BUILTIN_NAME)
        .map_err(|_| subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT))?;
    let p = scalar_from_value(&position, BUILTIN_NAME)
        .map_err(|_| subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT))?;
    let zero_based = p.saturating_sub(1);
    configure_subplot(m, n, zero_based).map_err(map_subplot_figure_error)?;
    let axes = current_axes_state();
    Ok(encode_axes_handle(axes.handle, axes.active_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn subplot_returns_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        assert!(matches!(props, Value::Struct(_)));
    }

    #[test]
    fn subplot_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SUBPLOT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"ax = subplot(rows, cols, position)"));
    }
}
