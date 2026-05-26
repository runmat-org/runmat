//! MATLAB-compatible `figure` builtin for selecting/creating plotting windows.

use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::op_common::handles::parse_optional_figure_handle;
use super::state::{new_figure_handle, select_figure};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const FIGURE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure handle.",
}];

const FIGURE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const FIGURE_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"next\""),
    description: "Figure handle or 'next' to create/select the next figure.",
}];

const FIGURE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "fig = figure()",
        inputs: &FIGURE_INPUTS_NONE,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "fig = figure(h)",
        inputs: &FIGURE_INPUTS_HANDLE,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
];

const FIGURE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE.INVALID_ARGUMENT",
    identifier: Some("RunMat:figure:InvalidArgument"),
    when: "Provided figure handle argument is invalid.",
    message: "figure: invalid argument",
};

const FIGURE_ERRORS: [BuiltinErrorDescriptor; 1] = [FIGURE_ERROR_INVALID_ARGUMENT];

pub const FIGURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIGURE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FIGURE_ERRORS,
};

#[runtime_builtin(
    name = "figure",
    category = "plotting",
    summary = "Create or select a plotting figure.",
    keywords = "figure,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::figure::FIGURE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::figure"
)]
pub fn figure_builtin(rest: Vec<Value>) -> crate::BuiltinResult<f64> {
    let handle = if rest.is_empty() {
        new_figure_handle()
    } else {
        match parse_optional_figure_handle(&rest[0], "figure")? {
            Some(handle) => {
                select_figure(handle);
                handle
            }
            None => new_figure_handle(),
        }
    };
    Ok(handle.as_u32() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, current_figure_handle, reset_hold_state_for_run,
    };

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn figure_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FIGURE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"fig = figure()"));
        assert!(labels.contains(&"fig = figure(h)"));
    }

    #[test]
    fn figure_creates_and_selects_handles() {
        let _guard = setup();
        let first = figure_builtin(Vec::new()).unwrap();
        assert!(first > 0.0);
        let selected = figure_builtin(vec![Value::Num(first)]).unwrap();
        assert_eq!(selected, first);
        assert_eq!(current_figure_handle().as_u32() as f64, first);
    }
}
