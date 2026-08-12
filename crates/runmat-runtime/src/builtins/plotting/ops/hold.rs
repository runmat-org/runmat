//! MATLAB-compatible `hold` builtin.

use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::op_common::cmd_parsing::parse_hold_mode;
use super::state::{set_hold, HoldMode};

const HOLD_TYPED_INTEGER_STATE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "hold-typed-integer-state",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "hold accepts typed integer 0 and 1 only as a RunMat extension because the public numeric-state documentation does not enumerate integer storage classes",
        error_identifier: Some("RunMat:compatibility:HoldTypedIntegerStateExtension"),
    };
const HOLD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [HOLD_TYPED_INTEGER_STATE_EXTENSION];
const HOLD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "state",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are inspected exactly and only values 0 and 1 are accepted behind the typed-state extension.",
}];
pub const HOLD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "hold(integer_state)", inputs: &HOLD_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "[integer-audit-open] The state is read without floating conversion. hold has no public output and mutates plotting state only after validation. Documented axes-target and matlab.lang.OnOffSwitchState forms remain implementation gaps and are not claimed by this integer-role record." }];

const HOLD_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const HOLD_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Hold mode ('on'|'off'|'all') or numeric/logical scalar.",
}];

const HOLD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "hold()",
        inputs: &HOLD_INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "hold(mode)",
        inputs: &HOLD_INPUTS_MODE,
        outputs: &[],
    },
];

const HOLD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HOLD.INVALID_ARGUMENT",
    identifier: Some("RunMat:hold:InvalidArgument"),
    when: "Hold mode argument is unsupported.",
    message: "hold: invalid argument",
};

const HOLD_ERRORS: [BuiltinErrorDescriptor; 1] = [HOLD_ERROR_INVALID_ARGUMENT];

pub const HOLD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HOLD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HOLD_ERRORS,
};

#[runtime_builtin(
    name = "hold",
    category = "plotting",
    summary = "Control plot replacement versus appending.",
    keywords = "hold,plotting",
    suppress_auto_output = true,
    extensions(HOLD_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::hold::HOLD_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::plotting::hold::HOLD_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::hold"
)]
pub fn hold_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.iter().any(|value| {
        matches!(value, Value::Int(_))
            || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HOLD_TYPED_INTEGER_STATE_EXTENSION,
            "hold",
        )?;
    }
    let mode = match rest.as_slice() {
        [] => HoldMode::Toggle,
        [value] => parse_hold_mode(value)?,
        _ => {
            return Err(crate::builtins::plotting::plotting_error(
                "hold",
                "hold: expected at most one input",
            ))
        }
    };
    set_hold(mode);
    Ok(Value::OutputList(Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = HOLD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"hold()"));
        assert!(labels.contains(&"hold(mode)"));
        assert!(HOLD_DESCRIPTOR
            .signatures
            .iter()
            .all(|signature| signature.outputs.is_empty()));
    }

    #[test]
    fn hold_toggle_and_explicit_modes_work() {
        let _ = hold_builtin(Vec::new()).unwrap();
        assert_eq!(
            hold_builtin(vec![Value::String("on".into())]).unwrap(),
            Value::OutputList(Vec::new())
        );
        assert_eq!(
            hold_builtin(vec![Value::String("off".into())]).unwrap(),
            Value::OutputList(Vec::new())
        );
        assert!(hold_builtin(vec![Value::from("on"), Value::from("off")]).is_err());
    }

    #[test]
    fn hold_typed_integer_state_is_exact_and_explicitly_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = hold_builtin(vec![Value::Int(runmat_builtins::IntValue::U8(1))])
            .expect_err("typed state must be gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:HoldTypedIntegerStateExtension")
        );
    }
}
