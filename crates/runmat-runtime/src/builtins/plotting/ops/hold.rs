//! MATLAB-compatible `hold` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::axes_target::{apply_axes_target, split_leading_axes_handle};
use super::op_common::cmd_parsing::parse_hold_mode;
use super::state::{set_hold, HoldMode};

const HOLD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "state",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Documented numeric scalar states accept every built-in integer class; all classes are inspected exactly and only values 0 and 1 are accepted.",
}];
pub const HOLD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "hold([ax,] integer_state)", inputs: &HOLD_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The scalar state is read without floating conversion and mutates only the selected axes hold policy after validation." }];

const HOLD_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const HOLD_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Hold mode ('on'|'off'|'all') or numeric/logical scalar.",
}];

const HOLD_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Hold mode ('on'|'off'|'all') or numeric/logical scalar.",
    },
];

const HOLD_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
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
    BuiltinSignatureDescriptor {
        label: "hold(ax, mode)",
        inputs: &HOLD_INPUTS_AX_MODE,
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
    integer_capabilities(crate::builtins::plotting::hold::HOLD_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::plotting::hold::HOLD_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::hold"
)]
pub fn hold_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (axes_target, rest) = split_leading_axes_handle(args, "hold")?;
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
    apply_axes_target(axes_target, "hold")?;
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
    fn hold_typed_integer_state_is_exact_in_compatibility_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        assert!(hold_builtin(vec![Value::Int(runmat_value::IntValue::U8(1))]).is_ok());
        assert!(hold_builtin(vec![Value::Int(runmat_value::IntValue::I64(0))]).is_ok());
    }
}
