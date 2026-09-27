//! Fixed-output bounds derived from the owning builtin's semantic descriptor.

use runmat_builtins::{
    BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode, BuiltinParamArity,
};

/// Check the fixed upper bound before the caller performs effects or provider work.
///
/// This does not select overloads or validate dynamic output counts. Those remain
/// with the identity that knows the arguments and requested-output semantics.
pub(crate) fn validate_fixed_outputs(
    descriptor: &'static BuiltinDescriptor,
    error: &'static BuiltinErrorDescriptor,
    builtin: &str,
) -> crate::BuiltinResult<()> {
    let Some(requested) = super::current_output_count() else {
        return Ok(());
    };
    if descriptor.output_mode != BuiltinOutputMode::Fixed
        || descriptor.signatures.iter().any(|signature| {
            signature
                .outputs
                .iter()
                .any(|output| output.arity == BuiltinParamArity::Variadic)
        })
    {
        return Ok(());
    }
    let Some(maximum) = descriptor
        .signatures
        .iter()
        .map(|signature| signature.outputs.len())
        .max()
    else {
        return Ok(());
    };
    if requested <= maximum {
        return Ok(());
    }
    let mut builder = crate::build_runtime_error(error.message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    Err(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{
        BuiltinCompletionPolicy, BuiltinParamDescriptor, BuiltinParamType,
        BuiltinSignatureDescriptor,
    };

    const ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
        code: "RM.OUTPUT_TEST.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:output_test:TooManyOutputs"),
        when: "Too many outputs requested.",
        message: "output_test: too many output arguments",
    };
    const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Result",
    };
    const FIXED: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &[BuiltinSignatureDescriptor {
            label: "value = output_test()",
            inputs: &[],
            outputs: &[OUTPUT],
        }],
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::HiddenInternal,
        errors: &[ERROR],
    };
    const DYNAMIC: BuiltinDescriptor = BuiltinDescriptor {
        output_mode: BuiltinOutputMode::ByRequestedOutputCount,
        ..FIXED
    };
    const VARIADIC: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &[BuiltinSignatureDescriptor {
            label: "[varargout] = output_test()",
            inputs: &[],
            outputs: &[BuiltinParamDescriptor {
                arity: BuiltinParamArity::Variadic,
                ..OUTPUT
            }],
        }],
        ..FIXED
    };

    #[test]
    fn fixed_bounds_preserve_zero_and_implicit_outputs() {
        for count in [None, Some(0), Some(1)] {
            let _guard = super::super::push_output_count(count);
            assert!(validate_fixed_outputs(&FIXED, &ERROR, "output_test").is_ok());
        }
        let _guard = super::super::push_output_count(Some(2));
        let error = validate_fixed_outputs(&FIXED, &ERROR, "output_test").unwrap_err();
        assert_eq!(error.identifier(), ERROR.identifier);
        assert_eq!(error.message(), ERROR.message);
    }

    #[test]
    fn dynamic_and_variadic_policies_stay_with_the_caller() {
        let _guard = super::super::push_output_count(Some(20));
        for descriptor in [&DYNAMIC, &VARIADIC] {
            assert!(validate_fixed_outputs(descriptor, &ERROR, "output_test").is_ok());
        }
    }

    #[test]
    fn overload_bound_is_the_maximum_not_the_first_signature() {
        const OVERLOADED: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[
                BuiltinSignatureDescriptor {
                    label: "value = output_test()",
                    inputs: &[],
                    outputs: &[OUTPUT],
                },
                BuiltinSignatureDescriptor {
                    label: "[first, second] = output_test(option)",
                    inputs: &[OUTPUT],
                    outputs: &[OUTPUT, OUTPUT],
                },
            ],
            ..FIXED
        };
        for count in [0, 1, 2] {
            let _guard = super::super::push_output_count(Some(count));
            assert!(validate_fixed_outputs(&OVERLOADED, &ERROR, "output_test").is_ok());
        }
        let _guard = super::super::push_output_count(Some(3));
        assert!(validate_fixed_outputs(&OVERLOADED, &ERROR, "output_test").is_err());
    }

    #[test]
    fn empty_signatures_do_not_mean_a_declared_zero_output_signature() {
        const UNDECLARED: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[],
            ..FIXED
        };
        const ZERO: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: "output_test()",
                inputs: &[],
                outputs: &[],
            }],
            ..FIXED
        };
        for count in [None, Some(0)] {
            let _guard = super::super::push_output_count(count);
            assert!(validate_fixed_outputs(&ZERO, &ERROR, "output_test").is_ok());
        }
        let _guard = super::super::push_output_count(Some(1));
        assert!(validate_fixed_outputs(&UNDECLARED, &ERROR, "output_test").is_ok());
        assert!(validate_fixed_outputs(&ZERO, &ERROR, "output_test").is_err());
    }
}
