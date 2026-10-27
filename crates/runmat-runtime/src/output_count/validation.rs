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
    if descriptor.output_mode != BuiltinOutputMode::Fixed {
        return Ok(());
    }
    validate_bounded_outputs(descriptor, error, builtin)
}

/// Opt-in bound for a descriptor whose advertised fixed signatures enumerate
/// every output prefix. Other requested-count owners may have unbounded output.
pub(crate) fn validate_bounded_outputs(
    descriptor: &'static BuiltinDescriptor,
    error: &'static BuiltinErrorDescriptor,
    builtin: &str,
) -> crate::BuiltinResult<()> {
    validate_bounded_count(descriptor, error, builtin, super::current_output_count())
}

/// Same descriptor bound under an explicitly selected invocation authority.
pub(crate) fn validate_bounded_outputs_in(
    descriptor: &'static BuiltinDescriptor,
    error: &'static BuiltinErrorDescriptor,
    builtin: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<()> {
    validate_bounded_count(
        descriptor,
        error,
        builtin,
        super::current_output_count_in(runtime),
    )
}

fn validate_bounded_count(
    descriptor: &'static BuiltinDescriptor,
    error: &'static BuiltinErrorDescriptor,
    builtin: &str,
    requested: Option<usize>,
) -> crate::BuiltinResult<()> {
    let Some(requested) = requested else {
        return Ok(());
    };
    if descriptor.signatures.iter().any(|signature| {
        signature
            .outputs
            .iter()
            .any(|output| output.arity == BuiltinParamArity::Variadic)
    }) {
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
    // Gathering inputs cannot change an output-arity contract.
    let mut builder = crate::build_runtime_error(error.message)
        .with_builtin(builtin)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    Err(builder.build())
}

/// Validate the selected fixed-arity form against the canonical descriptor.
/// A context-free Rust embedding call has no requested-output authority.
pub(crate) fn validate_fixed_form_outputs(
    descriptor: &'static BuiltinDescriptor,
    error: &'static BuiltinErrorDescriptor,
    builtin: &str,
    input_count: usize,
    requested_outputs: Option<usize>,
) -> crate::BuiltinResult<()> {
    let Some(requested) = requested_outputs else {
        return Ok(());
    };
    if descriptor.output_mode != BuiltinOutputMode::Fixed {
        return Ok(());
    }
    let matching = descriptor
        .signatures
        .iter()
        .filter(|signature| signature.inputs.len() == input_count);
    if matching.clone().any(|signature| {
        signature
            .outputs
            .iter()
            .any(|output| output.arity == BuiltinParamArity::Variadic)
    }) {
        return Ok(());
    }
    let Some(maximum) = matching.map(|signature| signature.outputs.len()).max() else {
        return Ok(());
    };
    if requested <= maximum {
        return Ok(());
    }
    let mut builder = crate::build_runtime_error(error.message)
        .with_builtin(builtin)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
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
            let _guard = super::super::push_output_count(count).expect("output-count scope");
            assert!(validate_fixed_outputs(&FIXED, &ERROR, "output_test").is_ok());
        }
        let _guard = super::super::push_output_count(Some(2)).expect("output-count scope");
        let error = validate_fixed_outputs(&FIXED, &ERROR, "output_test").unwrap_err();
        assert_eq!(error.identifier(), ERROR.identifier);
        assert_eq!(error.message(), ERROR.message);
    }

    #[test]
    fn dynamic_and_variadic_policies_stay_with_the_caller() {
        let _guard = super::super::push_output_count(Some(20)).expect("output-count scope");
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
            let _guard = super::super::push_output_count(Some(count)).expect("output-count scope");
            assert!(validate_fixed_outputs(&OVERLOADED, &ERROR, "output_test").is_ok());
        }
        let _guard = super::super::push_output_count(Some(3)).expect("output-count scope");
        assert!(validate_fixed_outputs(&OVERLOADED, &ERROR, "output_test").is_err());
    }

    #[test]
    fn fixed_form_bound_uses_matching_input_arity_not_global_maximum() {
        const OVERLOADED: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[
                BuiltinSignatureDescriptor {
                    label: "output_test()",
                    inputs: &[],
                    outputs: &[],
                },
                BuiltinSignatureDescriptor {
                    label: "value = output_test(option)",
                    inputs: &[OUTPUT],
                    outputs: &[OUTPUT],
                },
            ],
            ..FIXED
        };
        assert!(
            validate_fixed_form_outputs(&OVERLOADED, &ERROR, "output_test", 0, Some(0)).is_ok()
        );
        assert!(
            validate_fixed_form_outputs(&OVERLOADED, &ERROR, "output_test", 1, Some(1)).is_ok()
        );
        assert!(
            validate_fixed_form_outputs(&OVERLOADED, &ERROR, "output_test", 0, Some(1)).is_err()
        );
        assert!(
            validate_fixed_form_outputs(&OVERLOADED, &ERROR, "output_test", 1, Some(2)).is_err()
        );
        assert!(validate_fixed_form_outputs(&OVERLOADED, &ERROR, "output_test", 0, None).is_ok());
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
            let _guard = super::super::push_output_count(count).expect("output-count scope");
            assert!(validate_fixed_outputs(&ZERO, &ERROR, "output_test").is_ok());
        }
        let _guard = super::super::push_output_count(Some(1)).expect("output-count scope");
        assert!(validate_fixed_outputs(&UNDECLARED, &ERROR, "output_test").is_ok());
        assert!(validate_fixed_outputs(&ZERO, &ERROR, "output_test").is_err());
    }
}
