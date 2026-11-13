use runmat_value::Value;

#[runmat_macros::runtime_builtin(
    name = "feval",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    output_sequence = true,
    builtin_path = "crate::builtins::introspection::feval"
)]
pub async fn feval_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    f: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    let input_callsite = crate::callsite::current_callsite().and_then(|callsite| {
        (!callsite.arg_spans.is_empty()).then(|| callsite.without_first_argument())
    });
    crate::call::feval::invoke_retained(
        runtime.as_ref(),
        f,
        rest,
        requested_outputs
            .or_else(crate::output_count::current_output_count)
            .unwrap_or(1),
        input_callsite,
    )
    .await
}

#[cfg(test)]
mod tests {
    use runmat_builtins::{
        BuiltinIntegerBackendRule, BuiltinIntegerComputationDomain, FEVAL_EXTENSIONS,
        FEVAL_INTEGER_CAPABILITIES,
    };

    #[test]
    fn descriptor_declares_function_specific_integer_forwarding() {
        assert_eq!(FEVAL_INTEGER_CAPABILITIES.len(), 1);
        let capability = &FEVAL_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::FunctionSpecific
        );
        assert_eq!(
            capability.backend,
            BuiltinIntegerBackendRule::FunctionSpecific
        );
        assert_eq!(FEVAL_EXTENSIONS.len(), 2);
    }
}
