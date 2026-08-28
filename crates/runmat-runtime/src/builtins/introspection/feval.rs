use runmat_value::Value;

#[runmat_macros::runtime_builtin(
    name = "feval",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::feval"
)]
pub async fn feval_builtin_registered(f: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::feval_builtin(f, rest).await
}

#[cfg(test)]
mod tests {
    use runmat_builtins::{
        catalog::definitions::{FEVAL_EXTENSIONS, FEVAL_INTEGER_CAPABILITIES},
        BuiltinIntegerBackendRule, BuiltinIntegerComputationDomain,
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
