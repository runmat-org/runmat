//! Complete canonical semantic definition fingerprint shared by publication and invalidation.

pub(crate) fn fingerprint(
    function: &runmat_vm::FunctionBytecode,
) -> Result<String, serde_json::Error> {
    runmat_vm::semantic_definition_fingerprint(function)
}
