use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::*;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        vec![]
    } else {
        vec![argument_error(
            "RM-REGISTER-TEST-CLASSES-ARITY",
            "__register_test_classes requires zero arguments",
            request.arguments.len(),
        )]
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![crate::catalog::inference::default_double_scalar()],
        diagnostics,
    )
}
