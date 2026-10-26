//! Variable adapter to the single model-array dimension/label grammar.
use super::*;
fn policy() -> crate::builtins::math::optim::common::model_dimensions::Policy {
    crate::builtins::math::optim::common::model_dimensions::Policy {
        identity: runmat_builtins::OPTIMVAR_CATALOG_ENTRY.identity,
        invalid: &runmat_builtins::OPTIMVAR_ERROR_INVALID_INPUT,
    }
}
pub(super) async fn constructor(
    args: crate::call::FunctionArgumentView<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(
    runmat_value::SharedShape,
    runmat_value::optimization_model::ModelAxisNames,
)> {
    crate::builtins::math::optim::common::model_dimensions::constructor(
        args,
        cooperation,
        policy(),
        crate::builtins::math::optim::common::model_dimensions::ConstructorSyntax::Variable,
    )
    .await
}
pub(in crate::builtins::math::optim::optimvar) async fn property(
    value: &Value,
    shape: &[usize],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<runmat_value::optimization_model::ModelAxisNames> {
    crate::builtins::math::optim::common::model_dimensions::property(
        value,
        shape,
        cooperation,
        policy(),
    )
    .await
}
