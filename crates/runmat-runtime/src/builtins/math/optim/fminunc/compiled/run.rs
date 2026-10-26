//! One admitted invocation; all QN numerical transitions remain shared.
use super::*;
pub(super) async fn solve(
    initial: Vec<f64>,
    prepared: PreparedOptions,
    oracle: &mut dyn Oracle,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    let PreparedOptions { controls, values } = prepared;
    let shape = [initial.len(), 1];
    let fields =
        common::ScalarOptionFields::from_struct(values.as_ref().map(|value| value.structure()));
    let outcome = algorithm::minimize_internal(
        oracle,
        initial,
        &shape,
        &controls,
        fields.exact(runmat_builtins::OptimOptionField::OutputFcn),
        cooperation,
    )
    .await?;
    Ok(Outcome(outcome))
}
