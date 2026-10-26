//! Retire preparation before the existing admitted numerical child and summary.
use super::*;
use futures::future::Either;
pub(super) fn run<'a, V: Validation + 'a>(
    prepared: Either<Outcome, Ready<'a, V>>,
) -> impl std::future::Future<Output = BuiltinResult<Outcome>> + 'a {
    match prepared {
        Either::Left(outcome) => Either::Left(std::future::ready(Ok(outcome))),
        Either::Right(prepared) => {
            Either::Right(crate::execution::after_result(numerical(prepared), finish))
        }
    }
}
async fn numerical<'a, V: Validation>(prepared: Ready<'a, V>) -> BuiltinResult<Finished<'a>> {
    let Ready {
        problem,
        mut controls,
        products,
        cooperation,
        validation,
    } = prepared;
    let outcome = super::super::solve::solve_validated(
        problem,
        cooperation,
        &mut controls,
        products,
        validation,
    )?
    .await?;
    Ok(Finished {
        outcome,
        controls,
        cooperation,
    })
}
fn finish(prepared: Finished<'_>) -> std::future::Ready<BuiltinResult<Outcome>> {
    let Finished {
        outcome,
        controls,
        cooperation,
    } = prepared;
    std::future::ready(
        controls
            .summary(&outcome, cooperation.runtime())
            .map(|()| outcome),
    )
}
