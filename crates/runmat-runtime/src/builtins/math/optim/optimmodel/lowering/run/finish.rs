//! Original final candidate equality after the numerical child has retired.
use super::*;
pub(super) async fn run(completed: Completed<'_>) -> BuiltinResult<Solved> {
    let Completed {
        family,
        policy,
        cooperation,
    } = completed;
    cooperation.completed(0).await?;
    let completed = match family {
        Either::Left(completed) => completed,
        Either::Right(solved) => return Ok(Solved::Right(solved)),
    };
    let AlgebraicCompleted {
        program,
        formulation,
        rows,
        retained,
        mut outcome,
    } = completed;
    let point = if let Some(candidate) = outcome.point() {
        let Some(point) = retained else {
            return Err(
                policy.invalid("accepted numerical result has no original-model point evidence")
            );
        };
        if !point
            .matches(&program, &candidate, policy, cooperation)
            .await?
        {
            return Err(policy
                .invalid("accepted numerical result differs from original-model point evidence"));
        }
        Some(point)
    } else {
        None
    };
    Ok(Solved::Left(AlgebraicSolved {
        program,
        formulation,
        rows,
        point,
        outcome,
    }))
}
