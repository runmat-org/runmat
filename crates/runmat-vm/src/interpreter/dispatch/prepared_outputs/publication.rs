use super::commit;
use super::{admit, PreparedDispatchFuture, PreparedResult, Request};
use crate::bytecode::Instr;
use futures::FutureExt;
use runmat_runtime::RuntimeError;

#[inline(never)]
pub(super) fn commit_prepared_output_targets<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                stack,
                vars,
                state,
                context,
                before_overwrite,
                after_store,
                ..
            } = request;
            match instruction {
                Instr::CommitPreparedOutputTargets { retained_outputs } => {
                    if state.fixed_assignment_owner.is_some() {
                        return Err(crate::interpreter::errors::mex(
                            "RunMat:PreparedAssignmentOwner",
                            "fixed assignment requires its exact typed consumer",
                        ));
                    }
                    commit::commit(
                        stack,
                        vars,
                        state,
                        *retained_outputs,
                        &context.authority,
                        before_overwrite,
                        after_store,
                    )
                    .await?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn commit_fixed_assignment<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                stack,
                vars,
                state,
                context,
                before_overwrite,
                after_store,
                ..
            } = request;
            match instruction {
                Instr::CommitFixedAssignment {
                    source_statement,
                    root_slot,
                } => {
                    if state.fixed_assignment_owner != Some((*source_statement, *root_slot)) {
                        return Err(crate::interpreter::errors::mex(
                            "RunMat:PreparedAssignmentOwner",
                            "fixed assignment consumer differs from its source occurrence",
                        ));
                    }
                    commit::commit_fixed(
                        stack,
                        vars,
                        state,
                        &context.authority,
                        before_overwrite,
                        after_store,
                    )
                    .await?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}
