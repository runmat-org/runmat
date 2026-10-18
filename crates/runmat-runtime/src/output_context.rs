use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;

use crate::context::output_projection::{InvocationOutputProjection, PresentationOutputToken};
use crate::context::poll_projection::PollProjection;

runmat_thread_local! {
    static PRESENTATION_OUTPUT_PROJECTION: RefCell<InvocationOutputProjection> =
        const { RefCell::new(InvocationOutputProjection::new()) };
}

pub struct OutputCountGuard {
    token: PresentationOutputToken,
    owner: Option<PollProjection<InvocationOutputProjection>>,
}

impl Drop for OutputCountGuard {
    fn drop(&mut self) {
        let retired = if let Some(owner) = &self.owner {
            owner.borrow_mut().remove_presentation(self.token)
        } else {
            PRESENTATION_OUTPUT_PROJECTION
                .with(|projection| projection.borrow_mut().remove_presentation(self.token))
        };
        let _retired = retired;
    }
}

pub fn push_output_count(count: usize) -> Result<OutputCountGuard, crate::RuntimeError> {
    if let Some(runtime) = crate::context::legacy::active() {
        let owner = runtime.output_projection().clone();
        InvocationOutputProjection::ensure_presentation_policy(&owner, &runtime)
            .map_err(|error| output_projection_error(Some(&runtime), error))?;
        let snapshot = owner
            .borrow()
            .presentation_push_snapshot()
            .map_err(|error| output_projection_error(Some(&runtime), error))?;
        let prepared = InvocationOutputProjection::prepare_presentation(&runtime, snapshot, count)
            .map_err(|error| output_projection_error(Some(&runtime), error))?;
        let installed = { owner.borrow_mut().install_presentation(prepared) };
        let (token, retirement) = installed.map_err(|error| {
            let cause = error.cause;
            drop(error);
            output_projection_error(Some(&runtime), cause)
        })?;
        drop(retirement);
        return Ok(OutputCountGuard {
            token,
            owner: Some(owner),
        });
    }
    let token = PRESENTATION_OUTPUT_PROJECTION
        .with(|projection| projection.borrow_mut().push_presentation_unbounded(count))
        .map_err(|error| output_projection_error(None, error))?;
    Ok(OutputCountGuard { token, owner: None })
}

pub fn requested_output_count() -> Option<usize> {
    if let Some(owner) =
        crate::context::legacy::active().map(|runtime| runtime.output_projection().clone())
    {
        return owner.borrow().current_presentation();
    }
    PRESENTATION_OUTPUT_PROJECTION.with(|projection| projection.borrow().current_presentation())
}

fn output_projection_error(
    context: Option<&crate::context::RuntimeContext>,
    source: crate::context::host_allocation::HostAllocationError,
) -> crate::RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        context,
        "RunMat:Runtime:OutputPresentationProjection",
        "presentation-count projection growth failed",
        source,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn direct_guards_remove_their_exact_out_of_order_overlay() {
        let first = push_output_count(1).expect("output-count scope");
        let second = push_output_count(2).expect("output-count scope");
        assert_eq!(requested_output_count(), Some(2));

        drop(first);
        assert_eq!(requested_output_count(), Some(2));
        drop(second);
        assert_eq!(requested_output_count(), None);
    }
}
