use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::RefCell;

use crate::context::output_projection::{InvocationOutputProjection, RequestedOutputToken};
use crate::context::poll_projection::PollProjection;

mod validation;
pub(crate) use validation::{
    validate_bounded_outputs, validate_fixed_form_outputs, validate_fixed_outputs,
};

runmat_thread_local! {
    static OUTPUT_COUNT_PROJECTION: RefCell<InvocationOutputProjection> =
        const { RefCell::new(InvocationOutputProjection::new()) };
}

pub struct OutputCountGuard {
    token: RequestedOutputToken,
    owner: Option<PollProjection<InvocationOutputProjection>>,
}

/// Output request scoped to the current runtime builtin dispatch. `Implicit`
/// is deliberately distinct from `Unscoped`: a nested call that explicitly
/// requests no count must not inherit an enclosing presentation count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvocationOutputScope {
    Unscoped,
    Implicit,
    Requested(usize),
}

impl Drop for OutputCountGuard {
    fn drop(&mut self) {
        let retired = if let Some(owner) = &self.owner {
            owner.borrow_mut().remove_requested(self.token)
        } else {
            OUTPUT_COUNT_PROJECTION
                .with(|projection| projection.borrow_mut().remove_requested(self.token))
        };
        let _retired = retired;
    }
}

pub fn push_output_count(count: Option<usize>) -> Result<OutputCountGuard, crate::RuntimeError> {
    if let Some(runtime) = crate::context::legacy::active() {
        let owner = runtime.output_projection().clone();
        InvocationOutputProjection::ensure_requested_policy(&owner, &runtime)
            .map_err(|error| output_projection_error(Some(&runtime), error))?;
        let snapshot = owner
            .borrow()
            .requested_push_snapshot()
            .map_err(|error| output_projection_error(Some(&runtime), error))?;
        let prepared = InvocationOutputProjection::prepare_requested(&runtime, snapshot, count)
            .map_err(|error| output_projection_error(Some(&runtime), error))?;
        let installed = { owner.borrow_mut().install_requested(prepared) };
        let (token, retirement) = installed.map_err(|error| {
            let cause = error.cause;
            drop(error);
            output_projection_error(Some(&runtime), cause)
        })?;
        drop(retirement);
        Ok(OutputCountGuard {
            token,
            owner: Some(owner),
        })
    } else {
        let token = OUTPUT_COUNT_PROJECTION
            .with(|projection| projection.borrow_mut().push_requested_unbounded(count))
            .map_err(|error| output_projection_error(None, error))?;
        Ok(OutputCountGuard { token, owner: None })
    }
}

pub fn current_output_count() -> Option<usize> {
    match current_output_scope() {
        InvocationOutputScope::Requested(count) => Some(count),
        InvocationOutputScope::Unscoped | InvocationOutputScope::Implicit => None,
    }
}

pub fn current_output_scope() -> InvocationOutputScope {
    if let Some(owner) =
        crate::context::legacy::active().map(|runtime| runtime.output_projection().clone())
    {
        return match owner.borrow().current_requested() {
            Some(Some(count)) => InvocationOutputScope::Requested(count),
            Some(None) => InvocationOutputScope::Implicit,
            None => InvocationOutputScope::Unscoped,
        };
    }
    OUTPUT_COUNT_PROJECTION.with(|projection| match projection.borrow().current_requested() {
        Some(Some(count)) => InvocationOutputScope::Requested(count),
        Some(None) => InvocationOutputScope::Implicit,
        None => InvocationOutputScope::Unscoped,
    })
}

fn output_projection_error(
    context: Option<&crate::context::RuntimeContext>,
    source: crate::context::host_allocation::HostAllocationError,
) -> crate::RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        context,
        "RunMat:Runtime:OutputProjection",
        "output-count projection growth failed",
        source,
    )
}

pub fn output_list_with_padding(out_count: usize, mut outputs: Vec<Value>) -> Value {
    if outputs.len() > out_count {
        outputs.truncate(out_count);
    }
    if outputs.len() < out_count {
        outputs.resize(out_count, Value::Num(0.0));
    }
    Value::output_list(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn requested_prefix_bounds_are_explicit_and_descriptor_derived() {
        for (descriptor, maximum, error) in [
            (
                &runmat_builtins::LINPROG_DESCRIPTOR,
                4,
                &runmat_builtins::LINPROG_ERROR_INVALID_ARGUMENT,
            ),
            (
                &runmat_builtins::CONEPROG_DESCRIPTOR,
                5,
                &runmat_builtins::CONEPROG_ERROR_INVALID_ARGUMENT,
            ),
        ] {
            for requested in 0..=maximum {
                let _outputs = push_output_count(Some(requested)).expect("output-count scope");
                validate_bounded_outputs(descriptor, error, "solver").expect("valid prefix");
            }
            let _outputs = push_output_count(Some(maximum + 1)).expect("output-count scope");
            assert!(validate_bounded_outputs(descriptor, error, "solver").is_err());
            // ByRequestedOutputCount owners are not globally reinterpreted.
            validate_fixed_outputs(descriptor, error, "solver")
                .expect("fixed-only wrapper unchanged");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nested_implicit_scope_does_not_inherit_outer_requested_count() {
        assert_eq!(current_output_scope(), InvocationOutputScope::Unscoped);
        let outer = push_output_count(Some(0)).expect("output-count scope");
        assert_eq!(current_output_scope(), InvocationOutputScope::Requested(0));
        {
            let _inner = push_output_count(None).expect("output-count scope");
            assert_eq!(current_output_scope(), InvocationOutputScope::Implicit);
            assert_eq!(current_output_count(), None);
        }
        assert_eq!(current_output_scope(), InvocationOutputScope::Requested(0));
        drop(outer);
        assert_eq!(current_output_scope(), InvocationOutputScope::Unscoped);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn direct_guards_remove_their_exact_out_of_order_overlay() {
        let first = push_output_count(Some(1)).expect("output-count scope");
        let second = push_output_count(Some(2)).expect("output-count scope");
        assert_eq!(current_output_scope(), InvocationOutputScope::Requested(2));

        drop(first);
        assert_eq!(current_output_scope(), InvocationOutputScope::Requested(2));
        drop(second);
        assert_eq!(current_output_scope(), InvocationOutputScope::Unscoped);
    }
}
