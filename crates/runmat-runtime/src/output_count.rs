use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::RefCell;
use std::rc::Rc;

use crate::context::output_projection::{InvocationOutputProjection, RequestedOutputToken};

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
    owner: Option<Rc<RefCell<InvocationOutputProjection>>>,
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
        if let Some(owner) = &self.owner {
            owner.borrow_mut().remove_requested(self.token);
        } else {
            OUTPUT_COUNT_PROJECTION.with(|projection| {
                projection.borrow_mut().remove_requested(self.token);
            });
        }
    }
}

pub fn push_output_count(count: Option<usize>) -> OutputCountGuard {
    if let Some(owner) = active_projection() {
        let token = owner.borrow_mut().push_requested(count);
        OutputCountGuard {
            token,
            owner: Some(owner),
        }
    } else {
        let token = OUTPUT_COUNT_PROJECTION
            .with(|projection| projection.borrow_mut().push_requested(count));
        OutputCountGuard { token, owner: None }
    }
}

pub fn current_output_count() -> Option<usize> {
    match current_output_scope() {
        InvocationOutputScope::Requested(count) => Some(count),
        InvocationOutputScope::Unscoped | InvocationOutputScope::Implicit => None,
    }
}

pub fn current_output_scope() -> InvocationOutputScope {
    if let Some(owner) = active_projection() {
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

fn active_projection() -> Option<Rc<RefCell<InvocationOutputProjection>>> {
    crate::context::legacy::active().map(|context| Rc::clone(context.output_projection()))
}

pub fn output_list_with_padding(out_count: usize, mut outputs: Vec<Value>) -> Value {
    if outputs.len() > out_count {
        outputs.truncate(out_count);
    }
    if outputs.len() < out_count {
        outputs.resize(out_count, Value::Num(0.0));
    }
    Value::OutputList(outputs)
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
                let _outputs = push_output_count(Some(requested));
                validate_bounded_outputs(descriptor, error, "solver").expect("valid prefix");
            }
            let _outputs = push_output_count(Some(maximum + 1));
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
        let outer = push_output_count(Some(0));
        assert_eq!(current_output_scope(), InvocationOutputScope::Requested(0));
        {
            let _inner = push_output_count(None);
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
        let first = push_output_count(Some(1));
        let second = push_output_count(Some(2));
        assert_eq!(current_output_scope(), InvocationOutputScope::Requested(2));

        drop(first);
        assert_eq!(current_output_scope(), InvocationOutputScope::Requested(2));
        drop(second);
        assert_eq!(current_output_scope(), InvocationOutputScope::Unscoped);
    }
}
