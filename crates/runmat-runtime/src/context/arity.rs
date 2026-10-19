/// The arity of one selected language invocation, not a session-wide stack.
///
/// Scripts have no function-call arity. VM intrinsics historically report
/// zero for them, while the two-argument arity-check builtins require a
/// function invocation and report their existing context-unavailable error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeInvocationArity {
    Script,
    Function {
        supplied_inputs: usize,
        requested_outputs: usize,
    },
}

impl RuntimeInvocationArity {
    pub const fn function(supplied_inputs: usize, requested_outputs: usize) -> Self {
        Self::Function {
            supplied_inputs,
            requested_outputs,
        }
    }

    pub const fn function_counts(self) -> Option<(usize, usize)> {
        match self {
            Self::Script => None,
            Self::Function {
                supplied_inputs,
                requested_outputs,
            } => Some((supplied_inputs, requested_outputs)),
        }
    }

    pub const fn intrinsic_counts(self) -> (usize, usize) {
        match self.function_counts() {
            Some(counts) => counts,
            None => (0, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeInvocationArity;
    use crate::context::RuntimeContext;
    use std::rc::Rc;

    #[test]
    fn sibling_call_arity_is_selected_by_context_not_mutated_session_state() {
        let execution = Rc::new(crate::execution::RuntimeExecutionService::new());
        let parent = RuntimeContext::new(execution);
        let first = parent
            .clone()
            .with_invocation_arity(RuntimeInvocationArity::function(2, 1));
        let second = parent
            .clone()
            .with_invocation_arity(RuntimeInvocationArity::function(3, 0));
        assert_eq!(parent.invocation_arity(), None);
        assert_eq!(
            first
                .invocation_arity()
                .and_then(RuntimeInvocationArity::function_counts),
            Some((2, 1))
        );
        assert_eq!(
            second
                .invocation_arity()
                .and_then(RuntimeInvocationArity::function_counts),
            Some((3, 0))
        );
        assert_eq!(RuntimeInvocationArity::Script.function_counts(), None);
        assert_eq!(RuntimeInvocationArity::Script.intrinsic_counts(), (0, 0));
        let lab = second
            .fork_parallel_lab(crate::context::RuntimeServicePorts::default())
            .expect("idle session fork");
        assert_eq!(
            lab.invocation_arity(),
            Some(RuntimeInvocationArity::function(3, 0))
        );
    }
}
