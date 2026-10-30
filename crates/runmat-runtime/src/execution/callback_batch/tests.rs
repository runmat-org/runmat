use super::*;
use crate::call::descriptor::{
    FunctionNameResolver, NameResolutionCompleteness, RepeatableCallable,
};
use std::{cell::Cell, rc::Rc};

struct EmptyRegistry;
impl FunctionNameResolver for EmptyRegistry {
    fn resolve_function(&self, _: &str) -> Option<runmat_types::FunctionId> {
        None
    }
    fn name_resolution_completeness(&self) -> NameResolutionCompleteness {
        NameResolutionCompleteness::Complete
    }
}
fn selected(runtime: &RuntimeContext, name: &str) -> SelectedWorkerCallable {
    let target = RootedValue::in_context(Value::from(name), runtime).unwrap();
    let source = RepeatableCallable::select_rooted(Some(runtime), target, &EmptyRegistry).unwrap();
    assert_eq!(source.selected_target().selected_builtin_name(), Some(name));
    SelectedWorkerCallable::new(
        source.selected_descriptor().clone(),
        runmat_builtins::builtin_required_capabilities(name),
        runmat_types::SpawnSafetyFact::RequiresIsolation,
        source,
        // A routing marker only. The VM sibling control captures and executes
        // the real InterpreterV2 bytes; this test never executes this marker.
        Some(std::sync::Arc::<[u8]>::from(&[][..])),
    )
}
struct RefusingSnapshot(Rc<Cell<usize>>);
impl crate::execution::DeferredGraphSnapshot for RefusingSnapshot {
    fn snapshot(
        &self,
        _: &DeferredInvocation,
        _: runmat_execution::value::ValueLimits,
        _: Option<&crate::execution::value_codec::ValidationWorkspaceOwner>,
    ) -> Result<
        crate::execution::DeferredGraphCapture,
        crate::execution::value_codec::ValueCodecError,
    > {
        self.0.set(self.0.get() + 1);
        Err(
            crate::execution::value_codec::ValueCodecError::Unsupported {
                path: "$".into(),
                rule: "source unavailable".into(),
            },
        )
    }
}

#[test]
fn builtin_publication_keeps_numeric_route_and_bound_argument_graph_without_serial_replay() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let selected = selected(&runtime, "feval");
        let captures = Rc::new(Cell::new(0));
        let snapshot = DeferredGraphProgram::new(Rc::new(RefusingSnapshot(Rc::clone(&captures))));
        let numeric = selected
            .invocation(vec![Value::Num(1.)], 0, &runtime)
            .unwrap();
        assert!(matches!(numeric, DeferredInvocation::Program {
        callable: runmat_execution::ProgramCallable::Builtin { ref name }, requested_outputs: 0, ..
    } if name.0 == "feval"));
        assert!(matches!(
            selected
                .deferred_program(&numeric, &snapshot, &runtime)
                .unwrap(),
            DeferredProgram::InterpreterV2(_)
        ));

        let origin = runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256(b"builtin-bound-argument-fixture"),
            function: runmat_types::ProgramFunctionId(3),
        };
        // A portable origin alone is supported by the Direct codec. A shared
        // lexical activation genuinely requires the selected Graph transport.
        let (environment, _roots) = crate::call::lexical::allocate_environment(None).unwrap();
        let bound = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                function_name: "selectedArgument".into(),
                binding: Some(runmat_value::BoundFunctionBinding::portable(23, origin)),
                source_role: runmat_types::FunctionSourceRole::Nested,
                source_file: None,
                captures: Vec::new().into(),
                lexical_environment: Some(environment),
            },
        ));
        let invocation = selected.invocation(vec![bound], 1, &runtime).unwrap();
        let issuer = crate::execution::value_codec::GraphSourceIssuer::capture_session(&runtime);
        let (prepared, roots) = crate::execution::value_codec::prepare_deferred_invocation_graph(
            &invocation,
            &issuer,
            runmat_execution::value::ValueLimits::default(),
            None,
        )
        .unwrap();
        assert!(prepared.direct_bound_origins().contains(&origin));
        assert!(
            matches!(roots, runmat_execution::value::GraphInvocationRoots::Builtin { ref name, ref arguments }
        if name.0 == "feval" && arguments.len() == 1)
        );
        let DeferredProgram::SelectedGraph(program) = selected
            .deferred_program(&invocation, &snapshot, &runtime)
            .unwrap()
        else {
            panic!("source-bound argument cannot flatten into the numeric wire codec")
        };
        assert!(program.snapshot(&invocation, Default::default()).is_err());
        assert_eq!(
            captures.get(),
            1,
            "unavailable graph is terminal, never a serial replay"
        );
    });
}

#[test]
fn original_cancellation_precedes_owned_builtin_wire_name_publication() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let selected = selected(&runtime, "logical");
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    cancellation.cancel(CancellationReason::Deadline);
    let cancelled = runtime.with_cancellation_token(cancellation.token());
    let error = selected
        .invocation(vec![Value::Num(1.)], 1, &cancelled)
        .unwrap_err();
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
}
