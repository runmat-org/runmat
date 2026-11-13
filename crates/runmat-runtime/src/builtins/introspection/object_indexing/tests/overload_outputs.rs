use super::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

#[test]
fn overloaded_reads_execute_once_for_zero_and_multiple_outputs_and_failures() {
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _activation = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let fail = Arc::new(AtomicBool::new(false));
    let expected = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let should_fail = fail.clone();
    let requested_count = expected.clone();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "cutover_sequence_subsref").then_some(937)
        })))
        .expect("call-port scope");
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, _, requested| {
            assert_eq!(function, 937);
            assert_eq!(requested, requested_count.load(Ordering::SeqCst));
            observed.fetch_add(1, Ordering::SeqCst);
            let failure = should_fail.load(Ordering::SeqCst);
            Box::pin(async move {
                if failure {
                    return Err(crate::runtime_error::semantic_error(
                        "CutoverReadFailure",
                        "overload failed",
                    ));
                }
                let sequence = crate::sequence::ValueSequence::comma_separated(
                    (0..requested).map(|n| Value::Num(n as f64)).collect(),
                )
                .map_err(crate::sequence::sequence_error_to_runtime)?;
                crate::execution::RootedValueSequence::new(sequence)
                    .map_err(|error| error.into_runtime_error())
            })
        },
    )))
    .expect("call-port scope");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "CutoverSequenceRead".into(),
        parent: None,
        properties: Default::default(),
        methods: [(
            "subsref".into(),
            crate::class_registry::RuntimeMethod {
                name: "subsref".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "cutover_sequence_subsref".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]
        .into(),
    });
    let mut invocations = 0;
    for failure in [false, true] {
        fail.store(failure, Ordering::SeqCst);
        for count in [0, 3] {
            expected.store(count, Ordering::SeqCst);
            let _outputs =
                crate::output_count::push_output_count(Some(count)).expect("output-count scope");
            let result = block_on(subsref_builtin(
                Value::Object(ObjectInstance::new("CutoverSequenceRead")),
                substruct("()", cell(vec![Value::Num(1.0)])),
            ));
            invocations += 1;
            assert_eq!(calls.load(Ordering::SeqCst), invocations);
            if failure {
                assert_eq!(
                    result.unwrap_err().identifier(),
                    Some("RunMat:CutoverReadFailure")
                );
            } else {
                assert_eq!(
                    result.unwrap(),
                    Value::output_list(
                        (0..count).map(|n| Value::Num(n as f64)).collect::<Vec<_>>()
                    )
                );
            }
        }
    }
}
