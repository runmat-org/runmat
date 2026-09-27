use super::InterpretContext;
use crate::bytecode::program::ExecutionContext;
use crate::interpreter::dispatch::{SequenceRegister, SequenceState};
use runmat_value::Value;

#[test]
fn complete_frame_registration_allows_mutation_and_remains_live_across_await() {
    runmat_gc::gc_test_context(|| {
        let mut stack = Vec::new();
        let mut vars = Vec::new();
        let mut execution = ExecutionContext::default();
        let mut sequence = SequenceState::default();
        let registrations =
            InterpretContext::new_frame(&stack, &vars, &execution.locals, &sequence).unwrap();

        fn mutate_context(context: &mut ExecutionContext) {
            // Reallocate the Vec backing store without moving its frame owner.
            context.locals.reserve(128);
            context.locals.push(Value::Num(3.0));
            context.instruction_pointer += 1;
        }

        futures::executor::block_on(async {
            stack.push(Value::Num(1.0));
            vars.push(Value::Num(2.0));
            mutate_context(&mut execution);
            sequence.assignment = Some(SequenceRegister { start: 0, len: 1 });
            let mut resumed = false;
            futures::future::poll_fn(|context| {
                if resumed {
                    std::task::Poll::Ready(())
                } else {
                    resumed = true;
                    context.waker().wake_by_ref();
                    std::task::Poll::Pending
                }
            })
            .await;
            let mut observed = Vec::new();
            runmat_gc::gc_visit_retained_values(&[], &mut |value| {
                if let Value::Num(number) = value {
                    observed.push(*number as i32);
                }
                Some(())
            })
            .unwrap();
            observed.sort_unstable();
            assert_eq!(observed, vec![1, 2, 3]);
            assert_eq!(sequence.assignment.unwrap().len, 1);
        });

        drop(registrations);
        let mut remaining = 0;
        runmat_gc::gc_visit_retained_values(&[], &mut |_| {
            remaining += 1;
            Some(())
        })
        .unwrap();
        assert_eq!(remaining, 0);
    });
}
