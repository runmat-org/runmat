use super::InterpretContext;
use crate::bytecode::program::ExecutionContext;
use crate::interpreter::dispatch::{SequenceRegister, SequenceState};
use crate::interpreter::vars::InterpreterVars;
use runmat_value::Value;

#[test]
fn frame_registration_does_not_own_physical_gpu_retirement() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };
    runmat_gc::gc_test_context(|| {
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let returned_alias = allocation.handle.clone();
        let stack = vec![Value::GpuTensor(allocation.handle)];
        let vars = InterpreterVars::from_owned(Vec::new());
        let locals = Vec::new();
        let sequence = SequenceState::default();
        let registrations = InterpretContext::new_frame(&stack, &vars, &locals, &sequence).unwrap();
        drop(registrations);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(stack);
        assert_eq!(
            allocation.probe.retirements(),
            0,
            "returned alias owns the allocation"
        );
        drop(returned_alias);
        assert_eq!(allocation.probe.retirements(), 1);
    });
}

#[test]
fn complete_frame_registration_allows_mutation_and_remains_live_across_await() {
    runmat_gc::gc_test_context(|| {
        let mut stack = Vec::new();
        let mut vars = InterpreterVars::from_owned(Vec::new());
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
            vars.push_owned(Value::Num(2.0));
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
