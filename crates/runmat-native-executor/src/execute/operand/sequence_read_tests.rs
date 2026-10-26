use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_mir::MirLocalId;
use runmat_runtime::native::{NativeSitePhase, NativeSiteRequest};
use runmat_types::MemberName;
use runmat_value::UnboundedStructValueBuilder;
use std::rc::Rc;

#[test]
fn static_and_dynamic_member_reads_suspend_before_resolving_the_same_value() {
    runmat_gc::gc_test_context(|| {
        for dynamic in [false, true] {
            let mut state = observed_state(&Rc::new(Probe::default()));
            let request = NativeSiteRequest {
                function: 0,
                block: 0,
                position: 0,
                phase: NativeSitePhase::RVALUE,
                ordinal: 0,
                reserved: 0,
            };
            state.current_request = Some(request);
            let mut structure = UnboundedStructValueBuilder::with_capacity(1);
            structure.insert("answer", Value::Num(42.0));
            let base = state
                .arena
                .insert(Value::Struct(structure.finish()), Some(&state.runtime))
                .unwrap();
            state.locals.push(base);
            let value = if dynamic {
                let member = state
                    .arena
                    .insert(Value::String("answer".into()), Some(&state.runtime))
                    .unwrap();
                state.locals.push(member);
                MirRvalue::DynamicMember {
                    base: MirOperand::Local(MirLocalId(0)),
                    member: MirOperand::Local(MirLocalId(1)),
                    sequence_use: runmat_types::SequenceUse::RequireSingle,
                }
            } else {
                MirRvalue::Member {
                    base: MirOperand::Local(MirLocalId(0)),
                    member: MemberName("answer".into()),
                    sequence_use: runmat_types::SequenceUse::RequireSingle,
                }
            };
            assert!(matches!(
                evaluate_rvalue(&mut state, &value, 1, None),
                Err(NativeExecutorError::CallSuspended)
            ));
            let pending = state.pending_call.as_ref().unwrap();
            let (continuation, generation) = (pending.continuation, pending.generation);
            futures::executor::block_on(super::super::call_suspension::complete(
                &mut state,
                continuation,
                generation,
            ))
            .unwrap();
            let output = evaluate_rvalue(&mut state, &value, 1, None).unwrap();
            assert_eq!(output.len(), 1);
            assert_eq!(state.arena.get(output[0]).unwrap(), &Value::Num(42.0));
            assert!(state.completed_call.is_none());
        }
    });
}
