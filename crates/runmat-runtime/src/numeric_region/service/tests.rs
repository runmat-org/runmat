use super::*;
use std::rc::Rc;

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
}

fn site() -> NativeSiteRequest {
    NativeSiteRequest {
        function: 0,
        block: 0,
        position: 0,
        phase: crate::native::NativeSitePhase::RVALUE,
        ordinal: 0,
        reserved: 0,
    }
}

fn region() -> RegionId {
    RegionId {
        function: runmat_types::ProgramFunctionId(0),
        ordinal: 0,
    }
}

fn program() -> NumericRegionProgram {
    NumericRegionProgram {
        nodes: vec![
            super::super::NumericRegionNode::Input(0),
            super::super::NumericRegionNode::Unary {
                operation: super::super::NumericUnaryOperation::Minus,
                input: 0,
            },
        ],
        outputs: vec![1],
    }
}

#[test]
fn native_capture_retains_exact_payload_and_shape_without_changing_original() {
    runmat_gc::gc_test_context(|| {
        let runtime = runtime();
        let value = Value::Tensor(runmat_value::Tensor::new(vec![1.0, -2.0], vec![2, 1]).unwrap());
        let request = NativeNumericRegionRequest::capture_verified(
            &runtime,
            region(),
            site(),
            &program(),
            &[&value],
        )
        .unwrap()
        .unwrap();
        let (_program_owner, captured, right) = request.into_parts();
        let (Value::Tensor(original), Value::Tensor(retained)) = (&value, &captured.values()[0])
        else {
            panic!("dense inputs remain dense")
        };
        assert!(original
            .host_buffer()
            .shares_allocation_with(retained.host_buffer()));
        assert_eq!(original.shape().as_ptr(), retained.shape().as_ptr());
        assert!(right.is_current(&runtime));
        assert!(right.matches(region(), site()));
        assert!(!right.is_current(&self::runtime()));
        drop(captured);
        assert_eq!(original.as_f64_slice().unwrap(), &[1.0, -2.0]);
    });
}

#[test]
fn closed_capture_declines_non_floating_before_effects_and_cancellation_revokes_right() {
    runmat_gc::gc_test_context(|| {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = runtime().with_cancellation_token(cancellation.token());
        for value in [
            Value::String("not numeric".into()),
            Value::Int(runmat_value::IntValue::I32(1)),
            Value::Bool(true),
            Value::Complex(1.0, 1.0),
        ] {
            assert!(NativeNumericRegionRequest::capture_verified(
                &runtime,
                region(),
                site(),
                &program(),
                &[&value]
            )
            .unwrap()
            .is_none());
        }
        let request = NativeNumericRegionRequest::capture_verified(
            &runtime,
            region(),
            site(),
            &program(),
            &[&Value::Num(4.0)],
        )
        .unwrap()
        .unwrap();
        let (_program_owner, _, right) = request.into_parts();
        assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
        assert!(runtime.is_cancelled());
        assert!(!right.is_current(&runtime));
        assert!(NativeNumericRegionRequest::capture_verified(
            &runtime,
            region(),
            site(),
            &program(),
            &[&Value::Num(4.0)]
        )
        .unwrap()
        .is_none());
    });
}

#[test]
fn single_capture_retains_class_and_owner_without_claiming_f64_evaluator() {
    runmat_gc::gc_test_context(|| {
        let runtime = runtime();
        let value =
            Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0, -2.0], vec![2, 1]).unwrap());
        let request = NativeNumericRegionRequest::capture_verified(
            &runtime,
            region(),
            site(),
            &program(),
            &[&value],
        )
        .unwrap()
        .unwrap();
        let (program_owner, retained, _) = request.into_parts();
        assert!(program_owner.cpu_candidates().is_none());
        let (Value::Tensor(original), Value::Tensor(captured)) = (&value, &retained.values()[0])
        else {
            panic!("Single remains a dense typed tensor")
        };
        assert_eq!(captured.numeric_dtype(), runmat_value::NumericDType::F32);
        assert!(original
            .host_buffer()
            .shares_allocation_with(captured.host_buffer()));
        assert_eq!(original.shape().as_ptr(), captured.shape().as_ptr());
        assert!(matches!(
            super::super::execute_in_context(&program(), &[&value], &runtime).unwrap(),
            super::super::NumericRegionOwnedExecution::Ineligible
        ));
    });
}

#[test]
fn general_broadcast_capture_defers_to_typed_producer_without_widening_cpu_subset() {
    runmat_gc::gc_test_context(|| {
        let runtime = runtime();
        let program = NumericRegionProgram {
            nodes: vec![
                super::super::NumericRegionNode::Input(0),
                super::super::NumericRegionNode::Input(1),
                super::super::NumericRegionNode::Binary {
                    operation: super::super::NumericBinaryOperation::Add,
                    left: 0,
                    right: 1,
                },
            ],
            outputs: vec![2],
        };
        let left = Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let right =
            Value::Tensor(runmat_value::Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap());
        let request = NativeNumericRegionRequest::capture_verified(
            &runtime,
            region(),
            site(),
            &program,
            &[&left, &right],
        )
        .unwrap()
        .unwrap();
        assert!(request.into_parts().0.cpu_candidates().is_none());
        assert!(matches!(
            super::super::execute_in_context(&program, &[&left, &right], &runtime).unwrap(),
            super::super::NumericRegionOwnedExecution::Ineligible
        ));
    });
}

#[test]
fn native_right_retains_complete_domain_revision_identity_without_cloning_metadata() {
    use runmat_execution::{DomainContribution, ProgramEnvironment, ProgramRevision};
    runmat_gc::gc_test_context(|| {
        let digest = Digest::sha256(b"native-right-domain");
        let base = ProgramRevision::new(
            digest,
            digest,
            ProgramEnvironment::new(2, 2, digest, digest, "runmat").unwrap(),
        )
        .unwrap();
        let revision = base
            .clone()
            .with_domain_contribution(
                DomainContribution::new("native-owner", Digest::sha256(b"first")).unwrap(),
            )
            .unwrap();
        let runtime = runtime().with_program_revision(Some(revision));
        let request = NativeNumericRegionRequest::capture_verified(
            &runtime,
            region(),
            site(),
            &program(),
            &[&Value::Num(1.0)],
        )
        .unwrap()
        .unwrap();
        let (_program_owner, _, right) = request.into_parts();
        assert!(right.is_current(&runtime));
        let changed = base
            .with_domain_contribution(
                DomainContribution::new("native-owner", Digest::sha256(b"second")).unwrap(),
            )
            .unwrap();
        let changed = runtime.clone().with_program_revision(Some(changed));
        assert_eq!(changed.local_identity(), runtime.local_identity());
        assert!(!right.is_current(&changed));
    });
}

#[test]
fn native_publication_owner_retires_payload_before_claim_on_terminal_paths() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    };

    #[derive(Clone, Copy)]
    enum Terminal {
        Abandon,
        Complete,
        Cancelled,
        #[cfg(panic = "unwind")]
        Unwind,
    }
    struct Payload {
        live: Arc<AtomicBool>,
        events: Arc<Mutex<Vec<&'static str>>>,
        #[cfg(panic = "unwind")]
        terminal: Terminal,
    }
    impl NativeNumericRegionPublication for Payload {
        fn complete(self: Box<Self>, runtime: &RuntimeContext, right: NativeNumericRegionRight) {
            assert!(self.live.load(Ordering::SeqCst));
            self.events
                .lock()
                .unwrap()
                .push(if right.is_current(runtime) {
                    "published"
                } else {
                    "declined"
                });
            #[cfg(panic = "unwind")]
            if matches!(self.terminal, Terminal::Unwind) {
                panic!("injected publication unwind");
            }
        }
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            assert!(
                self.live.load(Ordering::SeqCst),
                "payload must retire while funding is retained"
            );
            self.events.lock().unwrap().push("payload retired");
        }
    }
    #[derive(Debug)]
    struct Claim {
        live: Arc<AtomicBool>,
        events: Arc<Mutex<Vec<&'static str>>>,
    }
    impl runmat_value::ExactHostAllocationReservation for Claim {}
    impl Drop for Claim {
        fn drop(&mut self) {
            assert!(self.live.swap(false, Ordering::SeqCst));
            assert_eq!(self.events.lock().unwrap().last(), Some(&"payload retired"));
            self.events.lock().unwrap().push("claim retired");
        }
    }
    for terminal in [
        Terminal::Abandon,
        Terminal::Complete,
        Terminal::Cancelled,
        #[cfg(panic = "unwind")]
        Terminal::Unwind,
    ] {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = runtime().with_cancellation_token(cancellation.token());
        let request = NativeNumericRegionRequest::capture_verified(
            &runtime,
            region(),
            site(),
            &program(),
            &[&Value::Num(1.0)],
        )
        .unwrap()
        .unwrap();
        let (_program, _inputs, right) = request.into_parts();
        let live = Arc::new(AtomicBool::new(true));
        let events = Arc::new(Mutex::new(Vec::new()));
        let owner = NativeNumericRegionPublicationOwner::new(
            Box::new(Payload {
                live: live.clone(),
                events: events.clone(),
                #[cfg(panic = "unwind")]
                terminal,
            }),
            Some(runmat_value::ExactHostAllocationClaim::Shared(Arc::new(
                Claim {
                    live: live.clone(),
                    events: events.clone(),
                },
            ))),
        );
        match terminal {
            Terminal::Abandon => drop(owner),
            Terminal::Complete => owner.complete(&runtime, right),
            Terminal::Cancelled => {
                assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
                assert!(runtime.is_cancelled());
                owner.complete(&runtime, right);
            }
            #[cfg(panic = "unwind")]
            Terminal::Unwind => {
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || owner.complete(&runtime, right)
                ))
                .is_err());
            }
        }
        let events = events.lock().unwrap();
        let expected: &[&str] = match terminal {
            Terminal::Abandon => &["payload retired", "claim retired"],
            Terminal::Cancelled => &["declined", "payload retired", "claim retired"],
            _ => &["published", "payload retired", "claim retired"],
        };
        assert_eq!(events.as_slice(), expected);
        assert!(!live.load(Ordering::SeqCst));
    }
}
