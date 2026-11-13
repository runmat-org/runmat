use super::super::TextValueTask;
use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::rc::Rc;

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

fn runtime(bytes: u64) -> (RuntimeContext, ManagedHostEffectLedger, CancellationSource) {
    let ledger = ManagedHostEffectLedger::new(bytes);
    let cancellation = CancellationSource::new();
    let context = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    );
    (context, ledger, cancellation)
}

#[test]
fn typed_string_decode_retains_only_final_slots_and_present_text_owners() {
    let source = [
        "<missing>".into(),
        runmat_execution::value::StringElement(None),
        "".into(),
        "雪".into(),
    ];
    let (context, ledger, _) = runtime(1 << 20);
    let control = ControlOwner::new(Some(&context)).unwrap();
    let mut task =
        TextValueTask::string_array(&[1, 4], &source, control.path("$").unwrap(), Some(&context))
            .unwrap();
    let value = loop {
        let (work, value) = task.advance(1).unwrap();
        assert!(work <= 1);
        if let Some(value) = value {
            break value;
        }
    };
    drop(task);
    drop(control);
    let Value::StringArray(array) = value else {
        panic!("string array");
    };
    assert_eq!(array.data()[0].as_str(), Some("<missing>"));
    assert!(array.data()[1].is_missing());
    assert!(!array.data()[1].has_retained_claim());
    assert_eq!(array.data()[2].as_str(), Some(""));
    assert_eq!(array.data()[3].as_str(), Some("雪"));
    assert!(array.data_owner().is_exact());
    assert!(array.data()[0].has_retained_claim());
    let retained = ledger.snapshot().live_reserved_bytes;
    let alias = array.clone();
    drop(array);
    assert!(retained > 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
    drop(alias);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let (denied, ledger, _) = runtime(1);
    let control = ControlOwner::new(Some(&denied));
    assert!(control.is_err());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

fn task<'a>(
    shape: &'a [u64],
    units: &'a [u16],
    context: &'a RuntimeContext,
) -> Result<TextValueTask<'a>, ValueCodecError> {
    let control = ControlOwner::new(Some(context))?;
    TextValueTask::character(shape, units, control.path("$")?, Some(context))
}

#[test]
fn raw_character_decode_is_bounded_no_copy_and_retained_through_last_alias() {
    let raw = [0x0061, 0xD800, 0xFFFF, 0x0000, 0xDFFF, 0xD801];
    for shape in [vec![2, 3], vec![1, 3, 2], vec![1, 6, 1]] {
        let (context, ledger, _) = runtime(1 << 20);
        let mut task = task(&shape, &raw, &context).unwrap();
        let mut original_pointer = None;
        let value = loop {
            let (work, value) = task.advance(1).unwrap();
            assert!(work <= 1);
            if let TextValueTask::ManagedCharacter { owner, .. } = &task {
                if let Some(output) = &owner.output {
                    if !output.initialized().is_empty() {
                        original_pointer = Some(output.initialized().as_ptr());
                    }
                }
            }
            if let Some(value) = value {
                break value;
            }
        };
        drop(task);
        let Value::CharArray(output) = value else {
            panic!("character result")
        };
        assert_eq!(output.data().as_ptr(), original_pointer.unwrap());
        assert_eq!(
            output
                .data()
                .iter()
                .map(|unit| unit.raw())
                .collect::<Vec<_>>(),
            raw
        );
        assert_eq!(
            output.shape(),
            shape
                .iter()
                .map(|&extent| extent as usize)
                .collect::<Vec<_>>()
        );
        assert_eq!(std::mem::size_of_val(output.data()), 12);
        let retained = ledger.snapshot().live_reserved_bytes;
        let alias = output.clone();
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
        assert!(retained > 0);
        drop(alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
    let (context, ledger, _) = runtime(1 << 20);
    let high_rank = vec![1; 4097];
    let mut high_rank_task = task(&high_rank, &[0xD800], &context).unwrap();
    let mut turns = 0;
    let value = loop {
        let (work, value) = high_rank_task.advance(64).unwrap();
        assert!(work <= 64);
        turns += 1;
        if let Some(value) = value {
            break value;
        }
    };
    assert!(
        turns > 100,
        "shape copying and final validation both remain bounded"
    );
    drop(high_rank_task);
    drop(value);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    // Later zero must erase an intermediate geometry overflow; no virtual
    // column or row payload is allocated. Rank-zero CharArray stays empty.
    for shape in [vec![u64::MAX, 2, 0], vec![0, u64::MAX, 7], vec![]] {
        if usize::BITS < 64 && shape.contains(&u64::MAX) {
            continue;
        }
        let mut task = task(&shape, &[], &context).unwrap();
        let value = loop {
            let (work, value) = task.advance(1).unwrap();
            assert!(work <= 1);
            if let Some(value) = value {
                break value;
            }
        };
        drop(task);
        let Value::CharArray(output) = value else {
            panic!("empty character result")
        };
        assert!(output.data().is_empty());
        assert_eq!(
            output.shape(),
            shape
                .iter()
                .map(|&extent| extent as usize)
                .collect::<Vec<_>>()
        );
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn denied_cancelled_or_rotated_character_decode_retires_partial_final_owners() {
    let shape = [1, 3];
    let raw = [0xD800, 0xFFFF, 0xDC00];
    let (denied, ledger, _) = runtime(1);
    assert!(matches!(
        task(&shape, &raw, &denied),
        Err(ValueCodecError::Allocation { .. })
    ));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert!(matches!(
        ManagedCharacter::new(&shape, &raw, Some(&denied), "$"),
        Err(ValueCodecError::Allocation { .. })
    ));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    for rotate in [false, true] {
        let (context, ledger, cancellation) = runtime(1 << 20);
        let mut task = task(&shape, &raw, &context).unwrap();
        task.advance(2).unwrap(); // final shape
        task.advance(1).unwrap(); // first raw unit in final exact payload
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        if rotate {
            context.invalidate_host_effect_session_authority();
        } else {
            cancellation.cancel(CancellationReason::User);
        }
        let error = task.advance(1).unwrap_err();
        if rotate {
            assert!(matches!(
                error,
                ValueCodecError::Allocation {
                    source: HostAllocationError::StaleAuthority,
                    ..
                }
            ));
        } else {
            assert!(matches!(
                error,
                ValueCodecError::Allocation {
                    source: HostAllocationError::Cancelled,
                    ..
                }
            ));
        }
        drop(task);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
