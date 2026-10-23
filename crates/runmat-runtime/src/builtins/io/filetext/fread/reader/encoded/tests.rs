use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
    RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use std::rc::Rc;

struct ManagedParallel(ManagedHostEffectLedger);
impl RuntimeParallelService for ManagedParallel {
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
fn runtime(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
}

#[test]
fn encoded_units_have_claimed_growth_exact_final_geometry_and_final_drop() {
    fn assert_send<T: Send>() {}
    assert_send::<crate::builtins::io::filetext::encoding::TextInputBox>();
    let denied = ManagedHostEffectLedger::new(1);
    assert!(StreamUnits::new(Some(&runtime(&denied))).is_err());
    assert_eq!(denied.snapshot().live_reserved_bytes, 0);
    for kind in [
        OutputKind::Char,
        OutputKind::Numeric(NumericDType::U16),
        OutputKind::Numeric(NumericDType::F32),
        OutputKind::Numeric(NumericDType::I8),
    ] {
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let cancellation = runmat_execution::CancellationSource::new();
        let runtime = runtime(&ledger).with_cancellation_token(cancellation.token());
        let mut stream = StreamUnits::new(Some(&runtime)).unwrap();
        for value in [0x41, 0xD800, 0xFFFF] {
            stream.push(Utf16CodeUnit::from_raw(value)).unwrap();
        }
        let workspace = ledger.snapshot().live_reserved_bytes;
        assert!(workspace > 0);
        if matches!(kind, OutputKind::Char) {
            let decoder = crate::builtins::io::filetext::encoding::EncodedInput::allocate(
                Some(&runtime),
                crate::builtins::io::filetext::encoding::TextEncoding::parse("UTF-8").unwrap(),
                true,
            )
            .unwrap();
            assert!(ledger.snapshot().live_reserved_bytes > workspace);
            drop(decoder);
            assert_eq!(ledger.snapshot().live_reserved_bytes, workspace);
        }
        let mut cooperation = crate::builtins::io::filetext::cooperation::FileCooperation::new(
            Some(&runtime),
            "fread",
        );
        let output = futures::executor::block_on(output::finish(
            &stream,
            &SizeSpec::Matrix {
                rows: 2,
                cols: Some(2),
            },
            presentation::Projection::select(kind, None).expect("host projection"),
            &mut cooperation,
        ))
        .unwrap();
        match &output {
            Value::CharArray(array) => {
                assert_eq!(array.shape(), &[2, 2]);
                assert_eq!(
                    array
                        .data()
                        .iter()
                        .map(|unit| unit.raw())
                        .collect::<Vec<_>>(),
                    [0x41, 0xFFFF, 0xD800, 0]
                );
            }
            Value::Tensor(tensor) => {
                let expected = if tensor.numeric_dtype() == NumericDType::I8 {
                    [65.0, 127.0, 127.0, 0.0]
                } else {
                    [65.0, 55296.0, 65535.0, 0.0]
                };
                assert_eq!(tensor.materialize_f64(), expected);
                assert_eq!(tensor.shape(), &[2, 2]);
            }
            other => panic!("unexpected output {other:?}"),
        }
        let retained = ledger.snapshot().live_reserved_bytes;
        assert!(retained > workspace);
        let alias = output.clone();
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
        drop(alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, workspace);
        cancellation.cancel(runmat_execution::CancellationReason::User);
        assert!(runtime.is_cancelled());
        assert!(stream.push(Utf16CodeUnit::from_raw(1)).is_err());
        drop(stream);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn character_precision_then_logical_like_keeps_source_column_order_and_exact_owner() {
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let runtime = runtime(&ledger);
    let mut stream = StreamUnits::new(Some(&runtime)).expect("managed input");
    for value in [0, 0x41, 0, 0x42] {
        stream
            .push(Utf16CodeUnit::from_raw(value))
            .expect("claimed unit");
    }
    let prototype = Value::LogicalArray(LogicalArray::zeros(vec![1, 1]));
    let projection = presentation::Projection::select(OutputKind::Char, Some(&prototype))
        .expect("character-to-logical projection");
    let mut cooperation =
        crate::builtins::io::filetext::cooperation::FileCooperation::new(Some(&runtime), "fread");
    let output = futures::executor::block_on(output::finish(
        &stream,
        &SizeSpec::Matrix {
            rows: 2,
            cols: Some(2),
        },
        projection,
        &mut cooperation,
    ))
    .expect("managed logical output");
    let Value::LogicalArray(output) = output else {
        panic!("expected logical output")
    };
    assert_eq!(output.shape.as_slice(), &[2, 2]);
    assert_eq!(output.data.as_slice(), &[0, 1, 0, 1]);
    drop(output);
    drop(stream);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn precision_conversion_precedes_like_conversion_without_scalar_staging() {
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let runtime = runtime(&ledger);
    let mut stream = StreamValues::<u16>::new(Some(&runtime)).expect("managed native input");
    stream.push(300).expect("source value");
    let prototype = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).expect("integer prototype"),
    );
    let projection =
        presentation::Projection::select(OutputKind::Numeric(NumericDType::I8), Some(&prototype))
            .expect("int8 then uint16 projection");
    let mut cooperation =
        crate::builtins::io::filetext::cooperation::FileCooperation::new(Some(&runtime), "fread");
    let output = futures::executor::block_on(output::finish(
        &stream,
        &SizeSpec::Count(1),
        projection,
        &mut cooperation,
    ))
    .expect("projected output");
    let Value::Tensor(output) = output else {
        panic!("expected numeric output")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::U16);
    assert_eq!(output.materialize_f64(), [127.0]);
    drop(output);
    drop(stream);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
