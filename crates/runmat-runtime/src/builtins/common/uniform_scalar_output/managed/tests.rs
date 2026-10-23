use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use runmat_value::{IntValue, NumericDType, Value};
use std::rc::Rc;

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            cpu_millicores_available: 1000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn runtime(capacity: u64) -> (RuntimeContext, ManagedHostEffectLedger, CancellationSource) {
    let ledger = ManagedHostEffectLedger::new_with_epoch(capacity, 1);
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    );
    (runtime, ledger, cancellation)
}
#[test]
fn managed_scalar_outputs_preserve_classes_bits_and_original_claims() {
    let (runtime, ledger, _) = runtime(1 << 20);
    let singles = [f32::from_bits(0x7fa0_0042), f32::from_bits(0x8000_0000)];
    let mut collector = Managed::new(&runtime, 2).unwrap().unwrap();
    for value in singles {
        collector
            .push(ClassifiedValue::ComplexF32((value, -0.0)))
            .unwrap();
    }
    let output = collector.finish(&[2, 1]).unwrap();
    drop(collector);
    let Value::ComplexTensor(tensor) = &output else {
        panic!("expected complex output")
    };
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    for (index, expected) in singles.into_iter().enumerate() {
        let (runmat_value::NumericScalar::F32(real), runmat_value::NumericScalar::F32(imaginary)) =
            tensor.numeric_value_at(index).unwrap()
        else {
            panic!("expected native single")
        };
        assert_eq!(real.to_bits(), expected.to_bits());
        assert_eq!(imaginary.to_bits(), (-0.0f32).to_bits());
    }
    let alias = output.clone();
    drop(output);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(alias);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    for value in [
        IntValue::I8(-1),
        IntValue::I16(-2),
        IntValue::I32(-3),
        IntValue::I64(i64::MIN),
        IntValue::U8(255),
        IntValue::U16(u16::MAX),
        IntValue::U32(u32::MAX),
        IntValue::U64(u64::MAX),
    ] {
        let mut collector = Managed::new(&runtime, 1).unwrap().unwrap();
        collector
            .push(ClassifiedValue::IntegerComplex(
                value.clone(),
                value.clone(),
            ))
            .unwrap();
        let output = collector.finish(&[1, 1]).unwrap();
        let Value::ComplexTensor(tensor) = output else {
            panic!("expected integer complex")
        };
        let (real, imaginary) = tensor.numeric_value_at(0).unwrap();
        assert_eq!(real.into_int_value(), Some(value.clone()));
        assert_eq!(imaginary.into_int_value(), Some(value));
    }
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
#[test]
fn managed_scalar_output_denial_cancellation_and_character_layout() {
    let (denied, ledger, _) = runtime(1);
    let error = match Managed::new(&denied, 4) {
        Err(error) => error,
        Ok(_) => panic!("expected managed capacity denial"),
    };
    assert!(error.into_failure().is_err());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let (runtime, ledger, cancellation) = runtime(1 << 20);
    let mut collector = Managed::new(&runtime, 4).unwrap().unwrap();
    for value in runmat_value::encode_utf16_code_units("abcd") {
        collector.push(ClassifiedValue::Char(value)).unwrap();
    }
    let value = collector.finish(&[2, 2]).unwrap();
    let Value::CharArray(value) = value else {
        panic!("expected character output")
    };
    assert_eq!(
        value.data(),
        runmat_value::encode_utf16_code_units("acbd").as_slice()
    );
    drop(value);
    cancellation.cancel(CancellationReason::User);
    let Err(error) = collector
        .finish(&[2, 2])
        .expect_err("cancelled completion")
        .into_failure()
    else {
        panic!("expected cancellation")
    };
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with(":Cancelled")));
    drop(collector);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
