use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
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
fn contextual_case_stream_matches_rust_string_mapping_and_exact_byte_quote() {
    assert_eq!(char::UNICODE_VERSION, (16, 0, 0));
    let samples = [
        "ΑΣ Α",
        "Α'Σ Α",
        "ΑΣ' Α",
        "Α'Σ' Α",
        "Α Σ",
        "Α 'Σ",
        "Σ",
        "ΑΣΑ",
        "ὈΔΥΣΣΕΎΣ",
        "İΣß",
        "<missing>",
        "aΣ\u{0301}b",
        "aΣ\u{0301} ",
        "aʕΣ",
        "aΣʕ",
        "a\u{1ACF}Σ",
    ];
    for sample in samples {
        for mode in [TextCase::Lower, TextCase::Upper] {
            let mut flags = vec![0; sample.len()];
            if mode == TextCase::Lower && sample.contains('Σ') {
                prepare_sigma_suffix(sample, &mut flags, || Ok(())).unwrap();
            }
            let mut actual = String::new();
            visit_case(sample, mode, Some(&flags), |character| {
                actual.push(character);
                Ok(())
            })
            .unwrap();
            let expected = if is_missing_string(sample) {
                sample.to_owned()
            } else {
                match mode {
                    TextCase::Lower => sample.to_lowercase(),
                    TextCase::Upper => sample.to_uppercase(),
                }
            };
            assert_eq!(actual, expected, "case mapping for {sample:?}");
            assert_eq!(
                transformed_bytes(sample, mode, None).unwrap(),
                expected.len(),
                "exact byte quote for {sample:?}"
            );
        }
    }
}

#[test]
fn managed_case_output_retains_original_bytes_and_refunds_or_refuses_before_output() {
    let source = StringArray::new(
        vec!["ΑΣ".to_owned(), "İΣß".to_owned(), "<missing>".to_owned()],
        vec![1, 3],
    )
    .unwrap();
    let map_error = |_: &RuntimeContext, cause: HostAllocationError| {
        crate::build_runtime_error(format!("case output: {cause}"))
            .with_source(cause)
            .build()
    };
    let invalid = |detail: String| crate::build_runtime_error(detail).build();
    let limited = ManagedHostEffectLedger::new(1);
    let denied = transform_string_array_case(
        Some(&runtime(&limited)),
        source.clone(),
        TextCase::Lower,
        map_error,
        invalid,
    );
    assert!(denied.is_err());
    assert_eq!(limited.snapshot().live_reserved_bytes, 0);

    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let context = runtime(&ledger);
    let output =
        transform_string_array_case(Some(&context), source, TextCase::Lower, map_error, invalid)
            .unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    assert_eq!(output.data()[0].as_str(), "ας");
    assert_eq!(output.data()[1].as_str(), "i\u{307}σß");
    assert_eq!(output.data()[2].as_str(), "<missing>");
    let retained = ledger.snapshot().live_reserved_bytes;
    assert!(retained > 0);
    let alias = output.clone();
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
    drop(alias);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let rank_zero = transform_string_array_case(
        Some(&context),
        StringArray::new(vec!["ΑΣ".to_owned()], vec![]).unwrap(),
        TextCase::Lower,
        map_error,
        invalid,
    )
    .unwrap();
    assert!(rank_zero.shape().is_empty());
    assert_eq!(rank_zero.data()[0].as_str(), "ας");
    drop(rank_zero);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let scalar = transform_scalar_case(
        Some(&context),
        Utf8Text::new("ΑΣ".to_owned()),
        TextCase::Lower,
        map_error,
    )
    .unwrap();
    assert_eq!(scalar.as_str(), "ας");
    let alias = scalar.clone();
    drop(scalar);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(alias);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
