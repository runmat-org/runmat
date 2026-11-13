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
            let expected = match mode {
                TextCase::Lower => sample.to_lowercase(),
                TextCase::Upper => sample.to_uppercase(),
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
    assert_eq!(output.data()[0].as_str(), Some("ας"));
    assert_eq!(output.data()[1].as_str(), Some("i\u{307}σß"));
    assert_eq!(output.data()[2].as_str(), Some("<missing>"));
    let retained = ledger.snapshot().live_reserved_bytes;
    assert!(retained > 0);
    let alias = output.clone();
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
    drop(alias);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    // Character rows are UTF-16 code-unit geometry, not UTF-8 or scalar
    // counts. Include contextual sigma, supplementary Deseret, expansion,
    // and independent row padding in the same physical-owner witness.
    let source = runmat_value::CharArray::from_code_units(
        runmat_value::encode_utf16_code_units("ΑΣ'  𐐀Σ  İß A "),
        3,
        5,
    )
    .unwrap();
    for (mode, expected) in [
        (
            TextCase::Lower,
            [
                0x03B1, 0x03C2, 39, 32, 32, 32, 0xD801, 0xDC28, 0x03C2, 32, 32, 32, 105, 0x0307,
                0x00DF, 32, 97, 32,
            ],
        ),
        (
            TextCase::Upper,
            [
                0x0391, 0x03A3, 39, 32, 32, 32, 0xD801, 0xDC00, 0x03A3, 32, 32, 32, 0x0130, 83, 83,
                32, 65, 32,
            ],
        ),
    ] {
        let expected = expected.map(runmat_value::Utf16CodeUnit::from_raw);
        let ordinary =
            transform_character_case(None, source.clone(), mode, map_error, invalid).unwrap();
        assert_eq!(ordinary.shape(), &[3, 6]);
        assert_eq!(ordinary.data(), expected);
        let managed =
            transform_character_case(Some(&context), source.clone(), mode, map_error, invalid)
                .unwrap();
        assert_eq!(managed.shape(), ordinary.shape());
        assert_eq!(managed.data(), expected);
        assert_eq!(std::mem::size_of_val(managed.data()), 36);
        let retained = ledger.snapshot().live_reserved_bytes;
        assert!(retained > 0);
        let alias = managed.clone();
        drop(managed);
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
        drop(alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        for (shape, expected_shape) in [
            (vec![1, 5, 3], vec![1, 6, 3]),
            (vec![3, 5, 1], vec![3, 6, 1]),
        ] {
            let paged =
                runmat_value::CharArray::from_code_units_with_shape(source.data().to_vec(), shape)
                    .unwrap();
            let output =
                transform_character_case(Some(&context), paged, mode, map_error, invalid).unwrap();
            assert_eq!(output.shape(), expected_shape);
            assert_eq!(output.data(), expected);
            drop(output);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    }
    assert!(transform_character_case(
        Some(&runtime(&limited)),
        source.clone(),
        TextCase::Lower,
        map_error,
        invalid
    )
    .is_err());
    assert_eq!(limited.snapshot().live_reserved_bytes, 0);
    let source_cancellation = runmat_execution::cancellation::CancellationSource::new();
    let cancelled = runtime(&ledger).with_cancellation_token(source_cancellation.token());
    source_cancellation.cancel(runmat_execution::CancellationReason::User);
    assert!(transform_character_case(
        Some(&cancelled),
        source,
        TextCase::Lower,
        map_error,
        invalid
    )
    .is_err());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    for shape in [vec![0, usize::MAX, 7], vec![usize::MAX, 2, 0]] {
        let empty = runmat_value::CharArray::from_code_units_with_shape(Vec::new(), shape).unwrap();
        let output = transform_character_case(
            Some(&context),
            empty.clone(),
            TextCase::Upper,
            map_error,
            invalid,
        )
        .unwrap();
        assert_eq!(output.shape(), empty.shape());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
    let malformed = runmat_value::CharArray::from_code_units(
        vec![runmat_value::Utf16CodeUnit::from_raw(0xD800)],
        1,
        1,
    )
    .unwrap();
    assert!(transform_character_case(
        Some(&context),
        malformed,
        TextCase::Lower,
        map_error,
        invalid
    )
    .is_err());
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
    assert_eq!(rank_zero.data()[0].as_str(), Some("ας"));
    drop(rank_zero);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let scalar = transform_scalar_case(
        Some(&context),
        StringElement::Text(Utf8Text::new("ΑΣ".to_owned())),
        TextCase::Lower,
        map_error,
    )
    .unwrap();
    assert_eq!(scalar.as_str(), Some("ας"));
    let alias = scalar.clone();
    drop(scalar);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(alias);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn typed_missing_and_literal_case_matrix_preserves_exact_owners() {
    let source = [
        StringElement::text("<missing>"),
        StringElement::text("<Missing>"),
        StringElement::text(""),
        StringElement::Missing,
    ];
    let error = |_: &RuntimeContext, cause: HostAllocationError| {
        crate::build_runtime_error(cause.to_string()).build()
    };
    let invalid = |detail: String| crate::build_runtime_error(detail).build();
    for mode in [TextCase::Lower, TextCase::Upper] {
        let expected = source
            .iter()
            .map(|element| match element.as_str() {
                Some(text) => StringElement::text(match mode {
                    TextCase::Lower => text.to_lowercase(),
                    TextCase::Upper => text.to_uppercase(),
                }),
                None => StringElement::Missing,
            })
            .collect::<Vec<_>>();
        for managed in [false, true] {
            let ledger = ManagedHostEffectLedger::new(1 << 20);
            let context = runtime(&ledger);
            let output = transform_string_array_case(
                managed.then_some(&context),
                StringArray::from_elements_unbounded(source.to_vec(), vec![1, 4]).unwrap(),
                mode,
                error,
                invalid,
            )
            .unwrap();
            assert_eq!(output.data(), expected.as_slice());
            assert!(!output.data()[3].has_retained_claim());
            let retained = ledger.snapshot().live_reserved_bytes;
            let alias = output.clone();
            drop(output);
            assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
            drop(alias);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    }
}
