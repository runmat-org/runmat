use super::*;

#[test]
fn audio_identity_is_typed_and_output_modes_remain_distinct() {
    use runmat_builtins::AudioReadDataType;

    let source = Path::new("/sound.wav");
    let metadata = HostEffectWorkload::AudioMetadata {
        source,
        output: HostEffectOutput::ReturnedValue,
    };
    let decode = |range, datatype, output| HostEffectWorkload::AudioDecode {
        source,
        range,
        datatype,
        output,
    };
    let selected = decode(
        Some(AudioFrameRange {
            first: 2,
            last: Some(5),
        }),
        AudioReadDataType::Native,
        AudioDecodeOutput::Samples,
    );
    assert_eq!(metadata.output(), HostEffectOutput::ReturnedValue);
    assert_eq!(
        decode(
            None,
            AudioReadDataType::Double,
            AudioDecodeOutput::EffectOnly
        )
        .output(),
        HostEffectOutput::EffectOnly
    );
    assert_eq!(selected.output(), HostEffectOutput::ReturnedValue);
    assert_eq!(
        decode(
            None,
            AudioReadDataType::Double,
            AudioDecodeOutput::SamplesAndRate
        )
        .output(),
        HostEffectOutput::ReturnedValue
    );
    assert_ne!(metadata.identity_bytes(), selected.identity_bytes());
    for other in [
        decode(None, AudioReadDataType::Native, AudioDecodeOutput::Samples),
        decode(
            Some(AudioFrameRange {
                first: 2,
                last: None,
            }),
            AudioReadDataType::Native,
            AudioDecodeOutput::Samples,
        ),
        decode(
            Some(AudioFrameRange {
                first: 2,
                last: Some(5),
            }),
            AudioReadDataType::Double,
            AudioDecodeOutput::Samples,
        ),
        decode(
            Some(AudioFrameRange {
                first: 2,
                last: Some(5),
            }),
            AudioReadDataType::Native,
            AudioDecodeOutput::SamplesAndRate,
        ),
    ] {
        assert_ne!(selected.identity_bytes(), other.identity_bytes());
    }
}

#[cfg(unix)]
#[test]
fn audio_identity_preserves_native_non_utf8_path_bytes() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;

    let source = |suffix| {
        let mut raw = b"/audio-".to_vec();
        raw.push(suffix);
        PathBuf::from(OsString::from_vec(raw))
    };
    let first = source(0xff);
    let second = source(0xfe);
    assert_eq!(first.to_string_lossy(), second.to_string_lossy());
    fn metadata(path: &Path) -> HostEffectWorkload<'_> {
        HostEffectWorkload::AudioMetadata {
            source: path,
            output: HostEffectOutput::ReturnedValue,
        }
    }
    fn decode(path: &Path) -> HostEffectWorkload<'_> {
        HostEffectWorkload::AudioDecode {
            source: path,
            range: None,
            datatype: runmat_builtins::AudioReadDataType::Double,
            output: AudioDecodeOutput::Samples,
        }
    }
    assert_ne!(
        metadata(&first).identity_bytes(),
        metadata(&second).identity_bytes()
    );
    assert_ne!(
        decode(&first).identity_bytes(),
        decode(&second).identity_bytes()
    );
}

#[test]
fn finite_audio_workload_without_a_certified_producer_rejects_before_selection() {
    let placement = Rc::new(Placement::default());
    let runtime = runtime_context(placement.clone(), None, Some(64));
    for workload in [
        HostEffectWorkload::AudioMetadata {
            source: Path::new("/sound.wav"),
            output: HostEffectOutput::EffectOnly,
        },
        HostEffectWorkload::AudioDecode {
            source: Path::new("/sound.wav"),
            range: Some(AudioFrameRange {
                first: 1,
                last: Some(4),
            }),
            datatype: runmat_builtins::AudioReadDataType::Native,
            output: AudioDecodeOutput::SamplesAndRate,
        },
    ] {
        let error = select_host_effect(
            Some(&runtime),
            HostEffectRequest {
                operation: "audio",
                workload,
                domain_facts: Digest::sha256(b"admitted-audio-controls"),
                filesystem: fixture_fs(),
            },
        )
        .err()
        .expect("uncertified audio route must reject finite quota");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
    }
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
}

#[test]
fn typed_audio_evidence_budgets_effect_only_and_retained_results() {
    struct AudioResources;
    impl RuntimeHostEffectResourceService for AudioResources {
        fn admission(
            &self,
            request: &HostEffectRequest<'_>,
            _: &RuntimeContext,
        ) -> Option<HostEffectResourceAdmission> {
            (request.operation == "audioread"
                && Arc::ptr_eq(request.filesystem, fixture_fs())
                && request.domain_facts == Digest::sha256(b"admitted-audio-controls")
                && matches!(
                    request.workload,
                    HostEffectWorkload::AudioDecode {
                        source,
                        range: Some(AudioFrameRange {
                            first: 1,
                            last: Some(4)
                        }),
                        datatype: runmat_builtins::AudioReadDataType::Native,
                        ..
                    } if source == Path::new("/sound.wav")
                ))
            .then_some(bounds(
                ResourceByteBound::Known(16),
                ResourceByteBound::Known(24),
            ))
            .map(test_admission)
        }
    }
    let placement = Rc::new(Placement::default());
    placement.revision_available.set(true);
    let context = |budget| {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_placement(placement.clone())
                    .with_host_effect_resources(Rc::new(AudioResources))
                    .with_parallel(Rc::new(Parallel(Some(budget)))),
            )
    };
    let request = |output| HostEffectRequest {
        operation: "audioread",
        workload: HostEffectWorkload::AudioDecode {
            source: Path::new("/sound.wav"),
            range: Some(AudioFrameRange {
                first: 1,
                last: Some(4),
            }),
            datatype: runmat_builtins::AudioReadDataType::Native,
            output,
        },
        domain_facts: Digest::sha256(b"admitted-audio-controls"),
        filesystem: fixture_fs(),
    };
    let effect_only =
        select_host_effect(Some(&context(24)), request(AudioDecodeOutput::EffectOnly))
            .unwrap()
            .unwrap();
    effect_only.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(placement.observations.borrow().len(), 1);

    assert!(select_host_effect(Some(&context(39)), request(AudioDecodeOutput::Samples)).is_err());
    let retained = select_host_effect(
        Some(&context(40)),
        request(AudioDecodeOutput::SamplesAndRate),
    )
    .unwrap()
    .unwrap();
    retained.finish(WholeWorkObservationOutcome::Success);
    assert_eq!(placement.selections.get(), 2);
    assert_eq!(placement.observations.borrow().len(), 2);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
}
