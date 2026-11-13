use super::*;

fn limits(records: u32, bytes: u64) -> ConsoleLimits {
    ConsoleLimits {
        max_records: records,
        max_text_bytes: bytes,
    }
}

#[test]
fn original_chunks_controls_and_unicode_survive_cold_roundtrip_in_write_order() {
    let mut capture = ConsoleCapture::new(limits(4, 10)).unwrap();
    capture.push(ConsoleStream::Stdout, "λ".into()).unwrap();
    capture.push(ConsoleStream::Stderr, "bad\n".into()).unwrap();
    capture
        .push(ConsoleStream::ClearScreen, String::new())
        .unwrap();
    capture.push(ConsoleStream::Stdout, "next".into()).unwrap();
    let original = capture.finish(true);
    assert_eq!(original.text_bytes(), 10);
    assert_eq!(original.completion(), ConsoleCompletion::Complete);
    assert_eq!(
        original
            .records()
            .iter()
            .map(|r| (r.sequence, r.stream, r.text.as_ref()))
            .collect::<Vec<_>>(),
        vec![
            (1, ConsoleStream::Stdout, "λ"),
            (2, ConsoleStream::Stderr, "bad\n"),
            (3, ConsoleStream::ClearScreen, ""),
            (4, ConsoleStream::Stdout, "next")
        ]
    );
    let cold: ConsoleTranscript =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    cold.validate().unwrap();
    assert_eq!(cold, original);
}

#[test]
fn overflow_rejects_before_append_and_cannot_be_repaired_by_smaller_later_writes() {
    for (limits, write, reason) in [
        (limits(1, 100), "x", ConsoleRejection::RecordLimit),
        (limits(10, 2), "λ", ConsoleRejection::TextByteLimit),
    ] {
        let mut capture = ConsoleCapture::new(limits).unwrap();
        capture.push(ConsoleStream::Stdout, "x".into()).unwrap();
        assert!(capture.push(ConsoleStream::Stdout, write.into()).is_err());
        assert!(capture.push(ConsoleStream::Stdout, String::new()).is_err());
        let transcript = capture.finish(true);
        transcript.validate().unwrap();
        assert_eq!(transcript.records().len(), 1);
        assert_eq!(transcript.text_bytes(), 1);
        assert_eq!(
            transcript.completion(),
            ConsoleCompletion::Rejected { reason }
        );
    }
}

#[test]
fn interruption_and_invalid_control_writes_never_become_complete_empty_output() {
    let empty = ConsoleCapture::new(limits(1, 1)).unwrap().finish(false);
    assert_eq!(empty.completion(), ConsoleCompletion::Interrupted);
    let mut capture = ConsoleCapture::new(limits(1, 1)).unwrap();
    assert!(capture
        .push(ConsoleStream::ClearScreen, "x".into())
        .is_err());
    let rejected = capture.finish(true);
    assert!(rejected.records().is_empty());
    assert_eq!(
        rejected.completion(),
        ConsoleCompletion::Rejected {
            reason: ConsoleRejection::InvalidControlRecord,
        }
    );
}

#[test]
fn cold_inventory_rejects_reordering_gaps_and_forged_utf8_counts() {
    let mut capture = ConsoleCapture::new(limits(2, 5)).unwrap();
    capture.push(ConsoleStream::Stdout, "λ".into()).unwrap();
    capture.push(ConsoleStream::Stderr, "x".into()).unwrap();
    let original = capture.finish(true);
    for mutation in 0..5 {
        let mut cold = serde_json::to_value(&original).unwrap();
        match mutation {
            0 => cold["records"][0]["sequence"] = 2.into(),
            1 => cold["records"][1]["sequence"] = 3.into(),
            2 => cold["text_bytes"] = 2.into(),
            3 => cold["records"][0]["stream"] = "clear_screen".into(),
            _ => cold["limits"]["max_records"] = 1.into(),
        }
        let decoded: ConsoleTranscript = serde_json::from_value(cold).unwrap();
        assert!(decoded.validate().is_err());
    }
}

#[test]
fn unsupported_limits_reject_before_allocation_and_storage_counts_utf8_and_slots() {
    for value in [
        limits(0, 1),
        limits(1, 0),
        limits(MAX_CONSOLE_RECORDS + 1, 1),
        limits(1, MAX_CONSOLE_TEXT_BYTES + 1),
    ] {
        assert!(ConsoleCapture::new(value).is_err());
        assert!(value.capture_storage_bytes().is_err());
    }
    assert_eq!(
        limits(2, 7).capture_storage_bytes().unwrap(),
        2 * std::mem::size_of::<ConsoleRecord>() as u64 + 7
    );
}
