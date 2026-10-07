use super::*;

fn source() -> ConsoleTranscript {
    let mut capture = ConsoleCapture::new(ConsoleLimits {
        max_records: 4,
        max_text_bytes: 100,
    })
    .unwrap();
    capture
        .push(ConsoleStream::Stdout, "original λ".into())
        .unwrap();
    capture
        .push(ConsoleStream::ClearScreen, String::new())
        .unwrap();
    capture
        .push(ConsoleStream::Stderr, "failed\n".into())
        .unwrap();
    capture.finish(true)
}

#[test]
fn original_order_and_receipt_survive_cold_transport() {
    let original = source();
    let mut receiver = ConsoleReceiver::new(original.limits()).unwrap();
    for record in original.records() {
        let wire = serde_json::to_vec(record).unwrap();
        receiver
            .push(serde_json::from_slice(&wire).unwrap())
            .unwrap();
    }
    let receipt: ConsoleReceipt =
        serde_json::from_slice(&serde_json::to_vec(&original.receipt().unwrap()).unwrap()).unwrap();
    let received = receiver.finish(Some(&receipt)).unwrap();
    assert_eq!(received.records(), original.records());
    assert_eq!(received.completion(), original.completion());
    assert_eq!(received.source_receipt(), Some(&receipt));
    let cold: ConsoleTranscript =
        serde_json::from_slice(&serde_json::to_vec(&received).unwrap()).unwrap();
    cold.validate().unwrap();
    assert_eq!(cold, received);
}

#[test]
fn missing_footer_preserves_interrupted_prefix_and_bad_sequences_reject() {
    let original = source();
    let mut receiver = ConsoleReceiver::new(original.limits()).unwrap();
    assert!(receiver.push(original.records()[1].clone()).is_err());
    receiver.push(original.records()[0].clone()).unwrap();
    assert!(receiver.push(original.records()[0].clone()).is_err());
    let prefix = receiver.finish(None).unwrap();
    assert_eq!(prefix.records(), &original.records()[..1]);
    assert_eq!(prefix.completion(), ConsoleCompletion::Interrupted);
}

#[test]
fn substituted_controls_text_counts_and_missing_writes_cannot_match_receipt() {
    let original = source();
    for change in 0..4 {
        let mut receiver = ConsoleReceiver::new(original.limits()).unwrap();
        for record in original
            .records()
            .iter()
            .take(if change == 3 { 2 } else { 3 })
        {
            receiver.push(record.clone()).unwrap();
        }
        let mut receipt = original.receipt().unwrap();
        match change {
            0 => receipt.limits.max_text_bytes += 1,
            1 => receipt.text_bytes += 1,
            2 => receipt.records_digest = Digest::sha256(b"foreign"),
            _ => {}
        }
        assert!(receiver.finish(Some(&receipt)).is_err());
    }
}

#[test]
fn retained_source_receipt_rejects_same_length_text_and_receipt_tampering() {
    let original = source();
    let mut receiver = ConsoleReceiver::new(original.limits()).unwrap();
    for record in original.records() {
        receiver.push(record.clone()).unwrap();
    }
    let received = receiver.finish(Some(&original.receipt().unwrap())).unwrap();
    for change in 0..3 {
        let mut wire = serde_json::to_value(&received).unwrap();
        match change {
            0 => wire["records"][2]["text"] = "forged\n".into(),
            1 => {
                wire["source_receipt"]["records_digest"] =
                    serde_json::to_value(Digest::sha256(b"foreign")).unwrap()
            }
            _ => wire["source_receipt"]["completion"] = serde_json::json!({"status":"interrupted"}),
        }
        let altered: ConsoleTranscript = serde_json::from_value(wire).unwrap();
        assert!(altered.validate().is_err());
    }
    let mut missing = serde_json::to_value(&received).unwrap();
    missing.as_object_mut().unwrap().remove("source_receipt");
    assert!(serde_json::from_value::<ConsoleTranscript>(missing).is_err());
}
