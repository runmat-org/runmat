use super::*;

#[test]
fn small_pages_preserve_utf8_bytes_and_original_write_identity() {
    let record = ConsoleRecord {
        sequence: 7,
        stream: ConsoleStream::Stderr,
        text: "λ🙂水 original".into(),
    };
    let mut cursor = ConsoleCursor {
        sequence: 7,
        text_offset: 0,
    };
    let mut reconstructed = String::new();
    while cursor.sequence == 7 {
        let fragment = ConsoleFragment::from_record(&record, cursor, 4).unwrap();
        assert_eq!(fragment.sequence, 7);
        assert_eq!(fragment.stream, record.stream);
        assert!(fragment.text.len() <= 4);
        assert!(!fragment.text.is_empty());
        reconstructed.push_str(&fragment.text);
        cursor = fragment.next_cursor().unwrap();
    }
    assert_eq!(reconstructed, record.text.as_ref());
    assert_eq!(
        cursor,
        ConsoleCursor {
            sequence: 8,
            text_offset: 0
        }
    );
    for offset in [1, 3, record.text.len() as u64, u64::MAX] {
        assert!(ConsoleFragment::from_record(
            &record,
            ConsoleCursor {
                sequence: 7,
                text_offset: offset
            },
            4
        )
        .is_err());
    }
}

#[test]
fn empty_control_writes_advance_and_hostile_page_counts_cannot_overflow() {
    let record = ConsoleRecord {
        sequence: 1,
        stream: ConsoleStream::ClearScreen,
        text: "".into(),
    };
    let fragment = ConsoleFragment::from_record(&record, ConsoleCursor::default(), 4).unwrap();
    let mut page = ConsolePage {
        limits: ConsoleLimits {
            max_records: 2,
            max_text_bytes: 4,
        },
        budget: ConsoleReadBudget {
            max_fragments: 1,
            max_text_bytes: 4,
        },
        cursor: ConsoleCursor::default(),
        next_cursor: fragment.next_cursor().unwrap(),
        fragments: vec![fragment],
        observed_records: 1,
        observed_text_bytes: 0,
        completion: ConsoleCompletion::Interrupted,
        source_receipt: None,
    };
    page.validate().unwrap();
    page.observed_records = u64::MAX;
    assert!(page.validate().is_err());
    page.observed_records = 1;
    page.completion = ConsoleCompletion::Complete;
    assert!(page.validate().is_err());
    page.completion = ConsoleCompletion::Interrupted;
    page.next_cursor.sequence = 1;
    assert!(page.validate().is_err());
    let fragment = ConsoleFragment {
        sequence: u64::MAX,
        stream: ConsoleStream::Stdout,
        text_offset: u64::MAX,
        original_text_bytes: u64::MAX,
        text: "a".into(),
    };
    assert!(fragment.next_cursor().is_err());
}

#[test]
fn split_fragments_cannot_change_original_stream_or_length() {
    let mut page = ConsolePage {
        limits: ConsoleLimits {
            max_records: 2,
            max_text_bytes: 8,
        },
        budget: ConsoleReadBudget {
            max_fragments: 2,
            max_text_bytes: 4,
        },
        cursor: ConsoleCursor::default(),
        next_cursor: ConsoleCursor {
            sequence: 2,
            text_offset: 0,
        },
        fragments: vec![
            ConsoleFragment {
                sequence: 1,
                stream: ConsoleStream::Stdout,
                text_offset: 0,
                original_text_bytes: 4,
                text: "ab".into(),
            },
            ConsoleFragment {
                sequence: 1,
                stream: ConsoleStream::Stdout,
                text_offset: 2,
                original_text_bytes: 4,
                text: "cd".into(),
            },
        ],
        observed_records: 1,
        observed_text_bytes: 4,
        completion: ConsoleCompletion::Interrupted,
        source_receipt: None,
    };
    page.validate().unwrap();
    page.fragments[1].stream = ConsoleStream::Stderr;
    assert!(page.validate().is_err());
    page.fragments[1].stream = ConsoleStream::Stdout;
    page.fragments[1].original_text_bytes = 5;
    page.next_cursor = ConsoleCursor {
        sequence: 1,
        text_offset: 4,
    };
    assert!(page.validate().is_err());
    page.fragments[1].original_text_bytes = 4;
    page.next_cursor = ConsoleCursor {
        sequence: 2,
        text_offset: 0,
    };
    page.observed_text_bytes = 3;
    assert!(page.validate().is_err());
}
