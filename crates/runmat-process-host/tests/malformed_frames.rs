use runmat_process_host::ipc::{read_payload, write_frame, FrameLimits};

#[tokio::test]
async fn reader_rejects_oversized_length_before_allocation() {
    let limits = FrameLimits {
        max_message_bytes: 8,
    };
    let mut bytes = std::io::Cursor::new(9_u32.to_be_bytes().to_vec());
    let error = read_payload(&mut bytes, limits).await.unwrap_err();
    assert!(error.to_string().contains("negotiated maximum"));
}

#[tokio::test]
async fn writer_rejects_a_mismatched_length_header() {
    let limits = FrameLimits {
        max_message_bytes: 32,
    };
    let mut output = Vec::new();
    let mut frame = 4_u32.to_be_bytes().to_vec();
    frame.extend_from_slice(b"abc");
    let error = write_frame(&mut output, &frame, limits).await.unwrap_err();
    assert!(error.to_string().contains("does not match"));
}
