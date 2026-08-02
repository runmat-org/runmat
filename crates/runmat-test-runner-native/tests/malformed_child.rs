use runmat_test::protocol::ProtocolLimits;
use runmat_test_runner_native::transport::read_response;

#[tokio::test]
async fn native_reader_rejects_a_frame_above_the_negotiated_bound_before_allocation() {
    let limits = ProtocolLimits {
        max_message_bytes: 8,
        ..ProtocolLimits::default()
    };
    let mut bytes = std::io::Cursor::new(9_u32.to_be_bytes().to_vec());
    let error = read_response(&mut bytes, limits).await.unwrap_err();
    assert!(error.to_string().contains("negotiated maximum"));
}
