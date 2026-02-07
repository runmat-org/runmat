use runmat_logging::with_signal_trace;

#[test]
fn with_signal_trace_returns_value() {
    let value = with_signal_trace(Some("0123456789abcdef0123456789abcdef"), "signal", || 7);
    assert_eq!(value, 7);
    let value = with_signal_trace(None, "signal", || 9);
    assert_eq!(value, 9);
}
