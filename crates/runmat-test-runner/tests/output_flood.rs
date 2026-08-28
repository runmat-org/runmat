use runmat_test::event::RedactionPolicy;

#[test]
fn captured_output_policy_truncates_on_a_utf8_boundary() {
    let policy = RedactionPolicy::new(Vec::<String>::new(), 5);
    let output = policy.redact("ééé");
    assert!(output.text.is_char_boundary(output.text.len()));
    assert!(output.truncated);
}
