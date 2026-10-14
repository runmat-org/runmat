use super::*;
use crate::InvocationControl;

#[test]
fn generated_capture_sequence_preserves_two_output_cardinality() {
    with_native_unit(|session, unit, _| {
        let control = InvocationControl::default().force_generic_native();
        let values =
            block_on(session.invoke_executable(unit, invoke("captureSequence", 2), &control))
                .expect("generated sequence capture");
        assert_eq!(values.as_slice(), &[Value::Num(7.0), Value::Num(3.0)]);
        let error =
            block_on(session.invoke_executable(unit, invoke("captureMismatch", 3), &control))
                .expect_err("three outputs from two captured values must fail");
        let crate::RunError::Runtime(error) = error else {
            panic!("expected typed capture-cardinality error: {error:?}");
        };
        assert_eq!(
            error.identifier(),
            Some("RunMat:deal:InputOutputCountMismatch"),
            "capture-cardinality error: {error:?}"
        );
    });
}
