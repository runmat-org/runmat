use super::*;
use crate::execution::RuntimeExecutionService;
use std::rc::Rc;

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

#[test]
fn ordinary_publication_enforces_native_four_class_and_exact_counts() {
    let runtime = runtime();
    assert!(matches!(
        SparsePublication::new(
            &runtime,
            ExactHostSparseClass::Numeric(NumericDType::I32),
            1,
            1,
            1,
        ),
        Err(SparsePublicationError::Invalid(_))
    ));

    let mut output = SparsePublication::new(
        &runtime,
        ExactHostSparseClass::Numeric(NumericDType::F64),
        1,
        1,
        1,
    )
    .unwrap();
    output.pointer(0, 0).unwrap();
    output.entry(0, 0, (2.0, 0.0)).unwrap();
    assert!(matches!(
        output.entry(1, 0, (3.0, 0.0)),
        Err(SparsePublicationError::Invalid(_))
    ));
    assert!(matches!(
        output.pointer(1, 2),
        Err(SparsePublicationError::Invalid(_))
    ));
}
