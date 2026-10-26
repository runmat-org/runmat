use super::*;
use crate::{NumericStorage, SharedShape, Tensor};

fn candidate() -> OptimizationVariableCandidate {
    OptimizationVariableCandidate {
        name: "decision".into(),
        domain: OptimizationVariableDomain::Continuous,
        shape: SharedShape::new_unmanaged(vec![2, 1]),
        index_names: crate::optimization_model::model_axis_names_unbounded(vec![None, None])
            .unwrap(),
        lower: OptimizationBoundStorage::Uniform(f64::NEG_INFINITY),
        upper: OptimizationBoundStorage::Uniform(f64::INFINITY),
    }
}

#[test]
fn original_bound_storage_is_moved_and_checked_without_class_erasure_or_eager_expansion() {
    let mut input = candidate();
    input.index_names = crate::optimization_model::model_axis_names_unbounded(vec![
        Some(vec!["first".into(), "second".into()]),
        None,
    ])
    .unwrap();
    let axis_alias = input.index_names.clone();
    let values = OptimizationBoundArray::try_from_tensor(
        Tensor::from_numeric_storage(NumericStorage::F64(vec![1.0, 2.0]), vec![2, 1]).unwrap(),
    )
    .unwrap();
    let retained = values.clone();
    input.lower = OptimizationBoundStorage::PerElement(values);
    let mut preparation = input.begin_incremental().unwrap();
    while preparation.prepare_next().unwrap() {}
    let mut validation = preparation.finish().unwrap();
    assert!(validation.validate_next().unwrap());
    assert!(validation.validate_next().unwrap());
    assert!(!validation.validate_next().unwrap());
    let data = validation.finish().unwrap();
    assert_eq!(data.index_names().as_ptr(), axis_alias.as_ptr());
    assert_eq!(
        data.index_names()[0].as_ref().unwrap().as_ptr(),
        axis_alias[0].as_ref().unwrap().as_ptr()
    );
    let OptimizationBoundView::PerElement(actual) = data.bound(OptimizationBoundRole::Lower) else {
        panic!("original double buffer");
    };
    assert_eq!(actual.as_ptr(), retained.as_f64_slice().unwrap().as_ptr());
    assert_eq!(data.bound_value(OptimizationBoundRole::Lower, 1), Some(2.0));
    assert_eq!(data.bound_value(OptimizationBoundRole::Lower, 2), None);

    let wrong = Tensor::from_numeric_storage(NumericStorage::I64(vec![1, 2]), vec![2, 1]).unwrap();
    assert!(OptimizationBoundArray::try_from_tensor(wrong).is_err());

    let mut large = candidate();
    large.shape = SharedShape::new_unmanaged(vec![usize::MAX, 1]);
    let validation = large.begin().unwrap();
    let data = validation.finish().unwrap();
    assert_eq!(data.element_count(), usize::MAX);
    assert_eq!(
        data.bound(OptimizationBoundRole::Upper),
        OptimizationBoundView::Uniform(f64::INFINITY)
    );
}

#[test]
fn incomplete_or_invalid_bounds_do_not_publish_a_definition_generation() {
    let mut input = candidate();
    input.upper = OptimizationBoundStorage::PerElement(
        OptimizationBoundArray::try_from_tensor(
            Tensor::from_numeric_storage(NumericStorage::F64(vec![1.0, f64::NAN]), vec![2, 1])
                .unwrap(),
        )
        .unwrap(),
    );
    let mut validation = input.begin().unwrap();
    assert!(validation.validate_next().unwrap());
    assert_eq!(
        validation.validate_next(),
        Err(OptimizationVariableLayoutError::BoundNaN(
            OptimizationBoundRole::Upper
        ))
    );
    assert!(matches!(
        validation.finish(),
        Err(OptimizationVariableLayoutError::IncompleteValidation)
    ));
    let mut input = candidate();
    input.lower = OptimizationBoundStorage::Uniform(f64::NAN);
    assert!(matches!(
        input.begin(),
        Err(OptimizationVariableLayoutError::BoundNaN(
            OptimizationBoundRole::Lower
        ))
    ));
}
