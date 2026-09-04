use super::super::plan;

#[test]
fn cartesian_repetition_plan_is_checked_before_materialization() {
    let plan = plan::build(&[2, 3, 4]).expect("plan");
    assert_eq!(plan.rows, 24);
    assert_eq!(plan.repetitions[0].outer, 1);
    assert_eq!(plan.repetitions[0].inner, 12);
    assert_eq!(plan.repetitions[1].outer, 2);
    assert_eq!(plan.repetitions[1].inner, 4);
    assert_eq!(plan.repetitions[2].outer, 6);
    assert_eq!(plan.repetitions[2].inner, 1);

    let too_large = plan::build(&[50_000_001]).unwrap_err();
    assert_eq!(too_large.identifier(), Some("RunMat:combinations:TooLarge"));
    let overflow = plan::build(&[usize::MAX, 2]).unwrap_err();
    assert_eq!(overflow.identifier(), Some("RunMat:combinations:TooLarge"));
}

#[test]
fn empty_products_retain_zero_rows_and_well_defined_strides() {
    let plan = plan::build(&[2, 0, 3]).expect("empty plan");
    assert_eq!(plan.rows, 0);
    assert_eq!(plan.repetitions[0].inner, 0);
    assert_eq!(plan.repetitions[1].outer, 2);
    assert_eq!(plan.repetitions[1].inner, 3);
    assert_eq!(plan.repetitions[2].outer, 0);
}
