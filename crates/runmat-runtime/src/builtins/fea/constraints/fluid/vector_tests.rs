use super::*;
use futures::executor::block_on;
use runmat_value::{CellArray, Tensor};

#[test]
fn runtime_fluid_vector_keeps_three_signed_typed_components_and_rejects_scalar_speed() {
    let scalar = |value| {
        block_on(
            crate::builtins::fea::values::velocity_meters_per_second_builtin(Value::Num(value)),
        )
        .unwrap()
    };
    let values = [-2.0, 3.0, -4.0].map(scalar);
    let vector = Value::Cell(CellArray::new(values.to_vec(), 1, 3).unwrap());
    assert_eq!(
        vector_velocity("test", &vector).unwrap(),
        PrescribedValue::Constant([-2.0, 3.0, -4.0].map(Velocity::new))
    );
    assert!(vector_velocity("test", &scalar(2.0)).is_err());
    assert!(vector_velocity("test", &Value::Num(2.0)).is_err());
    assert!(vector_velocity(
        "test",
        &Value::Cell(CellArray::new(vec![scalar(1.0), scalar(2.0)], 1, 2).unwrap())
    )
    .is_err());
    assert!(vector_velocity(
        "test",
        &Value::Cell(
            CellArray::new(
                vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
                1,
                3
            )
            .unwrap()
        )
    )
    .is_err());
}

#[test]
fn runtime_fluid_vector_signal_is_an_original_reference_not_a_magnitude() {
    let signal = block_on(crate::builtins::fea::signals::velocity_constant(
        "signed-inlet".into(),
        Value::Tensor(Tensor::new_2d(vec![-2.0, 3.0, -4.0], 1, 3).unwrap()),
    ))
    .unwrap();
    assert_eq!(
        vector_velocity("test", &signal).unwrap(),
        PrescribedValue::Signal(runmat_analysis_core::SignalId("signed-inlet".into()))
    );
}
