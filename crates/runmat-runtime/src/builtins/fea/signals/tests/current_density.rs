use super::*;

/// Regression: current density is a vector quantity in every constructor form.
/// Time samples were once admitted as scalars, unlike the constant and callable forms.
#[test]
fn current_density_time_samples_are_three_component_vectors() {
    let times = Value::Tensor(Tensor::new_2d(vec![0.0, 0.1], 2, 1).unwrap());
    let values = Value::Tensor(Tensor::new_2d(vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0e6], 2, 3).unwrap());
    let signal = block_on(time_samples::current_density_time_samples(
        "winding".into(),
        times.clone(),
        values,
        block_on(interpolation_linear()).unwrap(),
        block_on(extrapolation_reject()).unwrap(),
    ))
    .unwrap();
    let PhysicalSignal::ElectricCurrentDensity(signal) =
        physical_signal_from_value("test", &signal).unwrap()
    else {
        panic!("expected a current-density signal")
    };
    assert_eq!(signal.shape, SignalShape::Vector3);
    assert_eq!(signal.frame, Some(CoordinateFrame::Global));
    let SignalRepresentation::Tabulated(table) = signal.representation else {
        panic!("time samples must create a tabulated signal")
    };
    assert_eq!(table.knots.len(), 2);

    let scalar_values = Value::Tensor(Tensor::new_2d(vec![0.0, 2.0e6], 2, 1).unwrap());
    assert!(block_on(time_samples::current_density_time_samples(
        "winding".into(),
        times,
        scalar_values,
        block_on(interpolation_linear()).unwrap(),
        block_on(extrapolation_reject()).unwrap(),
    ))
    .is_err());
}
