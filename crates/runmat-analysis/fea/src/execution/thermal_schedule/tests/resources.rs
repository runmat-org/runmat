use super::*;

#[test]
fn original_uniform_schedule_admits_the_complete_construction_peak() {
    let model = crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let step = ProcedureStep::thermal_transient("counted-thermal");
    let procedure = ThermalProcedureControl::BackwardEuler {
        snapshot_count: 201,
    };
    let reference = ThermalSolveSchedule::derive(&model, &step, procedure).unwrap();
    let bound = reference.construction_resident_bytes();
    let (admitted, retained, peak) = crate::topology::measure_test_allocations(|| {
        ThermalSolveSchedule::derive_admitted(&model, &step, procedure, bound)
    });
    let admitted = admitted.unwrap();
    assert_eq!(admitted, reference);
    assert_eq!(admitted.points_s().len(), 201);
    for (i, point) in admitted.points_s().iter().enumerate() {
        assert_eq!(*point, i as f64 / 200.0);
    }
    assert!(peak as u64 <= bound, "{peak} exceeds {bound}");
    assert_eq!(retained, 201 * std::mem::size_of::<f64>());
    assert!(ThermalSolveSchedule::derive_admitted(&model, &step, procedure, bound - 1).is_err());
    assert_eq!(reference.points_s().last(), Some(&1.0));
}

#[test]
fn steady_schedule_needs_no_retained_heap_or_transient_projection() {
    let model = crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let step = ProcedureStep::thermal_steady("counted-steady");
    let (schedule, retained, peak) = crate::topology::measure_test_allocations(|| {
        ThermalSolveSchedule::derive_admitted(&model, &step, ThermalProcedureControl::Steady, 0)
    });
    let schedule = schedule.unwrap();
    assert_eq!(schedule.points_s(), &[0.0]);
    assert_eq!(schedule.construction_resident_bytes(), 0);
    assert_eq!((retained, peak), (0, 0));
}
