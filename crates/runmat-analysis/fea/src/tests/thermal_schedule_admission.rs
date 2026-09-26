use super::*;
use crate::{execution::ThermalWorkload, planning::resource::ThermalHistoryResourceFacts};

fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    let mut step = ProcedureStep::thermal_transient("thermal_schedule");
    for id in ["one", "two"] {
        step.observations.push(nodal_history_observation(
            id,
            runmat_analysis_core::ObservationQuantity::Temperature,
            runmat_analysis_core::SamplingPolicy::TimeSeconds(vec![0.0, 0.2, 0.65, 1.0]),
        ));
    }
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps("heat", vec![step]);
    ThermalWorkload {
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

#[test]
fn preflight_inventory_matches_actual_retained_fields_and_expanded_solve() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_samples = 5;
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let schedule = workload.solve_schedule().unwrap();
        assert_eq!(schedule.points_s(), &[0.0, 0.2, 0.5, 0.65, 1.0]);
        let facts = ThermalHistoryResourceFacts::derive(&mesh, &schedule).unwrap();
        let result = workload.execute(&mesh).unwrap();
        assert_eq!(result.time_points_s, schedule.points_s());
        assert_eq!(result.residual_norms.len(), schedule.solve_count());
        let field_values: usize = result
            .temperature_snapshots
            .iter()
            .chain(&result.temperature_gradient_snapshots)
            .chain(&result.heat_flux_snapshots)
            .chain(&result.energy_storage_rate_snapshots)
            .map(|field| field.as_host_f64().unwrap().len())
            .sum();
        assert_eq!(facts.retained_field_bytes.0, field_values as u64 * 8);
        // A nominal three-snapshot estimate would miss two mandatory recovered snapshots.
        assert_eq!(facts.samples.0, 5);
    }
}

#[test]
fn impossible_sample_or_retained_memory_ceilings_fail_before_thermal_assembly() {
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut workload = workload();
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = 4;
    assert!(matches!(
        workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "thermal solve samples",
                required: 5,
                maximum: 4
            }
        ))
    ));
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = 5;
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = 439;
    // Isolate mandatory field storage from the independently validated observation budget.
    workload.model.sequence.steps[0].observations.clear();
    workload.procedure = ThermalProcedureControl::BackwardEuler { snapshot_count: 5 };
    let error = workload.execute(&mesh).unwrap_err();
    assert!(
        matches!(
            error,
            crate::FeaRunError::Resource(
                crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    required: 440,
                    maximum: 439,
                    ..
                }
            )
        ),
        "{error}"
    );
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = 1024 * 1024;
    workload.procedure = ThermalProcedureControl::BackwardEuler {
        snapshot_count: usize::MAX,
    };
    assert!(matches!(
        workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(_))
    ));
}
