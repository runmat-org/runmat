use super::*;
use runmat_analysis_core::*;

#[test]
fn insulated_uniform_heating_publishes_storage_density_not_total_power() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        let material = &model.materials[0];
        let capacity = material
            .mass_density_kg_per_m3
            .map(|quantity| quantity.canonical_value())
            .unwrap()
            * material
                .thermal
                .as_ref()
                .unwrap()
                .specific_heat_j_per_kgk
                .canonical_value();
        // Analytic insulated uniform solution: dT/dt=q/(rho*c)=1 K/s,
        // hence stored energy rate per volume equals q independently of element volume.
        model.signals = vec![PhysicalSignal::HeatGenerationDensity(Signal {
            signal_id: SignalId("heat".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Constant(SignalValue::Scalar(
                HeatGenerationDensity::new(capacity),
            )),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
            provenance: SignalProvenance::Authored,
        })];
        model.excitations = vec![ExcitationAction::HeatGeneration(AnalysisExcitation {
            excitation_id: ExcitationId("heat".into()),
            target: ExcitationTarget::WholeModel,
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::IntensityPerVolume,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("heat".into()),
            active_step_ids: vec!["insulated".into()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-12,
            },
        })];
        model.excitation_cases = vec![ExcitationCase {
            case_id: ExcitationCaseId("heat".into()),
            action_ids: vec![ExcitationId("heat".into())],
            activation: ExcitationCaseActivation::Always,
        }];
        let mut step = ProcedureStep::thermal_transient("insulated");
        step.observations.push(history_observation(
            "storage",
            ObservationQuantity::ThermalEnergyStorageRateDensity,
            SamplingPolicy::TimeSeconds(vec![0.0, 0.5, 1.0]),
            FieldLocation::Element,
        ));
        model.sequence.steps = vec![step];
        let result = crate::run_thermal_with_options(
            &model,
            ComputeBackend::Cpu,
            ThermalSolveOptions {
                procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
                residual_target: 1e-10,
                solver_mesh: Some(
                    runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order),
                ),
                ..baseline_thermal_options()
            },
        )
        .unwrap();
        assert_eq!(result.residual_target, 1e-10);
        assert!(result.residual_norms.iter().all(|value| *value <= 1e-10));
        assert_eq!(result.energy_balances.len(), 2);
        assert!(result
            .energy_balances
            .iter()
            .all(|balance| balance.relative_residual <= 1e-10));
        let history = &result.run.observation_histories[0];
        assert_eq!(
            history.manifest().quantity,
            ObservationQuantity::ThermalEnergyStorageRateDensity
        );
        assert_eq!(history.samples().len(), 3);
        for sample in history.samples().iter().skip(1) {
            for value in sample.result.field.as_host_f64().unwrap() {
                assert!(
                    (value / capacity - 1.0).abs() < 1e-6,
                    "{order:?}: {value} vs {capacity}"
                );
            }
        }
    }
}
