use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    ComputeBackend, ThermalProcedureControl, ThermalSolveOptions,
};
use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn manifest_retains_identity_without_histories_and_rejects_malformed_catalogs() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        model.sequence = AnalysisSequence::from_steps(
            "thermal_manifest",
            vec![ProcedureStep::thermal_transient("thermal")],
        );
        let result = crate::run_thermal_with_options(
            &model,
            ComputeBackend::Cpu,
            ThermalSolveOptions {
                procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
                solver_mesh: Some(canonical_tetrahedron_solver_mesh(order)),
                ..ThermalSolveOptions::default()
            },
        )
        .unwrap();
        let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
        // Pin the workload-bound numerical identity tuple/domain, independently of the
        // shared metadata helper used by the two production entry points.
        let field_ids = manifest
            .fields
            .iter()
            .map(|entries| {
                [
                    entries[0].field,
                    entries[1].field,
                    entries[2].field,
                    entries[3].field,
                ]
            })
            .collect::<Vec<_>>();
        let expected_content = encode_contract(
            b"runmat-thermal-solver-result/v1\0",
            "thermal-solver-result-accepted-intent-field-content",
            &(
                result.numerical_intent,
                &result.run.identity,
                &result.time_points_s,
                &field_ids,
                &result.residual_norms,
                result.residual_target,
                &result.energy_balances,
                result.reference_temperature_k.canonical_value(),
                result.topology_digest,
                result.producer,
                &result.ordered_node_ids,
                &result.ordered_element_ids,
                &result.initial_state,
                &result.nonlinear_evidence,
            ),
            CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
        )
        .unwrap();
        assert_eq!(
            manifest.identity(),
            SolverResultIdentity::from_step_result(
                result.producer,
                Digest::sha256(&expected_content)
            )
            .unwrap()
        );
        assert_eq!(
            manifest.identity(),
            result.source_result_identity().unwrap()
        );
        let identity_shape = ThermalSolverResultManifest::numerical_identity_profile(
            &result.run.identity,
            result.ordered_node_ids.len(),
            result.ordered_element_ids.len(),
            result.time_points_s.len(),
            result.residual_norms.len(),
            result.interface_node_ids.len(),
        )
        .unwrap();
        let initial_shape =
            runmat_canonical_codec::profile_serialized(&result.initial_state, LIMITS).unwrap();
        let metadata = result.metadata();
        let identity_payload = metadata.identity_serialization(&field_ids);
        let actual_identity =
            runmat_canonical_codec::profile_serialized(&identity_payload, LIMITS).unwrap();
        assert!(
            actual_identity.encoded_bytes_upper_bound
                <= identity_shape.encoded_bytes_upper_bound
                    + initial_shape.encoded_bytes_upper_bound
        );
        assert!(
            actual_identity.materialization_resident_bytes_estimate
                <= identity_shape.materialization_resident_bytes_estimate
                    + initial_shape.materialization_resident_bytes_estimate
        );
        let streamed = super::thermal_result::identity::FieldIdentities {
            samples: &manifest.fields,
            interface_node_ids: &manifest.interface_node_ids,
            identities: super::thermal_result::identity::manifest_entry_identities,
        };
        let streamed_payload = metadata.identity_serialization(streamed);
        assert_eq!(
            encode_contract(
                b"runmat-thermal-solver-result/v1\0",
                "thermal-solver-result-accepted-intent-field-content",
                &streamed_payload,
                LIMITS
            )
            .unwrap(),
            expected_content
        );
        let budget = manifest.validation_resident_bytes_upper_bound().unwrap();
        let (_, _, peak) = crate::topology::measure_test_allocations(|| {
            assert_eq!(manifest.validate_admitted(budget).unwrap(), budget);
        });
        assert!(peak as u64 <= budget);
        let error = manifest.validate_admitted(budget - 1).unwrap_err();
        assert!(matches!(
            error,
            AnalysisExecutionError::Limit("thermal manifest validation workspace")
        ));
        assert!(manifest.run.fields.is_empty());
        let mut duplicate = manifest.clone();
        duplicate.fields[1][0].partitions[0].partition =
            duplicate.fields[0][0].partitions[0].partition;
        assert!(duplicate
            .validate()
            .unwrap_err()
            .to_string()
            .contains("partition coverage"));
        let schedule = crate::execution::ThermalSolveSchedule::derive(
            &model,
            &model.sequence.steps[0],
            ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        )
        .unwrap();
        let facts = crate::planning::resource::ThermalManifestCatalogResourceFacts::derive(
            &canonical_tetrahedron_solver_mesh(order),
            &schedule,
            0,
        )
        .unwrap();
        let actual = runmat_canonical_codec::profile_serialized(&manifest.fields, LIMITS).unwrap();
        assert!(actual.encoded_bytes_upper_bound <= facts.catalog.encoded_bytes_upper_bound);
        assert!(
            actual.materialization_resident_bytes_estimate
                <= facts.catalog.materialization_resident_bytes_estimate
        );
        assert_eq!(
            facts.validation_scratch_bytes.0,
            (12 * std::mem::size_of::<crate::results::field::RecoveredFieldPartitionIdentity>()
                + std::mem::size_of::<Vec<crate::results::field::RecoveredFieldPartitionIdentity>>(
                )) as u64
        );
        assert!(manifest.run.observation_histories.is_empty());
        assert_eq!(
            ThermalSolverResultManifest::from_canonical_bytes(&manifest.canonical_bytes().unwrap())
                .unwrap(),
            manifest
        );
        let mut malformed = manifest.clone();
        malformed.fields[0][0].partitions[0].first_source_index = 1;
        assert!(malformed.validate().is_err());
        let mut malformed = manifest.clone();
        malformed.fields[0][0].sample = RecoveredFieldSample::TimeSeconds(9.0);
        assert!(malformed.validate().is_err());
        let mut malformed = manifest.clone();
        malformed.fields[0].swap(0, 1);
        assert!(malformed.validate().is_err());
        let mut malformed = manifest.clone();
        malformed.fields.clear();
        assert!(malformed.validate().is_err());
        let mut malformed = manifest.clone();
        malformed
            .run
            .fields
            .push(result.temperature_snapshots[0].clone());
        assert!(malformed.validate().is_err());
        let mut malformed = manifest.clone();
        malformed.reference_temperature_k = runmat_types::physical::Quantity::new(
            malformed.reference_temperature_k.canonical_value() + (1.0),
        );
        assert!(malformed.validate().is_err());
        let mut malformed = manifest.clone();
        malformed.schema_version = 2;
        assert!(malformed.validate().is_err());
        let mut malformed = manifest.clone();
        malformed.energy_balances.pop();
        assert!(malformed.validate().is_err());
        let mut malformed = manifest.clone();
        malformed.energy_balances[0].residual_w = malformed.energy_balances[0]
            .residual_w
            .clone()
            .try_map_canonical_storage(|value| value + (1.0))
            .unwrap();
        assert!(malformed.validate().is_err());
        let mut excessive = result.clone();
        excessive.energy_balances[0] = crate::contracts::ThermalEnergyBalance::from_powers(
            runmat_types::physical::Power::new(1.0),
            runmat_types::physical::Power::new(0.0),
            runmat_types::physical::Power::new(0.0),
            runmat_types::physical::Power::new(0.0),
        );
        excessive.energy_balances[0].validate().unwrap();
        assert!(excessive.validate_complete().is_err());
        let mut changed_energy = result.clone();
        changed_energy.energy_balances[0] = crate::contracts::ThermalEnergyBalance::from_powers(
            runmat_types::physical::Power::new(1.0),
            runmat_types::physical::Power::new(0.0),
            runmat_types::physical::Power::new(1.0),
            runmat_types::physical::Power::new(0.0),
        );
        assert_ne!(
            changed_energy.source_result_identity().unwrap(),
            result.source_result_identity().unwrap()
        );
        for target in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut malformed = manifest.clone();
            malformed.residual_target = target;
            assert!(malformed.validate().is_err());
        }
        let mut presentation = manifest.clone();
        presentation.run.diagnostics.clear();
        presentation.validate().unwrap();
        assert_eq!(presentation.identity(), manifest.identity());
        let mut different_acceptance = result.clone();
        different_acceptance.residual_target *= 2.0;
        assert_ne!(
            different_acceptance.source_result_identity().unwrap(),
            result.source_result_identity().unwrap()
        );
    }
}
use crate::results::field::RecoveredFieldSample;
