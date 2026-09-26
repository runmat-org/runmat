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
                formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
                procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
                solver_mesh: Some(canonical_tetrahedron_solver_mesh(order)),
                ..ThermalSolveOptions::default()
            },
        )
        .unwrap();
        let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
        // Pin the pre-manifest numerical identity tuple/domain, independently of the
        // shared metadata helper used by the two production entry points.
        let field_ids = manifest
            .fields
            .iter()
            .map(|entries| entries.each_ref().map(|entry| entry.field))
            .collect::<Vec<_>>();
        let legacy_content = encode_contract(
            b"runmat-thermal-solver-result/v1\0",
            "thermal-solver-result-field-content",
            &(
                &result.run.identity,
                &result.time_points_s,
                &field_ids,
                &result.residual_norms,
                result.reference_temperature_k,
                result.topology_digest,
                result.producer,
                &result.ordered_node_ids,
                &result.ordered_element_ids,
                &result.initial_state,
            ),
            CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
        )
        .unwrap();
        assert_eq!(
            manifest.identity(),
            SolverResultIdentity::from_step_result(result.producer, Digest::sha256(legacy_content))
                .unwrap()
        );
        assert_eq!(
            manifest.identity(),
            result.source_result_identity().unwrap()
        );
        assert!(manifest.run.fields.is_empty());
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
        malformed.reference_temperature_k += 1.0;
        assert!(malformed.validate().is_err());
        let mut malformed = manifest.clone();
        malformed.schema_version = 2;
        assert!(malformed.validate().is_err());
    }
}
