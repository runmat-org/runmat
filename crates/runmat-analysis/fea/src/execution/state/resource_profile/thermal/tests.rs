use super::*;
use crate::{
    execution::{
        CanonicalSemanticState, SemanticStatePublicationProfileSource,
        SemanticStatePublicationSource,
    },
    fixtures::{fixture_model, FixtureId},
    ComputeBackend, ThermalProcedureControl, ThermalSolveOptions,
};
use runmat_analysis_core::{AnalysisSequence, ProcedureStep, StateId, Temperature, ThermalDomain};
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn thermal_profiles_bound_actual_state_for_orders_histories_and_declaration_lengths() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for snapshot_count in [2, 4] {
            let mut model = fixture_model(FixtureId::CantileverLinearStatic);
            model.thermal = Some(ThermalDomain {
                reference_temperature: Temperature::new(293.15),
            });
            model.sequence = AnalysisSequence::from_steps(
                "thermal_profile",
                vec![ProcedureStep::thermal_transient("thermal")],
            );
            let result = crate::run_thermal_with_options(
                &model,
                ComputeBackend::Cpu,
                ThermalSolveOptions {
                    procedure: ThermalProcedureControl::BackwardEuler { snapshot_count },
                    solver_mesh: Some(canonical_tetrahedron_solver_mesh(order)),
                    ..ThermalSolveOptions::default()
                },
            )
            .unwrap();
            let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
            let mut previous = None;
            for name in ["t".to_owned(), "thermal_terminal_".repeat(32)] {
                let mut declaration = model.sequence.steps[0].state_outputs[0].clone();
                declaration.state_id = StateId(name);
                let request = SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::ThermalPrimaryField,
                    result.producer,
                    declaration,
                    result.topology_digest,
                    manifest.identity(),
                )
                .unwrap();
                let estimate = request
                    .encoding_profile(SemanticStatePublicationProfileSource::Thermal(&manifest))
                    .unwrap();
                let state =
                    CanonicalSemanticState::from_thermal_publication(&request, &result).unwrap();
                let actual = runmat_canonical_codec::profile_serialized(
                    &state,
                    crate::execution::state::STATE_LIMITS,
                )
                .unwrap();
                assert_eq!(estimate, actual);
                assert!(
                    state.canonical_bytes().unwrap().len() as u64
                        <= estimate.encoded_bytes_upper_bound
                );
                if let Some(smaller) = previous {
                    assert!(estimate.encoded_bytes_upper_bound > smaller);
                }
                previous = Some(estimate.encoded_bytes_upper_bound);

                let mut wrong = request.clone();
                wrong.topology_digest = Digest::sha256(b"different topology");
                assert!(profile(&wrong, &manifest).is_err());
                let mut wrong = request.clone();
                wrong.schema_version = 2;
                assert!(profile(&wrong, &manifest).is_err());
                let mut malformed = manifest.clone();
                malformed.fields.last_mut().unwrap()[0].partitions[0].entity_count += 1;
                assert!(profile(&request, &malformed).is_err());
            }
        }
    }
}
