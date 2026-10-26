//! Cold typed-object execution/recovery, not a driver-crash or distributed-solve claim.
use super::{
    field_error, policy, rod_mesh, rod_model, ComputeBackend, SolverMeshArtifact, WAVE_SPEED,
};
use crate::{
    execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
        StructuralFrequencyEntryRequest, StructuralFrequencyEntryResult,
        StructuralFrequencySolverPlan, StructuralFrequencyWorkload,
        StructuralFrequencyWorkloadInput,
    },
    frequency::FrequencyResponsePlan,
    pipeline::structural_frequency::{
        execute_structural_frequency_entry, prepare_structural_frequency,
        recover_structural_frequency_entry_fields,
    },
    results::{
        field::{RecoveredFieldArtifact, StructuralRecoveredFieldTopology},
        observation::ObservationExecutionPolicy,
    },
};
use runmat_meshing_core::{CanonicalMeshingContract, StableDigest};

#[test]
fn solid_rod_harmonic_cold_workload_execution_and_field_recovery_preserve_physics() {
    cold_execution(false, false);
}

#[test]
fn solid_rod_tet10_harmonic_cold_workload_execution_and_field_recovery_preserve_physics() {
    cold_execution(true, false);
}

#[test]
fn damped_solid_rod_harmonic_cold_execution_and_recovery_preserve_phase() {
    cold_execution(false, true);
}

#[test]
fn damped_solid_rod_tet10_harmonic_cold_execution_and_recovery_preserve_phase() {
    cold_execution(true, true);
}

fn cold_execution(quadratic: bool, damped: bool) {
    let mesh = rod_mesh(if quadratic { 4 } else { 8 }, quadratic);
    cold_execution_on_mesh(mesh, quadratic, damped, None);
}

pub(super) fn cold_execution_on_mesh(
    mesh: SolverMeshArtifact,
    quadratic: bool,
    damped: bool,
    substitution: Option<SolverMeshArtifact>,
) {
    let k = 1.0;
    let f = k * WAVE_SPEED / std::f64::consts::TAU;
    let mut model = rod_model(&mesh, &[k]);
    if damped {
        super::damped::add_damping(&mut model);
    }
    let error = |real: &[f64], imaginary: &[f64]| {
        if damped {
            super::damped::field_error(&mesh, k, real, imaginary)
        } else {
            field_error(&mesh, k, real, imaginary)
        }
    };
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: runmat_analysis_core::ProcedureStepId("rod_response".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(f, &[], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct { policy: policy() },
        prestress_transition: None,
    })
    .unwrap();
    let prepared = prepare_structural_frequency(&workload, &mesh).unwrap();
    let request = StructuralFrequencyEntryRequest::new(
        &workload,
        prepared.study,
        prepared.step,
        prepared.execution_plan,
        prepared.scope.clone(),
        workload.plan().entries().unwrap()[0],
    )
    .unwrap();
    let hot = execute_structural_frequency_entry(&workload, &mesh, &request, None).unwrap();
    let cold_workload =
        StructuralFrequencyWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
            .unwrap();
    let cold_request =
        StructuralFrequencyEntryRequest::from_canonical_bytes(&request.canonical_bytes().unwrap())
            .unwrap();
    let cold_mesh: SolverMeshArtifact =
        serde_json::from_slice(&serde_json::to_vec(&mesh).unwrap()).unwrap();
    let cold = execute_structural_frequency_entry(&cold_workload, &cold_mesh, &cold_request, None)
        .unwrap();
    assert_eq!(hot.sample(), cold.sample());
    let imported =
        StructuralFrequencyEntryResult::from_canonical_bytes(&cold.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(imported, cold);
    let displacement = imported.sample().displacement();
    assert!(
        error(displacement.real(), displacement.imaginary())
            < if quadratic { 0.0005 } else { 0.005 }
    );
    let topology =
        StructuralRecoveredFieldTopology::derive(cold_workload.model(), Some(&cold_mesh)).unwrap();
    let recovery = FieldRecoveryRequest::new(
        prepared.step,
        FieldRecoverySource::StructuralFrequencyEntry(imported.identity()),
        FieldTopologyIdentity::Structural {
            topology: topology.identity,
        },
        FieldRecoveryPlan::resolve_structural(cold_workload.procedure_step().unwrap(), topology, 0)
            .unwrap(),
    )
    .unwrap();
    let fields =
        recover_structural_frequency_entry_fields(&cold_workload, &cold_mesh, &recovery, &imported)
            .unwrap();
    assert!(!fields.fields().is_empty());
    let recovered_displacement = fields
        .fields()
        .iter()
        .find(|field| {
            field.definition().quantity() == runmat_analysis_core::ObservationQuantity::Displacement
        })
        .unwrap();
    let interleaved = recovered_displacement.as_host_f64().unwrap();
    assert_eq!(interleaved.len(), displacement.real().len() * 2);
    let recovered_real = interleaved
        .chunks_exact(2)
        .map(|v| v[0])
        .collect::<Vec<_>>();
    let recovered_imaginary = interleaved
        .chunks_exact(2)
        .map(|v| v[1])
        .collect::<Vec<_>>();
    assert_eq!(recovered_real, displacement.real());
    assert_eq!(recovered_imaginary, displacement.imaginary());
    assert!(error(&recovered_real, &recovered_imaginary) < if quadratic { 0.0005 } else { 0.005 });
    assert_eq!(
        RecoveredFieldArtifact::from_canonical_bytes(&fields.canonical_bytes().unwrap()).unwrap(),
        fields
    );
    let hot_fields =
        recover_structural_frequency_entry_fields(&workload, &mesh, &recovery, &hot).unwrap();
    assert_eq!(hot_fields, fields);

    // Same node IDs and supports, but a different valid sealed physical geometry:
    // rejection must come from admitted identity, not missing constraint targets.
    let substituted_mesh = substitution.unwrap_or_else(|| {
        let mut substituted_mesh = cold_mesh.clone();
        {
            let quantity_change = 0.001;
            let quantity_index = 1;
            let quantity_slot = &mut substituted_mesh
                .topology
                .nodes
                .iter_mut()
                .find(|node| node.coordinates_m.canonical_value() == [0.5; 3])
                .unwrap()
                .coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] += quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        substituted_mesh.canonical_digest = StableDigest::ZERO;
        substituted_mesh.seal_canonical_digest().unwrap();
        substituted_mesh
    });
    substituted_mesh.validate_canonical().unwrap();
    assert!(matches!(
        execute_structural_frequency_entry(&cold_workload, &substituted_mesh, &cold_request, None),
        Err(crate::execution::AnalysisExecutionError::Identity(_))
    ));
    // An authored traction change cannot reuse the original admitted request authority.
    let mut substituted = cold_workload.model().clone();
    let runmat_analysis_core::PhysicalSignal::Traction(signal) = &mut substituted.signals[0] else {
        unreachable!()
    };
    let runmat_analysis_core::SignalRepresentation::Tabulated(table) = &mut signal.representation
    else {
        unreachable!()
    };
    table.knots[0] = runmat_analysis_core::SignalKnot::continuous(
        f,
        runmat_analysis_core::SignalOrdinate::Real(runmat_analysis_core::SignalValue::Vector3(
            [2.0, 0.0, 0.0].map(runmat_analysis_core::Traction::new),
        )),
    );
    signal.precision = runmat_analysis_core::SignalPrecision::Real64;
    let substituted = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model: substituted,
        step_id: runmat_analysis_core::ProcedureStepId("rod_response".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(f, &[], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct { policy: policy() },
        prestress_transition: None,
    })
    .unwrap();
    assert!(matches!(
        execute_structural_frequency_entry(&substituted, &cold_mesh, &cold_request, None),
        Err(crate::execution::AnalysisExecutionError::Identity(_))
    ));
}
