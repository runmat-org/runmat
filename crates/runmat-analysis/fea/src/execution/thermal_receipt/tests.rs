use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    ComputeBackend, ThermalProcedureControl,
};
use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let step = ProcedureStep::thermal_transient("thermal_receipt");
    let step_id = step.step_id.clone();
    model.sequence = AnalysisSequence::from_steps("thermal_receipt_sequence", vec![step]);
    ThermalWorkload {
        initial_fields: Vec::new(),
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-8,
    }
}

#[test]
fn receipts_bind_workload_manifest_and_accepted_thermal_evidence() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let workload = workload();
        let result = workload
            .execute(&canonical_tetrahedron_solver_mesh(order))
            .unwrap();
        let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
        let receipt = ThermalNumericalReceiptArtifact::new(&workload, &result).unwrap();
        let header_receipt =
            ThermalNumericalReceiptArtifact::from_manifest(&workload, &manifest).unwrap();
        // A completely unbalanced source cannot become acceptable merely by changing
        // its power scale. Exercise both full-result and persisted-manifest admission.
        for power_w in [1.0e-12, 1.0e-9, 1.0, 1.0e9] {
            let balance = crate::contracts::ThermalEnergyBalance::from_powers(
                runmat_types::physical::Power::new(0.0),
                runmat_types::physical::Power::new(0.0),
                runmat_types::physical::Power::new(power_w),
                runmat_types::physical::Power::new(0.0),
            );
            balance.validate().unwrap();
            let mut unbalanced = result.clone();
            unbalanced.energy_balances[0] = balance;
            assert!(
                ThermalNumericalReceiptArtifact::new(&workload, &unbalanced).is_err(),
                "accepted a fully unbalanced {power_w:e} W result"
            );
            let mut unbalanced = manifest.clone();
            unbalanced.energy_balances[0] = balance;
            assert!(
                ThermalNumericalReceiptArtifact::from_manifest(&workload, &unbalanced).is_err(),
                "accepted a fully unbalanced {power_w:e} W manifest"
            );
        }
        assert_eq!(header_receipt, receipt);
        assert_eq!(
            header_receipt.canonical_bytes().unwrap(),
            receipt.canonical_bytes().unwrap()
        );
        let mut invalid_header = manifest.clone();
        invalid_header.residual_norms[0] = workload.residual_target * 2.0;
        assert!(
            ThermalNumericalReceiptArtifact::from_manifest(&workload, &invalid_header).is_err()
        );
        assert_eq!(result.energy_balances.len(), result.residual_norms.len());
        assert_eq!(
            receipt.receipt().maximum_relative_energy_residual,
            result
                .energy_balances
                .iter()
                .map(|balance| balance.relative_residual)
                .fold(0.0_f64, f64::max)
        );
        assert_eq!(result.source_workload, workload.identity().unwrap());
        assert_eq!(
            workload.identity().unwrap(),
            AnalysisWorkloadIdentity::derive(
                "thermal-workload",
                &Digest::sha256(workload.canonical_bytes().unwrap())
            )
            .unwrap()
        );
        receipt.validate_against(&workload, &manifest).unwrap();
        assert_eq!(receipt.source_result(), manifest.identity());
        assert_eq!(receipt.receipt().accepted_sample_count, 3);
        assert_eq!(
            ThermalNumericalReceiptArtifact::from_canonical_bytes(
                &receipt.canonical_bytes().unwrap()
            )
            .unwrap(),
            receipt
        );
        let mut changed = workload.clone();
        changed.residual_target *= 2.0;
        assert!(receipt.validate_against(&changed, &manifest).is_err());
        let mut other_material = workload.clone();
        other_material.model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .conductivity_w_per_mk = runmat_types::physical::Quantity::new(
            other_material.model.materials[0]
                .thermal
                .as_mut()
                .unwrap()
                .conductivity_w_per_mk
                .canonical_value()
                * (2.0),
        );
        other_material.validate().unwrap();
        assert_eq!(other_material.model.sequence, workload.model.sequence);
        assert_ne!(other_material.identity().unwrap(), result.source_workload);
        assert_ne!(
            other_material.numerical_intent_identity().unwrap(),
            result.numerical_intent
        );
        assert!(ThermalNumericalReceiptArtifact::new(&other_material, &result).is_err());
        assert!(
            ThermalNumericalReceiptArtifact::from_manifest(&other_material, &manifest).is_err()
        );
        assert!(receipt
            .validate_against(&other_material, &manifest)
            .is_err());
        let mut malformed = result.clone();
        malformed.time_points_s[1] *= 0.5;
        assert!(ThermalNumericalReceiptArtifact::new(&workload, &malformed).is_err());
        let mut malformed = result.clone();
        malformed.residual_norms[0] = workload.residual_target * 2.0;
        assert!(ThermalNumericalReceiptArtifact::new(&workload, &malformed).is_err());
        let mut malformed = result.clone();
        malformed.energy_balances.clear();
        assert!(ThermalNumericalReceiptArtifact::new(&workload, &malformed).is_err());
        let mut malformed = result.clone();
        malformed.energy_balances[0].relative_residual = f64::NAN;
        assert!(ThermalNumericalReceiptArtifact::new(&workload, &malformed).is_err());
        let mut malformed = result.clone();
        malformed.residual_target *= 2.0;
        assert!(ThermalNumericalReceiptArtifact::new(&workload, &malformed).is_err());
        let mut presentation = result.clone();
        presentation.run.diagnostics.clear();
        assert_eq!(
            ThermalNumericalReceiptArtifact::new(&workload, &presentation).unwrap(),
            receipt
        );
        let mut malformed = result.clone();
        malformed.initial_state.procedure_step_id = "other".into();
        assert!(ThermalNumericalReceiptArtifact::new(&workload, &malformed).is_err());
        let mut malformed = receipt.clone();
        malformed.receipt.reference_temperature_k = runmat_types::physical::Quantity::new(
            malformed.receipt.reference_temperature_k.canonical_value() + (1.0),
        );
        assert!(malformed.validate().is_err());
        let mut excessive_energy = receipt.clone();
        excessive_energy.receipt.maximum_relative_energy_residual = workload.residual_target * 2.0;
        excessive_energy.identity = receipt_identity(
            excessive_energy.source_workload,
            excessive_energy.source_result,
            &excessive_energy.receipt,
        )
        .unwrap();
        assert!(excessive_energy.validate().is_err());
        let encoded =
            encode_contract(PREFIX, "thermal-numerical-receipt", &malformed, LIMITS).unwrap();
        assert!(ThermalNumericalReceiptArtifact::from_canonical_bytes(&encoded).is_err());
        let mut malformed = receipt.clone();
        malformed.source_workload = changed.identity().unwrap();
        assert!(malformed.validate().is_err());
        let mut changed_result = result.clone();
        changed_result.topology_digest = Digest::sha256(b"other mesh");
        let changed_manifest = ThermalSolverResultManifest::from_result(&changed_result).unwrap();
        assert!(receipt
            .validate_against(&workload, &changed_manifest)
            .is_err());
    }
}

#[test]
fn steady_receipts_bind_one_accepted_sample_and_mandatory_workload_lineage() {
    use runmat_analysis_core::{
        AnalysisConstraint, PrescribedValue, RegionTarget, ThermalConstraint, ThermalConstraintKind,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        let step = ProcedureStep::thermal_steady("thermal_receipt");
        workload.model.sequence.steps[0] = step;
        workload.procedure = ThermalProcedureControl::Steady;
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let face = projection.boundary_faces[0].provenance[0]
            .analysis_selection_id()
            .unwrap();
        workload
            .model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: "prescribed_temperature".into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                    ),
                    value: PrescribedValue::Constant(Temperature::new(315.0)),
                },
            }));
        let result = workload.execute(&mesh).unwrap();
        assert_eq!(result.time_points_s, vec![0.0]);
        assert_eq!(result.residual_norms.len(), 1);
        assert_eq!(result.energy_balances.len(), 1);
        let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
        let receipt = ThermalNumericalReceiptArtifact::new(&workload, &result).unwrap();
        receipt.validate_against(&workload, &manifest).unwrap();
        assert_eq!(receipt.receipt().accepted_sample_count, 1);
        assert_eq!(
            ThermalNumericalReceiptArtifact::from_manifest(&workload, &manifest).unwrap(),
            receipt
        );
        assert_eq!(
            receipt.receipt().maximum_relative_linear_residual,
            result.residual_norms[0]
        );
        assert!(result.residual_norms[0] <= workload.residual_target);
        assert_eq!(
            receipt.receipt().first_sample_time_s.canonical_value(),
            receipt.receipt().terminal_sample_time_s.canonical_value()
        );
        let mut other = workload.clone();
        other.model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .conductivity_w_per_mk = runmat_types::physical::Quantity::new(
            other.model.materials[0]
                .thermal
                .as_mut()
                .unwrap()
                .conductivity_w_per_mk
                .canonical_value()
                * (2.0),
        );
        let mut tampered_manifest = manifest.clone();
        tampered_manifest.source_workload = other.identity().unwrap();
        tampered_manifest.validate().unwrap();
        assert!(receipt
            .validate_against(&workload, &tampered_manifest)
            .is_err());
        tampered_manifest.numerical_intent = other.numerical_intent_identity().unwrap();
        assert!(tampered_manifest.validate().is_err());
        let mut tampered_result = result.clone();
        tampered_result.source_workload = other.identity().unwrap();
        assert_eq!(
            tampered_result.source_result_identity().unwrap(),
            result.source_result_identity().unwrap()
        );
        assert!(ThermalNumericalReceiptArtifact::new(&workload, &tampered_result).is_err());
        assert!(ThermalNumericalReceiptArtifact::new(&other, &result).is_err());
    }
}

#[test]
fn numerical_intent_excludes_placement_and_admission_but_exact_workload_does_not() {
    let cpu = workload();
    let mut gpu = cpu.clone();
    gpu.backend = ComputeBackend::Gpu;
    for other in [&gpu] {
        assert_eq!(
            cpu.numerical_intent_identity().unwrap(),
            other.numerical_intent_identity().unwrap()
        );
        assert_ne!(cpu.identity().unwrap(), other.identity().unwrap());
    }
    let mut changed_material = cpu.clone();
    changed_material.model.materials[0]
        .thermal
        .as_mut()
        .unwrap()
        .conductivity_w_per_mk = runmat_types::physical::Quantity::new(
        changed_material.model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .conductivity_w_per_mk
            .canonical_value()
            * (2.0),
    );
    assert_ne!(
        cpu.numerical_intent_identity().unwrap(),
        changed_material.numerical_intent_identity().unwrap()
    );

    // Metadata-only identity test, not GPU execution/equivalence evidence.
    let result = cpu
        .execute(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet4))
        .unwrap();
    let cpu_receipt = ThermalNumericalReceiptArtifact::new(&cpu, &result).unwrap();
    let mut alternate_request = result.clone();
    alternate_request.source_workload = gpu.identity().unwrap();
    alternate_request.run.requested_backend = ComputeBackend::Gpu;
    assert_eq!(
        result.source_result_identity().unwrap(),
        alternate_request.source_result_identity().unwrap()
    );
    let alternate_receipt = ThermalNumericalReceiptArtifact::new(&gpu, &alternate_request).unwrap();
    assert_ne!(cpu_receipt.identity(), alternate_receipt.identity());
}

#[test]
fn linear_thermal_gpu_request_without_a_device_falls_back_with_parity_unless_required() {
    use crate::contracts::accelerator_admission::{
        device_unavailable_reason, with_accelerator_admission_policy, AcceleratorAdmissionPolicy,
        ACCELERATOR_ADMISSION_FALLBACK_CODE,
    };
    if device_unavailable_reason().is_none() {
        // A provider is registered in this process: the device path is exercised elsewhere.
        return;
    }
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let cpu = workload();
    let reference = cpu.execute(&mesh).unwrap();
    let mut gpu = cpu;
    gpu.backend = ComputeBackend::Gpu;
    let fallback = gpu.execute(&mesh).unwrap();
    assert_eq!(fallback.run.requested_backend, ComputeBackend::Gpu);
    assert_eq!(fallback.run.execution_backend, crate::ExecutionBackend::Cpu);
    let warnings = fallback
        .run
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == ACCELERATOR_ADMISSION_FALLBACK_CODE)
        .count();
    assert_eq!(warnings, 1, "{:?}", fallback.run.diagnostics);
    assert_eq!(
        fallback.temperature_snapshots,
        reference.temperature_snapshots
    );
    fallback.validate_complete().unwrap();
    let strict = with_accelerator_admission_policy(AcceleratorAdmissionPolicy::Required, || {
        gpu.execute(&mesh)
    });
    assert!(matches!(
        strict,
        Err(crate::FeaRunError::AcceleratorRequired(ref required)) if required.family == "thermal"
    ));
}
