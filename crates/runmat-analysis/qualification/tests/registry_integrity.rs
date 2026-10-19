//! Register tests check linkage and drift, never numerical qualification.
use runmat_analysis_qualification::{requirement::*, requirements};
use std::path::PathBuf;
fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}
#[test]
fn register_preserves_required_native_contact_envelopes() {
    validate(requirements()).unwrap();
    let requirement = requirements()
        .find(|requirement| requirement.id == "FEA-NONLINEAR-STATE-001")
        .unwrap();
    assert_eq!(requirement.obligations.len(), 6);
    let obligation = requirement
        .obligations
        .iter()
        .find(|obligation| obligation.id == "producer_exit")
        .unwrap();
    for mesh in [MeshOrder::Tet4, MeshOrder::Tet10] {
        assert!(obligation.required_envelopes.contains(&Envelope {
            formulation: Formulation::SmallStrainJ2PenaltyCoulombContact,
            mesh,
            execution: ExecutionEnvelope::SerialCpuNativeProcess,
        }));
    }
}
#[test]
fn owning_sources_and_witnesses_exist() {
    for requirement in requirements() {
        for owner in requirement.owners {
            assert!(repo().join(owner).is_file(), "{owner}");
        }
        for witness in requirement.witnesses {
            if let WitnessLocator::CargoTest { name, source, .. } = witness.locator {
                let text = std::fs::read_to_string(repo().join(source)).unwrap();
                let function = name.rsplit("::").next().unwrap();
                assert!(
                    text.contains(&format!("fn {function}(")),
                    "{}: {name}",
                    witness.id
                );
            }
        }
    }
    // This source-level check is discovery only; Cargo listings and run results
    // must establish the qualified path and actual terminal verdict separately.
}
#[test]
fn duplicate_requirements_and_dangling_claims_cannot_enter_register() {
    let requirement = *requirements()
        .find(|requirement| requirement.id == "FEA-NONLINEAR-STATE-001")
        .unwrap();
    assert_eq!(
        validate(&[requirement, requirement]),
        Err(RegistryError::Duplicate(requirement.id))
    );
    static CLAIMS: &[CoverageClaim] = &[CoverageClaim {
        obligation: "absent",
        envelopes: &[],
        boundary: "invalid fixture",
    }];
    static WITNESSES: &[Witness] = &[Witness {
        id: "invalid",
        locator: WitnessLocator::CargoTest {
            package: "fixture",
            target: CargoTarget::Library,
            name: "fixture",
            source: "fixture.rs",
        },
        claims: CLAIMS,
    }];
    let invalid = Requirement {
        witnesses: WITNESSES,
        ..requirement
    };
    assert_eq!(
        validate(&[invalid]),
        Err(RegistryError::UnknownObligation("absent"))
    );
}
#[test]
fn witness_cannot_claim_an_execution_envelope_outside_its_obligation() {
    static ENVELOPES: &[Envelope] = &[Envelope {
        formulation: Formulation::SmallStrainJ2,
        mesh: MeshOrder::Tet4,
        execution: ExecutionEnvelope::SerialCpuNativeProcess,
    }];
    static CLAIMS: &[CoverageClaim] = &[CoverageClaim {
        obligation: "points",
        envelopes: ENVELOPES,
        boundary: "invalid fixture",
    }];
    static WITNESSES: &[Witness] = &[Witness {
        id: "invalid",
        locator: WitnessLocator::CargoTest {
            package: "fixture",
            target: CargoTarget::Library,
            name: "fixture",
            source: "fixture.rs",
        },
        claims: CLAIMS,
    }];
    let invalid = Requirement {
        witnesses: WITNESSES,
        ..*requirements()
            .find(|requirement| requirement.id == "FEA-NONLINEAR-STATE-001")
            .unwrap()
    };
    assert_eq!(
        validate(&[invalid]),
        Err(RegistryError::UnsupportedClaim("invalid"))
    );
}

#[test]
fn shared_witness_ids_reuse_one_locator_and_reject_conflicting_executables() {
    let requirement = *requirements()
        .find(|requirement| requirement.id == "FEA-NONLINEAR-STATE-001")
        .unwrap();
    let reused = Requirement {
        id: "REUSED-FIXTURE",
        ..requirement
    };
    validate(&[requirement, reused]).unwrap();
    static CLAIMS: &[CoverageClaim] = &[CoverageClaim {
        obligation: "points",
        envelopes: &[Envelope {
            formulation: Formulation::SmallStrainJ2,
            mesh: MeshOrder::Tet4,
            execution: ExecutionEnvelope::SerialCpuPreparedController,
        }],
        boundary: "invalid locator fixture",
    }];
    static WITNESSES: &[Witness] = &[Witness {
        id: "FEA-WITNESS-J2-BLOCKS",
        locator: WitnessLocator::CargoTest {
            package: "another-package",
            target: CargoTarget::Library,
            name: "another_test",
            source: "another.rs",
        },
        claims: CLAIMS,
    }];
    let conflict = Requirement {
        id: "CONFLICT-FIXTURE",
        witnesses: WITNESSES,
        ..requirement
    };
    assert_eq!(
        validate(&[requirement, conflict]),
        Err(RegistryError::ConflictingWitness("FEA-WITNESS-J2-BLOCKS"))
    );
}
#[test]
fn thermal_spatial_accuracy_keeps_steady_and_transient_orders_required() {
    let requirement = requirements()
        .find(|requirement| requirement.id == "FEA-THERMAL-ACCURACY-001")
        .unwrap();
    let obligation = requirement
        .obligations
        .iter()
        .find(|obligation| obligation.id == "space")
        .unwrap();
    for formulation in [Formulation::ThermalSteady, Formulation::ThermalTransient] {
        for mesh in [MeshOrder::Tet4, MeshOrder::Tet10] {
            assert!(obligation.required_envelopes.contains(&Envelope {
                formulation,
                mesh,
                execution: ExecutionEnvelope::SerialCpuPublicStudy,
            }));
        }
    }
}

#[test]
fn inventory_keeps_required_scopes_distinct_from_narrow_witness_claims() {
    let mut bytes = Vec::new();
    runmat_analysis_qualification::write_inventory(&mut bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.starts_with("inventory\t1\tdiscovery_only\n"));
    let lines = text.lines().collect::<Vec<_>>();
    for mesh in ["Tet4", "Tet10"] {
        let required = format!("required\tFEA-NONLINEAR-STATE-001\tproducer_exit\tSmallStrainJ2PenaltyCoulombContact\t{mesh}\tSerialCpuNativeProcess");
        assert!(lines.contains(&required.as_str()));
    }
    assert!(lines.iter().any(|row| row
        .starts_with("claim\tFEA-NONLINEAR-STATE-001\tproducer_exit\t")
        && row.contains("SmallStrainJ2\t")));
    for row in lines {
        assert!(!row.contains('\r'));
    }
}

#[test]
fn every_planned_formulation_and_stage_has_standing_obligations() {
    for formulation in [
        Formulation::SharedInfrastructure,
        Formulation::LinearStructural,
        Formulation::ModalStructural,
        Formulation::PrestressedStructural,
        Formulation::StructuralStability,
        Formulation::LinearStructuralDynamics,
        Formulation::NonlinearStructural,
        Formulation::SmallStrainJ2,
        Formulation::PenaltyCoulombContact,
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        Formulation::ThermalSteady,
        Formulation::ThermalTransient,
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell,
        Formulation::AcousticHarmonic,
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient,
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction,
        Formulation::AdaptiveAnalysis,
    ] {
        assert!(
            requirements().any(
                |requirement| requirement.obligations.iter().any(|obligation| obligation
                    .required_envelopes
                    .iter()
                    .any(|envelope| envelope.formulation == formulation))
            ),
            "Missing standing formulation: {formulation:?}"
        );
    }
    for stage in [
        Stage::Prerequisite,
        Stage::Baseline,
        Stage::Authoring,
        Stage::ExecutionContracts,
        Stage::SharedAuthorities,
        Stage::LinearStructures,
        Stage::Dynamics,
        Stage::Thermal,
        Stage::NonlinearContact,
        Stage::Electromagnetics,
        Stage::Acoustics,
        Stage::Fluid,
        Stage::FluidStructureInteraction,
        Stage::Adaptation,
        Stage::Release,
    ] {
        assert!(
            requirements().any(|requirement| requirement.stages.contains(&stage)),
            "Missing standing stage: {stage:?}"
        );
    }
}
