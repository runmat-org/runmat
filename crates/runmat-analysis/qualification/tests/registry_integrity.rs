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
            execution: ExecutionEnvelope::NativeProcess,
        }));
    }
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
                execution: ExecutionEnvelope::PublicStudy,
            }));
        }
    }
}

#[test]
fn inventory_lists_required_scopes_without_verdicts() {
    let mut bytes = Vec::new();
    runmat_analysis_qualification::write_inventory(&mut bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.starts_with("inventory\t1\tdiscovery_only\n"));
    let lines = text.lines().collect::<Vec<_>>();
    for mesh in ["Tet4", "Tet10"] {
        let required = format!("required\tFEA-NONLINEAR-STATE-001\tproducer_exit\tSmallStrainJ2PenaltyCoulombContact\t{mesh}\tNativeProcess");
        assert!(lines.contains(&required.as_str()));
    }
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
        Formulation::AcousticModal,
        Formulation::AcousticTransient,
        Formulation::AcousticStructureCoupled,
        Formulation::ElectromagneticTransient,
        Formulation::SaturatingMagnetics,
        Formulation::LowMachFlow,
        Formulation::GeneralizedNewtonianFlow,
        Formulation::ViscoelasticFlow,
        Formulation::PorousMediaFlow,
        Formulation::ScalarTransport,
        Formulation::FreeSurfaceFlow,
        Formulation::RotatingFrameFlow,
        Formulation::ThinFilmLubrication,
        Formulation::Electrokinetics,
        Formulation::ParticleTracing,
        Formulation::CompressibleLaminarFlow,
        Formulation::PipeNetwork,
        Formulation::QuasiStaticTem,
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
        Stage::Meshing,
        Stage::Scale,
        Stage::PublicSurface,
    ] {
        assert!(
            requirements().any(|requirement| requirement.stages.contains(&stage)),
            "Missing standing stage: {stage:?}"
        );
    }
}

// witness: FEA-RELEASE-REGISTER-001/register_integrity @ shared_infrastructure/mesh_independent/source_check;
//   boundary: Every requirement's owning source files exist in the repository; duplicate, empty and unknown-name checks are in the sibling tests.
#[test]
fn owning_sources_exist() {
    for requirement in requirements() {
        for owner in requirement.owners {
            assert!(repo().join(owner).is_file(), "{owner}");
        }
    }
}

// witness: FEA-RELEASE-REGISTER-001/register_integrity @ shared_infrastructure/mesh_independent/source_check;
//   boundary: Every formulation, mesh order and execution name round-trips through its wire name and every required envelope is drawn from the declared vocabulary.
#[test]
fn every_declared_envelope_name_round_trips() {
    for f in Formulation::ALL {
        assert_eq!(Formulation::parse(&f.wire_name()), Some(*f));
    }
    for m in MeshOrder::ALL {
        assert_eq!(MeshOrder::parse(&m.wire_name()), Some(*m));
    }
    for e in ExecutionEnvelope::ALL {
        assert_eq!(ExecutionEnvelope::parse(&e.wire_name()), Some(*e));
    }
    for requirement in requirements() {
        for obligation in requirement.obligations {
            for envelope in obligation.required_envelopes {
                assert!(Formulation::ALL.contains(&envelope.formulation));
                assert!(MeshOrder::ALL.contains(&envelope.mesh));
                assert!(ExecutionEnvelope::ALL.contains(&envelope.execution));
            }
        }
    }
}
