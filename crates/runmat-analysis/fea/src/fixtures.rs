use runmat_analysis_core::{
    AnalysisModel, AnalysisModelId, AnalysisStep, AnalysisStepKind, BeamElementModel,
    BeamSectionModel, BoundaryCondition, BoundaryConditionKind, EvidenceConfidence, LoadCase,
    LoadKind, MaterialAssignment, MaterialMechanicalModel, MaterialModel, MaterialThermalModel,
    ReferenceFrame, StructuralElement, StructuralElementKind, StructuralModel, StructuralNode,
};
use runmat_geometry_core::UnitSystem;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureId {
    CantileverLinearStatic,
    CantileverLoadSweep,
    CantileverLargeLoadSweep,
    StructuralAxialBarReference,
    StructuralBeamBendingReference,
    StructuralBeamCantileverEndMomentReference,
    StructuralBeamTorsionReference,
    StructuralBeamForceAndMomentReference,
    StructuralInvalidMomentWithoutRotationalDofs,
    ModalLarge,
    TransientLong,
    TransientShock,
    NonlinearAssembly,
    NonlinearAssemblyStress,
    NonlinearSofteningBenchmark,
    NonlinearLoadPathMix,
    NonlinearContactFrictionlessReference,
    NonlinearContactFrictionlessReferenceComplex,
    NonlinearPlasticHardeningReference,
    NonlinearPlasticHardeningReferenceComplex,
    ThermoMechanicalKickoff,
    ThermoGradientBenign,
    ThermoGradientPathological,
    ThermoRampSmooth,
    ThermoShockOscillatory,
    ElectroThermalJouleBenign,
    ElectroThermalJoulePathological,
    MultiMaterialAssembly,
    MissingMaterials,
    MissingLoads,
}

pub fn fixture_model(fixture: FixtureId) -> AnalysisModel {
    match fixture {
        FixtureId::CantileverLinearStatic => cantilever_linear_static(),
        FixtureId::CantileverLoadSweep => cantilever_load_sweep(),
        FixtureId::CantileverLargeLoadSweep => cantilever_large_load_sweep(),
        FixtureId::StructuralAxialBarReference => structural_axial_bar_reference(),
        FixtureId::StructuralBeamBendingReference => structural_beam_bending_reference(),
        FixtureId::StructuralBeamCantileverEndMomentReference => {
            structural_beam_cantilever_end_moment_reference()
        }
        FixtureId::StructuralBeamTorsionReference => structural_beam_torsion_reference(),
        FixtureId::StructuralBeamForceAndMomentReference => {
            structural_beam_force_and_moment_reference()
        }
        FixtureId::StructuralInvalidMomentWithoutRotationalDofs => {
            structural_invalid_moment_without_rotational_dofs()
        }
        FixtureId::ModalLarge => modal_large_fixture(),
        FixtureId::TransientLong => transient_long_fixture(),
        FixtureId::TransientShock => transient_shock_fixture(),
        FixtureId::NonlinearAssembly => nonlinear_assembly_fixture(),
        FixtureId::NonlinearAssemblyStress => nonlinear_assembly_stress_fixture(),
        FixtureId::NonlinearSofteningBenchmark => nonlinear_softening_benchmark_fixture(),
        FixtureId::NonlinearLoadPathMix => nonlinear_load_path_mix_fixture(),
        FixtureId::NonlinearContactFrictionlessReference => {
            nonlinear_contact_frictionless_reference_fixture()
        }
        FixtureId::NonlinearContactFrictionlessReferenceComplex => {
            nonlinear_contact_frictionless_reference_complex_fixture()
        }
        FixtureId::NonlinearPlasticHardeningReference => {
            nonlinear_plastic_hardening_reference_fixture()
        }
        FixtureId::NonlinearPlasticHardeningReferenceComplex => {
            nonlinear_plastic_hardening_reference_complex_fixture()
        }
        FixtureId::ThermoMechanicalKickoff => thermo_mechanical_kickoff_fixture(),
        FixtureId::ThermoGradientBenign => thermo_gradient_benign_fixture(),
        FixtureId::ThermoGradientPathological => thermo_gradient_pathological_fixture(),
        FixtureId::ThermoRampSmooth => thermo_ramp_smooth_fixture(),
        FixtureId::ThermoShockOscillatory => thermo_shock_oscillatory_fixture(),
        FixtureId::ElectroThermalJouleBenign => electro_thermal_joule_benign_fixture(),
        FixtureId::ElectroThermalJoulePathological => electro_thermal_joule_pathological_fixture(),
        FixtureId::MultiMaterialAssembly => multi_material_assembly(),
        FixtureId::MissingMaterials => missing_materials(),
        FixtureId::MissingLoads => missing_loads(),
    }
}

fn cantilever_linear_static() -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId("cantilever".to_string()),
        geometry_id: "geo:cantilever".to_string(),
        geometry_revision: 1,
        units: UnitSystem::Meter,
        frame: ReferenceFrame::Global,
        materials: vec![MaterialModel {
            material_id: "mat_steel".to_string(),
            name: "Steel".to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa: 200e9,
                poisson_ratio: 0.3,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -2.5e-4,
                ..MaterialThermalModel::default()
            },
            acoustic: None,
            electrical: None,
            plastic: None,
        }],
        material_assignments: vec![MaterialAssignment {
            region_id: "tip".to_string(),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        }],
        structural: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        boundary_conditions: vec![BoundaryCondition {
            bc_id: "bc_root".to_string(),
            region_id: "root".to_string(),
            kind: BoundaryConditionKind::Fixed,
        }],
        loads: vec![LoadCase {
            load_id: "tip_load".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Force {
                fx: 0.0,
                fy: -1000.0,
                fz: 0.0,
            },
        }],
        steps: vec![AnalysisStep {
            step_id: "static_1".to_string(),
            kind: AnalysisStepKind::Static,
        }],
    }
}

fn cantilever_load_sweep() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("cantilever_load_sweep".to_string());
    model.loads = (0..128)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.01;
            LoadCase {
                load_id: format!("tip_load_{i}"),
                region_id: format!("tip_{i}"),
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -1000.0 * scale,
                    fz: 0.0,
                },
            }
        })
        .collect();
    model
}

fn cantilever_large_load_sweep() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("cantilever_large_load_sweep".to_string());
    model.loads = (0..512)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.005;
            LoadCase {
                load_id: format!("tip_load_large_{i}"),
                region_id: format!("tip_large_{i}"),
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -800.0 * scale,
                    fz: 0.0,
                },
            }
        })
        .collect();
    model
}

fn structural_axial_bar_reference() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("structural_axial_bar_reference".to_string());
    model.geometry_id = "geo:structural_axial_bar".to_string();
    model.loads = (0..12)
        .map(|i| LoadCase {
            load_id: format!("axial_bar_tension_{i}"),
            region_id: format!("bar_station_{i}"),
            kind: LoadKind::Force {
                fx: 2_000.0,
                fy: 0.0,
                fz: 0.0,
            },
        })
        .collect();
    model.material_assignments = vec![MaterialAssignment {
        region_id: "bar_span".to_string(),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model
}

fn structural_beam_bending_reference() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("structural_beam_bending_reference".to_string());
    model.geometry_id = "geo:structural_beam_bending".to_string();
    model.loads = (0..12)
        .map(|i| {
            let span_fraction = (i + 1) as f64 / 12.0;
            LoadCase {
                load_id: format!("beam_bending_station_{i}"),
                region_id: format!("beam_station_{i}"),
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -500.0 * span_fraction,
                    fz: 0.0,
                },
            }
        })
        .collect();
    model.material_assignments = vec![MaterialAssignment {
        region_id: "beam_span".to_string(),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model
}

fn structural_beam_cantilever_end_moment_reference() -> AnalysisModel {
    structural_beam_reference_model(
        "structural_beam_cantilever_end_moment_reference",
        "geo:structural_beam_end_moment",
        vec![LoadCase {
            load_id: "tip_moment_z".to_string(),
            region_id: "node:2".to_string(),
            kind: LoadKind::Moment {
                mx: 0.0,
                my: 0.0,
                mz: 125.0,
            },
        }],
    )
}

fn structural_beam_torsion_reference() -> AnalysisModel {
    structural_beam_reference_model(
        "structural_beam_torsion_reference",
        "geo:structural_beam_torsion",
        vec![LoadCase {
            load_id: "tip_torque_x".to_string(),
            region_id: "node:2".to_string(),
            kind: LoadKind::Moment {
                mx: 80.0,
                my: 0.0,
                mz: 0.0,
            },
        }],
    )
}

fn structural_beam_force_and_moment_reference() -> AnalysisModel {
    structural_beam_reference_model(
        "structural_beam_force_and_moment_reference",
        "geo:structural_beam_force_and_moment",
        vec![
            LoadCase {
                load_id: "tip_force_y".to_string(),
                region_id: "node:2".to_string(),
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: 500.0,
                    fz: 0.0,
                },
            },
            LoadCase {
                load_id: "tip_moment_z".to_string(),
                region_id: "node:2".to_string(),
                kind: LoadKind::Moment {
                    mx: 0.0,
                    my: 0.0,
                    mz: 90.0,
                },
            },
        ],
    )
}

fn structural_invalid_moment_without_rotational_dofs() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("structural_invalid_moment_without_rotational_dofs".into());
    model.geometry_id = "geo:structural_invalid_moment_without_rotational_dofs".to_string();
    model.loads = vec![LoadCase {
        load_id: "solid_tip_moment".to_string(),
        region_id: "tip".to_string(),
        kind: LoadKind::Moment {
            mx: 0.0,
            my: 0.0,
            mz: 125.0,
        },
    }];
    model
}

fn structural_beam_reference_model(
    model_id: &str,
    geometry_id: &str,
    loads: Vec<LoadCase>,
) -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId(model_id.to_string());
    model.geometry_id = geometry_id.to_string();
    model.boundary_conditions = vec![BoundaryCondition {
        bc_id: "fixed_root".to_string(),
        region_id: "node:1".to_string(),
        kind: BoundaryConditionKind::Fixed,
    }];
    model.loads = loads;
    model.material_assignments = vec![MaterialAssignment {
        region_id: "beam_span".to_string(),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model.structural = Some(StructuralModel {
        nodes: vec![
            StructuralNode {
                node_id: 1,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            StructuralNode {
                node_id: 2,
                coordinates_m: [1.0, 0.0, 0.0],
            },
        ],
        elements: vec![StructuralElement {
            element_id: "beam_1".to_string(),
            region_id: "beam_span".to_string(),
            kind: StructuralElementKind::Beam(BeamElementModel {
                node_ids: [1, 2],
                section_id: "rect".to_string(),
                reference_axis: [0.0, 1.0, 0.0],
            }),
        }],
        beam_sections: vec![BeamSectionModel {
            section_id: "rect".to_string(),
            area_m2: 2.0e-4,
            iy_m4: 1.6e-9,
            iz_m4: 6.4e-9,
            torsion_j_m4: 2.4e-9,
            outer_fiber_y_m: 0.01,
            outer_fiber_z_m: 0.005,
            torsion_outer_radius_m: 0.011_180_339_887_498_949,
        }],
    });
    model
}

fn modal_large_fixture() -> AnalysisModel {
    let mut model = cantilever_large_load_sweep();
    model.model_id = AnalysisModelId("modal_large_fixture".to_string());
    model.steps = vec![AnalysisStep {
        step_id: "modal_large_1".to_string(),
        kind: AnalysisStepKind::Modal,
    }];
    model
}

fn transient_long_fixture() -> AnalysisModel {
    let mut model = cantilever_load_sweep();
    model.model_id = AnalysisModelId("transient_long_fixture".to_string());
    model.steps = vec![AnalysisStep {
        step_id: "transient_long_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    model
}

fn transient_shock_fixture() -> AnalysisModel {
    let mut model = cantilever_large_load_sweep();
    model.model_id = AnalysisModelId("transient_shock_fixture".to_string());
    model.boundary_conditions.push(BoundaryCondition {
        bc_id: "bc_mid_prescribed".to_string(),
        region_id: "mid_support".to_string(),
        kind: BoundaryConditionKind::PrescribedDisplacement,
    });
    model.loads = (0..256)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let scale = 1.0 + (i as f64) * 0.01;
            LoadCase {
                load_id: format!("shock_load_{i}"),
                region_id: format!("shock_region_{i}"),
                kind: LoadKind::Force {
                    fx: 50.0 * scale,
                    fy: sign * -1500.0 * scale,
                    fz: 0.0,
                },
            }
        })
        .collect();
    model.steps = vec![AnalysisStep {
        step_id: "transient_shock_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    model
}

fn nonlinear_assembly_fixture() -> AnalysisModel {
    let mut model = transient_shock_fixture();
    model.model_id = AnalysisModelId("nonlinear_assembly_fixture".to_string());
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_assembly_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    model
}

fn nonlinear_assembly_stress_fixture() -> AnalysisModel {
    let mut model = nonlinear_assembly_fixture();
    model.model_id = AnalysisModelId("nonlinear_assembly_stress_fixture".to_string());
    model.boundary_conditions.push(BoundaryCondition {
        bc_id: "bc_stress_mid_support".to_string(),
        region_id: "mid_support_stress".to_string(),
        kind: BoundaryConditionKind::PrescribedDisplacement,
    });
    model.loads = (0..640)
        .map(|i| {
            let phase = if i % 3 == 0 { -1.0 } else { 1.0 };
            let scale = 1.0 + (i as f64) * 0.003;
            LoadCase {
                load_id: format!("nonlinear_stress_load_{i}"),
                region_id: format!("nonlinear_stress_region_{}", i % 48),
                kind: LoadKind::Force {
                    fx: 75.0 * scale,
                    fy: phase * -1800.0 * scale,
                    fz: 20.0 * scale,
                },
            }
        })
        .collect();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_stress_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    model
}

fn nonlinear_softening_benchmark_fixture() -> AnalysisModel {
    let mut model = nonlinear_assembly_stress_fixture();
    model.model_id = AnalysisModelId("nonlinear_softening_benchmark_fixture".to_string());
    model.materials = vec![
        MaterialModel {
            material_id: "mat_soft_polymer".to_string(),
            name: "Soft Polymer".to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa: 1.4e9,
                poisson_ratio: 0.39,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -1.2e-3,
                ..MaterialThermalModel::default()
            },
            acoustic: None,
            electrical: None,
            plastic: None,
        },
        MaterialModel {
            material_id: "mat_aluminum".to_string(),
            name: "Aluminum".to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa: 69e9,
                poisson_ratio: 0.33,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -3.6e-4,
                ..MaterialThermalModel::default()
            },
            acoustic: None,
            electrical: None,
            plastic: None,
        },
    ];
    model.material_assignments = vec![
        MaterialAssignment {
            region_id: "nonlinear_soft_region_root".to_string(),
            expected_material_id: "mat_soft_polymer".to_string(),
            assigned_material_id: "mat_soft_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "nonlinear_soft_region_tip".to_string(),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_aluminum".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
    ];
    model.loads = (0..720)
        .map(|i| {
            let phase = if i % 4 == 0 { -1.0 } else { 1.0 };
            let drift = 1.0 + (i as f64) * 0.0025;
            LoadCase {
                load_id: format!("nonlinear_softening_load_{i}"),
                region_id: format!("nonlinear_softening_region_{}", i % 64),
                kind: LoadKind::Force {
                    fx: 65.0 * drift,
                    fy: phase * -2100.0 * drift,
                    fz: 28.0 * drift,
                },
            }
        })
        .collect();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_softening_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    model
}

fn nonlinear_load_path_mix_fixture() -> AnalysisModel {
    let mut model = multi_material_assembly();
    model.model_id = AnalysisModelId("nonlinear_load_path_mix_fixture".to_string());
    model.boundary_conditions.push(BoundaryCondition {
        bc_id: "bc_mix_path_support".to_string(),
        region_id: "mix_path_support".to_string(),
        kind: BoundaryConditionKind::PrescribedDisplacement,
    });
    model.loads = (0..480)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.0035;
            if i % 3 == 0 {
                LoadCase {
                    load_id: format!("mix_force_{i}"),
                    region_id: format!("mix_force_region_{}", i % 36),
                    kind: LoadKind::Force {
                        fx: 90.0 * scale,
                        fy: -1300.0 * scale,
                        fz: 40.0 * scale,
                    },
                }
            } else if i % 3 == 1 {
                LoadCase {
                    load_id: format!("mix_pressure_{i}"),
                    region_id: format!("mix_pressure_region_{}", i % 24),
                    kind: LoadKind::Pressure {
                        magnitude_pa: 9.0e5 * scale,
                    },
                }
            } else {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                LoadCase {
                    load_id: format!("mix_body_{i}"),
                    region_id: format!("mix_body_region_{}", i % 18),
                    kind: LoadKind::BodyForce {
                        gx: 0.35 * scale,
                        gy: sign * -9.81 * scale,
                        gz: 0.12 * scale,
                    },
                }
            }
        })
        .collect();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_mix_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    model
}

fn nonlinear_contact_frictionless_reference_fixture() -> AnalysisModel {
    let mut model = nonlinear_load_path_mix_fixture();
    model.model_id =
        AnalysisModelId("nonlinear_contact_frictionless_reference_fixture".to_string());
    model.material_assignments = vec![
        MaterialAssignment {
            region_id: "tip_steel".to_string(),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "mid_aluminum".to_string(),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_aluminum".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "polymer_segment".to_string(),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
    ];
    model
}

fn nonlinear_contact_frictionless_reference_complex_fixture() -> AnalysisModel {
    let mut model = nonlinear_contact_frictionless_reference_fixture();
    model.model_id =
        AnalysisModelId("nonlinear_contact_frictionless_reference_complex_fixture".to_string());
    model.loads = (0..560)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.004;
            if i % 2 == 0 {
                LoadCase {
                    load_id: format!("contact_ref_force_{i}"),
                    region_id: format!("contact_ref_force_region_{}", i % 32),
                    kind: LoadKind::Force {
                        fx: 80.0 * scale,
                        fy: -1400.0 * scale,
                        fz: 32.0 * scale,
                    },
                }
            } else {
                LoadCase {
                    load_id: format!("contact_ref_pressure_{i}"),
                    region_id: format!("contact_ref_pressure_region_{}", i % 28),
                    kind: LoadKind::Pressure {
                        magnitude_pa: 1.05e6 * scale,
                    },
                }
            }
        })
        .collect();
    model
}

fn nonlinear_plastic_hardening_reference_fixture() -> AnalysisModel {
    let mut model = nonlinear_load_path_mix_fixture();
    model.model_id = AnalysisModelId("nonlinear_plastic_hardening_reference_fixture".to_string());
    model.material_assignments = vec![
        MaterialAssignment {
            region_id: "tip_steel".to_string(),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "mid_aluminum".to_string(),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_aluminum".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "polymer_segment".to_string(),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
    ];
    model
}

fn nonlinear_plastic_hardening_reference_complex_fixture() -> AnalysisModel {
    let mut model = nonlinear_plastic_hardening_reference_fixture();
    model.model_id =
        AnalysisModelId("nonlinear_plastic_hardening_reference_complex_fixture".to_string());
    model.loads = (0..620)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.0045;
            if i % 3 == 0 {
                LoadCase {
                    load_id: format!("plastic_ref_force_{i}"),
                    region_id: format!("plastic_ref_force_region_{}", i % 36),
                    kind: LoadKind::Force {
                        fx: 85.0 * scale,
                        fy: -1450.0 * scale,
                        fz: 28.0 * scale,
                    },
                }
            } else if i % 3 == 1 {
                LoadCase {
                    load_id: format!("plastic_ref_pressure_{i}"),
                    region_id: format!("plastic_ref_pressure_region_{}", i % 30),
                    kind: LoadKind::Pressure {
                        magnitude_pa: 1.1e6 * scale,
                    },
                }
            } else {
                LoadCase {
                    load_id: format!("plastic_ref_body_{i}"),
                    region_id: format!("plastic_ref_body_region_{}", i % 22),
                    kind: LoadKind::BodyForce {
                        gx: 0.28 * scale,
                        gy: -9.81 * scale,
                        gz: 0.09 * scale,
                    },
                }
            }
        })
        .collect();
    model
}

fn thermo_mechanical_kickoff_fixture() -> AnalysisModel {
    let mut model = multi_material_assembly();
    model.model_id = AnalysisModelId("thermo_mechanical_kickoff_fixture".to_string());
    model.loads = (0..240)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.004;
            LoadCase {
                load_id: format!("thermo_mech_force_{i}"),
                region_id: format!("thermo_mech_region_{}", i % 24),
                kind: LoadKind::Force {
                    fx: 35.0 * scale,
                    fy: -900.0 * scale,
                    fz: 12.0 * scale,
                },
            }
        })
        .collect();
    model.steps = vec![AnalysisStep {
        step_id: "thermo_mech_transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    model
}

fn thermo_gradient_benign_fixture() -> AnalysisModel {
    let mut model = multi_material_assembly();
    model.model_id = AnalysisModelId("thermo_gradient_benign_fixture".to_string());
    model.material_assignments = vec![
        MaterialAssignment {
            region_id: "tip_steel".to_string(),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "mid_aluminum".to_string(),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_aluminum".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "polymer_segment".to_string(),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Probable,
        },
    ];
    model.loads = (0..260)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.003;
            LoadCase {
                load_id: format!("thermo_grad_benign_load_{i}"),
                region_id: format!("thermo_grad_benign_region_{}", i % 28),
                kind: LoadKind::Force {
                    fx: 30.0 * scale,
                    fy: -850.0 * scale,
                    fz: 14.0 * scale,
                },
            }
        })
        .collect();
    model.steps = vec![AnalysisStep {
        step_id: "thermo_grad_benign_transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    model
}

fn thermo_gradient_pathological_fixture() -> AnalysisModel {
    let mut model = multi_material_assembly();
    model.model_id = AnalysisModelId("thermo_gradient_pathological_fixture".to_string());
    model.material_assignments = vec![
        MaterialAssignment {
            region_id: "tip_steel".to_string(),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "mid_aluminum".to_string(),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "polymer_segment".to_string(),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Inferred,
        },
    ];
    model.loads = (0..320)
        .map(|i| {
            let scale = 1.0 + (i as f64) * 0.0038;
            LoadCase {
                load_id: format!("thermo_grad_pathological_load_{i}"),
                region_id: format!("thermo_grad_pathological_region_{}", i % 32),
                kind: LoadKind::Force {
                    fx: 52.0 * scale,
                    fy: -1150.0 * scale,
                    fz: 26.0 * scale,
                },
            }
        })
        .collect();
    model.steps = vec![AnalysisStep {
        step_id: "thermo_grad_pathological_transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    model
}

fn thermo_ramp_smooth_fixture() -> AnalysisModel {
    let mut model = thermo_gradient_benign_fixture();
    model.model_id = AnalysisModelId("thermo_ramp_smooth_fixture".to_string());
    model.loads = (0..280)
        .map(|i| {
            let scale = 0.6 + (i as f64) * 0.0025;
            LoadCase {
                load_id: format!("thermo_ramp_smooth_load_{i}"),
                region_id: format!("thermo_ramp_smooth_region_{}", i % 30),
                kind: LoadKind::Force {
                    fx: 24.0 * scale,
                    fy: -760.0 * scale,
                    fz: 10.0 * scale,
                },
            }
        })
        .collect();
    model
}

fn thermo_shock_oscillatory_fixture() -> AnalysisModel {
    let mut model = thermo_gradient_pathological_fixture();
    model.model_id = AnalysisModelId("thermo_shock_oscillatory_fixture".to_string());
    model.loads = (0..360)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let scale = 1.0 + (i as f64) * 0.004;
            LoadCase {
                load_id: format!("thermo_shock_osc_load_{i}"),
                region_id: format!("thermo_shock_osc_region_{}", i % 36),
                kind: LoadKind::Force {
                    fx: sign * 48.0 * scale,
                    fy: sign * -1320.0 * scale,
                    fz: sign * 28.0 * scale,
                },
            }
        })
        .collect();
    model
}

fn electro_thermal_joule_benign_fixture() -> AnalysisModel {
    let mut model = transient_long_fixture();
    model.model_id = AnalysisModelId("electro_thermal_joule_benign_fixture".to_string());
    model
}

fn electro_thermal_joule_pathological_fixture() -> AnalysisModel {
    let mut model = transient_shock_fixture();
    model.model_id = AnalysisModelId("electro_thermal_joule_pathological_fixture".to_string());
    model
}

fn multi_material_assembly() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("multi_material_assembly".to_string());
    model.materials = vec![
        MaterialModel {
            material_id: "mat_steel".to_string(),
            name: "Steel".to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa: 200e9,
                poisson_ratio: 0.3,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -2.5e-4,
                ..MaterialThermalModel::default()
            },
            acoustic: None,
            electrical: None,
            plastic: None,
        },
        MaterialModel {
            material_id: "mat_aluminum".to_string(),
            name: "Aluminum".to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa: 69e9,
                poisson_ratio: 0.33,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -3.6e-4,
                ..MaterialThermalModel::default()
            },
            acoustic: None,
            electrical: None,
            plastic: None,
        },
        MaterialModel {
            material_id: "mat_polymer".to_string(),
            name: "Polymer".to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa: 3.2e9,
                poisson_ratio: 0.37,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -8.0e-4,
                ..MaterialThermalModel::default()
            },
            acoustic: None,
            electrical: None,
            plastic: None,
        },
    ];

    model.boundary_conditions = vec![
        BoundaryCondition {
            bc_id: "bc_root".to_string(),
            region_id: "root".to_string(),
            kind: BoundaryConditionKind::Fixed,
        },
        BoundaryCondition {
            bc_id: "bc_interface".to_string(),
            region_id: "interface".to_string(),
            kind: BoundaryConditionKind::PrescribedDisplacement,
        },
    ];

    model.loads = vec![
        LoadCase {
            load_id: "load_tip_force".to_string(),
            region_id: "tip_steel".to_string(),
            kind: LoadKind::Force {
                fx: 0.0,
                fy: -1200.0,
                fz: 0.0,
            },
        },
        LoadCase {
            load_id: "load_mid_pressure".to_string(),
            region_id: "mid_aluminum".to_string(),
            kind: LoadKind::Pressure {
                magnitude_pa: 8.5e5,
            },
        },
        LoadCase {
            load_id: "load_body".to_string(),
            region_id: "polymer_segment".to_string(),
            kind: LoadKind::BodyForce {
                gx: 0.0,
                gy: -9.81,
                gz: 0.0,
            },
        },
    ];

    model.material_assignments = vec![
        MaterialAssignment {
            region_id: "tip_steel".to_string(),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        },
        MaterialAssignment {
            region_id: "mid_aluminum".to_string(),
            expected_material_id: "mat_aluminum".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Inferred,
        },
        MaterialAssignment {
            region_id: "polymer_segment".to_string(),
            expected_material_id: "mat_polymer".to_string(),
            assigned_material_id: "mat_polymer".to_string(),
            confidence: EvidenceConfidence::Probable,
        },
    ];

    model
}

fn missing_materials() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("missing_materials".to_string());
    model.materials.clear();
    model
}

fn missing_loads() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("missing_loads".to_string());
    model.loads.clear();
    model
}
