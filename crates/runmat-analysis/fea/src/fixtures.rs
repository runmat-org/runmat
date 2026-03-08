use runmat_analysis_core::{
    AnalysisModel, AnalysisModelId, AnalysisStep, AnalysisStepKind, BoundaryCondition,
    BoundaryConditionKind, EvidenceConfidence, LoadCase, LoadKind, MaterialAssignment,
    MaterialModel, ReferenceFrame,
};
use runmat_geometry_core::UnitSystem;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureId {
    CantileverLinearStatic,
    CantileverLoadSweep,
    CantileverLargeLoadSweep,
    ModalLarge,
    TransientLong,
    TransientShock,
    NonlinearAssembly,
    NonlinearAssemblyStress,
    MultiMaterialAssembly,
    MissingMaterials,
    MissingLoads,
}

pub fn fixture_model(fixture: FixtureId) -> AnalysisModel {
    match fixture {
        FixtureId::CantileverLinearStatic => cantilever_linear_static(),
        FixtureId::CantileverLoadSweep => cantilever_load_sweep(),
        FixtureId::CantileverLargeLoadSweep => cantilever_large_load_sweep(),
        FixtureId::ModalLarge => modal_large_fixture(),
        FixtureId::TransientLong => transient_long_fixture(),
        FixtureId::TransientShock => transient_shock_fixture(),
        FixtureId::NonlinearAssembly => nonlinear_assembly_fixture(),
        FixtureId::NonlinearAssemblyStress => nonlinear_assembly_stress_fixture(),
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
            youngs_modulus_pa: 200e9,
            poisson_ratio: 0.3,
        }],
        material_assignments: vec![MaterialAssignment {
            region_id: "tip".to_string(),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        }],
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

fn multi_material_assembly() -> AnalysisModel {
    let mut model = cantilever_linear_static();
    model.model_id = AnalysisModelId("multi_material_assembly".to_string());
    model.materials = vec![
        MaterialModel {
            material_id: "mat_steel".to_string(),
            name: "Steel".to_string(),
            youngs_modulus_pa: 200e9,
            poisson_ratio: 0.3,
        },
        MaterialModel {
            material_id: "mat_aluminum".to_string(),
            name: "Aluminum".to_string(),
            youngs_modulus_pa: 69e9,
            poisson_ratio: 0.33,
        },
        MaterialModel {
            material_id: "mat_polymer".to_string(),
            name: "Polymer".to_string(),
            youngs_modulus_pa: 3.2e9,
            poisson_ratio: 0.37,
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
