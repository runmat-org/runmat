use runmat_analysis_core::{
    AnalysisModel, AnalysisModelId, AnalysisStep, AnalysisStepKind, BoundaryCondition,
    BoundaryConditionKind, LoadCase, LoadKind, MaterialModel, ReferenceFrame,
};
use runmat_geometry_core::UnitSystem;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureId {
    CantileverLinearStatic,
    MissingMaterials,
    MissingLoads,
}

pub fn fixture_model(fixture: FixtureId) -> AnalysisModel {
    match fixture {
        FixtureId::CantileverLinearStatic => cantilever_linear_static(),
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
