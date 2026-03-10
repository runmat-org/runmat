//! Solver-agnostic analysis problem model and validation contracts.

pub mod problem {
    pub mod bc;
    pub mod loads;
    pub mod material_assignment;
    pub mod materials;
    pub mod model;
    pub mod steps;
}
pub mod field;
pub mod validate;

pub use field::{AnalysisField, AnalysisFieldValues, DeviceFieldRef};
pub use problem::bc::{BoundaryCondition, BoundaryConditionKind};
pub use problem::loads::{LoadCase, LoadKind};
pub use problem::material_assignment::{EvidenceConfidence, MaterialAssignment};
pub use problem::materials::MaterialModel;
pub use problem::model::{AnalysisModel, AnalysisModelId, ReferenceFrame};
pub use problem::steps::{AnalysisStep, AnalysisStepKind};
pub use validate::{validate_model, validate_model_against_geometry, AnalysisValidationError};

#[cfg(test)]
mod tests {
    use runmat_geometry_core::UnitSystem;

    use super::*;

    fn valid_model() -> AnalysisModel {
        AnalysisModel {
            model_id: AnalysisModelId("analysis_model_1".to_string()),
            geometry_id: "geo:model_1".to_string(),
            geometry_revision: 1,
            units: UnitSystem::Meter,
            frame: ReferenceFrame::Global,
            materials: vec![MaterialModel {
                material_id: "mat_steel".to_string(),
                name: "Steel".to_string(),
                youngs_modulus_pa: 200e9,
                poisson_ratio: 0.3,
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -2.5e-4,
            }],
            material_assignments: Vec::new(),
            boundary_conditions: vec![BoundaryCondition {
                bc_id: "bc_fixed_root".to_string(),
                region_id: "root".to_string(),
                kind: BoundaryConditionKind::Fixed,
            }],
            loads: vec![LoadCase {
                load_id: "load_tip".to_string(),
                region_id: "tip".to_string(),
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -1000.0,
                    fz: 0.0,
                },
            }],
            steps: vec![AnalysisStep {
                step_id: "step_static".to_string(),
                kind: AnalysisStepKind::Static,
            }],
        }
    }

    #[test]
    fn missing_material_bc_load_validation_failures() {
        let mut model = valid_model();
        model.materials.clear();
        assert_eq!(
            validate_model(&model).expect_err("expected material validation failure"),
            AnalysisValidationError::MissingMaterials
        );

        let mut model = valid_model();
        model.boundary_conditions.clear();
        assert_eq!(
            validate_model(&model).expect_err("expected boundary condition validation failure"),
            AnalysisValidationError::MissingBoundaryConditions
        );

        let mut model = valid_model();
        model.loads.clear();
        assert_eq!(
            validate_model(&model).expect_err("expected load validation failure"),
            AnalysisValidationError::MissingLoads
        );
    }

    #[test]
    fn unit_frame_mismatch_rejection() {
        let mut model = valid_model();
        model.units = UnitSystem::Inch;
        let err =
            validate_model_against_geometry(&model, UnitSystem::Meter, &ReferenceFrame::Global)
                .expect_err("expected unit mismatch");
        assert!(matches!(
            err,
            AnalysisValidationError::UnitMismatch {
                model: UnitSystem::Inch,
                geometry: UnitSystem::Meter
            }
        ));

        let mut model = valid_model();
        model.frame = ReferenceFrame::Local("fixture_frame".to_string());
        let err =
            validate_model_against_geometry(&model, UnitSystem::Meter, &ReferenceFrame::Global)
                .expect_err("expected frame mismatch");
        assert!(matches!(
            err,
            AnalysisValidationError::FrameMismatch {
                model: ReferenceFrame::Local(_),
                geometry: ReferenceFrame::Global
            }
        ));
    }

    #[test]
    fn valid_model_is_accepted() {
        let model = valid_model();
        validate_model(&model).expect("model should be valid");
        validate_model_against_geometry(&model, UnitSystem::Meter, &ReferenceFrame::Global)
            .expect("model should match geometry context");
    }
}
