//! Solver-agnostic analysis problem model and validation contracts.

pub mod problem {
    pub mod bc;
    pub mod domains;
    pub mod interfaces;
    pub mod loads;
    pub mod material_assignment;
    pub mod materials;
    pub mod model;
    pub mod steps;
    pub mod structure;
}
pub mod field;
pub mod validate;

pub use field::{AnalysisField, AnalysisFieldValues, DeviceFieldRef};
pub use problem::bc::{BoundaryCondition, BoundaryConditionKind};
pub use problem::domains::{
    CfdDomain, CfdSolveFamily, CfdTimeProfilePoint, ElectroRegionConductivityScale,
    ElectroThermalDomain, ElectroTimeProfilePoint, ElectromagneticDomain,
    ThermoFieldInterpolationMode, ThermoFieldSource, ThermoMechanicalDomain,
    ThermoRegionTemperatureDelta, ThermoTimeProfilePoint,
};
pub use problem::interfaces::{
    AnalysisInterface, AnalysisInterfaceKind, ConjugateHeatTransferInterfaceModel,
    ContactInterfaceModel, FluidStructureInterfaceModel,
};
pub use problem::loads::{LoadCase, LoadKind};
pub use problem::material_assignment::{EvidenceConfidence, MaterialAssignment};
pub use problem::materials::{
    ConductivityFrequencyPoint, MaterialAcousticModel, MaterialElectricalModel,
    MaterialMechanicalModel, MaterialModel, MaterialPlasticModel, MaterialThermalModel,
};
pub use problem::model::{AnalysisModel, AnalysisModelId, ReferenceFrame};
pub use problem::steps::{AnalysisStep, AnalysisStepKind};
pub use problem::structure::{
    BeamElementModel, BeamSectionModel, ShellElementModel, ShellSectionModel, StructuralElement,
    StructuralElementKind, StructuralModel, StructuralNode,
};
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
                mechanical: MaterialMechanicalModel {
                    youngs_modulus_pa: 200e9,
                    poisson_ratio: 0.3,
                    density_kg_per_m3: 7850.0,
                },
                thermal: MaterialThermalModel::default(),
                acoustic: None,
                electrical: None,
                plastic: None,
            }],
            material_assignments: Vec::new(),
            structural: None,
            thermo_mechanical: None,
            electro_thermal: None,
            electromagnetic: None,
            cfd: None,
            interfaces: Vec::new(),
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
    fn invalid_moment_vectors_fail_validation() {
        let mut model = valid_model();
        model.loads[0].kind = LoadKind::Moment {
            mx: f64::NAN,
            my: 0.0,
            mz: 1.0,
        };
        assert_eq!(
            validate_model(&model).expect_err("expected nonfinite moment validation failure"),
            AnalysisValidationError::InvalidMomentVector {
                load_id: "load_tip".to_string()
            }
        );

        let mut model = valid_model();
        model.loads[0].kind = LoadKind::Moment {
            mx: 0.0,
            my: 0.0,
            mz: 0.0,
        };
        assert_eq!(
            validate_model(&model).expect_err("expected zero moment validation failure"),
            AnalysisValidationError::ZeroMomentVector {
                load_id: "load_tip".to_string()
            }
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

    #[test]
    fn moment_load_kind_serializes_as_snake_case() {
        let load = LoadCase {
            load_id: "tip_moment".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Moment {
                mx: 1.0,
                my: 2.0,
                mz: 3.0,
            },
        };

        let json = serde_json::to_value(&load).expect("load should serialize");
        assert_eq!(json["kind"]["moment"]["mx"], 1.0);
        assert_eq!(json["kind"]["moment"]["my"], 2.0);
        assert_eq!(json["kind"]["moment"]["mz"], 3.0);

        let decoded: LoadCase = serde_json::from_value(json).expect("load should deserialize");
        assert_eq!(decoded, load);
    }

    #[test]
    fn prescribed_rotation_bc_serializes_as_snake_case() {
        let bc = BoundaryCondition {
            bc_id: "root_rotation".to_string(),
            region_id: "root".to_string(),
            kind: BoundaryConditionKind::PrescribedRotation {
                rx: 0.0,
                ry: 0.0,
                rz: 0.125,
            },
        };

        let json = serde_json::to_value(&bc).expect("bc should serialize");
        assert_eq!(json["kind"]["prescribed_rotation"]["rx"], 0.0);
        assert_eq!(json["kind"]["prescribed_rotation"]["ry"], 0.0);
        assert_eq!(json["kind"]["prescribed_rotation"]["rz"], 0.125);

        let decoded: BoundaryCondition =
            serde_json::from_value(json).expect("bc should deserialize");
        assert_eq!(decoded, bc);
    }

    #[test]
    fn structural_beam_model_round_trips() {
        let mut model = valid_model();
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
                    section_id: "section_1".to_string(),
                    reference_axis: [0.0, 0.0, 1.0],
                }),
            }],
            beam_sections: vec![BeamSectionModel {
                section_id: "section_1".to_string(),
                area_m2: 2.0e-4,
                iy_m4: 1.6e-9,
                iz_m4: 6.4e-9,
                torsion_j_m4: 2.4e-9,
                outer_fiber_y_m: 0.01,
                outer_fiber_z_m: 0.005,
                torsion_outer_radius_m: 0.011_180_339_887_498_949,
            }],
            shell_sections: Vec::new(),
        });

        let json = serde_json::to_value(&model).expect("model should serialize");
        assert_eq!(
            json["structural"]["elements"][0]["kind"]["beam"]["node_ids"][1],
            2
        );

        let decoded: AnalysisModel =
            serde_json::from_value(json).expect("model should deserialize");
        assert_eq!(decoded, model);
    }

    #[test]
    fn structural_shell_model_round_trips() {
        let mut model = valid_model();
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
                StructuralNode {
                    node_id: 3,
                    coordinates_m: [0.0, 1.0, 0.0],
                },
            ],
            elements: vec![StructuralElement {
                element_id: "shell_1".to_string(),
                region_id: "shell_panel".to_string(),
                kind: StructuralElementKind::Shell(ShellElementModel {
                    node_ids: [1, 2, 3],
                    section_id: "panel_2mm".to_string(),
                    reference_axis: [1.0, 0.0, 0.0],
                }),
            }],
            beam_sections: Vec::new(),
            shell_sections: vec![ShellSectionModel {
                section_id: "panel_2mm".to_string(),
                thickness_m: 0.002,
                shear_correction: 5.0 / 6.0,
                drilling_stiffness_scale: 1.0e-4,
            }],
        });

        let json = serde_json::to_value(&model).expect("model should serialize");
        assert_eq!(
            json["structural"]["elements"][0]["kind"]["shell"]["node_ids"][2],
            3
        );

        let decoded: AnalysisModel =
            serde_json::from_value(json).expect("model should deserialize");
        assert_eq!(decoded, model);
    }

    #[test]
    fn electrical_model_defaults_frequency_response() {
        let electrical = MaterialElectricalModel::default();
        assert!(electrical.conductivity_frequency_response.is_empty());
    }
}
