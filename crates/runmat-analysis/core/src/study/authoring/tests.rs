use super::*;
use crate::{AnalysisProcedure, PhysicsFamily, ProcedureStep, SolutionSpace};
use runmat_types::PhysicalDimension;

fn model() -> AuthoredModel {
    AuthoredModel {
        module_path: "project/model.fea".into(),
        model_id: "bracket".into(),
        electromagnetic: None,
        thermal: None,
        thermo_mechanical: None,
        inputs: vec![AuthoredInputDefinition {
            input_id: "load".into(),
            value_type: AuthoredInputType::Quantity(PhysicalDimension::FORCE.into()),
            default: None,
            minimum: None,
            maximum: None,
        }],
        includes: Vec::new(),
        sections: Vec::new(),
        layups: Vec::new(),
        orientations: Vec::new(),
        material_assignments: Vec::new(),
        property_assignments: Vec::new(),
        constraints: Vec::new(),
        boundaries: Vec::new(),
        connectors: Vec::new(),
        interfaces: Vec::new(),
        signals: Vec::new(),
        parameterized_signals: Vec::new(),
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
        response_spectra: Vec::new(),
        random_excitations: Vec::new(),
        initial_conditions: Vec::new(),
    }
}

#[test]
fn authored_thermal_domain_temperatures_are_explicit_finite_and_identity_sensitive() {
    let baseline = model();
    let baseline_bytes = serde_json::to_vec(&baseline).unwrap();
    for kelvin in [0.0, 293.15, 303.15] {
        let mut thermal = baseline.clone();
        thermal.thermal = Some(crate::ThermalDomain {
            reference_temperature: crate::Temperature::new(kelvin),
        });
        thermal.validate().unwrap();
        assert_ne!(serde_json::to_vec(&thermal).unwrap(), baseline_bytes);
        let previous = serde_json::to_vec(&thermal).unwrap();
        thermal.thermo_mechanical = Some(crate::ThermoMechanicalDomain {
            reference_temperature: crate::Temperature::new(kelvin),
        });
        thermal.validate().unwrap();
        assert_ne!(serde_json::to_vec(&thermal).unwrap(), previous);
        assert_eq!(
            serde_json::from_slice::<AuthoredModel>(&serde_json::to_vec(&thermal).unwrap())
                .unwrap(),
            thermal
        );
    }
    for kelvin in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let mut thermal = baseline.clone();
        thermal.thermal = Some(crate::ThermalDomain {
            reference_temperature: crate::Temperature::new(kelvin),
        });
        assert!(matches!(
            thermal.validate(),
            Err(crate::AuthoredModelError::InvalidDomainReferenceTemperature("thermal"))
        ));
        thermal.thermal = None;
        thermal.thermo_mechanical = Some(crate::ThermoMechanicalDomain {
            reference_temperature: crate::Temperature::new(kelvin),
        });
        assert!(matches!(
            thermal.validate(),
            Err(crate::AuthoredModelError::InvalidDomainReferenceTemperature("thermo_mechanical"))
        ));
    }
    let mut first = baseline.clone();
    first.thermal = Some(crate::ThermalDomain {
        reference_temperature: crate::Temperature::new(293.15),
    });
    let mut second = first.clone();
    second.thermal.as_mut().unwrap().reference_temperature = crate::Temperature::new(303.15);
    assert_ne!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap()
    );
}

fn study() -> AuthoredStudy {
    let mut step = ProcedureStep::structural_static("static");
    step.family = PhysicsFamily::Structural;
    step.procedure = AnalysisProcedure::Static;
    step.solution_space = SolutionSpace::Direct;
    AuthoredStudy {
        module_path: "project/main.fea".into(),
        study_id: "response".into(),
        geometry: AuthoredGeometryInstance {
            definition: GeometryDefinitionRef {
                module_path: "project/main.fea".into(),
                geometry_id: "part".into(),
            },
            arguments: Vec::new(),
        },
        mesh: AuthoredMeshRef {
            module_path: "project/main.fea".into(),
            mesh_id: "part".into(),
        },
        model: Some(AuthoredModelInstance {
            model: AuthoredModelRef {
                module_path: "project/model.fea".into(),
                model_id: "bracket".into(),
            },
            arguments: vec![AuthoredModelArgument {
                input_id: "load".into(),
                value: AuthoredModelArgumentValue::StudyInput(AuthoredStudyInputRef {
                    input_id: "service_load".into(),
                    value_type: AuthoredInputType::Quantity(PhysicalDimension::FORCE.into()),
                }),
            }],
        }),
        inputs: vec![AuthoredInputDefinition {
            input_id: "service_load".into(),
            value_type: AuthoredInputType::Quantity(PhysicalDimension::FORCE.into()),
            default: None,
            minimum: None,
            maximum: None,
        }],
        assessments: Vec::new(),
        outputs: Vec::new(),
        sequence: AnalysisSequence {
            sequence_id: "response".into(),
            steps: vec![step],
            dependencies: Vec::new(),
            transitions: Vec::new(),
            branches: Vec::new(),
        },
    }
}

fn geometry() -> GeometryDefinition {
    GeometryDefinition {
        module_path: "project/main.fea".into(),
        geometry_id: "part".into(),
        parameters: Vec::new(),
        source: runmat_geometry_core::GeometryDefinitionSource::Constructed(
            runmat_geometry_core::GeometryFeatureGraph {
                features: vec![runmat_geometry_core::GeometryFeature {
                    feature_id: "body".into(),
                    operation: runmat_geometry_core::GeometryFeatureOperation::Box {
                        width_m: 1.0.into(),
                        depth_m: 1.0.into(),
                        height_m: 1.0.into(),
                    },
                }],
                result: runmat_geometry_core::GeometryFeatureRef("body".into()),
            },
        ),
        regions: Vec::new(),
        cad_parameters: Vec::new(),
    }
}

#[test]
fn required_model_input_can_bind_one_exact_study_input() {
    study().validate(&model(), &geometry()).unwrap();
}

#[test]
fn realization_captures_the_geometry_source_definition() {
    let geometry = geometry();
    let realized = study()
        .realize(
            &model(),
            &geometry,
            &[crate::AuthoredStudyInputBinding {
                input_id: "service_load".into(),
                value: AuthoredInputValue::Quantity(
                    runmat_types::physical::DimensionedScalar::try_from_dimension(
                        PhysicalDimension::FORCE,
                        None,
                        1000.0,
                    )
                    .unwrap(),
                ),
            }],
        )
        .unwrap();
    assert_eq!(realized.geometry.source, geometry.source);
}

#[test]
fn direct_authored_studies_reject_invalid_output_storage_policy() {
    let mut study = study();
    study.outputs.push(AuthoredOutput {
        output_id: "displacement".into(),
        expression: AuthoredOutputExpression::AnalysisField {
            step_id: "static".into(),
            quantity: crate::ObservationQuantity::Displacement,
            location: crate::FieldLocation::Node,
        },
        policy: crate::AuthoredOutputPolicy {
            compression: crate::CompressionPolicy::ErrorBounded {
                maximum_relative_error: 1.0e-3,
            },
            purpose: crate::ObservationPurpose::SolverControl,
            ..Default::default()
        },
    });

    let error = study.validate(&model(), &geometry()).unwrap_err();
    assert!(matches!(error, AuthoredStudyError::InvalidOutput(_)));
    assert!(error.to_string().contains("error-bounded compression"));
}
