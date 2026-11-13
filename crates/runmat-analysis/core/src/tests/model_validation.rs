use super::*;

#[test]
fn magnetostatic_domain_requires_positive_permeability_without_frequency_or_conductivity() {
    let mut model = valid_model();
    model.electromagnetic = Some(ElectromagneticDomain::Magnetostatic {
        formulation: MagnetostaticFormulation::MagneticVectorPotential,
    });
    assert!(crate::validate_domains::validate_domains(&model).is_err());
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(0.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        conductivity_frequency_response: Vec::new(),
    });
    crate::validate_domains::validate_domains(&model).unwrap();
    for permeability in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        model.materials[0]
            .electrical
            .as_mut()
            .unwrap()
            .relative_permeability =
            runmat_types::physical::RelativePermeability::new(permeability);
        assert!(matches!(
            crate::validate_domains::validate_domains(&model),
            Err(AnalysisValidationError::InvalidMaterial { .. })
        ));
    }
}

#[test]
fn static_electrical_domain_does_not_invent_frequency_and_conduction_requires_positive_sigma() {
    let mut model = valid_model();
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(0.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(2.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        conductivity_frequency_response: Vec::new(),
    });
    model.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::Electrostatics,
    });
    crate::validate_domains::validate_domains(&model).unwrap();
    model.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::SteadyConduction,
    });
    assert!(matches!(
        crate::validate_domains::validate_domains(&model),
        Err(AnalysisValidationError::InvalidMaterial { .. })
    ));
    model.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .conductivity_s_per_m = runmat_types::physical::ElectricalConductivity::new(1.0);
    crate::validate_domains::validate_domains(&model).unwrap();
    for frequency in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
            reference_frequency: Frequency::new(frequency),
        });
        assert!(matches!(
            crate::validate_domains::validate_domains(&model),
            Err(AnalysisValidationError::InvalidDomain {
                domain: "electromagnetic",
                ..
            })
        ));
    }
}

#[test]
fn missing_material_and_condition_validation_failures() {
    let mut model = valid_model();
    model.materials.clear();
    assert_eq!(
        validate_model(&model).expect_err("expected material validation failure"),
        AnalysisValidationError::MissingMaterials
    );

    let mut model = valid_model();
    model.constraints.clear();
    assert_eq!(
        validate_model(&model).expect_err("expected boundary condition validation failure"),
        AnalysisValidationError::MissingConstraintsOrBoundaries
    );
}

#[test]
fn solver_identity_excludes_material_reconciliation_audit_fields() {
    let mut first = valid_model();
    first.material_assignments = vec![MaterialAssignment {
        target: PropertyRegionTarget::all_solids(),
        expected_material_id: "cad_steel".into(),
        assigned_material_id: "mat_steel".into(),
        confidence: EvidenceConfidence::Verified,
    }];
    let mut second = first.clone();
    second.material_assignments[0].expected_material_id = "legacy_material_name".into();
    second.material_assignments[0].confidence = EvidenceConfidence::Inferred;
    second.materials[0].name = "Imported steel display name".into();
    second.materials[0].provenance =
        MaterialProvenance::ImportedCad(ImportedCadMaterialProvenance {
            geometry_module_path: "parts/bracket.fea".into(),
            geometry_id: "bracket".into(),
            source_material_id: "cad-material-17".into(),
            source_material_name: "STEEL-GENERIC".into(),
            source_entity_id: 17,
            persistent_id: Some("material:steel".into()),
            specification: Some("imported evidence only".into()),
            associations: Vec::new(),
            adopted_properties: Vec::new(),
        });

    assert_eq!(
        serde_json::to_vec(&first.solver_identity_contract()).unwrap(),
        serde_json::to_vec(&second.solver_identity_contract()).unwrap()
    );

    second.material_assignments[0].assigned_material_id = "another_material".into();
    assert_ne!(
        serde_json::to_vec(&first.solver_identity_contract()).unwrap(),
        serde_json::to_vec(&second.solver_identity_contract()).unwrap()
    );
}

#[test]
fn frame_mismatch_rejection() {
    let mut model = valid_model();
    model.frame = CoordinateFrame::Cartesian(CartesianFrame {
        frame_id: "fixture_frame".to_string(),
        origin: Position::new([0.0, 0.0, 0.0]),
        x_axis: [1.0, 0.0, 0.0],
        y_axis: [0.0, 1.0, 0.0],
    });
    let err = validate_model_against_geometry(&model, &CoordinateFrame::Global)
        .expect_err("expected frame mismatch");
    let AnalysisValidationError::FrameMismatch { model, geometry } = err else {
        panic!("expected typed frame mismatch");
    };
    assert!(matches!(*model, CoordinateFrame::Cartesian(_)));
    assert_eq!(*geometry, CoordinateFrame::Global);
}

#[test]
fn valid_model_is_accepted() {
    let model = valid_model();
    validate_model(&model).expect("model should be valid");
    validate_model_against_geometry(&model, &CoordinateFrame::Global)
        .expect("model should match geometry context");
}

#[test]
fn prescribed_rotation_constraint_serializes_as_snake_case() {
    let constraint = AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "root_rotation".to_string(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::boundary("root"),
            translations: Vec::new(),
            rotations: vec![RotationDofValue {
                dof: StructuralRotationDof::Rz,
                value: PrescribedValue::Constant(Rotation::new(0.125)),
            }],
        }),
    });

    let json = serde_json::to_value(&constraint).expect("constraint should serialize");
    assert_eq!(
        json["structural"]["kind"]["prescribed"]["rotations"][0]["value"]["constant"]["radians"],
        0.125
    );

    let decoded: AnalysisConstraint =
        serde_json::from_value(json).expect("constraint should deserialize");
    assert_eq!(decoded, constraint);
}

#[test]
fn structural_beam_model_round_trips() {
    let mut model = valid_model();
    model.structural = Some(StructuralModel {
        nodes: vec![
            StructuralNode {
                node_id: 1,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.0, 0.0, 0.0]),
                solid_attachment: None,
            },
            StructuralNode {
                node_id: 2,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([1.0, 0.0, 0.0]),
                solid_attachment: None,
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
            area_m2: runmat_types::physical::Quantity::<runmat_types::physical::kind::Area, _>::new(
                2.0e-4,
            ),
            iy_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(1.6e-9),
            iz_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(6.4e-9),
            torsion_j_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(2.4e-9),
            outer_fiber_y_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.01),
            outer_fiber_z_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.005),
            torsion_outer_radius_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.011_180_339_887_498_949),
        }],
        shell_sections: Vec::new(),
    });

    let json = serde_json::to_value(&model).expect("model should serialize");
    assert_eq!(
        json["structural"]["elements"][0]["kind"]["beam"]["node_ids"][1],
        2
    );

    let decoded: AnalysisModel = serde_json::from_value(json).expect("model should deserialize");
    assert_eq!(decoded, model);
}

#[test]
fn structural_shell_model_round_trips() {
    let mut model = valid_model();
    model.structural = Some(StructuralModel {
        nodes: vec![
            StructuralNode {
                node_id: 1,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.0, 0.0, 0.0]),
                solid_attachment: None,
            },
            StructuralNode {
                node_id: 2,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([1.0, 0.0, 0.0]),
                solid_attachment: None,
            },
            StructuralNode {
                node_id: 3,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.0, 1.0, 0.0]),
                solid_attachment: None,
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
        shell_sections: vec![ShellSectionModel::Homogeneous(
            HomogeneousShellSectionModel {
                section_id: "panel_2mm".to_string(),
                thickness_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Length,
                    _,
                >::new(0.002),
                reference_surface: HomogeneousShellReferenceSurface::MidSurface,
                shear_correction: 5.0 / 6.0,
            },
        )],
    });

    let json = serde_json::to_value(&model).expect("model should serialize");
    assert_eq!(
        json["structural"]["elements"][0]["kind"]["shell"]["node_ids"][2],
        3
    );

    let decoded: AnalysisModel = serde_json::from_value(json).expect("model should deserialize");
    assert_eq!(decoded, model);
}

#[test]
fn electrical_frequency_response_is_explicitly_empty() {
    let electrical = MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        conductivity_frequency_response: Vec::new(),
    };
    assert!(electrical.conductivity_frequency_response.is_empty());
}

#[test]
fn duplicate_structural_dofs_are_rejected() {
    let mut model = valid_model();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "duplicate_ux".to_string(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::boundary("root"),
            translations: vec![
                TranslationDofValue {
                    dof: StructuralTranslationDof::Ux,
                    value: PrescribedValue::Constant(Displacement::new(0.0)),
                },
                TranslationDofValue {
                    dof: StructuralTranslationDof::Ux,
                    value: PrescribedValue::Constant(Displacement::new(1.0)),
                },
            ],
            rotations: Vec::new(),
        }),
    })];

    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidConstraint { .. })
    ));
}

#[test]
fn unknown_prescribed_signal_is_rejected() {
    let mut model = valid_model();
    model.constraints = vec![AnalysisConstraint::Thermal(ThermalConstraint {
        constraint_id: "temperature".to_string(),
        kind: ThermalConstraintKind::Prescribed {
            target: RegionTarget::boundary("root"),
            value: PrescribedValue::Signal(SignalId("missing_temperature".to_string())),
        },
    })];

    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidConstraint { .. })
    ));
}

#[test]
fn incompatible_target_topology_is_rejected() {
    let mut model = valid_model();
    let AnalysisConstraint::Structural(constraint) = &mut model.constraints[0] else {
        panic!("expected structural constraint")
    };
    let StructuralConstraintKind::Fixed { target } = &mut constraint.kind else {
        panic!("expected fixed constraint")
    };
    target.entity_dimension = EntityDimension::Volume;

    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidTarget { .. })
    ));
}

#[test]
fn invalid_frame_and_boundary_coefficients_are_rejected() {
    let mut model = valid_model();
    model.frame = CoordinateFrame::Cartesian(CartesianFrame {
        frame_id: "parallel_axes".to_string(),
        origin: Position::new([0.0, 0.0, 0.0]),
        x_axis: [1.0, 0.0, 0.0],
        y_axis: [2.0, 0.0, 0.0],
    });
    assert_eq!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidFrame)
    );

    let mut model = valid_model();
    model.boundaries = vec![AnalysisBoundary {
        boundary_id: "convection".to_string(),
        target: RegionTarget::boundary("root"),
        operator: BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(293.15)),
            coefficient: HeatTransferCoefficient::new(-1.0),
        },
    }];
    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidBoundary { .. })
    ));
}

#[test]
fn radiation_boundary_requires_physical_emissivity_and_absolute_ambient() {
    for (ambient, emissivity, valid) in [
        (0.0, 0.0, true),
        (300.0, 1.0, true),
        (300.0, 0.8, true),
        (-1.0, 0.8, false),
        (f64::NAN, 0.8, false),
        (300.0, -0.1, false),
        (300.0, 1.1, false),
        (300.0, f64::INFINITY, false),
    ] {
        let mut model = valid_model();
        model.boundaries = vec![AnalysisBoundary {
            boundary_id: "radiation".into(),
            target: RegionTarget::boundary("root"),
            operator: BoundaryOperator::ThermalRadiation {
                ambient_temperature: PrescribedValue::Constant(Temperature::new(ambient)),
                emissivity: SurfaceEmissivity::new(emissivity),
            },
        }];
        assert_eq!(
            validate_model(&model).is_ok(),
            valid,
            "ambient={ambient}, emissivity={emissivity}"
        );
        if valid {
            let wire = serde_json::to_vec(&model.boundaries[0]).unwrap();
            let restored: AnalysisBoundary = serde_json::from_slice(&wire).unwrap();
            assert_eq!(restored, model.boundaries[0]);
        }
    }
}
