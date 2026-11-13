use runmat_analysis_core::{
    AnalysisProcedure, AnalysisSequence, AuthoredMeshRef, AuthoredModel, CoupledPhysicsFamily,
    ProcedureStep, RealizedGeometryInstance, RealizedStudy,
};
use runmat_geometry_core::{
    GeometryAuthoringCapabilities, GeometryDefinitionRef, GeometryDefinitionSource, GeometryDigest,
    GeometrySourceFormat, ImportedGeometryDefinition, LengthUnit,
};
use runmat_meshing_core::{fixtures, ElementOrder};

use super::{
    compile_analysis_study, compile_study_properties, validate_realized_geometry_source,
    AnalysisStudyCompilationError, AnalysisStudyCompilationInput, CompiledAnalysisStudy,
};

// witness: FEA-SHARED-PROPERTIES-001/optional_law_capabilities @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Structural study compilation requires a capable solid material before planning;
//   acoustic, electrical and fluid capability checks are sibling cases.
#[test]
fn structural_study_compilation_requires_capable_solid_material_before_planning() {
    let model = crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let valid = compile_study_properties(&model, &mesh).expect("valid structural properties");

    let mut thermal_only = model.clone();
    thermal_only.materials[0].elasticity = None;
    let generic = crate::properties::compile_property_map(&thermal_only, Some(&mesh))
        .expect("generic material ownership is still valid");
    assert!(matches!(
        compile_study_properties(&thermal_only, &mesh),
        Err(
            crate::properties::PropertyCompilationError::MissingCapability {
                role: "structural solid",
                capability: runmat_analysis_core::MaterialCapability::IsotropicLinearElasticity,
                ..
            }
        )
    ));

    let substituted = CompiledAnalysisStudy {
        module_path: "tests/structural.fea".into(),
        study_id: "structural".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        model: thermal_only.clone(),
        properties: generic,
        solver_mesh_digest: mesh.canonical_digest,
    };
    assert!(matches!(
        substituted.definition_digest(),
        Err(AnalysisStudyCompilationError::Properties(
            crate::properties::PropertyCompilationError::MissingCapability {
                capability: runmat_analysis_core::MaterialCapability::IsotropicLinearElasticity,
                ..
            }
        ))
    ));
    assert!(matches!(
        substituted.validate_against(&mesh),
        Err(AnalysisStudyCompilationError::Properties(
            crate::properties::PropertyCompilationError::MissingCapability {
                capability: runmat_analysis_core::MaterialCapability::IsotropicLinearElasticity,
                ..
            }
        ))
    ));

    thermal_only.sequence =
        AnalysisSequence::from_steps("thermal", vec![ProcedureStep::thermal_steady("steady")]);
    assert_eq!(
        compile_study_properties(&thermal_only, &mesh)
            .expect("thermal study accepts a material without structural elasticity")
            .identity,
        crate::properties::compile_property_map(&thermal_only, Some(&mesh))
            .unwrap()
            .identity
    );
    assert_eq!(
        valid.solid_materials().len(),
        mesh.topology.volume_elements.len()
    );
}

#[test]
fn thermal_study_compilation_rejects_uncapable_and_substituted_properties() {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.sequence =
        AnalysisSequence::from_steps("thermal", vec![ProcedureStep::thermal_steady("steady")]);
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    compile_study_properties(&model, &mesh).expect("thermal fixture capabilities");

    model.materials[0].thermal = None;
    let generic = crate::properties::compile_property_map(&model, Some(&mesh))
        .expect("ownership is valid without thermal capability");
    assert!(matches!(
        compile_study_properties(&model, &mesh),
        Err(
            crate::properties::PropertyCompilationError::MissingCapability {
                role: "thermal solid",
                capability: runmat_analysis_core::MaterialCapability::Thermal,
                ..
            }
        )
    ));

    let substituted = CompiledAnalysisStudy {
        module_path: "tests/thermal.fea".into(),
        study_id: "thermal".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        model,
        properties: generic,
        solver_mesh_digest: mesh.canonical_digest,
    };
    assert!(matches!(
        substituted.definition_digest(),
        Err(AnalysisStudyCompilationError::Properties(
            crate::properties::PropertyCompilationError::MissingCapability {
                role: "thermal solid",
                capability: runmat_analysis_core::MaterialCapability::Thermal,
                ..
            }
        ))
    ));
    assert!(matches!(
        substituted.validate_against(&mesh),
        Err(AnalysisStudyCompilationError::Properties(
            crate::properties::PropertyCompilationError::MissingCapability {
                role: "thermal solid",
                capability: runmat_analysis_core::MaterialCapability::Thermal,
                ..
            }
        ))
    ));
}

// witness: FEA-SHARED-PROPERTIES-001/optional_law_capabilities @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A thermomechanical study requires the union of solid capabilities.
#[test]
fn thermomechanical_study_requires_the_union_of_solid_capabilities() {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.sequence = AnalysisSequence::from_steps(
        "thermomechanical",
        vec![ProcedureStep::coupled(
            "static",
            CoupledPhysicsFamily::Thermomechanical,
            AnalysisProcedure::Static,
        )],
    );
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    compile_study_properties(&model, &mesh).expect("coupled solid has both capabilities");

    let mut no_thermal = model.clone();
    no_thermal.materials[0].thermal = None;
    assert!(matches!(
        compile_study_properties(&no_thermal, &mesh),
        Err(
            crate::properties::PropertyCompilationError::MissingCapability {
                role: "thermal solid",
                capability: runmat_analysis_core::MaterialCapability::Thermal,
                ..
            }
        )
    ));
    let substituted = CompiledAnalysisStudy {
        module_path: "tests/thermomechanical.fea".into(),
        study_id: "thermomechanical".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        properties: crate::properties::compile_property_map(&no_thermal, Some(&mesh))
            .expect("generic ownership remains valid"),
        model: no_thermal,
        solver_mesh_digest: mesh.canonical_digest,
    };
    assert!(matches!(
        substituted.definition_digest(),
        Err(AnalysisStudyCompilationError::Properties(
            crate::properties::PropertyCompilationError::MissingCapability {
                role: "thermal solid",
                capability: runmat_analysis_core::MaterialCapability::Thermal,
                ..
            }
        ))
    ));

    model.materials[0].elasticity = None;
    assert!(matches!(
        compile_study_properties(&model, &mesh),
        Err(
            crate::properties::PropertyCompilationError::MissingCapability {
                role: "structural solid",
                capability: runmat_analysis_core::MaterialCapability::IsotropicLinearElasticity,
                ..
            }
        )
    ));
}

fn imported_study(document: &runmat_geometry_core::GeometryDocument) -> RealizedStudy {
    RealizedStudy {
        assessments: Vec::new(),
        module_path: "project/main.fea".into(),
        study_id: "static".into(),
        geometry: RealizedGeometryInstance {
            definition: GeometryDefinitionRef {
                module_path: "project/main.fea".into(),
                geometry_id: "part".into(),
            },
            source: GeometryDefinitionSource::Imported(Box::new(ImportedGeometryDefinition {
                preview_extent_m: None,
                unit_evidence: document.source.unit_evidence.clone(),
                logical_path: "project/part.brep".into(),
                format: document.source.format,
                authored_units: document.source.unit_evidence.uniform_unit(),
                content_digest: document.source.content_digest,
                topology_digest: GeometryDigest::from_bytes([1; 32]),
                capabilities: GeometryAuthoringCapabilities {
                    topology: true,
                    persistent_ids: true,
                    ..Default::default()
                },
                exchange: None,
                shared_interfaces: false,
            })),
            bindings: Default::default(),
        },
        mesh: AuthoredMeshRef {
            module_path: "project/main.fea".into(),
            mesh_id: "part".into(),
        },
        model: AuthoredModel {
            module_path: "project/main.fea".into(),
            model_id: "body".into(),
            electromagnetic: None,
            thermal: None,
            thermo_mechanical: None,
            inputs: Vec::new(),
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
        },
        included_models: Vec::new(),
        study_inputs: Vec::new(),
        outputs: Vec::new(),
        sequence: AnalysisSequence::from_steps(
            "static",
            vec![ProcedureStep::structural_static("solve")],
        ),
    }
}

#[test]
fn imported_source_digest_format_and_units_must_match_the_document() {
    let (document, _, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let study = imported_study(&document);
    validate_realized_geometry_source(&study, &document).unwrap();

    let mut wrong_digest = study.clone();
    let GeometryDefinitionSource::Imported(imported) = &mut wrong_digest.geometry.source else {
        unreachable!()
    };
    imported.content_digest = GeometryDigest::from_bytes([2; 32]);
    assert!(matches!(
        validate_realized_geometry_source(&wrong_digest, &document),
        Err(AnalysisStudyCompilationError::ImportedGeometrySourceMismatch)
    ));

    let mut wrong_format = study.clone();
    let GeometryDefinitionSource::Imported(imported) = &mut wrong_format.geometry.source else {
        unreachable!()
    };
    imported.format = if document.source.format == GeometrySourceFormat::Brep {
        GeometrySourceFormat::Step
    } else {
        GeometrySourceFormat::Brep
    };
    assert!(matches!(
        validate_realized_geometry_source(&wrong_format, &document),
        Err(AnalysisStudyCompilationError::ImportedGeometrySourceMismatch)
    ));

    let mut wrong_units = study;
    let GeometryDefinitionSource::Imported(imported) = &mut wrong_units.geometry.source else {
        unreachable!()
    };
    imported.authored_units = Some(
        if document.source.unit_evidence.uniform_unit() == Some(LengthUnit::Meter) {
            LengthUnit::Millimeter
        } else {
            LengthUnit::Meter
        },
    );
    assert!(matches!(
        validate_realized_geometry_source(&wrong_units, &document),
        Err(AnalysisStudyCompilationError::ImportedGeometrySourceMismatch)
    ));
}

#[test]
fn compilation_rejects_a_substituted_imported_document_before_model_resolution() {
    let (document, _, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let mut study = imported_study(&document);
    let GeometryDefinitionSource::Imported(imported) = &mut study.geometry.source else {
        unreachable!()
    };
    imported.content_digest = GeometryDigest::from_bytes([3; 32]);
    let mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);

    let error = compile_analysis_study(AnalysisStudyCompilationInput {
        study: &study,
        geometry: &document,
        regions: &Default::default(),
        materials: &Default::default(),
        mesh: &mesh,
    })
    .unwrap_err();
    assert!(matches!(
        error,
        AnalysisStudyCompilationError::ImportedGeometrySourceMismatch
    ));
}
