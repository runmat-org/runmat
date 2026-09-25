use runmat_analysis_core::{
    AnalysisSequence, AuthoredMeshRef, AuthoredModel, FormulationRequirement, ProcedureStep,
    RealizedGeometryInstance, RealizedStudy,
};
use runmat_geometry_core::{
    GeometryAuthoringCapabilities, GeometryDefinitionRef, GeometryDefinitionSource, GeometryDigest,
    GeometrySourceFormat, ImportedGeometryDefinition, UnitSystem,
};
use runmat_meshing_core::{fixtures, ElementOrder};

use super::{
    compile_analysis_study, validate_realized_geometry_source, AnalysisStudyCompilationError,
    AnalysisStudyCompilationInput,
};

fn imported_study(document: &runmat_geometry_core::GeometryDocument) -> RealizedStudy {
    RealizedStudy {
        module_path: "project/main.fea".into(),
        study_id: "static".into(),
        geometry: RealizedGeometryInstance {
            definition: GeometryDefinitionRef {
                module_path: "project/main.fea".into(),
                geometry_id: "part".into(),
            },
            source: GeometryDefinitionSource::Imported(ImportedGeometryDefinition {
                logical_path: "project/part.brep".into(),
                format: document.source.format,
                units: document.source.source_units,
                content_digest: document.source.content_digest,
                topology_digest: GeometryDigest::from_bytes([1; 32]),
                capabilities: GeometryAuthoringCapabilities {
                    topology: true,
                    persistent_ids: true,
                    ..Default::default()
                },
                exchange: None,
            }),
            bindings: Default::default(),
        },
        mesh: AuthoredMeshRef {
            module_path: "project/main.fea".into(),
            mesh_id: "part".into(),
        },
        model: AuthoredModel {
            module_path: "project/main.fea".into(),
            model_id: "body".into(),
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
        formulation_requirement: FormulationRequirement::Experimental,
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
    imported.units = if document.source.source_units == UnitSystem::Meter {
        UnitSystem::Millimeter
    } else {
        UnitSystem::Meter
    };
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
