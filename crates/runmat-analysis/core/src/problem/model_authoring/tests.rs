use super::*;

#[test]
fn material_assignment_requires_declared_strongly_typed_inputs() {
    let material = AuthoredModelInputRef {
        input_id: "material".into(),
        value_type: AuthoredInputType::Material,
    };
    let body = AuthoredModelInputRef {
        input_id: "body".into(),
        value_type: AuthoredInputType::Region(GeometryRegionRole::Solid),
    };
    AuthoredModel {
        module_path: "project/main.fea".into(),
        model_id: "bracket".into(),
        inputs: vec![
            AuthoredInputDefinition {
                input_id: material.input_id.clone(),
                value_type: material.value_type,
                default: None,
                minimum: None,
                maximum: None,
            },
            AuthoredInputDefinition {
                input_id: body.input_id.clone(),
                value_type: body.value_type,
                default: None,
                minimum: None,
                maximum: None,
            },
        ],
        includes: Vec::new(),
        sections: Vec::new(),
        layups: Vec::new(),
        orientations: Vec::new(),
        material_assignments: vec![AuthoredMaterialAssignment {
            material: AuthoredMaterialSource::ModelInput(material),
            target: AuthoredPropertyTarget::Solid(AuthoredRegionSource::ModelInput(body)),
        }],
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
    .validate()
    .unwrap();
}

#[test]
fn boundary_vertices_are_not_material_domains() {
    let model = AuthoredModel {
        module_path: "project/main.fea".into(),
        model_id: "invalid".into(),
        inputs: Vec::new(),
        includes: Vec::new(),
        sections: Vec::new(),
        layups: Vec::new(),
        orientations: Vec::new(),
        material_assignments: vec![AuthoredMaterialAssignment {
            material: AuthoredMaterialSource::Declaration(AuthoredMaterialRef {
                module_path: "project/main.fea".into(),
                material_id: "steel".into(),
            }),
            target: AuthoredPropertyTarget::Solid(AuthoredRegionSource::Geometry(
                GeometryRegionRef {
                    module_path: "project/main.fea".into(),
                    geometry_id: "part".into(),
                    region_id: "point".into(),
                    role: GeometryRegionRole::Vertex,
                },
            )),
        }],
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
    };
    assert_eq!(
        model.validate(),
        Err(AuthoredModelError::WrongPropertyTargetRole {
            expected: GeometryRegionRole::Solid,
            actual: GeometryRegionRole::Vertex,
        })
    );
}
