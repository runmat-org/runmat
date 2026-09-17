use super::*;

#[test]
fn material_assignment_requires_declared_strongly_typed_inputs() {
    let material = AuthoredModelInputRef {
        input_id: "material".into(),
        kind: AuthoredModelInputKind::Material,
    };
    let body = AuthoredModelInputRef {
        input_id: "body".into(),
        kind: AuthoredModelInputKind::Region(GeometryRegionRole::Solid),
    };
    AuthoredModel {
        module_path: "project/main.fea".into(),
        model_id: "bracket".into(),
        property_inputs: vec![material.clone(), body.clone()],
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
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
    }
    .validate()
    .unwrap();
}

#[test]
fn boundary_vertices_are_not_material_domains() {
    let model = AuthoredModel {
        module_path: "project/main.fea".into(),
        model_id: "invalid".into(),
        property_inputs: Vec::new(),
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
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
    };
    assert_eq!(
        model.validate(),
        Err(AuthoredModelError::WrongPropertyTargetRole {
            expected: GeometryRegionRole::Solid,
            actual: GeometryRegionRole::Vertex,
        })
    );
}
