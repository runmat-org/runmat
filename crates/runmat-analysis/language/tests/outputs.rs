mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(source: &str) -> CheckedFeaProgram {
    let host = TestHost::default()
        .with_module("project/unused.fea", "", 2)
        .with_geometry("project/unused.step", 3)
        .with_function("project/unused.m", "unused", Vec::new(), Vec::new());
    FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

const HARMONIC_STUDY: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {
}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis harmonic {
        structural harmonic {
            frequencies = [10 Hz, 20 Hz]
        }
    }
    output stress = analysis.harmonic.field.element.stress
    output real_stress = real(output.stress)
    output von_mises_stress = von_mises(output.real_stress)
    output displacement_x = component(
        analysis.harmonic.field.node.displacement,
        component = x,
    )
    output displacement_phase = component(
        phase(analysis.harmonic.field.node.displacement),
        component = x,
    )
    output peak_von_mises = max(
        output.von_mises_stress,
        over = geometry.part.region.body,
    )
}
"#;

#[test]
fn complex_outputs_require_explicit_representations_before_physical_reductions() {
    let program = check(HARMONIC_STUDY);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let outputs = &program.studies.values().next().unwrap().outputs;
    assert_eq!(outputs.len(), 6);
    let step = &program.studies.values().next().unwrap().sequence.steps[0];
    assert_eq!(step.observations.len(), 6);
    let projected = step.observations[2].spec().unwrap();
    assert_eq!(
        step.observations[0].spec().unwrap().storage.precision,
        runmat_analysis_core::SignalPrecision::Complex128
    );
    assert_eq!(
        projected.source_location,
        runmat_analysis_core::FieldLocation::Element
    );
    assert_eq!(
        projected.storage.precision,
        runmat_analysis_core::SignalPrecision::Real64
    );
    assert_eq!(
        projected.component,
        runmat_analysis_core::ObservationComponent::VonMises
    );
    assert_eq!(
        projected.result_quantity(),
        runmat_analysis_core::ObservationResultQuantity::Physical(
            runmat_analysis_core::ObservationQuantity::VonMisesStress
        )
    );
    assert_eq!(
        projected.cost,
        runmat_analysis_core::ObservationCostPolicy::Automatic
    );
    assert!(matches!(
        outputs[1].expression,
        runmat_analysis_core::AuthoredOutputExpression::Projection {
            projection: runmat_analysis_core::AuthoredFieldProjection::Real,
            ..
        }
    ));
    assert!(matches!(
        outputs[2].expression,
        runmat_analysis_core::AuthoredOutputExpression::Projection {
            projection: runmat_analysis_core::AuthoredFieldProjection::VonMises,
            ..
        }
    ));
    assert!(matches!(
        outputs[3].expression,
        runmat_analysis_core::AuthoredOutputExpression::Projection {
            projection: runmat_analysis_core::AuthoredFieldProjection::Component(
                runmat_analysis_core::AuthoredVectorComponent::X
            ),
            ..
        }
    ));
}

// witness: FEA-AUTHORING-OUTPUT-002/typed_output_queries @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Integration-point fields are typed output locations; node, element and constraint
//   locations are covered by sibling tests.
#[test]
fn integration_point_fields_are_typed_output_locations() {
    let source = HARMONIC_STUDY.replacen(
        "analysis.harmonic.field.element.stress",
        "analysis.harmonic.field.integration_point.stress",
        1,
    );
    let program = check(&source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let stress = program.studies.values().next().unwrap().sequence.steps[0].observations[0]
        .spec()
        .unwrap();
    assert_eq!(
        stress.source_location,
        runmat_analysis_core::FieldLocation::IntegrationPoint
    );
    assert_eq!(
        stress.output_location,
        runmat_analysis_core::FieldLocation::IntegrationPoint
    );
}

#[test]
fn complex_stress_cannot_claim_a_real_von_mises_invariant() {
    let source =
        HARMONIC_STUDY.replace("von_mises(output.real_stress)", "von_mises(output.stress)");
    let program = check(&source);
    assert!(!program.is_valid());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("real symmetric stress field")));
    assert!(program.studies.is_empty());
}

// witness: FEA-AUTHORING-OUTPUT-002/closed_query_surface @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Projections and spatial reductions lower with exact dimensions and locations; the
//   whole closed list is not enumerated.
// witness: FEA-AUTHORING-OUTPUT-002/physical_query_dimensions @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Reductions carry exact output dimensions; principal-stress admission rules are not
//   asserted.
#[test]
fn projections_and_spatial_reductions_lower_with_exact_dimensions_and_locations() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
    region loaded : face {
        select result
        expect one_or_more
    }
}

mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}

model structure {}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis static { structural static {} }
    output displacement = analysis.static.field.node.displacement
    output stress = analysis.static.field.element.stress
    output maximum_principal = principal(output.stress, which = maximum)
    output integrated_displacement = integral(output.displacement, over = geometry.part.region.body)
    output mean_displacement = mean(output.displacement, over = geometry.part.region.body)
    output rms_displacement = rms(output.displacement, over = geometry.part.region.body)
    output displacement_norm = norm(output.displacement, over = geometry.part.region.body)
    output force = resultant(output.stress, over = geometry.part.region.loaded)
    output force_moment = moment(
        output.stress,
        over = geometry.part.region.loaded,
        about = [0, 0, 0] m,
    )
}
"#;
    let program = check(source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let declaration = program
        .declarations
        .values()
        .find(|declaration| declaration.key.name == "response")
        .unwrap();
    assert_eq!(
        declaration.outputs["maximum_principal"],
        FeaType::Field(FeaFieldType {
            value: Box::new(FeaType::Quantity(
                runmat_types::physical::DynamicDimension::try_new(
                    PhysicalDimension::PRESSURE,
                    Some(runmat_types::physical::QuantityKind::Pressure),
                )
                .unwrap(),
            )),
            location: Some(runmat_analysis_core::FieldLocation::Element),
            quantity: Some(runmat_analysis_core::ObservationQuantity::Stress),
        })
    );
    assert_eq!(
        declaration.outputs["integrated_displacement"],
        FeaType::Vector {
            element: Box::new(FeaType::Quantity(
                (PhysicalDimension::LENGTH
                    .multiply(PhysicalDimension::VOLUME)
                    .unwrap())
                .into()
            )),
            length: Some(3),
        }
    );
    assert_eq!(
        declaration.outputs["force"],
        FeaType::Vector {
            element: Box::new(FeaType::Quantity((PhysicalDimension::FORCE).into())),
            length: Some(3),
        }
    );
    assert_eq!(
        declaration.outputs["force_moment"],
        FeaType::Vector {
            element: Box::new(FeaType::Quantity((PhysicalDimension::MOMENT).into())),
            length: Some(3),
        }
    );
    let study = program.studies.values().next().unwrap();
    assert_eq!(study.sequence.steps[0].observations.len(), 9);
    assert_eq!(
        study.sequence.steps[0].observations[2]
            .spec()
            .unwrap()
            .component,
        runmat_analysis_core::ObservationComponent::MaximumPrincipal
    );
    assert!(matches!(
        study.sequence.steps[0].observations[8]
            .spec()
            .unwrap()
            .reduction
            .as_ref()
            .unwrap()
            .spatial,
        runmat_analysis_core::SpatialReduction::Moment { .. }
    ));
    assert_eq!(
        study.sequence.steps[0].observations[3]
            .spec()
            .unwrap()
            .result_dimension(),
        Some(
            PhysicalDimension::LENGTH
                .multiply(PhysicalDimension::VOLUME)
                .unwrap()
        )
    );
    assert_eq!(
        study.sequence.steps[0].observations[7]
            .spec()
            .unwrap()
            .result_dimension(),
        Some(PhysicalDimension::FORCE)
    );
    assert_eq!(
        study.sequence.steps[0].observations[8]
            .spec()
            .unwrap()
            .result_dimension(),
        Some(PhysicalDimension::MOMENT)
    );
}

// witness: FEA-AUTHORING-OUTPUT-002/typed_output_queries @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Recovery is an explicit typed element-to-node operation.
#[test]
fn recovery_is_an_explicit_typed_element_to_node_operation() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis static { structural static {} }
    output element_stress = analysis.static.field.element.stress
    output element_von_mises = von_mises(output.element_stress)
    output nodal_von_mises = recover(
        output.element_von_mises,
        to = node,
        using = volume_weighted_average,
    )
}
"#;
    let program = check(source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let study = program.studies.values().next().unwrap();
    let recovered = study.sequence.steps[0].observations[2].spec().unwrap();
    assert_eq!(
        recovered.source_location,
        runmat_analysis_core::FieldLocation::Element
    );
    assert_eq!(
        recovered.output_location,
        runmat_analysis_core::FieldLocation::Node
    );
    assert_eq!(
        recovered.recovery,
        runmat_analysis_core::RecoveryPolicy::VolumeWeightedAverage
    );
    assert_eq!(
        recovered.quantity,
        runmat_analysis_core::ObservationQuantity::VonMisesStress
    );
    assert_eq!(
        recovered.value_shape,
        runmat_analysis_core::ObservationValueShape::Scalar
    );
    assert!(matches!(
        study.outputs[2].expression,
        runmat_analysis_core::AuthoredOutputExpression::Recovery { .. }
    ));

    let invalid = check(&source.replace("to = node", "to = face"));
    assert!(!invalid.is_valid());
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("von Mises stress to nodes")));
}
