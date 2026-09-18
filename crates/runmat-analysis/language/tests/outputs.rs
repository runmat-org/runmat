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
        projected.source_location,
        runmat_analysis_core::FieldLocation::Element
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
