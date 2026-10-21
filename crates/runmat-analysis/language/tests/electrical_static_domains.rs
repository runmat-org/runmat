pub mod support;
use runmat_analysis_core::{ElectromagneticDomain, StaticElectricalEquation};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

const SOURCE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh part on geometry.part {
    tetrahedra { order = linear
        target_size = 0.5 m
        maximum_deviation = 0.01 m }
}
model electrical {
    electromagnetic { equation = electrostatics }
}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.electrical
    analysis potential { electromagnetic static {} }
}
"#;

fn check(source: &str) -> CheckedFeaProgram {
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

#[test]
fn explicit_static_equations_survive_checked_model_composition() {
    for (name, equation) in [
        ("electrostatics", StaticElectricalEquation::Electrostatics),
        (
            "steady_conduction",
            StaticElectricalEquation::SteadyConduction,
        ),
    ] {
        let checked =
            check(&SOURCE.replace("equation = electrostatics", &format!("equation = {name}")));
        assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
        let resolved = checked
            .resolve_study("project/main.fea", "response")
            .unwrap();
        let realized = resolved
            .study
            .realize_with_models(&resolved.models, &resolved.geometry, &[])
            .unwrap();
        assert_eq!(
            realized.model.electromagnetic,
            Some(ElectromagneticDomain::Static { equation })
        );
    }
}

#[test]
fn malformed_or_harmonic_domains_do_not_become_static_defaults() {
    for replacement in [
        "electromagnetic {}",
        "electromagnetic { equation = harmonic }",
        "electromagnetic { equation = unknown }",
        "electromagnetic { equation = 1 }",
        "electromagnetic { equation = electrostatics\nreference_frequency = 60 Hz }",
        "electromagnetic { equation = electrostatics\nequation = steady_conduction }",
        "electromagnetic static { equation = electrostatics }",
        "electromagnetic { equation = electrostatics } electromagnetic { equation = electrostatics }",
    ] {
        let checked = check(&SOURCE.replace("electromagnetic { equation = electrostatics }", replacement));
        assert!(!checked.is_valid(), "accepted {replacement}");
    }
}
