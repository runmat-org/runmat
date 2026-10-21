pub mod support;
use runmat_analysis_core::{AccuracyTolerance, ErrorContributionKind};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(procedure: &str, model: &str, accuracy: &str) -> CheckedFeaProgram {
    let source = format!(
        r#"
geometry part {{
    construct {{
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }}
}}
mesh part on geometry.part {{
    tetrahedra {{ order = linear
        target_size = 0.5 m
        maximum_deviation = 0.01 m }}
}}
model physics {{ {model} }}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.physics
    analysis solve {{
        {procedure}
        {accuracy}
    }}
}}
"#
    );
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &source, 1),
        },
    })
}

#[test]
fn explicit_algebraic_accuracy_preserves_every_other_step_contract_across_physics() {
    for (procedure, model) in [
        ("structural static {}", ""),
        ("structural modal { modes = 2 }", ""),
        ("structural harmonic { frequencies = [0.2] Hz }", ""),
        (
            "structural transient { start_time = 0 s\nend_time = 1 s\ntime_step = 0.1 s }",
            "",
        ),
        ("acoustic harmonic { frequencies = [0.2] Hz }", ""),
        (
            "thermal steady {}",
            "thermal { reference_temperature = 300 K }",
        ),
        (
            "thermal transient { start_time = 0 s\nend_time = 1 s\ntime_step = 0.1 s }",
            "thermal { reference_temperature = 300 K }",
        ),
        (
            "electromagnetic static {}",
            "electromagnetic { equation = electrostatics }",
        ),
    ] {
        let baseline = check(procedure, model, "");
        assert!(
            baseline.is_valid(),
            "{procedure}: {:#?}",
            baseline.diagnostics
        );
        let authored = check(
            procedure,
            model,
            "accuracy { algebraic_relative_tolerance = 3e-11 }",
        );
        assert!(
            authored.is_valid(),
            "{procedure}: {:#?}",
            authored.diagnostics
        );
        let mut expected = baseline.studies.values().next().unwrap().sequence.steps[0].clone();
        assert!(expected
            .accuracy
            .criteria
            .iter()
            .any(|c| c.contribution == ErrorContributionKind::Algebraic));
        for criterion in &mut expected.accuracy.criteria {
            if criterion.contribution == ErrorContributionKind::Algebraic {
                criterion.tolerance = AccuracyTolerance::Relative(3e-11);
            }
        }
        assert_eq!(
            authored.studies.values().next().unwrap().sequence.steps[0],
            expected
        );
    }
}

#[test]
fn malformed_ambiguous_or_dimensional_accuracy_is_not_silently_ignored() {
    for accuracy in [
        "accuracy {}",
        "accuracy named { algebraic_relative_tolerance = 1e-11 }",
        "accuracy { algebraic_relative_tolerance = 0 }",
        "accuracy { algebraic_relative_tolerance = -1 }",
        "accuracy { algebraic_relative_tolerance = 1 }",
        "accuracy { algebraic_relative_tolerance = 1 m }",
        "accuracy { algebraic_relative_tolerance = [1e-11] }",
        "accuracy { algebraic_relative_tolerance = unknown }",
        "accuracy { unknown = 1e-11 }",
        "accuracy { qualified.algebraic_relative_tolerance = 1e-11 }",
        "accuracy { algebraic_relative_tolerance = 1e-11\nalgebraic_relative_tolerance = 1e-10 }",
        "accuracy { nested { algebraic_relative_tolerance = 1e-11 } }",
        "accuracy { algebraic_relative_tolerance = 1e-11 }\naccuracy { algebraic_relative_tolerance = 1e-10 }",
    ] {
        let checked = check("structural static {}", "", accuracy);
        assert!(!checked.is_valid(), "accepted {accuracy}");
    }
}
