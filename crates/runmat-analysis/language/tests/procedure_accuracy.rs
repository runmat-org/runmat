pub mod support;
use runmat_analysis_core::{
    AccuracyTolerance, ErrorContributionKind, ModalSolverMethod, RequiredAccuracyEvidence,
};
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

fn algebraic(step: &runmat_analysis_core::ProcedureStep) -> (RequiredAccuracyEvidence, f64) {
    let criterion = step
        .accuracy
        .criteria
        .iter()
        .find(|c| c.contribution == ErrorContributionKind::Algebraic)
        .unwrap();
    let AccuracyTolerance::Relative(tolerance) = criterion.tolerance else {
        panic!("algebraic tolerance is relative");
    };
    (criterion.evidence, tolerance)
}

fn step(checked: &CheckedFeaProgram) -> runmat_analysis_core::ProcedureStep {
    checked.studies.values().next().unwrap().sequence.steps[0].clone()
}

#[test]
fn modal_steps_qualify_on_the_backward_error_and_may_require_the_relative_residual() {
    let modal = "structural modal { modes = 2 }";
    let baseline = check(modal, "", "");
    assert!(baseline.is_valid(), "{:#?}", baseline.diagnostics);
    assert_eq!(
        algebraic(&step(&baseline)),
        (RequiredAccuracyEvidence::BackwardErrorBound, 1.0e-12)
    );
    for (accuracy, expected) in [
        (
            "accuracy { algebraic_measure = relative_residual }",
            (RequiredAccuracyEvidence::ResidualBound, 1.0e-8),
        ),
        (
            "accuracy { algebraic_measure = relative_residual\nalgebraic_relative_tolerance = 1e-6 }",
            (RequiredAccuracyEvidence::ResidualBound, 1.0e-6),
        ),
        (
            "accuracy { algebraic_relative_tolerance = 1e-10\nalgebraic_measure = backward_error }",
            (RequiredAccuracyEvidence::BackwardErrorBound, 1.0e-10),
        ),
        (
            "accuracy { algebraic_relative_tolerance = 1e-13 }",
            (RequiredAccuracyEvidence::BackwardErrorBound, 1.0e-13),
        ),
    ] {
        let authored = check(modal, "", accuracy);
        assert!(authored.is_valid(), "{accuracy}: {:#?}", authored.diagnostics);
        assert_eq!(algebraic(&step(&authored)), expected, "{accuracy}");
    }
    // Static steps publish the relative residual; asking them for the backward error is refused
    // by check (and so by run, which compiles through the same rule).
    let static_step = check("structural static {}", "", "");
    assert_eq!(
        algebraic(&step(&static_step)),
        (RequiredAccuracyEvidence::ResidualBound, 1.0e-8)
    );
    assert!(check(
        "structural static {}",
        "",
        "accuracy { algebraic_measure = relative_residual }"
    )
    .is_valid());
    for (procedure, accuracy) in [
        ("structural static {}", "accuracy { algebraic_measure = backward_error }"),
        (modal, "accuracy { algebraic_measure = residual }"),
        (modal, "accuracy { algebraic_measure = 1e-12 }"),
        (
            modal,
            "accuracy { algebraic_measure = backward_error\nalgebraic_measure = relative_residual }",
        ),
    ] {
        assert!(!check(procedure, "", accuracy).is_valid(), "accepted {accuracy}");
    }
}

#[test]
fn modal_solver_method_is_authored_on_the_extraction() {
    for (setting, expected) in [
        ("", ModalSolverMethod::Auto),
        ("\nsolver = auto", ModalSolverMethod::Auto),
        ("\nsolver = factorization", ModalSolverMethod::Factorization),
        ("\nsolver = iterative", ModalSolverMethod::Iterative),
        (
            "\nsolver = iterative\nresidual_vectors = [[0, 0, 1]]",
            ModalSolverMethod::Iterative,
        ),
    ] {
        let checked = check(&format!("structural modal {{ modes = 2{setting} }}"), "", "");
        assert!(checked.is_valid(), "{setting}: {:#?}", checked.diagnostics);
        let step = step(&checked);
        assert_eq!(step.modal_solver_method(), expected, "{setting}");
        if setting.contains("residual_vectors") {
            assert_eq!(step.modal_residual_vector_directions().len(), 1);
        }
    }
    for procedure in [
        "structural modal { modes = 2\nsolver = direct }",
        "structural modal { modes = 2\nsolver = 1 }",
        "structural static {\nsolver = factorization }",
    ] {
        assert!(!check(procedure, "", "").is_valid(), "accepted {procedure}");
    }
}
