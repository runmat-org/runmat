mod support;
use runmat_analysis_language::*;
use support::{source_module, TestHost};
const BASE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid { select result; expect exactly_one }
}
mesh part on geometry.part {
    tetrahedra { order = linear; target_size = 0.1 m; maximum_deviation = 0.01 m }
}
model structure {}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis response { structural transient { start_time = 0 s; end_time = 0.1 s; time_step = 0.001 s } }
    output spatial_peak = max(magnitude(analysis.response.field.node.displacement), over = geometry.part.region.body)
    output history = sample(output.spatial_peak, over = time, at = [0 s, 0.05 s, 0.1 s])
    output peak = max(output.history, over = time)
    assessment deflection {
        value = output.peak
        maximum = 1 mm
        evidence = authored_samples
    }
}
"#;
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
#[test]
fn threshold_is_typed_and_retained() {
    use runmat_analysis_core::AssessmentEvidenceRequirement;
    for bound in [FeaContextualWord::Minimum, FeaContextualWord::Maximum] {
        for evidence in [
            AssessmentEvidenceRequirement::AuthoredSamples,
            AssessmentEvidenceRequirement::CertifiedBound,
        ] {
            let source = BASE
                .replace("maximum = 1 mm", &format!("{} = 1 mm", bound.as_source()))
                .replace(
                    "authored_samples",
                    FeaContextualWord::EvidenceValue(evidence).as_source(),
                );
            let checked = check(&source);
            assert!(
                !checked
                    .diagnostics
                    .iter()
                    .any(|d| d.severity == FeaDiagnosticSeverity::Error),
                "{:?}",
                checked.diagnostics
            );
            let study = checked.studies.values().next().unwrap();
            assert_eq!(study.assessments.len(), 1);
            assert_eq!(study.assessments[0].evidence, evidence);
            let (minimum, maximum) = study.assessments[0].rule.bounds();
            let retained = match bound {
                FeaContextualWord::Minimum => minimum,
                FeaContextualWord::Maximum => maximum,
                _ => unreachable!("the matrix contains only assessment bound properties"),
            };
            assert_eq!(retained.unwrap().canonical_value(), 0.001);
        }
    }
}
#[test]
fn wrong_units_and_duplicate_settings_are_rejected() {
    for source in [
        BASE.replace("1 mm", "1 N"),
        BASE.replace("maximum = 1 mm", "maximum = 1 mm\n maximum = 2 mm"),
    ] {
        assert!(check(&source)
            .diagnostics
            .iter()
            .any(|d| d.severity == FeaDiagnosticSeverity::Error));
    }
}

#[test]
fn imported_assessment_requires_a_complete_logical_record() {
    use runmat_types::{OperationParameter, OperationParameterArity, OperationTypePattern};
    use std::collections::BTreeMap;
    let source = format!(
        "import function verify from \"./verify.m\"\n{}",
        BASE.replace(
            "maximum = 1 mm",
            "function = function.verify\n parameters = [1 mm]"
        )
    );
    let complete = OperationTypePattern::Record(BTreeMap::from([
        ("evaluated".into(), OperationTypePattern::LogicalScalar),
        ("accepted".into(), OperationTypePattern::LogicalScalar),
    ]));
    for (output, valid) in [
        (complete, true),
        (OperationTypePattern::Any, false),
        (OperationTypePattern::LogicalScalar, false),
    ] {
        let host = TestHost::default().with_function(
            "project/verify.m",
            "verify",
            vec![
                OperationParameter {
                    name: "value".into(),
                    value: OperationTypePattern::NumericScalar,
                    arity: OperationParameterArity::Required,
                    default: None,
                },
                OperationParameter {
                    name: "limit".into(),
                    value: OperationTypePattern::NumericScalar,
                    arity: OperationParameterArity::Required,
                    default: None,
                },
            ],
            vec![output],
        );
        let checked = FeaCompiler::new(&host).check(CheckRequest {
            modules: ModuleGraphRequest {
                source_root: LogicalSourcePath::new("project").unwrap(),
                root: source_module("project/main.fea", &source, 1),
            },
        });
        assert_eq!(checked.is_valid(), valid, "{:#?}", checked.diagnostics);
        if valid {
            let definition = &checked.studies.values().next().unwrap().assessments[0];
            let runmat_analysis_core::AssessmentRule::Function {
                function,
                parameters,
            } = &definition.rule
            else {
                panic!("wrong assessment rule")
            };
            assert_eq!(function.module_path, "project/verify.m");
            assert_eq!(function.source_digest, [9; 32]);
            assert_eq!(parameters[0].canonical_value(), 0.001);
        }
    }
}
