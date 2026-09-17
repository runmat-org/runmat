mod support;

use runmat_analysis_core::{AuthoredSweepDefinition, SweepAxisCombination};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

const STUDY: &str = r#"
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
    input load : force {
        minimum = 0 N
        maximum = 2 kN
    }
    input damping : real = 0.02
    analysis static {
        structural static {
        }
    }
    output displacement = analysis.static.field.displacement
    output displacement_magnitude = magnitude(output.displacement)
    output peak_displacement = max(
        output.displacement_magnitude,
        over = geometry.part.region.body,
    )
}
"#;

#[test]
fn axis_sweeps_expand_product_and_zip_with_stable_case_identities() {
    let product = check(&format!(
        "{STUDY}\n{}",
        r#"
sweep envelope : study.response {
    axis load {
        half = 0.5 kN
        full = 1 kN
    }
    axis damping {
        half = 0.01
        full = 0.02
    }
    combine = product
    output peaks = collect(case.output.peak_displacement.value)
    output governing = max(case.output.peak_displacement.value, over = cases)
    output fields = retain_each(case.output.displacement)
}
"#
    ));
    assert!(product.is_valid(), "{:#?}", product.diagnostics);
    let sweep = product.sweeps.values().next().unwrap();
    assert!(matches!(
        sweep.definition,
        AuthoredSweepDefinition::Axes {
            combination: SweepAxisCombination::Product,
            ..
        }
    ));
    let study = product.studies.values().next().unwrap();
    let cases = sweep.expand_cases(study).unwrap();
    assert_eq!(cases.len(), 4);
    assert_eq!(sweep.outputs.len(), 3);
    assert_eq!(cases[0].case_id, "damping=full__load=full");
    assert_eq!(cases[3].case_id, "damping=half__load=half");

    let zipped = check(&format!(
        "{STUDY}\n{}",
        r#"
sweep envelope : study.response {
    axis load {
        half = 0.5 kN
        full = 1 kN
    }
    axis damping {
        full = 0.02
        half = 0.01
    }
    combine = zip
}
"#
    ));
    assert!(zipped.is_valid(), "{:#?}", zipped.diagnostics);
    let sweep = zipped.sweeps.values().next().unwrap();
    let study = zipped.studies.values().next().unwrap();
    let cases = sweep.expand_cases(study).unwrap();
    assert_eq!(
        cases
            .iter()
            .map(|case| case.case_id.as_str())
            .collect::<Vec<_>>(),
        ["full", "half"]
    );
}

#[test]
fn explicit_cases_use_defaults_and_reject_missing_or_invalid_required_inputs() {
    let valid = check(&format!(
        "{STUDY}\n{}",
        r#"
sweep qualification : study.response {
    case nominal {
        load = 1 kN
    }
    case overload {
        load = 2 kN
        damping = 0.01
    }
}
"#
    ));
    assert!(valid.is_valid(), "{:#?}", valid.diagnostics);
    let sweep = valid.sweeps.values().next().unwrap();
    let study = valid.studies.values().next().unwrap();
    assert_eq!(sweep.expand_cases(study).unwrap().len(), 2);

    for body in [
        "case missing { damping = 0.01 }",
        "case outside { load = 3 kN }",
        "axis unknown { nominal = 1 }
         combine = product",
        "axis load { low = 1 kN }
         axis damping { high = 0.02 }
         combine = zip",
    ] {
        let invalid = check(&format!(
            "{STUDY}\nsweep invalid : study.response {{\n{body}\n}}"
        ));
        assert!(!invalid.is_valid(), "invalid sweep was accepted: {body}");
    }
}

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
