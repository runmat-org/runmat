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
    output displacement = analysis.static.field.node.displacement
    output displacement_magnitude = magnitude(output.displacement)
    output peak_displacement = max(
        output.displacement_magnitude,
        over = geometry.part.region.body,
    )
}
"#;

// witness: FEA-AUTHORING-LANGUAGE-002/sweep_identity @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Axis sweeps expand product and zip with stable named case identities; independence of
//   identity from source order and the placement envelopes are not covered.
#[test]
fn axis_sweeps_expand_product_and_zip_with_stable_case_identities() {
    let product = check(&format!(
        "{STUDY}\n{}",
        r#"
sweep envelope over study.response {
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
sweep envelope over study.response {
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
fn sweep_relationship_uses_the_explicit_over_keyword() {
    let retired_spelling = check(&format!(
        "{STUDY}\nsweep envelope : study.response {{ case nominal {{ load = 1 kN }} }}"
    ));
    assert!(!retired_spelling.is_valid());
    assert!(retired_spelling
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("after `over`")));
}

#[test]
fn explicit_cases_use_defaults_and_reject_missing_or_invalid_required_inputs() {
    let valid = check(&format!(
        "{STUDY}\n{}",
        r#"
sweep qualification over study.response {
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
            "{STUDY}\nsweep invalid over study.response {{\n{body}\n}}"
        ));
        assert!(!invalid.is_valid(), "invalid sweep was accepted: {body}");
    }
}

#[test]
fn geometry_inputs_are_bound_by_each_sweep_case_before_materialization() {
    let source = r#"
geometry part {
    input width : length
    input depth : length = 1 m
    construct {
        feature body = box(width = input.width, depth = input.depth, height = 1 m)
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
    input stock_width : length {
        minimum = 0.5 m
        maximum = 2 m
    }
    geometry = geometry.part(width = input.stock_width)
    mesh = mesh.part
    model = model.structure
    analysis static { structural static {} }
}
sweep sizes over study.response {
    axis stock_width {
        narrow = 0.75 m
        wide = 1.5 m
    }
    combine = product
}
"#;
    let program = check(source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let study = program.studies.values().next().unwrap();
    let geometry = program.geometries.values().next().unwrap();
    let model = program.models.values().next().unwrap();
    let sweep = program.sweeps.values().next().unwrap();
    let cases = sweep.expand_cases(study).unwrap();
    assert_eq!(cases.len(), 2);

    let widths = cases
        .iter()
        .map(|case| {
            let bindings = case
                .bindings
                .iter()
                .map(|binding| runmat_analysis_core::AuthoredStudyInputBinding {
                    input_id: binding.input_id.clone(),
                    value: binding.value.clone(),
                })
                .collect::<Vec<_>>();
            let realized = study.realize(model, geometry, &bindings).unwrap();
            assert_eq!(realized.geometry.bindings.len(), 2);
            assert!(matches!(
                realized.geometry.bindings["depth"],
                runmat_geometry_core::GeometryAuthoringValue::Scalar(_)
            ));
            let runmat_geometry_core::GeometryAuthoringValue::Scalar(
                runmat_geometry_core::GeometryScalarExpression::Canonical { value },
            ) = &realized.geometry.bindings["width"]
            else {
                panic!("geometry width must realize to one canonical SI scalar");
            };
            assert_eq!(value.descriptor().dimension(), PhysicalDimension::LENGTH);
            assert_eq!(value.descriptor().angle_power(), 0);
            value.canonical_value()
        })
        .collect::<Vec<_>>();
    assert_eq!(widths, [0.75, 1.5]);
    assert_eq!(geometry.parameters.len(), 2);
}

#[test]
fn geometry_bindings_reject_missing_unknown_and_wrong_typed_arguments() {
    let prefix = r#"
geometry part {
    input width : length
    construct {
        feature body = box(width = input.width, depth = 1 m, height = 1 m)
        result = feature.body
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
"#;
    for geometry_binding in [
        "geometry.part",
        "geometry.part(missing = input.stock_width)",
        "geometry.part(width = input.ratio)",
    ] {
        let inputs = if geometry_binding.contains("ratio") {
            "input stock_width : length = 1 m\n    input ratio : real = 1"
        } else {
            "input stock_width : length = 1 m"
        };
        let source = format!(
            "{prefix}\nstudy response {{\n    {inputs}\n    geometry = {geometry_binding}\n    mesh = mesh.part\n    model = model.structure\n    analysis static {{ structural static {{}} }}\n}}"
        );
        let program = check(&source);
        assert!(
            !program.is_valid(),
            "invalid geometry binding was accepted: {geometry_binding}"
        );
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
