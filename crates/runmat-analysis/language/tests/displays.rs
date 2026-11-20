pub mod support;
use runmat_analysis_core::{AuthoredOutputDisplay, DisplayRange, DisplayUnitSource};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

/// A thermal study of a copper block whose model writes its reference temperature as
/// `reference`, with `display` statements appended to the study.
fn check(reference: &str, display: &str) -> CheckedFeaProgram {
    let source = format!(
        r#"
geometry part {{
    construct {{
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }}
    region body : solid {{ select result; expect exactly_one }}
    region lid : solid {{ select result; expect exactly_one }}
    region skin : face {{ select result; expect one_or_more }}
}}
mesh part on geometry.part {{
    tetrahedra {{ order = linear; target_size = 0.5 m; maximum_deviation = 0.01 m }}
}}
material copper {{
    mass_density = 8960 kg/m^3
    thermal {{
        reference_temperature = 293.15 K
        modulus_temperature_coefficient = 0 K^-1
        conductivity = 400 W/m/K
        specific_heat = 385 J/kg/K
        expansion_coefficient = 0.0000165 K^-1
    }}
}}
model physics {{
    thermal {{ reference_temperature = {reference} }}
    assign material.copper to geometry.part.region.body
    load heater {{ heat_generation on geometry.part.region.body = 2e6 W/m^3 }}
    load_case running {{ include load.heater }}
}}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.physics
    analysis heat {{ thermal steady {{}} }}
    output temperature = analysis.heat.field.node.temperature
    output heat_flux = analysis.heat.field.element.heat_flux
    output peak = max(analysis.heat.field.node.temperature, over = geometry.part.region.body)
    {display}
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

fn displays(program: &CheckedFeaProgram) -> Vec<AuthoredOutputDisplay> {
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    program.studies.values().next().unwrap().displays.clone()
}

#[test]
fn a_display_statement_compiles_to_canonical_values() {
    let program = check(
        "293.15 K",
        "display temperature { unit = degC; range = [20 degC, 80 degC]; precision = 3 }",
    );
    let [display] = displays(&program).try_into().unwrap();
    assert_eq!(display.output_id, "temperature");
    assert_eq!(display.unit.as_deref(), Some("degC"));
    assert_eq!(display.unit_source, DisplayUnitSource::Display);
    let DisplayRange { minimum, maximum } = display.range.unwrap();
    assert!((minimum - 293.15).abs() < 1e-9 && (maximum - 353.15).abs() < 1e-9);
    assert_eq!(display.precision, Some(3));
    // A list with one unit and a compound unit as a quoted symbol.
    let program = check(
        "293.15 K",
        "display heat_flux { unit = \"kW/m^2\"; range = [0, 50] kW/m^2 }",
    );
    let [display] = displays(&program).try_into().unwrap();
    assert_eq!(display.unit.as_deref(), Some("kW/m^2"));
    assert_eq!(
        display.range,
        Some(DisplayRange {
            minimum: 0.0,
            maximum: 50_000.0
        })
    );
}

#[test]
fn without_a_unit_the_legend_takes_the_unit_the_study_writes() {
    // The model's reference temperature is written in °C, so the temperature field shows °C
    // with or without a display statement; the heat flux has no authored unit and stays SI.
    for display in ["", "display temperature { precision = 4 }"] {
        let program = check("20 degC", display);
        let [temperature] = displays(&program).try_into().unwrap();
        assert_eq!(temperature.output_id, "temperature");
        assert_eq!(temperature.unit.as_deref(), Some("degC"));
        assert_eq!(temperature.unit_source, DisplayUnitSource::Study);
    }
    // Written in kelvin: the SI unit, so nothing is recorded.
    assert!(displays(&check("293.15 K", "")).is_empty());
    // An explicit unit wins over the authored one.
    let program = check("20 degC", "display temperature { unit = K }");
    let [temperature] = displays(&program).try_into().unwrap();
    assert_eq!(temperature.unit.as_deref(), Some("K"));
    assert_eq!(temperature.unit_source, DisplayUnitSource::Display);
}

#[test]
fn malformed_displays_are_refused_with_their_codes() {
    for (display, code) in [
        (
            "display peak { precision = 3 }",
            "fea.semantic.display_output",
        ),
        (
            "display missing { precision = 3 }",
            "fea.semantic.display_output",
        ),
        (
            "display temperature { precision = 3 }\n    display temperature { precision = 4 }",
            "fea.semantic.display_output",
        ),
        ("display { precision = 3 }", "fea.semantic.display_output"),
        ("display temperature {}", "fea.semantic.display_member"),
        (
            "display temperature { colour = red }",
            "fea.semantic.display_member",
        ),
        (
            "display temperature { precision = 3; precision = 4 }",
            "fea.semantic.display_member",
        ),
        (
            "display temperature { unit = MPa }",
            "fea.semantic.display_unit",
        ),
        (
            "display temperature { unit = 3 }",
            "fea.semantic.display_unit",
        ),
        (
            "display temperature { range = [20 MPa, 30 MPa] }",
            "fea.semantic.display_range",
        ),
        (
            "display temperature { range = [80 degC, 20 degC] }",
            "fea.semantic.display_range",
        ),
        (
            "display temperature { range = [20 degC] }",
            "fea.semantic.display_range",
        ),
        (
            "display temperature { precision = 7 }",
            "fea.semantic.display_precision",
        ),
        (
            "display temperature { precision = 0 }",
            "fea.semantic.display_precision",
        ),
        (
            "display temperature { precision = 2.5 }",
            "fea.semantic.display_precision",
        ),
        // Hidden and context regions are solid regions of the study geometry, in one list.
        (
            "display temperature { hide = [geometry.part.region.missing] }",
            "fea.semantic.display_region",
        ),
        (
            "display temperature { context = [geometry.part.region.skin] }",
            "fea.semantic.display_region",
        ),
        (
            "display temperature { hide = [material.copper] }",
            "fea.semantic.display_region",
        ),
        (
            "display temperature { hide = [3] }",
            "fea.semantic.display_region",
        ),
        (
            "display temperature { hide = [geometry.part.region.lid]; \
             context = [geometry.part.region.lid] }",
            "fea.semantic.display_region",
        ),
        (
            "display temperature { hide = [geometry.part.region.lid, geometry.part.region.lid] }",
            "fea.semantic.display_region",
        ),
        (
            "display temperature { deformation = 0 }",
            "fea.semantic.display_deformation",
        ),
        (
            "display temperature { deformation = -2 }",
            "fea.semantic.display_deformation",
        ),
        (
            "display temperature { deformation = 2 mm }",
            "fea.semantic.display_deformation",
        ),
    ] {
        let program = check("293.15 K", display);
        assert!(
            program
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == code
                    && diagnostic.severity == FeaDiagnosticSeverity::Error),
            "{display}: expected {code}, got {:#?}",
            program.diagnostics
        );
    }
}

#[test]
fn hidden_and_context_regions_and_the_deformation_scale_compile_by_region_name() {
    let program = check(
        "293.15 K",
        "display temperature { hide = [geometry.part.region.lid]; \
         context = geometry.part.region.body; deformation = 200 }",
    );
    let [display] = displays(&program).try_into().unwrap();
    assert_eq!(display.hide, ["lid"]);
    assert_eq!(display.context, ["body"]);
    assert_eq!(display.deformation, Some(200.0));
    assert!(!display.shows_every_region());
    // Without a unit, a range or a precision, the view settings alone record a display; its
    // unit is still the one the study writes temperatures in.
    assert_eq!(display.unit.as_deref(), Some("K"));
    assert_eq!(display.unit_source, DisplayUnitSource::Study);
    let program = check("293.15 K", "display heat_flux { deformation = 2.5 }");
    let [display] = displays(&program).try_into().unwrap();
    assert_eq!(display.output_id, "heat_flux");
    assert_eq!(display.deformation, Some(2.5));
    assert!(display.shows_every_region());
}
