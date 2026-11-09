pub mod support;

use runmat_analysis_language::*;
use runmat_geometry_core::{
    CrossSectionLayerRole, CrossSectionTraceRole, GeometryDefinitionSource,
};
use support::{source_module, TestHost};

const INLINE: &str = r#"
geometry pair {
    cross_section {
        layer "F.Mask" : solder_mask { thickness = 20 um; permittivity = 3.8 }
        layer "F.Cu" : signal { thickness = 35 um }
        layer "dielectric 1" : dielectric { thickness = 0.2 mm; permittivity = 4.1 }
        layer "In1.Cu" : plane { thickness = 35 um }
        trace p : signal { layer = "F.Cu"; center = -0.2 mm; width = 0.23 mm; etch_angle = 60 deg }
        trace n : signal { layer = "F.Cu"; center = 0.2 mm; width = 0.23 mm; etch_angle = 60 deg }
        trace g : ground { layer = "F.Cu"; center = 1 mm; width = 0.5 mm; base = bottom }
    }
}
mesh pair on geometry.pair { triangles { edge_size = 0.1 um; growth = 1.05 } }
"#;

const FROM_LAYOUT: &str = r#"
geometry line {
    cross_section {
        stackup = "./board.circuit.layout"
        stackup_name = rigid
        trace line : signal { layer = "F.Cu"; width = 0.36 mm }
    }
}
mesh line on geometry.line { triangles {} }
"#;

const LAYOUT: &str = r#"
layout board {
    stackup rigid {
        finish = ENIG
        layer "F.Mask" = { kind = mask, thickness = 0.02, material = solder_mask }
        layer "F.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
        layer "dielectric 1" = { kind = prepreg, thickness = 0.2, material = { name = "prepreg", dk = 4.1, df = 0.02 } }
        layer "In1.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
        layer "dielectric 2" = { kind = core, thickness = 1.2, material = fr4 }
        layer "B.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
    }
    stackup flex {
        layer "F.Cu" = { kind = copper, thickness = 0.018, material = copper }
    }
}
"#;

fn check(host: &TestHost, source: &str) -> CheckedFeaProgram {
    FeaCompiler::new(host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

fn errors(program: &CheckedFeaProgram) -> Vec<String> {
    program
        .bound
        .graph
        .diagnostics
        .iter()
        .chain(&program.bound.diagnostics)
        .chain(&program.diagnostics)
        .map(|d| d.message.clone())
        .collect()
}

fn section(program: &CheckedFeaProgram) -> &runmat_geometry_core::CrossSectionDefinition {
    let geometry = program.geometries.values().next().expect("geometry lowers");
    let GeometryDefinitionSource::CrossSection(section) = &geometry.source else {
        panic!("not a cross-section: {:?}", geometry.source);
    };
    section
}

#[test]
fn an_inline_cross_section_lowers_to_layers_traces_and_triangle_controls() {
    let program = check(&TestHost::default(), INLINE);
    assert!(errors(&program).is_empty(), "{:?}", errors(&program));
    let section = section(&program);
    assert_eq!(section.layers.len(), 4);
    assert_eq!(
        section.layers[0].role,
        CrossSectionLayerRole::SolderMask { permittivity: 3.8 }
    );
    assert!((section.layers[1].thickness_m - 35e-6).abs() < 1e-15);
    assert_eq!(section.layers[3].role, CrossSectionLayerRole::Plane);
    assert_eq!(section.traces.len(), 3);
    assert_eq!(section.traces[2].role, CrossSectionTraceRole::Ground);
    assert!((section.traces[0].center_m + 0.2e-3).abs() < 1e-15);
    assert!((section.traces[0].etch_angle_rad - 60f64.to_radians()).abs() < 1e-12);
    assert!((section.traces[2].etch_angle_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    let mesh = program.meshes.values().next().expect("mesh lowers");
    let controls = mesh.settings.cross_section.expect("cross-section controls");
    assert!((controls.edge_size_m.unwrap() - 0.1e-6).abs() < 1e-18);
    assert_eq!(controls.growth, Some(1.05));
    assert_eq!(controls.max_size_m, None);
}

#[test]
fn a_cross_section_reads_its_stackup_from_the_board_layout() {
    let host = TestHost::default().with_module("project/board.circuit.layout", LAYOUT, 7);
    let program = check(&host, FROM_LAYOUT);
    assert!(errors(&program).is_empty(), "{:?}", errors(&program));
    let section = section(&program);
    let roles = section
        .layers
        .iter()
        .map(|layer| (layer.name.as_str(), layer.role))
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        [
            (
                "F.Mask",
                CrossSectionLayerRole::SolderMask { permittivity: 3.8 }
            ),
            (
                "F.Cu",
                CrossSectionLayerRole::Signal {
                    fill_permittivity: None
                }
            ),
            (
                "dielectric 1",
                CrossSectionLayerRole::Dielectric { permittivity: 4.1 }
            ),
            ("In1.Cu", CrossSectionLayerRole::Plane),
            (
                "dielectric 2",
                CrossSectionLayerRole::Dielectric { permittivity: 4.3 }
            ),
            ("B.Cu", CrossSectionLayerRole::Plane),
        ]
    );
    let stackup = section.stackup.as_ref().expect("stackup source");
    assert_eq!(stackup.logical_path, "project/board.circuit.layout");
    assert_eq!(stackup.stackup_name.as_deref(), Some("rigid"));
    // The layout is a source of the program: editing it invalidates the check.
    assert!(program
        .bound
        .graph
        .source_digests()
        .contains_key(&LogicalSourcePath::new("project/board.circuit.layout").unwrap()));
}

#[test]
fn cross_section_mistakes_are_named() {
    let host = TestHost::default().with_module("project/board.circuit.layout", LAYOUT, 7);
    let unnamed = FROM_LAYOUT.replace("        stackup_name = rigid\n", "");
    let messages = errors(&check(&host, &unnamed)).join("\n");
    assert!(messages.contains("rigid, flex"), "{messages}");

    let missing = errors(&check(&TestHost::default(), FROM_LAYOUT)).join("\n");
    assert!(missing.contains("board.circuit.layout"), "{missing}");

    let role = INLINE.replace(": plane {", ": copper {");
    let messages = errors(&check(&TestHost::default(), &role)).join("\n");
    assert!(
        messages.contains("`copper` is not a layer role"),
        "{messages}"
    );

    let mesh = INLINE.replace("growth = 1.05", "order = linear");
    let messages = errors(&check(&TestHost::default(), &mesh)).join("\n");
    assert!(
        messages.contains("accepts edge_size, growth and max_size"),
        "{messages}"
    );
}
