pub mod support;

use runmat_analysis_language::*;
use runmat_meshing_core::{MeshSizeControlTarget, MeshingRequestSettings};
use support::{source_module, TestHost};

fn check(tetrahedra_extra: &str) -> CheckedFeaProgram {
    let source = format!(
        r#"
geometry part {{
    construct {{
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }}
    region body : solid {{
        select result
        expect exactly_one
    }}
    region wall : face {{
        select result
        expect one_or_more
    }}
    region rim : edge {{
        select result
        expect one_or_more
    }}
}}
geometry other {{
    construct {{
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }}
    region skin : face {{
        select result
        expect one_or_more
    }}
}}
mesh part on geometry.part {{
    tetrahedra {{
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
{tetrahedra_extra}
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

fn settings(program: &CheckedFeaProgram) -> MeshingRequestSettings {
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    program.meshes.values().next().unwrap().settings.clone()
}

fn codes(program: &CheckedFeaProgram) -> Vec<String> {
    program
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.code.to_string())
        .collect()
}

#[test]
fn refine_blocks_lower_to_ordered_region_face_and_edge_controls() {
    let program = check(
        r#"
        cells_across_gap = 3
        refine walls {
            on = geometry.part.region.wall
            target_size = 20 mm
        }
        refine core {
            on = geometry.part.region.body
            target_size = 0.05 m
        }
        refine edges {
            on = geometry.part.region.rim
            target_size = 5 mm
        }
"#,
    );
    let settings = settings(&program);
    assert_eq!(settings.cells_across_gap, Some(3));
    let lowered = settings
        .size_controls
        .iter()
        .map(|control| {
            (
                control.control_id.as_str(),
                control.region_id.as_str(),
                control.target,
                control.target_size_m,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lowered,
        vec![
            ("core", "body", MeshSizeControlTarget::Solid, 0.05),
            ("edges", "rim", MeshSizeControlTarget::Edge, 0.005),
            ("walls", "wall", MeshSizeControlTarget::Face, 0.02),
        ]
    );
}

#[test]
fn meshes_without_controls_lower_to_the_default_settings() {
    let settings = settings(&check(""));
    assert_eq!(settings.cells_across_gap, None);
    assert!(settings.size_controls.is_empty());
    let value = serde_json::to_value(&settings).unwrap();
    assert!(value.get("size_controls").is_none() && value.get("cells_across_gap").is_none());
}

#[test]
fn refine_blocks_reject_unknown_regions_bad_sizes_and_conflicts() {
    // Unknown selector.
    let unknown = check("refine a {\n on = geometry.part.region.missing\n target_size = 5 mm\n }");
    assert!(!unknown.is_valid());
    // A region of another geometry.
    let foreign = check("refine a {\n on = geometry.other.region.skin\n target_size = 5 mm\n }");
    assert!(codes(&foreign).contains(&"fea.semantic.mesh.control_region".to_string()));
    // Non-positive and non-refining sizes.
    for size in ["0 m", "-5 mm", "0.2 m"] {
        let program = check(&format!(
            "refine a {{\n on = geometry.part.region.wall\n target_size = {size}\n }}"
        ));
        assert!(
            codes(&program).contains(&"fea.semantic.mesh.control_size".to_string()),
            "{size}: {:#?}",
            program.diagnostics
        );
    }
    // Two controls on the same region.
    let conflict = check(
        "refine a {\n on = geometry.part.region.wall\n target_size = 5 mm\n }\n\
         refine b {\n on = geometry.part.region.wall\n target_size = 2 mm\n }",
    );
    assert!(codes(&conflict).contains(&"fea.semantic.mesh.control_conflict".to_string()));
    // Unsupported control settings and gap counts.
    let setting =
        check("refine a {\n on = geometry.part.region.wall\n target_size = 5 mm\n growth = 2\n }");
    assert!(codes(&setting).contains(&"fea.semantic.mesh.control_setting".to_string()));
    for cells in ["0", "65"] {
        let program = check(&format!("cells_across_gap = {cells}"));
        assert!(codes(&program).contains(&"fea.semantic.mesh.cells_across_gap".to_string()));
    }
}
