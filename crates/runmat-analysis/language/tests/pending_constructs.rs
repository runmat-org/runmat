pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

/// A study that uses a planned construct fails `check` with that construct's pending id, the
/// same key docs/simulation/pending-features.json and the release-status page carry.
const BASE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid { select result; expect exactly_one }
    region skin : face { SELECT; expect one_or_more }
}
material steel {
    mass_density = 7850 kg/m^3
    elastic { youngs_modulus = 200 GPa; poisson_ratio = 0.3 }
}
model base {
    assign material.steel to geometry.part.region.body
}
model structure {
    BODY
}
EXTRA
"#;

fn diagnostics(select: &str, body: &str) -> Vec<String> {
    diagnostics_with(select, body, "")
}

fn diagnostics_with(select: &str, body: &str, extra: &str) -> Vec<String> {
    let source = BASE
        .replace("SELECT", select)
        .replace("BODY", body)
        .replace("EXTRA", extra);
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &source, 1),
        },
    });
    program.diagnostics.iter().map(|d| d.message.clone()).collect()
}

fn assert_pending(messages: &[String], id: &str) {
    let marker = format!("[pending:{id}]");
    assert!(
        messages.iter().any(|message| message.contains(&marker)),
        "expected {marker} in {messages:#?}"
    );
}

#[test]
fn baseline_checks_clean() {
    let messages = diagnostics("select result", "include model.base");
    assert!(
        !messages.iter().any(|message| message.contains("[pending:")),
        "{messages:#?}"
    );
}

#[test]
fn geometric_predicate_selector_reports_its_pending_id() {
    assert_pending(
        &diagnostics("select normal [0, 0, 1]", "include model.base"),
        "geometry.select.predicate",
    );
}

#[test]
fn parameterized_include_reports_its_pending_id() {
    assert_pending(
        &diagnostics("select result", "include model.base(load = 2 kN)"),
        "language.include.parameterized",
    );
}

#[test]
fn trailing_unit_vector_reports_its_pending_id() {
    let body = "include model.base\n    orientation axes {\n        cartesian {\n            origin = [0, 0, 0] m\n            x_axis = [1, 0, 0]\n            y_axis = [0, 1, 0]\n        }\n    }";
    assert_pending(
        &diagnostics("select result", body),
        "language.vector.trailing-unit",
    );
}

#[test]
fn keyword_declared_name_reports_its_pending_id() {
    let body = "input load : force = 1.5 kN\n    include model.base";
    assert_pending(&diagnostics("select result", body), "language.keyword-names");
}

#[test]
fn species_material_member_reports_its_pending_id() {
    let extra = "material dyed {\n    mass_density = 1000 kg/m^3\n    species dye { diffusivity = 1e-10 m^2/s }\n}";
    assert_pending(
        &diagnostics_with("select result", "include model.base", extra),
        "fluid.species-transport",
    );
}

#[test]
fn lubrication_constraint_reports_its_pending_id() {
    let body = "include model.base\n    constraint bearing {\n        lubrication.journal { target = geometry.part.region.skin }\n    }";
    assert_pending(&diagnostics("select result", body), "lubrication.family");
}

#[test]
fn total_power_heat_generation_reports_its_pending_id() {
    let body = "include model.base\n    load heater {\n        heat_generation on geometry.part.region.body = 2 W\n    }";
    assert_pending(
        &diagnostics("select result", body),
        "thermal.heat-generation.total-power",
    );
}
