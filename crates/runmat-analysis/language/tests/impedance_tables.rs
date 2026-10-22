pub mod support;

use runmat_analysis_core::{AcousticImpedanceLaw, AuthoredBoundaryKind};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(source: &str) -> CheckedFeaProgram {
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

const GEOMETRY: &str = r#"
geometry room {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region air : solid {
        select result
        expect exactly_one
    }
    region ceiling : face {
        select result
        expect one_or_more
    }
}
"#;

fn law(program: &CheckedFeaProgram, model: &str) -> AcousticImpedanceLaw {
    let model = program
        .models
        .values()
        .find(|candidate| candidate.model_id == model)
        .expect("model lowered");
    model
        .boundaries
        .iter()
        .find_map(|boundary| match &boundary.kind {
            AuthoredBoundaryKind::AcousticImpedanceLaw { law } => Some(law.clone()),
            _ => None,
        })
        .expect("impedance law boundary")
}

// witness: FEA-PUBLIC-LANGUAGE-001/input_forms @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Delany-Bazley and sampled impedance tables lower into their acoustic boundaries; the
//   other input forms (film coefficient, CSV samples, species, porous, lubrication) are separate.
// witness: FEA-ACOUSTIC-BOUNDARY-002/complex_frequency_dependent_impedance @
//   acoustic_harmonic/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Language lowering of Delany-Bazley and sampled impedance tables only; no solve
//   against the closed-form absorption.
#[test]
fn delany_bazley_and_sampled_impedance_tables_lower_into_their_boundaries() {
    let source = format!(
        r#"{GEOMETRY}
table lining : impedance over frequency {{
    model = delany_bazley
    flow_resistivity = 20000 Pa*s/m^2
    thickness = 50 mm
}}
table measured : impedance over frequency {{
    samples = [
        {{ at = 125 Hz, resistance = 400 Pa*s/m, reactance = -200 Pa*s/m }},
        {{ at = 250 Hz, resistance = 420 Pa*s/m, reactance = -100 Pa*s/m }},
    ]
}}
model treated {{
    load ceiling {{
        acoustic.impedance {{
            target = geometry.room.region.ceiling
            impedance = table.lining
        }}
    }}
}}
model measured_wall {{
    load ceiling {{
        acoustic.impedance {{
            target = geometry.room.region.ceiling
            impedance = table.measured
        }}
    }}
}}
"#
    );
    let program = check(&source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(
        law(&program, "treated"),
        AcousticImpedanceLaw::DelanyBazley {
            flow_resistivity_pa_s_per_m2: 20000.0,
            thickness_m: 0.05,
        }
    );
    assert_eq!(
        law(&program, "measured_wall"),
        AcousticImpedanceLaw::Table {
            frequencies_hz: vec![125.0, 250.0],
            resistance_pa_s_per_m: vec![400.0, 420.0],
            reactance_pa_s_per_m: vec![-200.0, -100.0],
        }
    );
}

// witness: FEA-PUBLIC-LANGUAGE-001/input_forms @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Impedance tables reject unknown models, wrong quantities and unsorted samples.
#[test]
fn impedance_tables_reject_unknown_models_quantities_and_unsorted_samples() {
    for table in [
        "table bad : impedance over frequency {\n    model = miki\n    flow_resistivity = 1 Pa*s/m^2\n    thickness = 1 mm\n}",
        "table bad : pressure over frequency {\n    model = delany_bazley\n    flow_resistivity = 1 Pa*s/m^2\n    thickness = 1 mm\n}",
        "table bad : impedance over frequency {\n    samples = [\n        { at = 250 Hz, resistance = 1 Pa*s/m, reactance = 0 Pa*s/m },\n        { at = 125 Hz, resistance = 1 Pa*s/m, reactance = 0 Pa*s/m },\n    ]\n}",
    ] {
        let source = format!(
            "{GEOMETRY}\n{table}\nmodel m {{\n    load c {{\n        acoustic.impedance {{\n            target = geometry.room.region.ceiling\n            impedance = table.bad\n        }}\n    }}\n}}\n"
        );
        assert!(!check(&source).is_valid(), "{table}");
    }
}
