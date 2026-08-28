use lsp_types::Position;
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerInputAvailability, BuiltinIntegerOutputClassRule,
};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 8] = [
    "lsqcurvefit",
    "null",
    "ode15s",
    "ode23",
    "ode45",
    "optimoptions",
    "optimset",
    "quad",
];

#[test]
fn optimization_ode_and_null_integer_metadata_is_explicit() {
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        for capability in builtin.integer_capabilities {
            for input in capability.inputs {
                assert_eq!(input.classes.len(), 8, "{name}: {}", input.name);
            }
        }
    }
}

#[test]
fn optimization_and_ode_integer_extensions_describe_checked_host_boundaries() {
    for name in [
        "lsqcurvefit",
        "ode15s",
        "ode23",
        "ode45",
        "optimoptions",
        "optimset",
        "quad",
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.integer_capabilities.iter().any(|capability| {
                capability
                    .inputs
                    .iter()
                    .any(|input| input.availability == BuiltinIntegerInputAvailability::RunMatOnly)
            }),
            "{name}"
        );
        assert!(
            builtin.integer_capabilities.iter().any(|capability| {
                capability.backend == BuiltinIntegerBackendRule::GatherFallback
            }),
            "{name}"
        );
        assert!(!builtin.extensions.is_empty(), "{name}");
    }
}

#[test]
fn null_and_quad_distinguish_documented_integer_controls_from_extensions() {
    let null = runmat_builtins::builtin_function_by_name("null").expect("null builtin");
    assert!(null.integer_capabilities.iter().any(|capability| {
        capability.inputs.iter().any(|input| {
            input.name == "tol" && input.availability == BuiltinIntegerInputAvailability::Documented
        }) && capability.output_class == BuiltinIntegerOutputClassRule::PreserveNondoubleInput
    }));

    let quad = runmat_builtins::builtin_function_by_name("quad").expect("quad builtin");
    for input_name in ["trace", "p1, p2, ..."] {
        assert!(
            quad.integer_capabilities.iter().any(|capability| {
                capability.inputs.iter().any(|input| {
                    input.name == input_name
                        && input.availability == BuiltinIntegerInputAvailability::Documented
                })
            }),
            "{input_name}"
        );
    }
}

#[test]
fn optimization_ode_and_null_signatures_are_visible_to_lsp() {
    for (name, source) in [
        (
            "lsqcurvefit",
            "x=lsqcurvefit(@(p,x) p,uint16(0),uint16(0),uint16(0));",
        ),
        ("null", "z=null(uint16([1 2;2 4]));"),
        ("ode15s", "y=ode15s(@(t,y) -y,uint16([0 1]),uint16(1));"),
        ("ode23", "y=ode23(@(t,y) -y,uint16([0 1]),uint16(1));"),
        ("ode45", "y=ode45(@(t,y) -y,uint16([0 1]),uint16(1));"),
        (
            "optimoptions",
            "o=optimoptions('fsolve','MaxIter',uint16(5));",
        ),
        ("optimset", "o=optimset('MaxIter',uint16(5));"),
        ("quad", "q=quad(@sin,uint16(0),uint16(1));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
