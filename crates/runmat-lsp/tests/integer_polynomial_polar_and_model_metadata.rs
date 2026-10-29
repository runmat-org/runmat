use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const PACKET: [(&str, usize, &str); 15] = [
    (
        "perfcurve",
        4,
        "[x,y]=perfcurve(uint8([0 1]),[0 1],uint8(1));",
    ),
    ("pie", 2, "h=pie(uint8([1 2]));"),
    ("pivot", 2, "t=table(uint8([1;2])); p=pivot(t,uint8(1));"),
    ("pol2cart", 3, "[x,y]=pol2cart(uint8([0 1]),[1 1]);"),
    ("polarhistogram", 4, "h=polarhistogram(uint8([0 1]));"),
    ("polarplot", 3, "h=polarplot(uint8([0 1]),uint8([1 2]));"),
    (
        "polarscatter",
        3,
        "h=polarscatter(uint8([0 1]),uint8([1 2]));",
    ),
    ("pole", 1, "sys=tf([1],[1 1]); p=pole(sys,uint8(1));"),
    ("polyder", 1, "d=polyder(uint8([1 2 3]));"),
    (
        "polyfit",
        4,
        "p=polyfit(uint8([0 1]),uint8([0 1]),uint8(1));",
    ),
    ("polyval", 3, "v=polyval(uint8([1 2]),uint8([0 1]));"),
    ("pow2", 3, "v=pow2(uint8([0 1]));"),
    (
        "ppval",
        1,
        "pp=pchip([0 1],[0 1]); v=ppval(pp,uint8([0 1]));",
    ),
    (
        "predict",
        3,
        "model=struct(); y=predict(model,uint8([0 1]));",
    ),
    ("print", 1, "ok=print(uint8(1));"),
];

#[test]
fn polynomial_polar_and_model_metadata_is_class_complete() {
    for (name, expected_forms, _) in PACKET {
        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        if name == "pivot" {
            use runmat_builtins::{
                BuiltinIntegerBackendRule as Backend, BuiltinIntegerComputationDomain as Domain,
                BuiltinIntegerInputAvailability as Availability,
                BuiltinIntegerOutputClassRule as Output, BuiltinIntegerOverflowRule as Overflow,
                BuiltinIntegerOverloadKind as Overload, BuiltinIntegerScalarDoubleRule as Scalar,
            };
            let [data, selectors] = builtin.integer_capabilities else {
                panic!("pivot must distinguish table data from positional selectors");
            };
            assert_eq!(
                data.inputs[0].name,
                "integer grouping or DataVariable table columns"
            );
            assert_eq!(
                selectors.inputs[0].name,
                "rowvars, colvars, or datavar variable indices"
            );
            for capability in [data, selectors] {
                assert_eq!(capability.inputs.len(), 1);
                assert_eq!(
                    capability.inputs[0].classes,
                    &runmat_builtins::ALL_INTEGER_CLASSES
                );
                assert_eq!(capability.inputs[0].availability, Availability::Documented);
                assert_eq!(capability.backend, Backend::HostOnly);
                assert_eq!(capability.output_class, Output::FunctionSpecific);
            }
            assert_eq!(data.computation_domain, Domain::FunctionSpecific);
            assert_eq!(data.overflow, Overflow::FunctionSpecific);
            assert_eq!(data.overload, Overload::Multiple);
            assert_eq!(data.inputs[0].scalar_double, Scalar::NotApplicable);
            assert_eq!(selectors.computation_domain, Domain::Structural);
            assert_eq!(selectors.overflow, Overflow::Error);
            assert_eq!(selectors.overload, Overload::StructuralParameter);
            assert_eq!(selectors.inputs[0].scalar_double, Scalar::Allowed);
            assert!(selectors.notes.contains("not implemented"));
        }
        for capability in builtin.integer_capabilities {
            assert!(!capability.inputs.is_empty(), "{name}: {}", capability.form);
            for input in capability.inputs {
                assert_eq!(input.classes.len(), 8, "{name}: {}", input.name);
            }
        }
    }
}

#[test]
fn polynomial_polar_and_model_signatures_remain_visible_to_lsp() {
    for (name, _, source) in PACKET {
        let analysis = analyze_document_with_compat(source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{name}: {source}");
        assert!(
            analysis.lowering_error.is_none(),
            "{name}: {source}: {:?}",
            analysis.lowering_error
        );
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}

#[test]
fn runmat_only_forms_publish_stable_compatibility_identifiers() {
    for name in [
        "perfcurve",
        "pie",
        "pol2cart",
        "polarplot",
        "polyder",
        "polyfit",
        "polyval",
        "pow2",
        "ppval",
        "predict",
        "print",
    ] {
        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
        assert!(!builtin.extensions.is_empty(), "{name}");
        for extension in builtin.extensions {
            assert_eq!(
                extension.mode,
                runmat_builtins::BuiltinExtensionMode::RunMatOnly,
                "{name}: {}",
                extension.id
            );
            assert!(
                extension.error_identifier.is_some(),
                "{name}: {}",
                extension.id
            );
        }
    }
}
