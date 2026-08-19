use lsp_types::{HoverContents, Position};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinExtensionMode, BuiltinIntegerClass, BuiltinOutputMode,
};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const PACKET: [&str; 8] = [
    "gamrnd",
    "groupcounts",
    "groupsummary",
    "isoutlier",
    "kmeans",
    "knnsearch",
    "lhsdesign",
    "linkage",
];

const INTEGER_CLASSES: [BuiltinIntegerClass; 8] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];

#[test]
fn grouping_random_and_clustering_packet_is_public_and_audited() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name)
            .unwrap_or_else(|| panic!("registered builtin {name}"));
        let descriptor = builtin
            .descriptor
            .unwrap_or_else(|| panic!("{name} descriptor"));
        assert_eq!(
            descriptor.completion_policy,
            BuiltinCompletionPolicy::Public
        );
        assert!(!descriptor.signatures.is_empty(), "{name} signatures");
        assert!(builtin.integer_audit.is_none(), "{name} capability-bearing");
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} integer capabilities"
        );
        if !matches!(name, "groupcounts" | "groupsummary") {
            assert!(
                builtin
                    .integer_capabilities
                    .iter()
                    .all(|capability| !capability.notes.contains("[integer-audit-open]")),
                "{name} must not retain an open integer form"
            );
        }
        assert!(
            completions
                .iter()
                .any(|item| item.label.eq_ignore_ascii_case(name)),
            "public completion for {name}"
        );
    }
}

#[test]
fn packet_capabilities_publish_complete_integer_class_masks() {
    for name in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        for capability in builtin.integer_capabilities {
            assert!(!capability.inputs.is_empty(), "{name} {}", capability.form);
            for input in capability.inputs {
                assert_eq!(
                    input.classes, &INTEGER_CLASSES,
                    "{name} {} {}",
                    capability.form, input.name
                );
            }
        }
    }
}

#[test]
fn packet_calls_expose_descriptor_hover_and_signature_help() {
    for (lookup, source, expected) in [
        ("gamrnd", "x=gamrnd(2,3);", "r = gamrnd(a, b)"),
        (
            "groupcounts",
            "x=groupcounts([1;2]);",
            "varargout = groupingBuiltin(args...)",
        ),
        (
            "groupsummary",
            "x=groupsummary([1;2],[1;1],'sum');",
            "G = groupsummary(T, groupvars, method, datavars)",
        ),
        ("isoutlier", "x=isoutlier([1;2]);", "TF = isoutlier(A)"),
        ("kmeans", "x=kmeans([1;2],1);", "idx = kmeans(X, k)"),
        (
            "knnsearch",
            "x=knnsearch([1;2],[1]);",
            "Idx = knnsearch(X, Y)",
        ),
        ("lhsdesign", "x=lhsdesign(2,1);", "X = lhsdesign(n, p)"),
        ("linkage", "x=linkage([1;2]);", "Z = linkage(X)"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(lookup).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected),
            "{lookup}: {:?}",
            help.signatures
                .iter()
                .map(|signature| signature.label.as_str())
                .collect::<Vec<_>>()
        );
        let hover = hover_at(source, &analysis, &position).expect("hover");
        let HoverContents::Markup(markup) = hover.contents else {
            panic!("Markdown hover for {lookup}");
        };
        assert!(
            markup.value.contains(expected),
            "{lookup}: {}",
            markup.value
        );
    }
}

#[test]
fn grouping_control_extensions_and_groupsummary_forms_are_explicit() {
    for (name, extension_id) in [
        ("groupcounts", "groupcounts-integer-control"),
        ("groupsummary", "groupsummary-integer-control"),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).unwrap();
        assert!(builtin.extensions.iter().any(|extension| {
            extension.id == extension_id && extension.mode == BuiltinExtensionMode::RunMatOnly
        }));
    }

    let builtin = runmat_builtins::builtin_function_by_name("groupsummary").unwrap();
    let descriptor = builtin.descriptor.unwrap();
    assert_eq!(
        descriptor.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
    let array_binned = descriptor
        .signatures
        .iter()
        .find(|signature| signature.label.starts_with("[B, BG, BC]"))
        .expect("binned array groupsummary signature");
    assert_eq!(
        array_binned
            .inputs
            .iter()
            .map(|input| input.name)
            .collect::<Vec<_>>(),
        vec!["A", "groupvars", "groupbins", "method", "nameValuePairs"]
    );
    assert_eq!(
        array_binned
            .outputs
            .iter()
            .map(|output| output.name)
            .collect::<Vec<_>>(),
        vec!["B", "BG", "BC"]
    );
    let table_binned = descriptor
        .signatures
        .iter()
        .find(|signature| {
            signature.label.starts_with("G = groupsummary(T")
                && signature.label.contains("groupbins")
        })
        .expect("binned table groupsummary signature");
    assert_eq!(
        table_binned
            .inputs
            .iter()
            .map(|input| input.name)
            .collect::<Vec<_>>(),
        vec!["T", "groupvars", "groupbins", "method", "datavars"]
    );
    let ordinary_table = descriptor
        .signatures
        .iter()
        .find(|signature| signature.label == "G = groupsummary(T, groupvars, method, datavars)")
        .expect("ordinary table groupsummary signature");
    assert_eq!(ordinary_table.inputs[0].name, "T");
    assert_eq!(ordinary_table.inputs[0].description, "Input table.");
    assert!(builtin
        .integer_capabilities
        .iter()
        .any(|capability| capability.notes.contains("[integer-audit-open]")));
}
