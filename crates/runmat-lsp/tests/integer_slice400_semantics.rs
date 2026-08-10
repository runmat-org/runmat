use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const SLICE400_BUILTINS: [&str; 10] = [
    "ecdf",
    "eig",
    "eigs",
    "empty",
    "encode",
    "endsWith",
    "envelope",
    "eq",
    "erase",
    "eraseBetween",
];

const INTEGER_EMPTY_BUILTINS: [&str; 8] = [
    "int8.empty",
    "int16.empty",
    "int32.empty",
    "int64.empty",
    "uint8.empty",
    "uint16.empty",
    "uint32.empty",
    "uint64.empty",
];

#[test]
fn slice400_descriptors_are_visible_to_signature_help_and_hover() {
    for (name, source, expected_signature) in [
        ("ecdf", "[f, x] = ecdf(uint8([1 2]));", "[f, x] = ecdf(y)"),
        ("eig", "d = eig(uint8(1));", "d = eig(A)"),
        ("eigs", "d = eigs(uint8(1));", "d = eigs(A)"),
        ("empty", "a = empty(0);", "A = ClassName.empty"),
        (
            "encode",
            "counts = encode(1, 'word');",
            "counts = encode(bag, documentsOrWords, Name, Value, ...)",
        ),
        (
            "endsWith",
            "tf = endsWith('runmat', 'mat');",
            "tf = endsWith(str, pat)",
        ),
        (
            "envelope",
            "upper = envelope(uint8([1 2]));",
            "yupper = envelope(x)",
        ),
        ("eq", "tf = eq(uint8(1), uint8(1));", "tf = eq(A, B)"),
        (
            "erase",
            "out = erase('runmat', 'run');",
            "newStr = erase(str, pattern)",
        ),
        (
            "eraseBetween",
            "out = eraseBetween('runmat', 1, 3);",
            "newText = eraseBetween(str, start, end)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");

        let column = source.find(name).expect("builtin call") as u32;
        let position = Position::new(0, column);
        let help = signature_help_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("descriptor-backed signature help for {name}"));
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected_signature),
            "expected {expected_signature} for {name}, got {:?}",
            help.signatures
                .iter()
                .map(|signature| signature.label.as_str())
                .collect::<Vec<_>>()
        );

        let hover = hover_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("descriptor-backed hover for {name}"));
        let markdown = match hover.contents {
            HoverContents::Markup(markup) => markup.value,
            other => panic!("expected Markdown hover for {name}, got {other:?}"),
        };
        assert!(
            markdown.contains(expected_signature),
            "expected descriptor signature in {name} hover, got:\n{markdown}"
        );
        assert!(
            markdown
                .to_ascii_lowercase()
                .contains(&name.to_ascii_lowercase()),
            "expected {name} documentation in hover, got:\n{markdown}"
        );
    }
}

#[test]
fn slice400_builtins_are_public_completions_with_integer_metadata() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in SLICE400_BUILTINS {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        let descriptor = builtin
            .descriptor
            .unwrap_or_else(|| panic!("{name} must expose a descriptor"));
        assert_eq!(
            descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name} must remain a public LSP completion"
        );
        assert!(
            !descriptor.signatures.is_empty(),
            "{name} must expose descriptor-backed signatures"
        );
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} must expose its settled integer capability records"
        );

        let completion = completions
            .iter()
            .find(|item| item.label.eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("public completion for {name}"));
        let detail = completion.detail.as_deref().unwrap_or_default();
        assert!(
            detail.contains(&format!("{name}(")),
            "{name} completion detail must be rendered from its descriptor, got {detail:?}"
        );
    }
}

#[test]
fn integer_empty_static_registrations_follow_the_public_completion_policy() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in INTEGER_EMPTY_BUILTINS {
        let builtin = runmat_builtins::builtin_function_by_name(name)
            .unwrap_or_else(|| panic!("registered static builtin {name}"));
        let descriptor = builtin
            .descriptor
            .unwrap_or_else(|| panic!("{name} must expose a descriptor"));
        assert_eq!(
            descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name} inherits the public empty completion policy"
        );
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} must expose the empty integer capability"
        );
        assert!(
            builtin.extensions.iter().any(|extension| {
                extension.id == "empty-resident-size"
                    && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
            }),
            "{name} must expose the resident-size extension"
        );

        let completion = completions
            .iter()
            .find(|item| item.label.eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("public completion for {name}"));
        let detail = completion.detail.as_deref().unwrap_or_default();
        assert!(
            detail.contains(&format!("{name}(")),
            "{name} completion detail must be rendered from the shared empty descriptor, got {detail:?}"
        );
    }
}

#[test]
fn matlab_mode_keeps_slice400_extension_metadata_visible() {
    for (name, extension_id) in [
        ("ecdf", "ecdf-integer-y"),
        ("eig", "eig-nonfloating-coefficient"),
        ("eigs", "eigs-nonfloating-matrix"),
        ("empty", "empty-global-call"),
        ("encode", "encode-numeric-force-cell-output"),
        ("endsWith", "endswith-numeric-ignore-case"),
        ("envelope", "envelope-integer-data"),
        ("eraseBetween", "erasebetween-resident-position"),
    ] {
        let source = format!("x = {name}(uint8(1));");
        let analysis = analyze_document_with_compat(&source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{source}");

        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.extensions.iter().any(|extension| {
                extension.id == extension_id
                    && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
            }),
            "{name} must expose {extension_id} to LSP consumers"
        );
    }

    for name in ["eq", "erase"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.extensions.is_empty(),
            "{name} has settled integer capability metadata but no compatibility extension"
        );
    }
}
