use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const SLICE399_BUILTINS: [&str; 10] = [
    "discretize",
    "dividerand",
    "dlmread",
    "dlmwrite",
    "doc2sequence",
    "dot",
    "double",
    "downsample",
    "dummyvar",
    "duration",
];

#[test]
fn slice399_descriptors_are_visible_to_signature_help_and_hover() {
    for (name, source, expected_signature) in [
        (
            "discretize",
            "y = discretize([1 2], [0 2 3]);",
            "Y = discretize(X, edges)",
        ),
        (
            "dividerand",
            "[train, val, test] = dividerand(uint8(10));",
            "[trainInd, valInd, testInd] = dividerand(Q)",
        ),
        (
            "dlmread",
            "m = dlmread('data.csv');",
            "M = dlmread(filename)",
        ),
        (
            "dlmwrite",
            "dlmwrite('data.csv', uint8(1));",
            "bytesWritten = dlmwrite(filename, M)",
        ),
        (
            "doc2sequence",
            "sequences = doc2sequence(1, 1);",
            "sequences = doc2sequence(emb, documents)",
        ),
        (
            "dot",
            "c = dot(uint8([1 2]), uint8([3 4]));",
            "C = dot(A, B)",
        ),
        ("double", "y = double(uint8(1));", "Y = double(X)"),
        (
            "downsample",
            "y = downsample(uint8([1 2]), 2);",
            "Y = downsample(X, N)",
        ),
        (
            "dummyvar",
            "d = dummyvar(uint8([1 2]));",
            "D = dummyvar(group)",
        ),
        ("duration", "t = duration(uint8([1 2]));", "t = duration(X)"),
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
            markdown.to_ascii_lowercase().contains(name),
            "expected {name} documentation in hover, got:\n{markdown}"
        );
    }
}

#[test]
fn slice399_builtins_are_public_completions_with_integer_metadata() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in SLICE399_BUILTINS {
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
        let call_head = format!("{name}(");
        assert!(
            detail.contains(&call_head),
            "expected a descriptor signature in {name} completion detail, got {detail:?}"
        );
    }
}

#[test]
fn matlab_mode_keeps_slice399_extension_metadata_visible() {
    for (name, extension_id) in [
        ("dividerand", "dividerand-resident-argument"),
        ("dlmread", "dlmread-colon-spreadsheet-range"),
        ("dlmwrite", "dlmwrite-byte-count-output"),
        ("dot", "dot-integer-data"),
        ("double", "double-like-prototype"),
        ("downsample", "downsample-integer-factor"),
        ("dummyvar", "dummyvar-integer-group"),
        ("duration", "duration-short-component-form"),
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
}
