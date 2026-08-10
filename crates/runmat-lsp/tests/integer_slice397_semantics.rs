use lsp_types::Position;
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, signature_help_at, CompatMode,
};

#[test]
fn slice397_descriptors_are_visible_to_lsp() {
    for (source, expected) in [
        (
            "t = datetime(uint16([2024 7 31]));",
            "t = datetime(dateVectors)",
        ),
        (
            "t2 = dateshift(datetime(2024, 7, 31), 'start', 'month');",
            "t2 = dateshift(t, boundary, unit)",
        ),
        ("d = day(datetime(2024, 7, 31));", "X = day(t)"),
        ("g = dcgain(tf(1, [1 1]));", "gain = dcgain(sys)"),
        ("[a, b] = deal(uint8(1));", "[varargout] = deal(varargin)"),
        (
            "dA = decomposition(int16([1 0; 0 1]));",
            "dA = decomposition(A)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let column = source.find('(').expect("call expression") as u32 - 1;
        let help = signature_help_at(source, &analysis, &Position::new(0, column))
            .unwrap_or_else(|| panic!("descriptor-backed signature help for {source}"));
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected),
            "expected {expected} for {source}"
        );
    }
}

#[test]
fn matlab_mode_keeps_slice397_capabilities_and_extensions_visible() {
    for name in [
        "datetime",
        "dateshift",
        "day",
        "dcgain",
        "deal",
        "decomposition",
        "decomposition.ctranspose",
        "decomposition.mldivide",
        "decomposition.mrdivide",
        "decomposition.mtimes",
        "decomposition.rdivide",
        "decomposition.subsref",
        "decomposition.times",
        "decomposition.uminus",
        "decomposition.uplus",
        "mldivide",
        "mrdivide",
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} must expose its settled integer capability records"
        );
    }

    let deal = runmat_builtins::builtin_function_by_name("deal").expect("registered deal");
    assert_eq!(deal.integer_capabilities.len(), 2);
    assert_eq!(deal.integer_capabilities[0].inputs[0].classes.len(), 8);
    assert_eq!(
        deal.integer_capabilities[0].inputs[0].availability,
        runmat_builtins::BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(
        deal.integer_capabilities[0].output_class,
        runmat_builtins::BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert_eq!(
        deal.integer_capabilities[0].backend,
        runmat_builtins::BuiltinIntegerBackendRule::HostOnly
    );
    assert_eq!(deal.integer_capabilities[1].inputs[0].classes.len(), 8);
    assert_eq!(
        deal.integer_capabilities[1].inputs[0].availability,
        runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly
    );
    assert_eq!(
        deal.integer_capabilities[1].output_class,
        runmat_builtins::BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert_eq!(
        deal.integer_capabilities[1].backend,
        runmat_builtins::BuiltinIntegerBackendRule::HostAndGpu
    );

    for (name, extension_id) in [
        ("datetime", "datetime-implicit-datenum"),
        ("dateshift", "datetime-resident-numeric-input"),
        ("day", "datetime-logical-numeric-input"),
        ("deal", "deal-resident-input"),
        ("decomposition", "decomposition-nonfloating-input"),
        (
            "decomposition.ctranspose",
            "decomposition-nonfloating-input",
        ),
        ("decomposition.mldivide", "decomposition-nonfloating-input"),
        ("decomposition.mrdivide", "decomposition-nonfloating-input"),
        ("decomposition.mtimes", "decomposition-nonfloating-input"),
        ("decomposition.rdivide", "decomposition-nonfloating-input"),
        ("decomposition.subsref", "decomposition-nonfloating-input"),
        ("decomposition.times", "decomposition-nonfloating-input"),
        ("decomposition.uminus", "decomposition-nonfloating-input"),
        ("decomposition.uplus", "decomposition-nonfloating-input"),
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
            "{name} must expose {extension_id} to LSP metadata"
        );
    }

    assert!(runmat_builtins::builtin_function_by_name("decomposition.ldivide").is_none());

    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));
    assert!(completions
        .iter()
        .all(|item| item.label != "decomposition.ldivide"));
    assert!(completions.iter().all(|item| {
        runmat_builtins::builtin_function_by_name(&item.label).is_none_or(|builtin| {
            builtin.descriptor.is_none_or(|descriptor| {
                descriptor.completion_policy
                    != runmat_builtins::BuiltinCompletionPolicy::HiddenInternal
            })
        })
    }));
}
