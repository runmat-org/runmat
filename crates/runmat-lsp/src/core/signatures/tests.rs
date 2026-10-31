use super::*;
use crate::core::{analysis, position::offset_to_position};

#[test]
fn call_context_selects_output_forms_and_preserves_all_signature_help() {
    for (text, offset, outputs) in [
        ("[~, k] = max(f);", 9, 2),
        ("[a,b] = max(max(f));", 8, 2),
        ("[a,b] = max(max(f));", 12, 1),
    ] {
        let document =
            analysis::analyze_document_with_compat(text, analysis::CompatMode::default());
        let site = CallSite::at(&document, offset).expect("call context");
        assert_eq!(site.outputs, Some(outputs));
        let entry = runmat_builtins::public_builtin_catalog_entry_by_name("max").unwrap();
        let selected = site.best(entry.descriptor.signatures);
        assert!(!selected.is_empty());
        assert!(selected
            .iter()
            .all(|index| entry.descriptor.signatures[*index].outputs.len() == outputs));
        let position = offset_to_position(text, offset);
        let help = analysis::signature_help_at(text, &document, &position).unwrap();
        assert_eq!(help.signatures.len(), entry.descriptor.signatures.len());
        assert_eq!(help.active_signature, Some(selected[0] as u32));
        let hover = analysis::hover_at(text, &document, &position).unwrap();
        let lsp_types::HoverContents::Markup(markdown) = hover.contents else {
            panic!("markdown hover")
        };
        let expected_header = selected
            .iter()
            .map(|index| entry.descriptor.signatures[*index].label)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            markdown
                .value
                .starts_with(&format!("```runmat\n{expected_header}\n```")),
            "hover must preserve all equally ranked signatures: {}",
            markdown.value
        );
        assert!(markdown.value.contains("more overloads"));
    }
    let text = "[f, xi] = ksdensity(cycle);";
    let document = analysis::analyze_document_with_compat(text, analysis::CompatMode::default());
    let site = CallSite::at(&document, text.find("ksdensity").unwrap()).unwrap();
    assert_eq!(site.outputs, Some(2));
    let entry = runmat_builtins::public_builtin_catalog_entry_by_name("ksdensity").unwrap();
    let selected = site.best(entry.descriptor.signatures);
    assert_eq!(selected, vec![0]);
    for (name, text) in [("false", "out = false(2);"), ("true", "out = true(2);")] {
        let offset = text.find(name).unwrap();
        let document =
            analysis::analyze_document_with_compat(text, analysis::CompatMode::default());
        let site = CallSite::at(&document, offset).expect("logical constructor call");
        assert_eq!(site.name, name);
        let entry = runmat_builtins::public_builtin_catalog_entry_by_name(name).unwrap();
        let selected = site.best(entry.descriptor.signatures);
        assert!(!selected.is_empty());
        assert_eq!(
            entry.descriptor.signatures[selected[0]].label,
            format!("L = {name}(n)")
        );
        let position = offset_to_position(text, offset);
        let help = analysis::signature_help_at(text, &document, &position).unwrap();
        assert_eq!(help.active_signature, Some(selected[0] as u32));
        let hover = analysis::hover_at(text, &document, &position).unwrap();
        let lsp_types::HoverContents::Markup(markdown) = hover.contents else {
            panic!("markdown hover")
        };
        let expected_header = selected
            .iter()
            .map(|index| entry.descriptor.signatures[*index].label)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            markdown
                .value
                .starts_with(&format!("```runmat\n{expected_header}\n```")),
            "hover must preserve all equally ranked signatures: {}",
            markdown.value
        );
    }
}

#[test]
fn parameter_cursor_uses_balanced_syntax_not_commas_inside_strings_or_arrays() {
    for marked in [
        "max(A, [1, 2], |)",
        "max(A, sin(2), |)",
        "max(A, \"x,y\", |)",
        "max(A, |",
    ] {
        let cursor = marked.find('|').unwrap();
        let text = marked.replace('|', "");
        let document =
            analysis::analyze_document_with_compat(&text, analysis::CompatMode::default());
        let site = CallSite::at(&document, cursor).unwrap();
        assert_eq!(site.name, "max");
        let expected = if text == "max(A, " { 1 } else { 2 };
        assert_eq!(site.active_argument, expected);
        let help =
            analysis::signature_help_at(&text, &document, &offset_to_position(&text, cursor))
                .unwrap();
        assert_eq!(help.active_parameter, Some(expected as u32));
        assert!(help
            .signatures
            .iter()
            .all(|signature| signature.parameters.is_some()));
    }
}

#[test]
fn unknown_arguments_retain_ties_and_known_kinds_prefer_specific_descriptors() {
    use runmat_builtins::BuiltinParamDescriptor;
    const ANY: BuiltinParamDescriptor = BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "",
    };
    const NUMERIC: BuiltinParamDescriptor = BuiltinParamDescriptor {
        ty: BuiltinParamType::NumericArray,
        ..ANY
    };
    let descriptors = [
        BuiltinSignatureDescriptor {
            label: "generic",
            inputs: &[ANY],
            outputs: &[ANY],
        },
        BuiltinSignatureDescriptor {
            label: "numeric",
            inputs: &[NUMERIC],
            outputs: &[ANY],
        },
    ];
    for (text, expected) in [("sample(x)", vec![0, 1]), ("sample([1,2])", vec![1])] {
        let document =
            analysis::analyze_document_with_compat(text, analysis::CompatMode::default());
        let site = CallSite::at(&document, 0).unwrap();
        assert_eq!(site.best(&descriptors), expected);
    }
    const REPEATED: BuiltinParamDescriptor = BuiltinParamDescriptor {
        name: "placeholder",
        default: Some("[]"),
        ..ANY
    };
    let descriptor = BuiltinSignatureDescriptor {
        label: "[x] = sample([], [], λ)",
        inputs: &[
            REPEATED,
            REPEATED,
            BuiltinParamDescriptor { name: "λ", ..ANY },
        ],
        outputs: &[ANY],
    };
    let labels = parameters(&descriptor)
        .into_iter()
        .map(|parameter| parameter.label)
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        vec![
            ParameterLabel::LabelOffsets([13, 15]),
            ParameterLabel::LabelOffsets([17, 19]),
            ParameterLabel::LabelOffsets([21, 22]),
        ]
    );
}
