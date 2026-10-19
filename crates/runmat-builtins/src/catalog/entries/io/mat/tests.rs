use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn literal_request(
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: usize,
) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

#[test]
fn exact_identity_inventory_and_user_documentation_are_present() {
    assert_eq!(ENTRIES.len(), 5);
    let names: Vec<_> = ENTRIES.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(
        names,
        [
            "load",
            "matfile",
            "matlab.io.MatFile.subsasgn",
            "matlab.io.MatFile.subsref",
            "save",
        ]
    );
    assert_eq!(documentation::LOAD.examples.len(), 7);
    assert_eq!(documentation::SAVE.examples.len(), 8);
    assert_eq!(documentation::MATFILE.examples.len(), 2);
    assert_eq!(documentation::LOAD.faqs.len(), 8);
    assert_eq!(documentation::SAVE.faqs.len(), 7);
    assert!(ENTRIES
        .iter()
        .all(|entry| !entry.documentation.examples.is_empty()));
}

#[test]
fn load_zero_output_is_workspace_effect_and_one_output_is_host_struct() {
    let zero = inference::load(&request(vec![text()], 0), &LOAD_CATALOG_ENTRY);
    assert!(zero.outputs.is_empty());
    assert!(zero.effects.0.contains(&EffectKind::WorkspaceWrite));
    assert!(zero.effects.0.contains(&EffectKind::FilesystemRead));
    let one = inference::load(&request(vec![text()], 1), &LOAD_CATALOG_ENTRY);
    assert!(one.diagnostics.is_empty());
    assert!(matches!(one.outputs[0].kind, ValueKindFact::Struct(_)));
    assert_eq!(one.outputs[0].residency, ResidencyFact::Host);
    assert!(!one.effects.0.contains(&EffectKind::WorkspaceWrite));
    assert!(
        !inference::load(&request(vec![text()], 2), &LOAD_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}

#[test]
fn save_status_and_matfile_object_keep_exact_known_classes() {
    let status = inference::save(&request(vec![text()], 1), &SAVE_CATALOG_ENTRY);
    assert!(matches!(
        status.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    ));
    let object = inference::matfile(&request(vec![text()], 1), &MATFILE_CATALOG_ENTRY);
    let ValueKindFact::Object(fact) = &object.outputs[0].kind else {
        panic!("matfile returns an object fact");
    };
    assert!(fact
        .runtime_class
        .as_ref()
        .is_some_and(|class| class.is(MATFILE_CLASS)));
    assert_eq!(fact.handle_semantics, Some(true));
}

#[test]
fn known_invalid_matfile_controls_and_receiver_reject_statically() {
    let invalid_flag = inference::matfile(
        &request(vec![text(), text(), text()], 1),
        &MATFILE_CATALOG_ENTRY,
    );
    assert!(!invalid_flag.diagnostics.is_empty());
    let bad_receiver = inference::subsref(
        &request(
            vec![
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            1,
        ),
        &MATFILE_SUBSREF_CATALOG_ENTRY,
    );
    assert!(!bad_receiver.diagnostics.is_empty());
    let unknown_receiver = inference::subsref(
        &request(
            vec![
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            1,
        ),
        &MATFILE_SUBSREF_CATALOG_ENTRY,
    );
    assert!(unknown_receiver.diagnostics.is_empty());

    let repeated = inference::matfile(
        &request(
            vec![
                text(),
                text(),
                ValueFact::scalar(ValueKindFact::Logical),
                text(),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Int8,
                    domain: NumericDomain::Real,
                })),
            ],
            1,
        ),
        &MATFILE_CATALOG_ENTRY,
    );
    assert!(repeated.diagnostics.is_empty());
}

#[test]
fn repeated_writable_and_literal_mat_options_match_runtime_grammar() {
    let repeated = literal_request(
        vec![
            text(),
            text(),
            ValueFact::scalar(ValueKindFact::Logical),
            text(),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::String("case.mat".into()),
            LiteralValue::String("Writable".into()),
            LiteralValue::Unknown,
            LiteralValue::String("writable".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(inference::matfile(&repeated, &MATFILE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let mut invalid = repeated;
    invalid.literals.literal_args[3] = LiteralValue::String("Other".into());
    assert!(!inference::matfile(&invalid, &MATFILE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    let unsupported_load = literal_request(
        vec![text(), text()],
        vec![
            LiteralValue::String("case.mat".into()),
            LiteralValue::String("-unknown".into()),
        ],
        1,
    );
    assert!(!inference::load(&unsupported_load, &LOAD_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let missing_save_regexp = literal_request(
        vec![text(), text()],
        vec![
            LiteralValue::String("case.mat".into()),
            LiteralValue::String("-regexp".into()),
        ],
        0,
    );
    assert!(!inference::save(&missing_save_regexp, &SAVE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn option_first_mat_calls_use_the_default_filename_without_false_path_diagnostics() {
    let load = literal_request(
        vec![text(), text()],
        vec![
            LiteralValue::String("-regexp".into()),
            LiteralValue::String("^x$".into()),
        ],
        1,
    );
    assert!(inference::load(&load, &LOAD_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let save = literal_request(
        vec![text(), text()],
        vec![
            LiteralValue::String("-struct".into()),
            LiteralValue::String("state".into()),
        ],
        0,
    );
    assert!(inference::save(&save, &SAVE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    assert!(inference::save(&save, &SAVE_CATALOG_ENTRY)
        .outputs
        .is_empty());
}
