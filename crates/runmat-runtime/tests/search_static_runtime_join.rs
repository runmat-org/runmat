//! Candidate `crates/runmat-runtime/tests/search_static_runtime_join.rs`.
//! Actual canonical Runtime dispatch versus catalog facts for all 11 forms.
use futures::executor::block_on;
use runmat_builtins::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_runtime::{call_builtin_async_with_outputs, value_fact::value_fact};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, OutputSelection, ValueKindFact,
};
use runmat_value::Value;

#[derive(Clone, Copy)]
enum ExpectedKind {
    Logical,
    DoubleIndices,
    CellOfDoubleIndices,
}

fn text(value: &str) -> Value {
    Value::String(value.into())
}

fn check_form(name: &str, args: Vec<Value>, literals: Vec<LiteralValue>, kind: ExpectedKind) {
    assert_eq!(args.len(), literals.len(), "{name}: source argument facts");
    let entry = builtin_catalog_entry_by_name(name).expect("canonical catalog binding");
    let inference = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: args.iter().map(value_fact).collect(),
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        },
    );
    assert!(inference.diagnostics.is_empty(), "{name}: {inference:?}");
    assert_eq!(inference.outputs.len(), 1, "{name}: fixed public output");
    let actual = block_on(call_builtin_async_with_outputs(name, &args, 1))
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    let actual = value_fact(&actual);
    let inferred = &inference.outputs[0];
    match kind {
        ExpectedKind::Logical => {
            assert!(matches!(inferred.kind, ValueKindFact::Logical));
            assert!(matches!(actual.kind, ValueKindFact::Logical));
            assert!(
                inferred.shape.is_proven_equivalent(&actual.shape),
                "{name}: inferred {:?}, actual {:?}",
                inferred.shape,
                actual.shape
            );
        }
        ExpectedKind::DoubleIndices => {
            assert!(
                matches!(&inferred.kind, ValueKindFact::Numeric(number) if number.class == NumericClass::Double)
            );
            assert!(
                matches!(&actual.kind, ValueKindFact::Numeric(number) if number.class == NumericClass::Double)
            );
            // Result length depends on contents: inference proves row rank and
            // Double class, not a fabricated count of substring matches.
            assert_eq!(inferred.shape.known_dims().unwrap()[0], Some(1));
            assert_eq!(actual.shape.known_dims().unwrap()[0], Some(1));
        }
        ExpectedKind::CellOfDoubleIndices => {
            let ValueKindFact::Cell(inferred_cell) = &inferred.kind else {
                panic!("{name}: inferred non-cell output: {inferred:?}")
            };
            let ValueKindFact::Cell(actual_cell) = &actual.kind else {
                panic!("{name}: runtime non-cell output: {actual:?}")
            };
            assert!(
                matches!(&inferred_cell.element.kind, ValueKindFact::Numeric(number) if number.class == NumericClass::Double)
            );
            assert!(
                matches!(&actual_cell.element.kind, ValueKindFact::Numeric(number) if number.class == NumericClass::Double)
            );
            assert!(inferred.shape.is_proven_equivalent(&actual.shape));
        }
    }
}

#[test]
fn quartet_all_eleven_source_forms_have_sound_runtime_output_facts() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let unknown = LiteralValue::Unknown;
    for name in ["contains", "endsWith", "startsWith"] {
        check_form(
            name,
            vec![text("RunMat"), text("Run")],
            vec![unknown.clone(), unknown.clone()],
            ExpectedKind::Logical,
        );
        check_form(
            name,
            vec![text("RunMat"), text("run"), Value::Bool(true)],
            vec![unknown.clone(), unknown.clone(), LiteralValue::Bool(true)],
            ExpectedKind::Logical,
        );
        check_form(
            name,
            vec![
                text("RunMat"),
                text("run"),
                text("IgnoreCase"),
                Value::Bool(true),
            ],
            vec![
                unknown.clone(),
                unknown.clone(),
                LiteralValue::String("IgnoreCase".into()),
                LiteralValue::Bool(true),
            ],
            ExpectedKind::Logical,
        );
    }
    check_form(
        "strfind",
        vec![text("aaaa"), text("aa")],
        vec![unknown.clone(), unknown.clone()],
        ExpectedKind::DoubleIndices,
    );
    check_form(
        "strfind",
        vec![
            text("mission"),
            text("s"),
            text("ForceCellOutput"),
            Value::Bool(true),
        ],
        vec![
            unknown.clone(),
            unknown,
            LiteralValue::String("ForceCellOutput".into()),
            LiteralValue::Bool(true),
        ],
        ExpectedKind::CellOfDoubleIndices,
    );
}
