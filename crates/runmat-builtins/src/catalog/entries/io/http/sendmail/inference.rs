use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    if !(2..=4).contains(&args.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SENDMAIL-ARITY",
            "sendmail expects recipient, subject, optional message, and optional attachments",
            0,
        ));
    }
    if let Some(recipient) = args.first() {
        let admissible = match &recipient.kind {
            ValueKindFact::Cell(cell) => text(&cell.element) && cell.elements.iter().all(text),
            ValueKindFact::String => true,
            ValueKindFact::Character | ValueKindFact::Unknown => text(recipient),
            _ => false,
        };
        if !admissible {
            diagnostics.push(argument_error(
                "RM-CATALOG-SENDMAIL-RECIPIENT",
                "recipient must be text or a collection of text addresses",
                0,
            ));
        }
    }
    for (index, argument) in args.iter().enumerate().take(3).skip(1) {
        if !text(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SENDMAIL-TEXT",
                "subject and message must be character rows or string scalars",
                index,
            ));
        }
    }
    if args.len() == 4 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SENDMAIL-ATTACHMENTS",
            "the current mail transports do not support attachments",
            3,
        ));
    }
    let mut output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, diagnostics)
}

fn text(value: &ValueFact) -> bool {
    super::super::super::repl_fs::text_input::scalar(value)
}
