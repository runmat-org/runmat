use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count == 0 || !(count - 1).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-READSTRUCT-ARITY",
            "readstruct requires a filename and complete name-value pairs",
            0,
        ));
    }
    super::super::inference::input(request, 0, &mut diagnostics);
    for index in (1..count).step_by(2) {
        super::super::inference::input(request, index, &mut diagnostics);
        if let Some(name) = request.literals.literal_string_at(index) {
            match super::super::ReadStructOption::from_name(&name) {
                Some(_) => {}
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-READSTRUCT-OPTION",
                    "unsupported readstruct option",
                    index,
                )),
            }
        }
        if index + 1 < count {
            super::super::inference::input(request, index + 1, &mut diagnostics);
            if request
                .literals
                .literal_string_at(index)
                .is_some_and(|name| {
                    super::super::ReadStructOption::from_name(&name)
                        == Some(super::super::ReadStructOption::FileType)
                })
                && request
                    .literals
                    .literal_string_at(index + 1)
                    .is_some_and(|value| !value.eq_ignore_ascii_case("xml"))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-READSTRUCT-FILETYPE",
                    "only XML FileType is supported",
                    index + 1,
                ));
            }
        }
    }
    let output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        false,
    )));
    super::super::inference::finish(request, entry, output, diagnostics)
}
