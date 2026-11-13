use super::super::facts::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

enum OptionName {
    ReadSize,
    BlockSize,
}
impl OptionName {
    fn decode(text: &str) -> Option<Self> {
        match text {
            "readsize" => Some(Self::ReadSize),
            "blocksize" => Some(Self::BlockSize),
            _ => None,
        }
    }
}
// The shared runtime guard rejects native integer payloads recursively.
// Callable contracts are not payloads; only captures participate in this check.
fn rejected_integer(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Numeric(numeric) => numeric.class.integer_class().is_some(),
        ValueKindFact::Cell(cell) => {
            rejected_integer(&cell.element) || cell.elements.iter().any(rejected_integer)
        }
        ValueKindFact::Struct(structure) => {
            structure.fields.values().any(rejected_integer)
                || structure
                    .elements
                    .iter()
                    .any(|fields| fields.values().any(rejected_integer))
        }
        ValueKindFact::Object(object) => object.properties.values().any(rejected_integer),
        ValueKindFact::Callable(callable) => callable.captures.iter().any(rejected_integer),
        _ => false,
    }
}
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    option_pairs(request, &mut diagnostics);
    for (index, value) in request.arguments.iter().enumerate() {
        if rejected_integer(value) {
            diagnostics.push(argument_error(
                "RM-PARQUET-DATASTORE-INTEGER",
                "typed-integer constructor arguments are not supported",
                index,
            ));
        }
    }
    let mut read_size = string();
    for index in (1..request.arguments.len().saturating_sub(1)).step_by(2) {
        text(request, index, &mut diagnostics);
        let Some(name) = request.literals.literal_string_at(index) else {
            read_size = ValueFact::unknown(DynamicReason::RuntimeValue);
            continue;
        };
        match OptionName::decode(&name) {
            Some(OptionName::BlockSize) => positive_double(request, index + 1, &mut diagnostics),
            Some(OptionName::ReadSize) => {
                let value = &request.arguments[index + 1];
                match value.kind {
                    ValueKindFact::String | ValueKindFact::Character => {
                        text(request, index + 1, &mut diagnostics);
                        if request
                            .literals
                            .literal_string_at(index + 1)
                            .is_some_and(|text| text != "file" && text != "rowgroup")
                        {
                            diagnostics.push(argument_error(
                                "RM-PARQUET-DATASTORE-READ-SIZE",
                                "ReadSize must be file, rowgroup, or a positive double integer",
                                index + 1,
                            ));
                        }
                        read_size = string();
                    }
                    ValueKindFact::Unknown => {
                        read_size = ValueFact::unknown(DynamicReason::RuntimeValue)
                    }
                    _ => {
                        positive_double(request, index + 1, &mut diagnostics);
                        read_size = double();
                    }
                }
            }
            None => diagnostics.push(argument_error(
                "RM-PARQUET-DATASTORE-OPTION",
                "unsupported parquetDatastore option",
                index,
            )),
        }
    }
    finish_fixed(
        entry,
        request,
        object(
            StaticClassIdentity::new("parquetDatastore"),
            &[
                ("Files", retained(request, 0)),
                ("ReadSize", read_size),
                ("BlockSize", double()),
            ],
        ),
        diagnostics,
    )
}
