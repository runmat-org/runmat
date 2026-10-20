use super::super::facts::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

enum OptionName {
    ReadSize,
    IterationDimension,
    OutputType,
}
impl OptionName {
    fn decode(text: &str) -> Option<Self> {
        match text {
            "readsize" => Some(Self::ReadSize),
            "iterationdimension" => Some(Self::IterationDimension),
            "outputtype" => Some(Self::OutputType),
            _ => None,
        }
    }
}
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    option_pairs(request, &mut diagnostics);
    let mut dimension = Some(1.0);
    let mut same = Some(false);
    for index in (1..request.arguments.len().saturating_sub(1)).step_by(2) {
        text(request, index, &mut diagnostics);
        let Some(name) = request.literals.literal_string_at(index) else {
            dimension = None;
            same = None;
            continue;
        };
        match OptionName::decode(&name) {
            Some(OptionName::ReadSize) => positive_double(request, index + 1, &mut diagnostics),
            Some(OptionName::IterationDimension) => {
                positive_double(request, index + 1, &mut diagnostics);
                dimension = request.literals.numeric_at(index + 1);
            }
            Some(OptionName::OutputType) => {
                text(request, index + 1, &mut diagnostics);
                same = request
                    .literals
                    .literal_string_at(index + 1)
                    .and_then(|value| match value.as_str() {
                        "same" => Some(true),
                        "cell" => Some(false),
                        _ => {
                            diagnostics.push(argument_error(
                                "RM-ARRAY-DATASTORE-OUTPUT-TYPE",
                                "OutputType must be cell or same",
                                index + 1,
                            ));
                            None
                        }
                    });
            }
            None => diagnostics.push(argument_error(
                "RM-ARRAY-DATASTORE-OPTION",
                "unsupported arrayDatastore option",
                index,
            )),
        }
    }
    if same == Some(true) && dimension.is_some_and(|dimension| dimension != 1.0) {
        diagnostics.push(argument_error(
            "RM-ARRAY-DATASTORE-DIMENSION",
            "OutputType same requires IterationDimension 1",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        object(
            StaticClassIdentity::new("arrayDatastore"),
            &[
                ("Data", retained(request, 0)),
                ("ReadSize", double()),
                ("IterationDimension", double()),
                ("OutputType", string()),
            ],
        ),
        diagnostics,
    )
}
