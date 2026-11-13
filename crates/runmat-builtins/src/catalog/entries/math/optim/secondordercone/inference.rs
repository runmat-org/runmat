use super::super::program_facts as facts;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let matrix_shape = inspect_inputs(request, &mut diagnostics);
    let mut fields = BTreeMap::new();
    fields.insert(
        "A".into(),
        facts::double(
            matrix_shape
                .map(ShapeFact::from)
                .unwrap_or(ShapeFact::Unknown),
            match request.arguments.first().map(|value| value.storage) {
                Some(StorageFact::Sparse) => StorageFact::Sparse,
                None | Some(StorageFact::Unknown) => StorageFact::Unknown,
                _ => StorageFact::Dense,
            },
        ),
    );
    for (name, index) in [("b", 1), ("d", 2)] {
        fields.insert(
            name.into(),
            facts::double(
                ShapeFact::from(vec![
                    request
                        .arguments
                        .get(index)
                        .and_then(|fact| fact.shape.element_count()),
                    Some(1),
                ]),
                StorageFact::Dense,
            ),
        );
    }
    fields.insert(
        "gamma".into(),
        facts::double(ShapeFact::Scalar, StorageFact::Scalar),
    );
    fields.insert("Type".into(), ValueFact::scalar(ValueKindFact::String));
    facts::finish(
        entry,
        request,
        vec![ValueFact::scalar(ValueKindFact::Struct(
            StructFact::scalar(fields, true),
        ))],
        diagnostics,
    )
}

fn inspect_inputs(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<Vec<Option<usize>>> {
    if request.arguments.len() != 4 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SOC-ARITY",
            "secondordercone requires A, b, d and gamma",
            0,
        ));
    }
    facts::numeric(
        request,
        0,
        facts::OptimNumericInput::NonemptySparseMatrix,
        diagnostics,
    );
    for index in [1, 2] {
        facts::numeric(
            request,
            index,
            facts::OptimNumericInput::OptionalVector,
            diagnostics,
        );
    }
    facts::numeric(request, 3, facts::OptimNumericInput::Scalar, diagnostics);
    let matrix_shape = request
        .arguments
        .first()
        .and_then(|fact| fact.shape.known_dims())
        .map(|dims| {
            vec![
                dims.first().copied().unwrap_or(Some(1)),
                dims.get(1).copied().unwrap_or(Some(1)),
            ]
        });
    for (index, dimension) in [(1, 0), (2, 1)] {
        if let (Some(expected), Some(actual)) = (
            matrix_shape.as_ref().and_then(|dims| dims[dimension]),
            request
                .arguments
                .get(index)
                .and_then(|fact| fact.shape.element_count()),
        ) {
            if expected != actual {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SOC-DIMENSION",
                    "cone vectors must match the admitted matrix rows and columns",
                    index,
                ));
            }
        }
    }
    if request
        .literals
        .numeric_at(3)
        .is_some_and(|value| !value.is_finite())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SOC-FINITE",
            "gamma must be finite",
            3,
        ));
    }
    matrix_shape
}

pub(in super::super) fn inspect_descriptor(
    descriptor: &StructFact,
    objective: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let start = diagnostics.len();
    let arguments = super::SECONDORDERCONE_DESCRIPTOR.signatures[0]
        .inputs
        .iter()
        .map(|input| {
            descriptor
                .fields
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(input.name))
                .map(|(_, fact)| fact.clone())
                .unwrap_or_else(|| {
                    if descriptor.fields_complete {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-SOC-FIELD",
                            "cone descriptor is missing a required constructor field",
                            1,
                        ));
                    }
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                })
        })
        .collect();
    let request = CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let matrix_shape = inspect_inputs(&request, diagnostics);
    for length in [
        matrix_shape.as_ref().and_then(|dims| dims[1]),
        request
            .arguments
            .get(2)
            .and_then(|fact| fact.shape.element_count()),
    ] {
        if length
            .zip(objective)
            .is_some_and(|(length, objective)| length != objective)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SOC-OBJECTIVE",
                "cone A columns and d length must match the objective length",
                1,
            ));
        }
    }
    for diagnostic in &mut diagnostics[start..] {
        diagnostic.argument = Some(1);
    }
}
