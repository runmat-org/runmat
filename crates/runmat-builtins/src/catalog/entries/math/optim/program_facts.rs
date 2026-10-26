//! Shared representation facts of the existing host LP/conic normalization.
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimNumericInput {
    OptionalVector,
    NonemptyVector,
    OptionalMatrix,
    /// Real sparse or dense matrix; only consumers with a CSC adapter use it.
    OptionalSparseMatrix,
    /// Column-major flattened arrays for bound and initial-point consumers.
    OptionalSparseArray,
    NonemptySparseMatrix,
    NonemptyMatrix,
    Scalar,
}

impl OptimNumericInput {
    /// Metadata preflight, not a payload/finite-value or allocation certificate.
    pub fn accepts_metadata(self, fact: &ValueFact) -> bool {
        if fact.storage == StorageFact::Sparse
            && !matches!(
                self,
                Self::OptionalSparseMatrix | Self::OptionalSparseArray | Self::NonemptySparseMatrix
            )
        {
            return false;
        }
        let vector = matches!(
            self,
            Self::OptionalVector | Self::NonemptyVector | Self::Scalar
        );
        let empty = matches!(
            self,
            Self::OptionalVector
                | Self::OptionalMatrix
                | Self::OptionalSparseMatrix
                | Self::OptionalSparseArray
        );
        match fact.kind {
            ValueKindFact::Unknown => return true,
            ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {}
            ValueKindFact::Logical
                if vector && matches!(fact.storage, StorageFact::Scalar | StorageFact::Unknown) => {
            }
            _ => return false,
        }
        if fact.shape.element_count() == Some(0) {
            return empty;
        }
        if self != Self::OptionalSparseArray && fact.shape.rank().is_some_and(|rank| rank > 2) {
            return false;
        }
        if self == Self::Scalar {
            return fact.shape.element_count().is_none_or(|count| count == 1);
        }
        if vector {
            return fact.shape.known_dims().is_none_or(|dims| {
                dims.len() < 2
                    || dims[0].is_none_or(|rows| rows == 1)
                    || dims[1].is_none_or(|cols| cols == 1)
            });
        }
        true
    }
}

pub(super) fn numeric(
    request: &CallRequest,
    index: usize,
    role: OptimNumericInput,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request
        .arguments
        .get(index)
        .is_some_and(|fact| !role.accepts_metadata(fact))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPT-NUMERIC",
            "solver operand has unsupported real numeric geometry or storage",
            index,
        ));
    }
}

pub(super) fn constraint_geometry(
    request: &CallRequest,
    objective: usize,
    pairs: &[(usize, usize)],
    bounds: &[usize],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let length = request
        .arguments
        .get(objective)
        .and_then(|fact| fact.shape.element_count());
    for &(matrix, rhs) in pairs {
        let Some(a) = request.arguments.get(matrix) else {
            continue;
        };
        let b_count = request
            .arguments
            .get(rhs)
            .and_then(|fact| fact.shape.element_count());
        if a.shape.element_count() == Some(0) {
            if b_count.is_some_and(|count| count != 0) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPT-EMPTY-PAIR",
                    "a nonempty right-hand side requires a constraint matrix",
                    rhs,
                ));
            }
            continue;
        }
        if let Some(dims) = a.shape.known_dims() {
            let rows = dims.first().copied().unwrap_or(Some(1));
            let cols = dims.get(1).copied().unwrap_or(Some(1));
            if length
                .zip(cols)
                .is_some_and(|(length, cols)| length != cols)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPT-COLUMNS",
                    "constraint columns must match the objective length",
                    matrix,
                ));
            }
            if rows.zip(b_count).is_some_and(|(rows, count)| rows != count) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPT-ROWS",
                    "constraint rows must match the right-hand side length",
                    rhs,
                ));
            }
        }
    }
    for &index in bounds {
        if length
            .zip(
                request
                    .arguments
                    .get(index)
                    .and_then(|fact| fact.shape.element_count()),
            )
            .is_some_and(|(n, count)| count != 0 && n != count)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPT-BOUND",
                "a nonempty bound must match the objective length",
                index,
            ));
        }
    }
}

pub(super) fn double(shape: ShapeFact, storage: StorageFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    )
}

pub(super) fn vector() -> ValueFact {
    double(ShapeFact::from(vec![None, Some(1)]), StorageFact::Dense)
}

pub(super) fn diagnostics() -> ValueFact {
    let mut fields = BTreeMap::new();
    for name in ["iterations", "constrviolation"] {
        fields.insert(name.into(), double(ShapeFact::Scalar, StorageFact::Scalar));
    }
    for name in ["algorithm", "message"] {
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)))
}

pub(super) fn lambda() -> ValueFact {
    let mut fields = BTreeMap::new();
    for name in ["ineqlin", "eqlin", "lower", "upper"] {
        fields.insert(name.into(), vector());
    }
    fields.insert(
        "soc".into(),
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(double(ShapeFact::Scalar, StorageFact::Scalar)),
                elements: Vec::new(),
                elements_complete: false,
            }),
            ShapeFact::Unknown,
            StorageFact::Dense,
        ),
    );
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)))
}

pub(super) fn solver_outputs(conic: bool) -> Vec<ValueFact> {
    let mut outputs = vec![
        double(ShapeFact::Unknown, StorageFact::Dense),
        double(ShapeFact::Unknown, StorageFact::Unknown),
        double(ShapeFact::Scalar, StorageFact::Scalar),
        diagnostics(),
    ];
    if conic {
        outputs.push(lambda());
    }
    outputs
}

pub(super) fn finish(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    outputs: Vec<ValueFact>,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let maximum = entry
        .descriptor
        .signatures
        .iter()
        .map(|signature| signature.outputs.len())
        .max()
        .unwrap_or(0);
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > maximum)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPT-OUTPUTS",
            "too many outputs requested",
            maximum,
        ));
        return CallInference {
            outputs: Vec::new(),
            dynamic_outputs: false,
            discarded: request.outputs.discarded.iter().copied().collect(),
            effects: entry.contract.effect_set(),
            capabilities: entry.contract.capability_set(),
            diagnostics,
        };
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !diagnostics.is_empty() {
        result
            .outputs
            .iter_mut()
            .for_each(|fact| *fact = ValueFact::unknown(DynamicReason::UnsupportedRepresentation));
    }
    result.diagnostics.extend(diagnostics);
    result
}
