use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy)]
pub(super) enum StructureOperation {
    Bandwidth,
    Symrcm,
    Isdiag,
    Istril,
    Istriu,
    Issymmetric,
    Ishermitian,
}

impl StructureOperation {
    const fn maximum_inputs(self) -> usize {
        match self {
            Self::Bandwidth => 2,
            Self::Issymmetric | Self::Ishermitian => 3,
            Self::Symrcm | Self::Isdiag | Self::Istril | Self::Istriu => 1,
        }
    }

    const fn requires_square(self) -> bool {
        matches!(self, Self::Symrcm)
    }

    const fn rejects_higher_rank(self) -> bool {
        !matches!(self, Self::Isdiag | Self::Istril | Self::Istriu)
    }
}

fn matrix_admission(
    input: &ValueFact,
    operation: StructureOperation,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !matches!(
        &input.kind,
        ValueKindFact::Unknown | ValueKindFact::Numeric(_) | ValueKindFact::Logical
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINALG-STRUCTURE-CLASS",
            "matrix structure analysis requires numeric or logical input",
            0,
        ));
    }
    if let Some(dims) = input.shape.known_dims() {
        if operation.rejects_higher_rank()
            && dims
                .iter()
                .skip(2)
                .any(|dim| dim.is_some_and(|extent| extent != 1))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINALG-STRUCTURE-RANK",
                "matrix structure analysis requires a two-dimensional matrix",
                0,
            ));
        }
        if operation.requires_square()
            && dims.len() >= 2
            && dims[0]
                .zip(dims[1])
                .is_some_and(|(rows, cols)| rows != cols)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINALG-STRUCTURE-SQUARE",
                "symmetric reverse Cuthill-McKee requires a square matrix",
                0,
            ));
        }
    }
}

fn scalar_double() -> ValueFact {
    let mut fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    fact.residency = ResidencyFact::Host;
    fact
}

fn scalar_logical() -> ValueFact {
    let mut fact = ValueFact::scalar(ValueKindFact::Logical);
    fact.residency = ResidencyFact::Host;
    fact
}

fn permutation(input: Option<&ValueFact>) -> ValueFact {
    let rows = input.and_then(|fact| {
        fact.shape
            .known_dims()
            .and_then(|dims| dims.first().copied().flatten())
    });
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), rows]),
        StorageFact::Dense,
    );
    fact.residency = ResidencyFact::Host;
    fact
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: StructureOperation,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > operation.maximum_inputs() {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINALG-STRUCTURE-ARITY",
            "unsupported matrix structure argument count",
            request.arguments.len().min(operation.maximum_inputs()),
        ));
    }
    let input = request.arguments.first();
    if let Some(input) = input {
        matrix_admission(input, operation, &mut diagnostics);
    }
    if matches!(operation, StructureOperation::Bandwidth)
        && request.arguments.len() == 2
        && request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-BANDWIDTH-OUTPUTS",
            "selector form returns at most one bandwidth",
            1,
        ));
    }
    if matches!(operation, StructureOperation::Bandwidth) {
        if request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count > 2)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-BANDWIDTH-OUTPUTS",
                "bandwidth returns at most two outputs",
                2,
            ));
        }
        let mut contract = CallContract::fixed(vec![scalar_double(), scalar_double()]);
        contract.effects = entry.contract.effect_set();
        contract.capabilities = entry.contract.capability_set();
        let mut result = infer_call(&contract, request);
        result.diagnostics.extend(diagnostics);
        return result;
    }
    let output = match operation {
        StructureOperation::Symrcm => permutation(input),
        StructureOperation::Isdiag
        | StructureOperation::Istril
        | StructureOperation::Istriu
        | StructureOperation::Issymmetric
        | StructureOperation::Ishermitian => scalar_logical(),
        StructureOperation::Bandwidth => unreachable!(),
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize, trailing: Option<usize>) -> ValueFact {
        let mut dims = vec![Some(rows), Some(columns)];
        if let Some(page) = trailing {
            dims.push(Some(page));
        }
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(dims),
            StorageFact::Dense,
        )
    }

    fn request(input: ValueFact, outputs: usize) -> CallRequest {
        CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        }
    }

    #[test]
    fn structure_facts_keep_host_scalar_results_and_predicate_rank_policy() {
        let bandwidth = infer(
            &request(matrix(3, 4, None), 2),
            &super::super::bandwidth::BANDWIDTH_CATALOG_ENTRY,
            StructureOperation::Bandwidth,
        );
        assert!(
            bandwidth.diagnostics.is_empty(),
            "{:?}",
            bandwidth.diagnostics
        );
        assert_eq!(bandwidth.outputs.len(), 2);
        assert!(bandwidth
            .outputs
            .iter()
            .all(|output| output.shape == ShapeFact::Scalar
                && output.residency == ResidencyFact::Host));

        let permutation = infer(
            &request(matrix(4, 4, None), 1),
            &super::super::symrcm::SYMRCM_CATALOG_ENTRY,
            StructureOperation::Symrcm,
        );
        assert_eq!(
            permutation.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(4)])
        );

        let paged = matrix(2, 2, Some(3));
        let predicate = infer(
            &request(paged.clone(), 1),
            &super::super::isdiag::ISDIAG_CATALOG_ENTRY,
            StructureOperation::Isdiag,
        );
        assert!(
            predicate.diagnostics.is_empty(),
            "{:?}",
            predicate.diagnostics
        );
        assert!(matches!(&predicate.outputs[0].kind, ValueKindFact::Logical));
        let bandwidth = infer(
            &request(paged, 1),
            &super::super::bandwidth::BANDWIDTH_CATALOG_ENTRY,
            StructureOperation::Bandwidth,
        );
        assert!(bandwidth
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LINALG-STRUCTURE-RANK"));
    }

    #[test]
    fn sparse_structure_inputs_keep_host_metadata_facts() {
        let mut sparse = matrix(4, 4, None);
        sparse.storage = StorageFact::Sparse;
        let bandwidth = infer(
            &request(sparse.clone(), 2),
            &super::super::bandwidth::BANDWIDTH_CATALOG_ENTRY,
            StructureOperation::Bandwidth,
        );
        assert!(
            bandwidth.diagnostics.is_empty(),
            "{:?}",
            bandwidth.diagnostics
        );
        assert!(bandwidth.outputs.iter().all(|fact| {
            fact.shape == ShapeFact::Scalar && fact.residency == ResidencyFact::Host
        }));
        let permutation = infer(
            &request(sparse, 1),
            &super::super::symrcm::SYMRCM_CATALOG_ENTRY,
            StructureOperation::Symrcm,
        );
        assert!(
            permutation.diagnostics.is_empty(),
            "{:?}",
            permutation.diagnostics
        );
        assert_eq!(
            permutation.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(4)])
        );
        assert_eq!(permutation.outputs[0].storage, StorageFact::Dense);
    }

    #[test]
    fn seven_structure_entries_have_distinct_catalog_contracts_and_original_example_counts() {
        for (entry, examples) in [
            (&super::super::bandwidth::BANDWIDTH_CATALOG_ENTRY, 7),
            (&super::super::ishermitian::ISHERMITIAN_CATALOG_ENTRY, 6),
            (&super::super::issymmetric::ISSYMMETRIC_CATALOG_ENTRY, 6),
            (&super::super::isdiag::ISDIAG_CATALOG_ENTRY, 4),
            (&super::super::istril::ISTRIL_CATALOG_ENTRY, 4),
            (&super::super::istriu::ISTRIU_CATALOG_ENTRY, 4),
            (&super::super::symrcm::SYMRCM_CATALOG_ENTRY, 4),
        ] {
            assert_eq!(
                entry.documentation.authority,
                crate::BuiltinDocumentationAuthority::Catalog
            );
            assert_eq!(
                entry.documentation.examples.len(),
                examples,
                "{}",
                entry.identity.name
            );
            assert!(
                !entry.descriptor.signatures.is_empty(),
                "{}",
                entry.identity.name
            );
        }
    }
}
