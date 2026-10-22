use crate::OperationDiagnostic;
use runmat_types::{OperationFacts, OperationIdentity, OperationOverloadFacts};

mod geometry;
mod output;

pub use geometry::geometry_operation_extension;
pub use output::output_operation_extension;
pub(crate) use output::OutputOperation;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationResolutionRequest {
    pub operation: OperationIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOperationFacts {
    pub operation: OperationFacts,
}

pub trait AnalysisOperationCatalog {
    fn fingerprint(&self) -> [u8; 32];

    fn resolve(
        &self,
        request: &OperationResolutionRequest,
    ) -> Result<ResolvedOperationFacts, OperationDiagnostic>;
}

pub struct BorrowedAnalysisOperationCatalog<'a> {
    fingerprint: [u8; 32],
    operations: &'a [OperationFacts],
}

impl<'a> BorrowedAnalysisOperationCatalog<'a> {
    pub fn new(fingerprint: [u8; 32], operations: &'a [OperationFacts]) -> Self {
        Self {
            fingerprint,
            operations,
        }
    }

    pub fn overloads(&self, operation: &OperationIdentity) -> Option<&[OperationOverloadFacts]> {
        self.operations
            .iter()
            .find(|candidate| &candidate.identity == operation)
            .map(|candidate| candidate.overloads.as_slice())
    }
}

impl AnalysisOperationCatalog for BorrowedAnalysisOperationCatalog<'_> {
    fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    fn resolve(
        &self,
        request: &OperationResolutionRequest,
    ) -> Result<ResolvedOperationFacts, OperationDiagnostic> {
        self.operations
            .iter()
            .find(|candidate| candidate.identity == request.operation)
            .cloned()
            .map(|operation| ResolvedOperationFacts { operation })
            .ok_or_else(|| OperationDiagnostic {
                code: "fea.operation.unknown".to_string(),
                message: format!("unknown analysis operation `{}`", request.operation),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        OperationDistribution, OperationEffectFacts, OperationFrontend, OperationLoweringId,
        OperationOverloadIdentity, OperationQuantityRule, OperationSignature,
        TopologyApplicability,
    };

    fn operation() -> OperationFacts {
        OperationFacts {
            identity: OperationIdentity::new("maximum").unwrap(),
            category: "reduction".to_string(),
            summary: "Find the maximum value.".to_string(),
            overloads: vec![runmat_types::OperationOverloadFacts {
                identity: OperationOverloadIdentity::new("maximum:fea-field").unwrap(),
                frontends: vec![OperationFrontend::Fea],
                signature: OperationSignature {
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                },
                quantity: OperationQuantityRule::CheckedByLowering,
                topology: TopologyApplicability::Region,
                effects: OperationEffectFacts::default(),
                distribution: OperationDistribution::DeterministicReduction,
                lowering: OperationLoweringId::new("fea.maximum").unwrap(),
            }],
        }
    }

    // witness: FEA-AUTHORING-OUTPUT-002/shared_catalog_lowering @
    //   shared_infrastructure/mesh_independent/source_check; coverage: partial;
    //   boundary: The read-only catalog resolves without runtime bindings.
    #[test]
    fn read_only_catalog_resolves_without_runtime_bindings() {
        let operations = [operation()];
        let catalog = BorrowedAnalysisOperationCatalog::new([7; 32], &operations);
        let request = OperationResolutionRequest {
            operation: OperationIdentity::new("maximum").unwrap(),
        };
        let resolved = catalog.resolve(&request).unwrap();
        assert_eq!(resolved.operation.identity.as_str(), "maximum");
        assert_eq!(catalog.fingerprint(), [7; 32]);
    }

    #[test]
    fn unknown_operations_are_bounded_diagnostics() {
        let catalog = BorrowedAnalysisOperationCatalog::new([0; 32], &[]);
        let error = catalog
            .resolve(&OperationResolutionRequest {
                operation: OperationIdentity::new("missing").unwrap(),
            })
            .unwrap_err();
        assert_eq!(error.code, "fea.operation.unknown");
    }
}
