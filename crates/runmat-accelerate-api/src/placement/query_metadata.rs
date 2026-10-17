//! Requested host allocations for one owned provider-feasibility query.
//! This is a pure pre-construction extent, not an allocation grant or a
//! substitute for provider-owned observation/physical-work metadata.

use super::{ProviderFeasibilityQuery, ProviderRepresentation};

/// The operation identity and the exact ranks of every owned input/output
/// representation. Workloads with additional heap-bearing fields must quote
/// those separately at their typed owner before constructing the query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderQueryMetadataFootprint<'a> {
    pub operation: &'a str,
    pub input_ranks: &'a [usize],
    pub output_ranks: &'a [usize],
    /// Caller-local shapes simultaneously live with the finished query, for
    /// example the output descriptor retained beside an upload query.
    pub preparation_shape_ranks: &'a [usize],
    /// Caller-local representation Vec slots simultaneously live with it.
    pub preparation_representation_slots: usize,
}

impl ProviderQueryMetadataFootprint<'_> {
    /// Requested String, input/output Vec slot, and shape Vec storage. Empty
    /// Vecs request no storage. This counts no provider buffer or payload.
    pub fn requested_bytes(&self) -> Option<u64> {
        let slots = self
            .input_ranks
            .len()
            .checked_add(self.output_ranks.len())?
            .checked_add(self.preparation_representation_slots)?
            .checked_mul(std::mem::size_of::<ProviderRepresentation>())?;
        let shape_elements = self
            .input_ranks
            .iter()
            .chain(self.output_ranks)
            .chain(self.preparation_shape_ranks)
            .try_fold(0usize, |total, &rank| total.checked_add(rank))?;
        let shapes = shape_elements.checked_mul(std::mem::size_of::<u64>())?;
        u64::try_from(
            self.operation
                .len()
                .checked_add(slots)?
                .checked_add(shapes)?,
        )
        .ok()
    }

    /// Validate the structural allocation plan against the finished query
    /// before it is used for feasibility. This does not attest any separately
    /// quoted caller-local clone or optional workload payload.
    pub fn matches(&self, query: &ProviderFeasibilityQuery) -> bool {
        self.operation == query.operation.0.as_str()
            && self.input_ranks.len() == query.inputs.len()
            && self.output_ranks.len() == query.outputs.len()
            && self
                .input_ranks
                .iter()
                .zip(&query.inputs)
                .all(|(&rank, input)| rank == input.shape.len())
            && self
                .output_ranks
                .iter()
                .zip(&query.outputs)
                .all(|(&rank, output)| rank == output.shape.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ProviderElementType, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
        ProviderResidency, ProviderStorage, ProviderWorkload,
    };

    #[test]
    fn owned_query_extent_is_checked_and_matches_actual_ranks() {
        let plan = ProviderQueryMetadataFootprint {
            operation: "random.gpu_child.integer.batch",
            input_ranks: &[2, 2],
            output_ranks: &[2, 2],
            preparation_shape_ranks: &[],
            preparation_representation_slots: 0,
        };
        let expected = plan.operation.len()
            + 4 * std::mem::size_of::<ProviderRepresentation>()
            + 8 * std::mem::size_of::<u64>();
        assert_eq!(plan.requested_bytes(), u64::try_from(expected).ok());
        assert_eq!(
            ProviderQueryMetadataFootprint {
                preparation_shape_ranks: &[2],
                ..plan
            }
            .requested_bytes(),
            u64::try_from(expected + 2 * std::mem::size_of::<u64>()).ok()
        );
        let output = ProviderRepresentation {
            element_type: ProviderElementType::U32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 2],
            residency: ProviderResidency::Device,
        };
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(plan.operation),
            family: ProviderOperationFamily::Library,
            inputs: vec![output.clone(), output.clone()],
            outputs: vec![output.clone(), output],
            workload: ProviderWorkload::default(),
        };
        assert!(plan.matches(&query));
        assert!(!ProviderQueryMetadataFootprint {
            output_ranks: &[2],
            ..plan
        }
        .matches(&query));
        assert_eq!(
            ProviderQueryMetadataFootprint {
                operation: "x",
                input_ranks: &[usize::MAX],
                output_ranks: &[],
                preparation_shape_ranks: &[],
                preparation_representation_slots: 0,
            }
            .requested_bytes(),
            None
        );
    }
}
