//! Cost/feasibility evidence for a selected checked recipe. This serialized
//! projection cannot authenticate a buffer or authorize execution by itself.
use crate::{
    reduction::{ReductionSelection, ReductionStrategy, SelectedReductionWork},
    ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderTypedReductionWorkload {
    pub recipe_identity: Digest,
    pub selection_identity: Digest,
    pub strategy: ReductionStrategy,
    pub work: SelectedReductionWork,
    pub inputs: Vec<ProviderRepresentation>,
    pub output: ProviderRepresentation,
    // The checked execution inventory is serialized into local placement
    // identity, but a decoded profile/query cannot recreate execution facts.
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub input_preparation: Option<crate::reduction::ProviderReductionInputInventory>,
}

impl ProviderTypedReductionWorkload {
    pub fn from_selection(selection: &ReductionSelection<'_>) -> Self {
        let recipe = selection.recipe();
        Self {
            recipe_identity: recipe.identity(),
            selection_identity: selection.identity(),
            strategy: selection.strategy(),
            work: selection.work(),
            inputs: recipe
                .descriptor()
                .inputs
                .iter()
                .map(|binding| ProviderRepresentation {
                    element_type: ProviderElementType::from(binding.precision),
                    storage: ProviderStorage::DenseReal,
                    layout: match binding.layout {
                        crate::GpuTensorLayout::Contiguous => ProviderLayout::ColumnMajorContiguous,
                        crate::GpuTensorLayout::Transpose(_) => ProviderLayout::Strided,
                    },
                    shape: binding
                        .shape
                        .iter()
                        .map(|&extent| u64::try_from(extent).unwrap_or(u64::MAX))
                        .collect(),
                    residency: ProviderResidency::Device,
                })
                .collect(),
            output: ProviderRepresentation {
                element_type: ProviderElementType::from(recipe.descriptor().output_precision),
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: recipe
                    .descriptor()
                    .output_shape
                    .iter()
                    .map(|&extent| u64::try_from(extent).unwrap_or(u64::MAX))
                    .collect(),
                residency: ProviderResidency::Device,
            },
            input_preparation: None,
        }
    }

    pub fn with_input_preparation(
        mut self,
        inventory: crate::reduction::ProviderReductionInputInventory,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            inventory.recipe_identity() == self.recipe_identity,
            "typed preparation inventory belongs to a different recipe"
        );
        anyhow::ensure!(
            inventory
                .entries()
                .iter()
                .map(|entry| &entry.prepared)
                .eq(self.inputs.iter()),
            "typed preparation inventory differs from selected bindings"
        );
        self.input_preparation = Some(inventory);
        Ok(self)
    }

    /// Compare an execution workload with the complete canonical projection
    /// of its checked selection. Input preparation may augment that projection
    /// only after proving its recipe and prepared leaves are the selected ones.
    pub fn matches_selection(&self, selection: &ReductionSelection<'_>) -> bool {
        let expected = Self::from_selection(selection);
        let expected = match self.input_preparation.clone() {
            Some(inventory) => expected.with_input_preparation(inventory).ok(),
            None => Some(expected),
        };
        expected.as_ref() == Some(self)
    }
}
