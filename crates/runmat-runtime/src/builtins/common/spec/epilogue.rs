//! Operation-owned fixed-phase epilogues, independent of scalar/reduction templates.
/// These declarations authorize a form, not arbitrary arguments: the execution
/// planner must validate represented classes, exact arity and phase ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionEpilogueOperation {
    Maximum,
    Minimum,
    RealPower,
    Diagonal,
}

impl FusionEpilogueOperation {
    pub const fn input_arity(self) -> usize {
        match self {
            Self::Diagonal => 1,
            Self::Maximum | Self::Minimum | Self::RealPower => 2,
        }
    }
}

/// Resolve epilogue semantics only for the selected named implementation.
pub fn fusion_epilogue_operation(
    name: &str,
    context: Option<&crate::context::RuntimeContext>,
) -> Option<FusionEpilogueOperation> {
    super::selected_fusion_binding(name, None, context)?.epilogue
}
