use runmat_analysis_core::StructuralModel;

use super::PropertyRealizationError;

pub(super) fn merge_structural_topology(
    target: &mut StructuralModel,
    generated: StructuralModel,
) -> Result<(), PropertyRealizationError> {
    if !target.elements.is_empty()
        || !target.nodes.is_empty()
        || !target.beam_sections.is_empty()
        || !target.shell_sections.is_empty()
    {
        return Err(PropertyRealizationError::InvalidMesh(
            "authored explicit topology cannot be combined with geometry-derived property topology"
                .into(),
        ));
    }
    *target = generated;
    Ok(())
}
