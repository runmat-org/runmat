use runmat_types::{CapabilitySet, EffectKind, EffectSet};

use super::MirConstructKind;

pub(super) fn declared_requirements(
    region: &crate::MirExpressionRegion,
) -> (EffectSet, CapabilitySet) {
    let mut effects = EffectSet::default();
    let mut capabilities = CapabilitySet::default();
    for step in region.steps() {
        match step {
            crate::MirExpressionStep::Let { value, .. } => {
                let (nested_effects, nested_capabilities) =
                    super::rvalue_declared_requirements(value);
                effects.0.extend(nested_effects.0);
                capabilities.0.extend(nested_capabilities.0);
            }
            crate::MirExpressionStep::CaptureSequence { source, .. } => {
                effects.0.insert(EffectKind::Unknown);
                source.visit_direct_expression_regions_dyn(&mut |nested| {
                    let (nested_effects, nested_capabilities) = declared_requirements(nested);
                    effects.0.extend(nested_effects.0);
                    capabilities.0.extend(nested_capabilities.0);
                });
            }
        }
    }
    (effects, capabilities)
}

pub(super) fn inventory(region: &crate::MirExpressionRegion) -> Vec<MirConstructKind> {
    let mut constructs = Vec::new();
    for step in region.steps() {
        match step {
            crate::MirExpressionStep::Let { value, .. } => {
                constructs.extend(super::rvalue_construct_inventory(value));
            }
            crate::MirExpressionStep::CaptureSequence { source, .. } => {
                constructs.push(MirConstructKind::CaptureSequence);
                source.visit_direct_expression_regions_dyn(&mut |nested| {
                    constructs.extend(inventory(nested));
                });
            }
        }
    }
    constructs
}
