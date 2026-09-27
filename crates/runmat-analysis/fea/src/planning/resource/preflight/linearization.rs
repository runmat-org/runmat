//! Numerical payload envelope for physical initial-stress integration.
//! This is a conservative admission allowance, not an allocator-peak proof.

use super::ResourcePlanningError;
use runmat_analysis_core::{
    AnalysisProcedure, CoupledPhysicsFamily, PhysicsFamily, ProcedureStep, StateKind,
};

pub(super) fn requires_envelope(step: &ProcedureStep) -> bool {
    // Solved thermomechanical static execution always returns its physical
    // linearization, even when no semantic geometric-stiffness output is authored.
    (step.family == PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
        && step.procedure == AnalysisProcedure::Static)
        || step
            .state_outputs
            .iter()
            .any(|output| output.kind == StateKind::GeometricStiffness)
}

// Curved Tet10 may retain all 16³ integration-point tensors. Using this bound for
// every solid avoids silently treating curved geometry as an affine four-point rule.
const MAXIMUM_RETAINED_SAMPLES: u64 = 4096;
// Twelve³ samples grow a Vec to 2048 slots; this previous rule remains live while
// the final 4096-sample rule is evaluated. The final rule is counted in retained data.
const PREVIOUS_RULE_CAPACITY: u64 = 2048;
// A tensor has nine f64 values plus nested-sequence CBOR tags. 96 bytes also exceeds
// its 72-byte numerical resident footprint; units below are eight-byte value slots.
const TENSOR_VALUE_SLOTS: u64 = 12;

pub(super) fn value_envelope(
    dofs: u64,
    elements: u64,
    stiffness_nonzeros: u64,
) -> Result<u64, ResourcePlanningError> {
    let retained = elements
        .checked_mul(MAXIMUM_RETAINED_SAMPLES)
        .and_then(|samples| samples.checked_mul(TENSOR_VALUE_SLOTS));
    // Integration is serial per solid. At most two 30x30 matrices, one 30-value
    // local displacement and the preceding rule coexist with the retained samples.
    let scratch = PREVIOUS_RULE_CAPACITY * TENSOR_VALUE_SLOTS + 2 * 30 * 30 + 30;
    dofs.checked_mul(2)
        .and_then(|values| values.checked_add(retained?))
        .and_then(|values| values.checked_add(stiffness_nonzeros.checked_mul(2)?))
        .and_then(|values| values.checked_add(dofs.checked_add(1)?))
        .and_then(|values| values.checked_add(scratch))
        .ok_or(ResourcePlanningError::Overflow(
            "structural linearization value count",
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupled_static_reserves_linearization_without_authored_output() {
        let mut step = ProcedureStep::coupled(
            "thermal_static",
            CoupledPhysicsFamily::Thermomechanical,
            AnalysisProcedure::Static,
        );
        step.state_outputs.clear();
        assert!(requires_envelope(&step));
        step.family = PhysicsFamily::Structural;
        assert!(!requires_envelope(&step));
        let mut output = ProcedureStep::thermal_steady("source")
            .state_outputs
            .remove(0);
        output.kind = StateKind::GeometricStiffness;
        step.state_outputs.push(output);
        assert!(requires_envelope(&step));
        step.state_outputs.clear();
        step.family = PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical);
        step.procedure = AnalysisProcedure::TransientResponse;
        assert!(!requires_envelope(&step));
    }

    #[test]
    fn stress_inventory_scales_with_full_curved_rule_not_legacy_four_points() {
        let one = value_envelope(30, 1, 900).unwrap();
        let two = value_envelope(30, 2, 900).unwrap();
        assert_eq!(two - one, 4096 * 12);
        assert!(one * 8 > (4096 + 2048) * 9 * 8 + 2 * 30 * 30 * 8);
        assert!(one > 30 * 2 + 36 + 900 * 2 + 31);
    }

    #[test]
    fn linearization_inventory_overflow_fails_closed() {
        assert!(value_envelope(30, u64::MAX, 900).is_err());
        assert!(value_envelope(u64::MAX, 1, 900).is_err());
        assert!(value_envelope(30, 1, u64::MAX).is_err());
    }
}
