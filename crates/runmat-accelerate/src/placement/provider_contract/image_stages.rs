//! Fail-closed physical admission for the held staged image program. A future
//! WGPU owner must replace UnknownResourceEstimate with kernel-route evidence
//! from the same planners that actually execute the hooks and bind cache state.
use super::*;

pub(super) fn validate(query: &ProviderFeasibilityQuery) -> Option<ProviderFeasibility> {
    let program = query.workload.image_stages.as_ref()?;
    if query.operation.0 != fusion_operation_token(&FusionKind::ImageNormalize)
        || query.family != ProviderOperationFamily::Fusion
        || !program.validate()
    {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::InvalidShape,
            "fusion.image_stages.invalid_program",
        ));
    }
    // A logical transcript is neither the WGPU reduction route nor its
    // strong moments-cache allocation/epoch. It must not inherit the generic
    // one-shot image feasibility or one-dispatch cost prior.
    Some(ProviderFeasibility::rejected(
        ProviderRejectionCode::UnknownResourceEstimate,
        "fusion.image_stages.physical_route_unresolved",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        ProviderImageMeanCall, ProviderImageMeanClass, ProviderImageMeanHook,
        ProviderImageOutputCall, ProviderImageOutputHook, ProviderImageOutputOperand,
        ProviderImageOutputProgram, ProviderImageStageWorkload, ProviderWorkload,
    };

    fn query() -> ProviderFeasibilityQuery {
        let mean = ProviderImageMeanCall {
            hook: ProviderImageMeanHook::ReduceNd([1, 2]),
            output_class: ProviderImageMeanClass::Native,
            shape: [1, 1, 1],
        };
        ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(fusion_operation_token(
                &FusionKind::ImageNormalize,
            )),
            family: ProviderOperationFamily::Fusion,
            inputs: Vec::new(),
            outputs: Vec::new(),
            workload: ProviderWorkload {
                image_stages: Some(ProviderImageStageWorkload {
                    input_shape: [1, 2, 2],
                    mu: vec![mean],
                    variance: vec![mean],
                    square_exponent_f32_bits: 2.0f32.to_bits(),
                    final_exponent_f32_bits: 1.0f32.to_bits(),
                    selected_powers: None,
                    epsilon: runmat_accelerate_api::scalar_program::ScalarLiteral::F32Bits(
                        0.25f32.to_bits(),
                    ),
                    output_program: None,
                    moment_program: None,
                }),
                ..Default::default()
            },
        }
    }

    #[test]
    fn staged_candidate_never_inherits_one_shot_physical_cost() {
        let valid = query();
        assert!(
            matches!(validate(&valid), Some(ProviderFeasibility::Rejected { rejection })
            if matches!(rejection.code, ProviderRejectionCode::UnknownResourceEstimate))
        );
        let mut tampered = valid.clone();
        tampered.workload.image_stages.as_mut().unwrap().variance[0].shape = [1, 2, 1];
        assert!(
            matches!(validate(&tampered), Some(ProviderFeasibility::Rejected { rejection })
            if matches!(rejection.code, ProviderRejectionCode::InvalidShape))
        );
        let mut wrong_operation = valid;
        wrong_operation.operation = ProviderOperationIdentity::new("fusion.elementwise");
        assert!(
            matches!(validate(&wrong_operation), Some(ProviderFeasibility::Rejected { rejection })
            if matches!(rejection.code, ProviderRejectionCode::InvalidShape))
        );

        let mut ordered = query();
        ordered
            .workload
            .image_stages
            .as_mut()
            .unwrap()
            .output_program = Some(ProviderImageOutputProgram {
            calls: vec![ProviderImageOutputCall {
                source_output: 41,
                hook: ProviderImageOutputHook::ScalarMax,
                operands: vec![
                    ProviderImageOutputOperand::Input,
                    ProviderImageOutputOperand::ScalarF32Bits(0.0f32.to_bits()),
                ],
                output_shape: [1, 2, 2],
                output_precision: runmat_accelerate_api::ProviderPrecision::F32,
                checkpoint: true,
            }],
        });
        assert!(matches!(
            validate(&ordered),
            Some(ProviderFeasibility::Rejected { rejection })
                if matches!(rejection.code, ProviderRejectionCode::UnknownResourceEstimate)
        ));
    }
}
