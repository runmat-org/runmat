use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{HostTensorView, ProviderWorkload};

#[test]
fn declared_rounding_hooks_execute_the_admitted_representations() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload(&HostTensorView {
            data: &[0.5, -0.5, 1.75, -1.75],
            shape: &[2, 1, 2],
        })
        .unwrap();
    for (index, entry) in ENTRIES.iter().enumerate() {
        let mut query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(entry.identity.name),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![ProviderRepresentation {
                element_type: ProviderElementType::F64,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: vec![2, 1, 2],
                residency: ProviderResidency::Host,
            }],
            outputs: vec![],
            workload: ProviderWorkload::default(),
        };
        assert!(
            provider.query_feasibility(&query).is_supported(),
            "{}",
            entry.identity.name
        );
        let output = block_on(match index {
            0 => provider.unary_ceil(&input),
            1 => provider.unary_fix(&input),
            2 => provider.unary_floor(&input),
            _ => provider.unary_round(&input),
        })
        .unwrap();
        let expected = match index {
            0 => vec![1.0, 0.0, 2.0, -1.0],
            1 => vec![0.0, 0.0, 1.0, -1.0],
            2 => vec![0.0, -1.0, 1.0, -2.0],
            _ => vec![1.0, -1.0, 2.0, -2.0],
        };
        assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
        assert_eq!(output.shape(), input.shape());
        assert!(!output.shares_allocation_with(&input));
        output.release().unwrap();
        query.inputs[0].element_type = ProviderElementType::F32;
        assert!(!provider.query_feasibility(&query).is_supported());
        query.inputs[0].element_type = ProviderElementType::F64;
        query.inputs[0].storage = ProviderStorage::Sparse;
        assert!(!provider.query_feasibility(&query).is_supported());
    }
    input.release().unwrap();
}

#[cfg(feature = "wgpu")]
#[test]
fn digit_work_counts_real_chunking_uniform_transfers_and_scaling_bound() {
    use runmat_accelerate_api::{NumericalDomainEvidence, RoundDigitsMode};
    let launch = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(
        crate::backend::wgpu::config::WORKGROUP_SIZE,
        crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS,
    )
    .unwrap();
    let capacity = usize::try_from(crate::backend::wgpu::config::WORKGROUP_SIZE).unwrap()
        * usize::try_from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS).unwrap();
    let mut query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(ROUND_CATALOG_ENTRY.identity.name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, u64::try_from(capacity).unwrap() + 1],
            residency: ProviderResidency::Device,
        }],
        outputs: vec![],
        workload: ProviderWorkload {
            numerical_domain: NumericalDomainEvidence::RoundDigits {
                digits: 2,
                mode: RoundDigitsMode::Significant,
            },
            ..Default::default()
        },
    };
    let work = wgpu_work(ProviderPrecision::F64, &query, launch).unwrap();
    assert_eq!(work.dispatches, 2);
    assert_eq!(work.queue_submissions, 2);
    assert_eq!(work.allocations, 2);
    assert_eq!(work.scratch_bytes, 16);
    assert_eq!(work.extra_upload_bytes, 32);
    assert!(work.execution_units > (capacity as u64 + 1) * 96);
    assert_eq!(work.execution_units, (capacity as u64 + 1) * (96 + 15 + 7));
    query.inputs[0].shape = vec![0, u64::MAX];
    let work = wgpu_work(ProviderPrecision::F64, &query, launch).unwrap();
    assert_eq!(work.dispatches, 0);
    assert_eq!(work.queue_submissions, 0);
    assert_eq!(work.allocations, 1);
    assert_eq!(work.scratch_bytes, 0);
    assert_eq!(work.execution_units, 0);
    query.inputs[0].shape = vec![u64::MAX, u64::MAX];
    assert!(wgpu_work(ProviderPrecision::F64, &query, launch).is_none());
}

#[cfg(feature = "wgpu")]
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn digit_work_dispatches_follow_selected_launch_not_static_default() {
    use runmat_accelerate_api::{NumericalDomainEvidence, RoundDigitsMode};
    let selected = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(256, 3).unwrap();
    let baseline = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(512, 3).unwrap();
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(ROUND_CATALOG_ENTRY.identity.name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::F32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 769],
            residency: ProviderResidency::Device,
        }],
        outputs: vec![],
        workload: ProviderWorkload {
            numerical_domain: NumericalDomainEvidence::RoundDigits {
                digits: 2,
                mode: RoundDigitsMode::Decimals,
            },
            ..Default::default()
        },
    };
    assert_eq!(
        wgpu_work(ProviderPrecision::F32, &query, selected)
            .unwrap()
            .dispatches,
        2
    );
    assert_eq!(
        wgpu_work(ProviderPrecision::F32, &query, baseline)
            .unwrap()
            .dispatches,
        1
    );
}

#[test]
fn digit_capability_preserves_parsed_strategy_and_validates_evidence() {
    use runmat_accelerate_api::{NumericalDomainEvidence, RoundDigitsMode};
    let provider = InProcessProvider::new();
    let input = provider
        .upload(&HostTensorView {
            data: &[23.75, 1253.0],
            shape: &[1, 1, 2],
        })
        .unwrap();
    let mut query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(ROUND_CATALOG_ENTRY.identity.name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 1, 2],
            residency: ProviderResidency::Device,
        }],
        outputs: vec![],
        workload: ProviderWorkload::default(),
    };
    for (digits, mode, expected) in [
        (20, RoundDigitsMode::Decimals, vec![23.75, 1253.0]),
        (2, RoundDigitsMode::Significant, vec![24.0, 1300.0]),
        (i32::MIN, RoundDigitsMode::Decimals, vec![0.0, 0.0]),
    ] {
        query.workload.numerical_domain = NumericalDomainEvidence::RoundDigits { digits, mode };
        assert!(provider.query_feasibility(&query).is_supported());
        let output =
            block_on(provider.round_digits(&input, digits, mode == RoundDigitsMode::Significant))
                .unwrap();
        assert_eq!(output.shape(), input.shape());
        assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
        output.release().unwrap();
    }
    query.workload.numerical_domain = NumericalDomainEvidence::RoundDigits {
        digits: 0,
        mode: RoundDigitsMode::Significant,
    };
    assert!(!provider.query_feasibility(&query).is_supported());
    query.workload.numerical_domain = NumericalDomainEvidence::RoundDigits {
        digits: 1,
        mode: RoundDigitsMode::Decimals,
    };
    query.operation = ProviderOperationIdentity::new(CEIL_CATALOG_ENTRY.identity.name);
    assert!(!provider.query_feasibility(&query).is_supported());
    input.release().unwrap();
}
