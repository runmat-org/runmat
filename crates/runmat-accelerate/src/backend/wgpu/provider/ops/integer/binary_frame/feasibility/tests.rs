use super::*;
use runmat_accelerate_api::{ProviderOperationIdentity, ProviderRepresentation, ProviderWorkload};

fn representation(class: ProviderElementType, shape: &[u64]) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: class,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.to_vec(),
        residency: ProviderResidency::Device,
    }
}

fn query(class: ProviderElementType, lhs: &[u64], rhs: &[u64]) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(
            runmat_builtins::LDIVIDE_CATALOG_ENTRY.identity.name,
        ),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![representation(class, lhs), representation(class, rhs)],
        outputs: Vec::new(),
        workload: ProviderWorkload::default(),
    }
}

fn feasibility(
    query: &ProviderFeasibilityQuery,
    limits: &wgpu::Limits,
    precision: ProviderPrecision,
) -> ProviderFeasibility {
    let launch = FrozenLinearLaunch::new(64, limits.max_compute_workgroups_per_dimension)
        .expect("test launch");
    feasibility_for_geometry(limits, launch, precision, query).expect("integer division query")
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integer_division_feasibility_covers_all_word_classes_and_both_scalar_orders() {
    let limits = wgpu::Limits::default();
    for (class, expected_bytes) in [
        (ProviderElementType::I8, 12),
        (ProviderElementType::I16, 12),
        (ProviderElementType::I32, 12),
        (ProviderElementType::I64, 24),
        (ProviderElementType::U8, 12),
        (ProviderElementType::U16, 12),
        (ProviderElementType::U32, 12),
        (ProviderElementType::U64, 24),
    ] {
        for (lhs, rhs) in [(&[1, 3][..], &[1, 1][..]), (&[1, 1][..], &[1, 3][..])] {
            let mut request = query(class, lhs, rhs);
            request.outputs.push(representation(class, &[1, 3]));
            for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
                let ProviderFeasibility::Supported { estimate } =
                    feasibility(&request, &limits, precision)
                else {
                    panic!("native integer {class:?} must be feasible")
                };
                assert_eq!(estimate.output_bytes, Some(expected_bytes));
                assert_eq!(estimate.transient_bytes, None);
                assert_eq!(estimate.dispatches, None);
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integer_division_query_rejects_mixed_or_nondevice_and_invalid_output() {
    let limits = wgpu::Limits::default();
    let mut request = query(ProviderElementType::I32, &[1, 3], &[1, 1]);
    request.inputs[1].element_type = ProviderElementType::F32;
    assert!(!feasibility(&request, &limits, ProviderPrecision::F32).is_supported());
    request.inputs[1].element_type = ProviderElementType::I64;
    assert!(!feasibility(&request, &limits, ProviderPrecision::F32).is_supported());
    request.inputs[1].element_type = ProviderElementType::I32;
    request.inputs[0].residency = ProviderResidency::Host;
    assert!(!feasibility(&request, &limits, ProviderPrecision::F32).is_supported());
    request.inputs[0].residency = ProviderResidency::Device;
    request.inputs[0].layout = ProviderLayout::Strided;
    assert!(!feasibility(&request, &limits, ProviderPrecision::F32).is_supported());
    request.inputs[0].layout = ProviderLayout::ColumnMajorContiguous;
    request
        .outputs
        .push(representation(ProviderElementType::I32, &[3, 1]));
    assert!(!feasibility(&request, &limits, ProviderPrecision::F32).is_supported());
    let incompatible = query(ProviderElementType::I32, &[2, 3], &[4, 1]);
    assert!(!feasibility(&incompatible, &limits, ProviderPrecision::F32).is_supported());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integer_division_query_checks_kernel_limit_and_retains_empty_native_route() {
    let limits = wgpu::Limits::default();
    let nonempty = query(ProviderElementType::I32, &[1, 3], &[1, 1]);
    assert!(feasibility(&nonempty, &limits, ProviderPrecision::F32).is_supported());
    let restricted = wgpu::Limits {
        max_storage_buffer_binding_size: 8,
        ..limits
    };
    assert!(!feasibility(&nonempty, &restricted, ProviderPrecision::F32).is_supported());
    let empty = query(ProviderElementType::I32, &[0, 3], &[1, 1]);
    for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let ProviderFeasibility::Supported { estimate } =
            feasibility(&empty, &restricted, precision)
        else {
            panic!("empty native integer division emits no compute binding")
        };
        assert_eq!(estimate.output_bytes, Some(4));
    }
}
