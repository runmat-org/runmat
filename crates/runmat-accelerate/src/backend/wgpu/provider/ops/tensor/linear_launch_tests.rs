use runmat_accelerate_api::{
    AccelProvider, HostTensorView, ProviderNdgridAxis, ProviderNdgridRequest,
};

async fn actual_linear_launch_tail_witness() {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU provider required");
    #[cfg(target_arch = "wasm32")]
    let provider = crate::backend::wgpu::provider::register_wgpu_provider_async(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .await
    .expect("actual browser WebGPU provider required");
    #[cfg(not(target_arch = "wasm32"))]
    assert_eq!(provider.linear_launch.workgroup(), 64);

    let bootstrap = crate::backend::wgpu::pipelines::ImageNormalizeBootstrap {
        batch_tile: 1,
        values_per_thread: 1,
        lane_count: 32,
        spatial_tile: 1,
    };
    let first_launch = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(
        64,
        provider.adapter_limits.max_compute_workgroups_per_dimension,
    )
    .expect("first enabled launch");
    let second_launch = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(
        128,
        provider.adapter_limits.max_compute_workgroups_per_dimension,
    )
    .expect("second enabled launch");
    let first = crate::backend::wgpu::pipelines::WgpuPipelines::new(
        &provider.access,
        crate::backend::wgpu::types::NumericPrecision::F32,
        bootstrap,
        first_launch,
        provider.global_reduce_launch,
        provider.workgroup_config.matmul_tile,
    );
    let second = crate::backend::wgpu::pipelines::WgpuPipelines::new(
        &provider.access,
        crate::backend::wgpu::types::NumericPrecision::F32,
        bootstrap,
        second_launch,
        provider.global_reduce_launch,
        provider.workgroup_config.matmul_tile,
    );
    assert_ne!(first.repmat.linear_launch(), second.repmat.linear_launch());
    for (first_bundle, second_bundle) in [
        (&first.eye, &second.eye),
        (&first.fill, &second.fill),
        (&first.fspecial, &second.fspecial),
        (&first.linspace, &second.linspace),
        (&first.diag_from_vector, &second.diag_from_vector),
        (&first.diag_extract, &second.diag_extract),
    ] {
        assert_eq!(first_bundle.linear_dispatch_groups(65).unwrap(), 2);
        assert_eq!(second_bundle.linear_dispatch_groups(65).unwrap(), 1);
        assert_eq!(
            first_bundle.linear_chunk_capacity().unwrap(),
            first_launch.chunk_capacity().unwrap()
        );
        assert_eq!(
            second_bundle.linear_chunk_capacity().unwrap(),
            second_launch.chunk_capacity().unwrap()
        );
        assert!(!first_bundle.frozen_shader_source().contains("@WG@"));
        assert!(!second_bundle.frozen_shader_source().contains("@WG@"));
        assert!(first_bundle
            .frozen_shader_source()
            .contains("@workgroup_size(64)"));
        assert!(second_bundle
            .frozen_shader_source()
            .contains("@workgroup_size(128)"));
    }
    provider
        .access
        .with_shared(|emission| -> anyhow::Result<()> {
            let first_cold = first.repmat.get(emission)?;
            let second_cold = second.repmat.get(emission)?;
            assert!(std::ptr::eq(first_cold, first.repmat.get(emission)?));
            assert!(std::ptr::eq(second_cold, second.repmat.get(emission)?));
            for bundle in [
                &first.eye,
                &first.fill,
                &first.fspecial,
                &first.linspace,
                &first.diag_from_vector,
                &first.diag_extract,
            ] {
                bundle.get(emission)?;
            }
            for bundle in [
                &second.eye,
                &second.fill,
                &second.fspecial,
                &second.linspace,
                &second.diag_from_vector,
                &second.diag_extract,
            ] {
                bundle.get(emission)?;
            }
            assert_eq!(first.repmat.linear_dispatch_groups(65)?, 2);
            assert_eq!(second.repmat.linear_dispatch_groups(65)?, 1);
            Ok(())
        })
        .expect("WGPU access")
        .expect("cold and warm pipeline compilation");

    let selected = runmat_accelerate_api::provider().expect("selected actual WGPU provider");
    assert!(runmat_accelerate_api::same_provider_instance(
        selected, provider
    ));
    let expected_class = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => runmat_types::standard::SINGLE.owned(),
        runmat_accelerate_api::ProviderPrecision::F64 => runmat_types::standard::DOUBLE.owned(),
    };
    let assert_floating_output = |handle: &runmat_accelerate_api::GpuTensorHandle,
                                  shape: &[usize]| {
        assert_eq!(handle.shape(), shape);
        assert_eq!(
            runmat_accelerate_api::handle_precision(handle),
            Some(provider.precision())
        );
        assert_eq!(
            runmat_accelerate_api::handle_class_identity(handle),
            Some(expected_class.clone())
        );
    };

    let eye = selected.eye(&[65, 65]).expect("eye tail dispatch");
    assert_floating_output(&eye, &[65, 65]);
    let eye_host = selected.download(&eye).await.expect("eye readback");
    assert_eq!(eye_host.shape, vec![65, 65]);
    for col in 0..65 {
        for row in 0..65 {
            assert_eq!(
                eye_host.data[row + col * 65],
                if row == col { 1.0 } else { 0.0 }
            );
        }
    }

    let average = selected
        .fspecial(&runmat_accelerate_api::FspecialRequest {
            filter: runmat_accelerate_api::FspecialFilter::Average { rows: 5, cols: 13 },
        })
        .expect("fspecial average tail dispatch");
    assert_floating_output(&average, &[5, 13]);
    let average_host = selected
        .download(&average)
        .await
        .expect("fspecial average readback");
    assert_eq!(average_host.shape, vec![5, 13]);
    let average_value = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => f64::from(1.0_f32 / 65.0_f32),
        runmat_accelerate_api::ProviderPrecision::F64 => 1.0 / 65.0,
    };
    assert_eq!(average_host.data, vec![average_value; 65]);

    let linspace = selected
        .linspace(-1.0, 1.0, 129)
        .expect("linspace tail dispatch");
    assert_floating_output(&linspace, &[1, 129]);
    let linspace_host = selected
        .download(&linspace)
        .await
        .expect("linspace readback");
    assert_eq!(linspace_host.shape, vec![1, 129]);
    let expected_linspace = (0..129)
        .map(|index| -1.0 + (index as f64) / 64.0)
        .collect::<Vec<_>>();
    assert_eq!(linspace_host.data, expected_linspace);

    let filled = selected
        .fill(&[1, 65], 3.25)
        .expect("fill tail dispatch")
        .expect("fill is supported by the selected hardware provider");
    assert_floating_output(&filled, &[1, 65]);
    let filled_host = selected.download(&filled).await.expect("fill readback");
    assert_eq!(filled_host.shape, vec![1, 65]);
    assert_eq!(filled_host.data, vec![3.25; 65]);

    let diagonal_data = (0..65).map(f64::from).collect::<Vec<_>>();
    let diagonal_input = selected
        .upload(&HostTensorView {
            data: &diagonal_data,
            shape: &[65, 1],
        })
        .expect("diagonal input upload");
    let diagonal = selected
        .diag_from_vector(&diagonal_input, 0)
        .expect("diag construction tail dispatch");
    assert_floating_output(&diagonal, &[65, 65]);
    let diagonal_host = selected
        .download(&diagonal)
        .await
        .expect("diagonal matrix readback");
    for col in 0..65 {
        for (row, value) in diagonal_data.iter().copied().enumerate() {
            assert_eq!(
                diagonal_host.data[row + col * 65],
                if row == col { value } else { 0.0 }
            );
        }
    }
    let extracted = selected
        .diag_extract(&diagonal, 0)
        .expect("diag extraction tail dispatch");
    assert_floating_output(&extracted, &[65, 1]);
    let extracted_host = selected
        .download(&extracted)
        .await
        .expect("diagonal extraction readback");
    assert_eq!(extracted_host.data, diagonal_data);
    let bounded_diagonal = selected
        .diag_from_vector_sized(&diagonal_input, 0, 3, 3)
        .expect("bounded diagonal construction");
    assert_floating_output(&bounded_diagonal, &[3, 3]);
    let bounded_diagonal_host = selected
        .download(&bounded_diagonal)
        .await
        .expect("bounded diagonal readback");
    assert_eq!(
        bounded_diagonal_host.data,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]
    );

    let first_axis_data = (0..65).map(f64::from).collect::<Vec<_>>();
    let second_axis_data = vec![10.0, 20.0];
    let first_axis = selected
        .upload(&HostTensorView {
            data: &first_axis_data,
            shape: &[65, 1],
        })
        .expect("first ndgrid axis upload");
    let second_axis = selected
        .upload(&HostTensorView {
            data: &second_axis_data,
            shape: &[2, 1],
        })
        .expect("second ndgrid axis upload");
    let grids = selected
        .ndgrid(&ProviderNdgridRequest {
            axes: &[
                ProviderNdgridAxis {
                    handle: &first_axis,
                },
                ProviderNdgridAxis {
                    handle: &second_axis,
                },
            ],
            output_shape: &[65, 2],
            output_count: 2,
        })
        .expect("ndgrid tail dispatch")
        .outputs;
    assert_eq!(grids.len(), 2);
    for grid in &grids {
        assert_floating_output(grid, &[65, 2]);
    }
    let first_grid = selected
        .download(&grids[0])
        .await
        .expect("first ndgrid output readback");
    assert_eq!(
        first_grid.data,
        [first_axis_data.as_slice(), first_axis_data.as_slice()].concat()
    );
    let second_grid = selected
        .download(&grids[1])
        .await
        .expect("second ndgrid output readback");
    assert_eq!(second_grid.data, [vec![10.0; 65], vec![20.0; 65]].concat());

    let input_data = (0..1_026).map(f64::from).collect::<Vec<_>>();
    let input = provider
        .upload(&HostTensorView {
            data: &input_data,
            shape: &[2, 513],
        })
        .expect("upload");
    let replicated = provider.repmat_exec(&input, &[1, 2]).expect("repmat");
    assert_eq!(replicated.shape(), vec![2, 1_026]);
    let repeated = provider
        .download_exec(&replicated)
        .await
        .expect("repmat readback");
    assert_eq!(
        repeated.data,
        [input_data.as_slice(), input_data.as_slice()].concat()
    );
    let rounded = provider
        .round_digits_exec(&replicated, 0, false)
        .await
        .expect("round digits");
    let rounded_host = provider
        .download_exec(&rounded)
        .await
        .expect("round readback");
    assert_eq!(rounded_host.data, repeated.data);
    let sqrt = provider
        .unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &rounded)
        .expect("fused sqrt");
    let sqrt_host = provider.download_exec(&sqrt).await.expect("sqrt readback");
    let tolerance = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-4,
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-11,
    };
    for (actual, input) in sqrt_host.data.iter().zip(&rounded_host.data) {
        assert!((*actual - input.sqrt()).abs() <= tolerance);
    }
    let diff = provider.diff_exec(&rounded, 0, 1).expect("diff");
    let downloaded = provider.download_exec(&diff).await.expect("readback");
    assert_eq!(downloaded.shape, vec![1, 1_026]);
    assert_eq!(downloaded.data, vec![1.0; 1_026]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "run explicitly on actual WGPU hardware with RUNMAT_WG=64"]
fn native_linear_launch_families_use_frozen_launch() {
    pollster::block_on(actual_linear_launch_tail_witness());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn browser_linear_launch_families_use_frozen_launch() {
    actual_linear_launch_tail_witness().await;
}
