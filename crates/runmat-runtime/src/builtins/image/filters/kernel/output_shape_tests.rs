use super::*;

#[test]
fn floating_accumulation_keeps_host_and_explicit_precision_policy() {
    assert_eq!(
        floating_accumulation(ProviderPrecision::F32, false),
        FilterAccumulation::HostDouble
    );
    assert_eq!(
        floating_accumulation(ProviderPrecision::F64, false),
        FilterAccumulation::HostDouble
    );
    assert_eq!(
        floating_accumulation(ProviderPrecision::F32, true),
        FilterAccumulation::GpuSingle
    );
    assert_eq!(
        floating_accumulation(ProviderPrecision::F64, true),
        FilterAccumulation::HostDouble
    );
}

#[test]
fn shape_only_provider_admission_matches_host_plan_for_all_modes() {
    let kernel = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
    for shape in [
        ImfilterShape::Same,
        ImfilterShape::Full,
        ImfilterShape::Valid,
    ] {
        let options = ImfilterOptions {
            shape,
            ..Default::default()
        };
        let plan = build_imfilter_plan(&[4, 5], &kernel, &options, "imfilter").unwrap();
        assert_eq!(
            filter_output_shape(&[4, 5], &kernel.shape, shape, "imfilter").unwrap(),
            plan.final_shape,
        );
    }
    assert!(filter_output_shape(&[4, 5], &[0, 3], ImfilterShape::Same, "imfilter").is_err());
}
