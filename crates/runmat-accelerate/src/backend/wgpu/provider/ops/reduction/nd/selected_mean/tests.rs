use super::ColdNdResources;

#[test]
fn one_hook_resources_retain_requested_pair_and_uniform_extents() {
    let limits = wgpu::Limits::downlevel_defaults();
    let route = ColdNdResources::new(2, 4, 16, &limits).expect("small route");
    assert_eq!(route.output_bytes, 8);
    assert_eq!(route.pair_bytes, 16);
    assert_eq!(
        route.uniform_bytes,
        std::mem::size_of::<crate::backend::wgpu::params::ReduceNdParams>() as u64
    );
}

#[test]
fn preparation_and_consumption_share_every_device_binding_limit() {
    let defaults = wgpu::Limits::downlevel_defaults();
    let mut limits = defaults.clone();
    limits.max_buffer_size = 16;
    assert!(ColdNdResources::new(2, 4, 32, &limits).is_none());
    assert!(ColdNdResources::new(5, 4, 4, &limits).is_none());

    let mut limits = defaults.clone();
    limits.max_storage_buffer_binding_size = 16;
    assert!(ColdNdResources::new(2, 4, 32, &limits).is_none());
    assert!(ColdNdResources::new(5, 4, 4, &limits).is_none());

    let mut limits = defaults.clone();
    limits.max_uniform_buffer_binding_size =
        std::mem::size_of::<crate::backend::wgpu::params::ReduceNdParams>() as u32 - 1;
    assert!(ColdNdResources::new(2, 4, 16, &limits).is_none());

    let mut limits = defaults;
    limits.max_compute_workgroups_per_dimension = 1;
    assert!(ColdNdResources::new(2, 4, 16, &limits).is_none());
    assert!(ColdNdResources::new(usize::MAX, usize::MAX, 16, &limits).is_none());
}
