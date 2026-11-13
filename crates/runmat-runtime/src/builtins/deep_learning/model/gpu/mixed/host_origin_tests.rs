use super::{host_origin, HostNetworkPlan};
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use crate::builtins::deep_learning::{
    layers::{feature_input_layer_builtin, layer_normalization_layer_builtin, relu_layer_builtin},
    model::dlnetwork_builtin,
};
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelProvider, DeepLearningForwardCapability,
    DeepLearningForwardResources, DeepLearningForwardRoute, GpuHandleProvenance, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostNumericTensorView, HostTensorView, NumericElementType,
};
use runmat_value::{CellArray, Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[path = "host_origin_tests/conv_cancellation.rs"]
mod conv_cancellation;

struct AdmissionProbe {
    effects: AtomicUsize,
    queries: AtomicUsize,
    device: u32,
    ready: bool,
}

impl AccelProvider for AdmissionProbe {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        panic!("host route admission uploaded before placement selected the route")
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        panic!("host route admission gathered a parameter")
    }

    fn device_id(&self) -> u32 {
        self.device
    }

    fn device_info(&self) -> String {
        "pure-host-route-admission-probe".into()
    }

    fn deep_learning_forward_route_capability(
        &self,
        route: &DeepLearningForwardRoute<'_>,
    ) -> anyhow::Result<Option<DeepLearningForwardCapability>> {
        assert!(matches!(
            route.input,
            runmat_accelerate_api::DeepLearningForwardSource::Host { .. }
        ));
        self.queries.fetch_add(1, Ordering::SeqCst);
        if !self.ready {
            return Ok(None);
        }
        let geometry = route
            .analyze()
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;
        Ok(Some(DeepLearningForwardCapability::Ready(
            DeepLearningForwardResources {
                output_bytes: geometry.output_bytes,
                largest_stage_value_bytes: geometry.largest_stage_value_bytes,
                known_normalization_scratch_bytes: 0,
                upload_bytes: 0,
                resident_bytes: 0,
                new_buffer_bytes_upper_bound:
                    runmat_accelerate_api::DeepLearningForwardBufferBound::Unavailable,
            },
        )))
    }
}

fn network() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let normalization = block_on(layer_normalization_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, normalization, relu], 3, 1).unwrap(),
    )]))
    .unwrap()
}

fn dense_relu_network() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::from_f32(vec![2.0, -1.0], vec![1, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense, relu], 3, 1).unwrap(),
    )]))
    .unwrap()
}

fn network_object(owner: &crate::execution::RootedValue) -> &runmat_value::ObjectInstance {
    let Value::Object(network) = owner.value() else {
        panic!("dlnetwork constructor must return an object")
    };
    network
}

fn with_explicit_weights(network: &mut runmat_value::ObjectInstance, weights: GpuTensorHandle) {
    use crate::builtins::deep_learning::model::parameters::ParameterRole;
    let Value::Cell(layers) = network.fixture_dl_property_mut("Layers").unwrap() else {
        panic!("network must retain Layers")
    };
    let Value::Object(dense) =
        &mut crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1]
    else {
        panic!("second layer must be fully connected")
    };
    dense
        .replace_dl_property(
            ParameterRole::Weights.name(),
            Value::GpuTensor(
                weights
                    .with_provenance(GpuHandleProvenance::Explicit)
                    .unwrap(),
            ),
        )
        .expect("fixture property");
}

fn with_explicit_scale(network: &mut runmat_value::ObjectInstance, scale: GpuTensorHandle) {
    use crate::builtins::deep_learning::model::parameters::ParameterRole;
    let Value::Cell(layers) = network.fixture_dl_property_mut("Layers").unwrap() else {
        panic!("network must retain Layers")
    };
    let Value::Object(normalization) =
        &mut crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1]
    else {
        panic!("second layer must be normalization")
    };
    normalization
        .replace_dl_property(
            ParameterRole::Scale.name(),
            Value::GpuTensor(
                scale
                    .with_provenance(GpuHandleProvenance::Explicit)
                    .unwrap(),
            ),
        )
        .expect("fixture property");
}

#[test]
fn direct_forward_with_explicit_learnable_never_replays_on_host_after_route_change() {
    use crate::builtins::deep_learning::model::execution::{
        execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout,
    };

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static LiveRouteChangeProbe = Box::leak(Box::new(LiveRouteChangeProbe {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        live_queries: AtomicUsize::new(0),
        retired: Mutex::new(Vec::new()),
        malformed_upload_shape: false,
        cancel_on_upload: None,
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let scale = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let network_owner = network();
    let mut network = network_object(&network_owner).clone();
    with_explicit_scale(&mut network, scale.handle.clone());
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let error = block_on(execute_network(NetworkExecutionRequest {
        runtime: None,
        network: &network,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: None,
    }))
    .err()
    .expect("selected provider route must report changed live capability");
    assert_eq!(
        error.identifier(),
        Some("RunMat:gpu:ProviderCapabilityChanged")
    );
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.live_queries.load(Ordering::SeqCst), 1);
    assert_eq!(scale.probe.retirements(), 0);
}

#[test]
fn direct_forward_with_explicit_learnable_rejects_unavailable_provider_without_upload() {
    use crate::builtins::deep_learning::model::execution::{
        execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout,
    };

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let provider: &'static AdmissionProbe = Box::leak(Box::new(AdmissionProbe {
        effects: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        device: owner.device_id(),
        ready: false,
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let scale = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let network_owner = network();
    let mut network = network_object(&network_owner).clone();
    with_explicit_scale(&mut network, scale.handle.clone());
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let result = block_on(execute_network(NetworkExecutionRequest {
        runtime: None,
        network: &network,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: None,
    }));
    assert!(result.is_err());
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    assert_eq!(scale.probe.retirements(), 0);
}

#[test]
fn direct_forward_requires_the_explicit_learnables_device_before_provider_query() {
    use crate::builtins::deep_learning::model::execution::{
        execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout,
    };

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let provider: &'static AdmissionProbe = Box::leak(Box::new(AdmissionProbe {
        effects: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        device: TestAllocationOwner::new().device_id(),
        ready: true,
    }));
    assert_ne!(provider.device_id(), owner.device_id());
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let scale = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let network_owner = network();
    let mut network = network_object(&network_owner).clone();
    with_explicit_scale(&mut network, scale.handle.clone());
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let result = block_on(execute_network(NetworkExecutionRequest {
        runtime: None,
        network: &network,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }));
    assert!(result.is_err());
    assert_eq!(provider.queries.load(Ordering::SeqCst), 0);
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    assert_eq!(scale.probe.retirements(), 0);
}

#[test]
fn direct_forward_cancellation_retires_only_the_uploaded_input() {
    use std::rc::Rc;

    use crate::builtins::deep_learning::model::execution::{
        execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout,
    };

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let provider: &'static LiveRouteChangeProbe = Box::leak(Box::new(LiveRouteChangeProbe {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        live_queries: AtomicUsize::new(0),
        retired: Mutex::new(Vec::new()),
        malformed_upload_shape: false,
        cancel_on_upload: Some(cancellation),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let scale = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let network_owner = network();
    let mut network = network_object(&network_owner).clone();
    with_explicit_scale(&mut network, scale.handle.clone());
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let error = block_on(execute_network(NetworkExecutionRequest {
        runtime: Some(&runtime),
        network: &network,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }))
    .err()
    .expect("cancellation must stop the selected provider route");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.live_queries.load(Ordering::SeqCst), 0);
    assert_eq!(scale.probe.retirements(), 0);
    let probes = provider.retired.lock().unwrap();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].retirements(), 1);
}

#[test]
fn direct_forward_without_context_honors_scoped_interrupt() {
    use crate::builtins::deep_learning::model::execution::{
        execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout,
    };

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let _interrupt = crate::interrupt::replace_interrupt(Some(cancellation.token()));
    let provider: &'static LiveRouteChangeProbe = Box::leak(Box::new(LiveRouteChangeProbe {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        live_queries: AtomicUsize::new(0),
        retired: Mutex::new(Vec::new()),
        malformed_upload_shape: false,
        cancel_on_upload: Some(cancellation),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let scale = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let network_owner = network();
    let mut network = network_object(&network_owner).clone();
    with_explicit_scale(&mut network, scale.handle.clone());
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let error = block_on(execute_network(NetworkExecutionRequest {
        runtime: None,
        network: &network,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }))
    .err()
    .expect("the scoped interrupt must stop the direct selected route");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.live_queries.load(Ordering::SeqCst), 0);
    assert_eq!(scale.probe.retirements(), 0);
    let probes = provider.retired.lock().unwrap();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].retirements(), 1);
}

#[test]
fn host_origin_admission_requires_whole_route_query_without_effects() {
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let unavailable = AdmissionProbe {
        effects: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        device: 23,
        ready: false,
    };
    let unavailable: &'static AdmissionProbe = Box::leak(Box::new(unavailable));
    assert!(matches!(
        host_origin::admit_host_route(&host, host.output_axes(), unavailable).unwrap(),
        Some(Err(super::GpuCapabilityReason::ProviderOperation))
    ));
    assert_eq!(unavailable.queries.load(Ordering::SeqCst), 1);
    assert_eq!(unavailable.effects.load(Ordering::SeqCst), 0);

    let ready: &'static AdmissionProbe = Box::leak(Box::new(AdmissionProbe {
        effects: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        device: 24,
        ready: true,
    }));
    let admitted = host_origin::admit_host_route(&host, host.output_axes(), ready)
        .unwrap()
        .expect("normalization and activation have a physical route")
        .expect("provider admitted the physical route");
    assert_eq!(
        admitted
            .physical_route()
            .unwrap()
            .unwrap()
            .analyze()
            .unwrap()
            .output_bytes,
        16
    );
    assert_eq!(ready.queries.load(Ordering::SeqCst), 1);
    assert_eq!(ready.effects.load(Ordering::SeqCst), 0);
}

#[test]
fn ordinary_dense_relu_is_an_effect_free_host_origin_candidate() {
    let network_owner = dense_relu_network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    static PROVIDER: AdmissionProbe = AdmissionProbe {
        effects: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        device: 29,
        ready: true,
    };
    let admitted = host_origin::admit_host_route(&host, host.output_axes(), &PROVIDER)
        .unwrap()
        .expect("ordinary dense and ReLU stages are structurally eligible")
        .expect("the provider admits the complete physical route");
    assert!(admitted
        .plan
        .stages
        .iter()
        .any(|stage| matches!(stage, super::plan::Stage::Dense { .. })));
    assert!(admitted
        .plan
        .stages
        .iter()
        .any(|stage| matches!(stage, super::plan::Stage::Relu)));
    assert_eq!(
        admitted
            .physical_route()
            .unwrap()
            .unwrap()
            .analyze()
            .unwrap()
            .output_bytes,
        8
    );
    assert_eq!(PROVIDER.queries.load(Ordering::SeqCst), 1);
    assert_eq!(PROVIDER.effects.load(Ordering::SeqCst), 0);
}

#[test]
fn explicit_dense_weights_require_a_provider_without_host_replay() {
    use crate::builtins::deep_learning::model::execution::{
        execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout,
    };

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let provider: &'static AdmissionProbe = Box::leak(Box::new(AdmissionProbe {
        effects: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        device: owner.device_id(),
        ready: false,
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let weights = owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let network_owner = dense_relu_network();
    let mut network = network_object(&network_owner).clone();
    with_explicit_weights(&mut network, weights.handle.clone());
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let result = block_on(execute_network(NetworkExecutionRequest {
        runtime: None,
        network: &network,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }));
    assert!(
        result.is_err(),
        "explicit weights cannot select the CPU route"
    );
    assert_eq!(provider.queries.load(Ordering::SeqCst), 1);
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    assert_eq!(weights.probe.retirements(), 0);
}

#[test]
fn admitted_host_route_is_bound_to_the_exact_network_even_with_identical_geometry() {
    let admitted_owner = network();
    let other_owner = network();
    let admitted_network = network_object(&admitted_owner);
    let other_network = network_object(&other_owner);
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let host = HostNetworkPlan::admit(admitted_network, &input, "forward").unwrap();
    static PROVIDER: AdmissionProbe = AdmissionProbe {
        effects: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        device: 25,
        ready: true,
    };
    let admitted = host_origin::admit_host_route(&host, host.output_axes(), &PROVIDER)
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(std::ptr::eq(
        admitted.host.admitted_network(),
        admitted_network
    ));
    assert!(!std::ptr::eq(
        admitted.host.admitted_network(),
        other_network
    ));
    // Execution accepts only this admitted token; there is no independent
    // network argument with which a same-shaped object could be substituted.
    assert_eq!(PROVIDER.effects.load(Ordering::SeqCst), 0);
}

#[test]
fn explicit_resident_parameter_precision_rejects_before_query_or_upload() {
    use crate::builtins::deep_learning::model::parameters::ParameterRole;

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let scale = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let provider: &'static AdmissionProbe = Box::leak(Box::new(AdmissionProbe {
        effects: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        device: owner.device_id(),
        ready: true,
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let network_owner = network();
    let mut network = network_object(&network_owner).clone();
    let Value::Cell(layers) = network.fixture_dl_property_mut("Layers").unwrap() else {
        panic!("network must retain Layers")
    };
    let Value::Object(normalization) =
        &mut crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1]
    else {
        panic!("second layer must be normalization")
    };
    normalization
        .replace_dl_property(
            ParameterRole::Scale.name(),
            Value::GpuTensor(
                scale
                    .handle
                    .clone()
                    .with_provenance(GpuHandleProvenance::Explicit)
                    .unwrap(),
            ),
        )
        .expect("fixture property");
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let host = HostNetworkPlan::admit(&network, &input, "forward").unwrap();
    let result = host_origin::admit_host_route(&host, host.output_axes(), provider);
    assert!(result.is_err());
    assert_eq!(provider.queries.load(Ordering::SeqCst), 0);
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    assert_eq!(scale.probe.retirements(), 0);
}

struct LiveRouteChangeProbe {
    owner: TestAllocationOwner,
    uploads: AtomicUsize,
    live_queries: AtomicUsize,
    retired: Mutex<Vec<runmat_accelerate_api::test_support::RetirementProbe>>,
    malformed_upload_shape: bool,
    cancel_on_upload: Option<runmat_execution::cancellation::CancellationSource>,
}

impl AccelProvider for LiveRouteChangeProbe {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("native upload must not be demoted to the legacy floating transfer")
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        let data = match host.data {
            HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
            HostNumericDataView::F64(values) => HostNumericDataOwned::F64(values.to_vec()),
            _ => {
                return Err(
                    anyhow::anyhow!("host-origin route requires native floating payload").into(),
                )
            }
        };
        let allocated = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: if self.malformed_upload_shape {
                vec![host.shape.iter().product(), 1]
            } else {
                host.shape.to_vec()
            },
            storage: host.storage,
        })?;
        self.retired.lock().unwrap().push(allocated.probe);
        if let Some(source) = &self.cancel_on_upload {
            source.cancel(runmat_execution::CancellationReason::User);
        }
        Ok(allocated.handle)
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("changed capability must fail without a host replay")
    }

    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "host-route-change-probe".into()
    }

    fn deep_learning_forward_route_capability(
        &self,
        route: &DeepLearningForwardRoute<'_>,
    ) -> anyhow::Result<Option<DeepLearningForwardCapability>> {
        if matches!(
            route.input,
            runmat_accelerate_api::DeepLearningForwardSource::Resident(_)
        ) {
            self.live_queries.fetch_add(1, Ordering::SeqCst);
            return Ok(Some(DeepLearningForwardCapability::UnsupportedOperation));
        }
        let geometry = route
            .analyze()
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;
        Ok(Some(DeepLearningForwardCapability::Ready(
            DeepLearningForwardResources {
                output_bytes: geometry.output_bytes,
                largest_stage_value_bytes: geometry.largest_stage_value_bytes,
                known_normalization_scratch_bytes: 0,
                upload_bytes: 0,
                resident_bytes: 0,
                new_buffer_bytes_upper_bound:
                    runmat_accelerate_api::DeepLearningForwardBufferBound::Unavailable,
            },
        )))
    }
}

#[test]
fn changed_live_capability_retires_the_uploaded_input_without_replay() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static LiveRouteChangeProbe = Box::leak(Box::new(LiveRouteChangeProbe {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        live_queries: AtomicUsize::new(0),
        retired: Mutex::new(Vec::new()),
        malformed_upload_shape: false,
        cancel_on_upload: None,
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let admitted = host_origin::admit_host_route(&host, host.output_axes(), provider)
        .unwrap()
        .expect("mixed route")
        .expect("host source is admitted");
    let failure = match block_on(host_origin::execute_host_route_selected(admitted, None)) {
        Ok(_) => panic!("changed live capability must fail the selected provider route"),
        Err(failure) => failure,
    };
    assert!(failure.is_operational());
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.live_queries.load(Ordering::SeqCst), 1);
    let probes = provider.retired.lock().unwrap();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].retirements(), 1);
}

#[test]
fn malformed_host_upload_retires_only_its_new_result_before_dispatch() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static LiveRouteChangeProbe = Box::leak(Box::new(LiveRouteChangeProbe {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        live_queries: AtomicUsize::new(0),
        retired: Mutex::new(Vec::new()),
        malformed_upload_shape: true,
        cancel_on_upload: None,
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let admitted = host_origin::admit_host_route(&host, host.output_axes(), provider)
        .unwrap()
        .unwrap()
        .unwrap();
    let failure = block_on(host_origin::execute_host_route_selected(admitted, None));
    assert!(failure.is_err());
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.live_queries.load(Ordering::SeqCst), 0);
    let probes = provider.retired.lock().unwrap();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].retirements(), 1);
}

#[test]
fn cancellation_after_host_upload_retires_input_before_stage_dispatch() {
    use std::rc::Rc;

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let provider: &'static LiveRouteChangeProbe = Box::leak(Box::new(LiveRouteChangeProbe {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        live_queries: AtomicUsize::new(0),
        retired: Mutex::new(Vec::new()),
        malformed_upload_shape: false,
        cancel_on_upload: Some(cancellation),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let admitted = host_origin::admit_host_route(&host, host.output_axes(), provider)
        .unwrap()
        .unwrap()
        .unwrap();
    let failure = match block_on(host_origin::execute_host_route_selected(
        admitted,
        Some(&runtime),
    )) {
        Ok(_) => panic!("cancellation must abandon the selected route"),
        Err(failure) => failure,
    };
    assert!(!failure.is_operational());
    assert_eq!(failure.into_error().identifier(), Some("RunMat:Cancelled"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.live_queries.load(Ordering::SeqCst), 0);
    let probes = provider.retired.lock().unwrap();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].retirements(), 1);
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires an actual compatible WGPU adapter"]
fn selected_host_origin_route_executes_native_single_without_a_host_replay() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU network route qualification requires a compatible adapter");
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap());
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let admitted = host_origin::admit_host_route(&host, host.output_axes(), provider)
        .unwrap()
        .expect("mixed normalization and activation route")
        .expect("actual provider admits complete host-origin route");
    let output = block_on(host_origin::execute_host_route_selected(admitted, None)).unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );
    assert!(!runmat_accelerate_api::handle_is_explicit(&output));
    let gathered = crate::builtins::common::test_support::gather(Value::GpuTensor(output)).unwrap();
    let actual = gathered.materialize_f64();
    assert_eq!(actual.len(), 4);
    for (index, value) in actual.into_iter().enumerate() {
        let expected = if index < 2 {
            0.0
        } else {
            1.0 / (1.0 + 4.0e-5_f64).sqrt()
        };
        assert!(
            (value - expected).abs() < 1.0e-4,
            "element {index}: {value} != {expected}"
        );
    }
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires an actual compatible WGPU adapter"]
fn ordinary_dense_relu_host_origin_executes_the_selected_provider_route() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU network route qualification requires a compatible adapter");
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let network_owner = dense_relu_network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap());
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let admitted = host_origin::admit_host_route(&host, host.output_axes(), provider)
        .unwrap()
        .expect("ordinary dense and ReLU route")
        .expect("actual provider admits the complete route");
    let output = block_on(host_origin::execute_host_route_selected(admitted, None)).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );
    let gathered = crate::builtins::common::test_support::gather(Value::GpuTensor(output)).unwrap();
    assert_eq!(gathered.materialize_f64(), vec![0.0, 0.5]);
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires an actual compatible WGPU adapter"]
fn direct_forward_preserves_explicit_learnable_output_presentation() {
    use crate::builtins::deep_learning::model::{
        execution::{execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout},
        output::{self, OutputPresentation},
    };

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("direct resident-learnable qualification requires a compatible adapter");
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let scale = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 1.0]),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let network_owner = network();
    let mut network = network_object(&network_owner).clone();
    with_explicit_scale(&mut network, scale);
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap());
    let outcome = block_on(execute_network(NetworkExecutionRequest {
        runtime: None,
        network: &network,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }))
    .unwrap();
    let Value::GpuTensor(handle) = &outcome.data else {
        panic!("explicit consumed learnable must retain resident presentation")
    };
    assert_eq!(handle.shape(), &[2, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(handle),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );
    assert!(runmat_accelerate_api::handle_is_explicit(handle));
    assert_eq!(handle.device_id(), provider.device_id());
    let output = output::present(
        &network,
        &input,
        outcome,
        OutputPresentation::Forward,
        false,
    )
    .unwrap();
    assert_eq!(output.len(), 1);
    let Value::GpuTensor(handle) = &output[0] else {
        panic!("public forward output must keep the admitted provider value")
    };
    assert_eq!(handle.shape(), &[2, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(handle),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );
    assert!(runmat_accelerate_api::handle_is_explicit(handle));
}
