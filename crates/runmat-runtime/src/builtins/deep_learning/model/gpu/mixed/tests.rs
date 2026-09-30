use super::*;
use crate::builtins::deep_learning::{layers::*, model::*, object};
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorStorage, HostNumericDataView,
    HostNumericTensorView,
};
use runmat_value::{CellArray, Value};

fn normalization_then_relu() -> Value {
    let input = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let normalization = block_on(layer_normalization_layer_builtin(vec![])).unwrap();
    let relu = block_on(relu_layer_builtin(vec![])).unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, normalization, relu], 3, 1).unwrap(),
    )]))
    .unwrap()
}

#[test]
fn later_single_relu_capability_rejects_mixed_route_before_effects() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = [1.0_f32, 2.0, 3.0, 4.0];
        let input_handle = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&source),
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(input_handle.clone())),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        );
        let Value::Object(network) = normalization_then_relu() else {
            panic!("network object expected")
        };
        let host = HostNetworkPlan::admit(&network, &input, "forward").unwrap();
        provider.reset_telemetry();
        let before = provider.telemetry_snapshot();
        let result = plan::admit(
            &network,
            &host,
            &input_handle,
            host.output_axes(),
            "forward",
        )
        .unwrap();
        assert!(matches!(
            result,
            Some(Err(GpuCapabilityReason::ProviderOperation))
        ));
        assert_eq!(provider.telemetry_snapshot(), before);
    });
}
