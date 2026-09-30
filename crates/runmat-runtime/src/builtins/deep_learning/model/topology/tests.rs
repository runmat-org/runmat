use super::*;
use crate::builtins::deep_learning::{layers, training};
use futures::executor::block_on;
use runmat_accelerate_api::{test_support::TestAllocationOwner, NumericElementType};

fn input() -> Value {
    block_on(layers::feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap()
}

fn transform() -> Value {
    block_on(layers::relu_layer_builtin(vec![])).unwrap()
}

fn collection(layers: Vec<Value>) -> Value {
    let count = layers.len();
    Value::Cell(CellArray::new(layers, count, 1).unwrap())
}

#[test]
fn constructor_and_resident_forward_reject_invalid_order_before_provider_access() {
    let output = block_on(layers::regression_layer_builtin(vec![])).unwrap();
    for layers in [
        vec![input(), input(), transform()],
        vec![input(), output, transform()],
        vec![transform(), input()],
    ] {
        let invalid = collection(layers);
        let error = block_on(dlnetwork_builtin(vec![invalid.clone()])).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));

        let mut network = block_on(dlnetwork_builtin(vec![collection(vec![
            input(),
            transform(),
        ])]))
        .unwrap();
        let Value::Object(object) = &mut network else {
            panic!("network expected")
        };
        object.properties.insert("Layers".into(), invalid);
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let resident = block_on(training::dlarray_builtin(
            Value::GpuTensor(allocation.handle.clone()),
            vec![],
        ))
        .unwrap();
        let error = block_on(forward_builtin(network, resident, vec![])).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(allocation.probe.retirements(), 0);
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}

#[test]
fn connected_graphs_are_not_silently_converted_to_sequential_networks() {
    let mut connections = StructValue::new();
    connections.insert("Source", Value::String("first".into()));
    connections.insert("Destination", Value::String("second".into()));
    let graph = object_with_identity(
        super::super::LAYER_GRAPH_CLASS,
        vec![
            ("Layers", collection(vec![input(), transform()])),
            ("Connections", Value::Struct(connections)),
        ],
    );
    assert!(layers_from_network_ref(&graph, "dlnetwork").is_err());
    assert!(layers_from_network_value(graph.clone(), "dlnetwork").is_err());
    let error = block_on(dlnetwork_builtin(vec![graph])).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:Unsupported"));
    assert!(error.to_string().contains("Connections"));
}
