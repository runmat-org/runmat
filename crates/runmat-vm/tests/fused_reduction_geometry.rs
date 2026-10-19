#![cfg(feature = "native-accel")]

#[path = "support/mod.rs"]
mod test_helpers;

use runmat_accelerate::FusionKind;
use runmat_accelerate_api::ThreadProviderGuard;
use runmat_value::Value;

#[test]
fn interpreter_keeps_empty_nd_broadcast_and_dimension_error_semantics() {
    runmat_gc::gc_test_context(|| {
        let _scope = ThreadProviderGuard::set(None);
        for (source, expected_shape, expected) in [
            ("A=zeros(0,3); y=sum(A,1);", vec![1, 3], vec![0., 0., 0.]),
            ("A=zeros(3,0); y=sum(A,2);", vec![3, 1], vec![0., 0., 0.]),
            (
                "A=reshape(1:8,2,2,2); y=sum(A+0,'all');",
                vec![1, 1],
                vec![36.],
            ),
            (
                "A=[1;2]; B=[1,2,3]; y=sum(A+B,'all');",
                vec![1, 1],
                vec![21.],
            ),
        ] {
            let bytecode = test_helpers::compile_source(source).unwrap();
            let groups = bytecode.runtime_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|group| group.kind == FusionKind::Reduction),
                "required reduction candidate: {source}"
            );
            let result = test_helpers::interpret(&bytecode).unwrap();
            let output_slot = bytecode
                .var_names
                .iter()
                .find_map(|(slot, name)| (name == "y").then_some(*slot))
                .expect("named output binding");
            let value = result.get(output_slot).expect("published y output");
            match value {
                Value::Num(number) => {
                    assert_eq!(expected_shape, [1, 1]);
                    assert_eq!(expected, [*number]);
                }
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), expected_shape);
                    assert_eq!(tensor.materialize_f64(), expected);
                }
                _ => panic!("numeric interpreter result: {value:?}"),
            }
        }
        let bytecode = test_helpers::compile_source("A=zeros(0,3); y=mean(A,'all');").unwrap();
        let output_slot = bytecode
            .var_names
            .iter()
            .find_map(|(slot, name)| (name == "y").then_some(*slot))
            .expect("named output binding");
        let result = test_helpers::interpret(&bytecode).unwrap();
        assert!(
            matches!(result.get(output_slot), Some(Value::Num(number)) if number.is_nan())
                || matches!(result.get(output_slot), Some(Value::Tensor(tensor)) if tensor.len() == 1 && tensor.materialize_f64()[0].is_nan())
        );
        assert!(test_helpers::execute_source("A=ones(2,3); y=sum(A,1.5);").is_err());
    });
}
