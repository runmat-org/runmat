use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};

pub const DEAL_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "deal-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "deal with a resident input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DealResidentInputExtension"),
};

pub const DEAL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DEAL_RESIDENT_INPUT_EXTENSION];

const DEAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Distributed output values.",
}];

const DEAL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargin",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Input values to distribute.",
}];

const DEAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = deal(varargin)",
    inputs: &DEAL_INPUTS,
    outputs: &DEAL_OUTPUT,
}];

const DEAL_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.DEAL.INPUT_OUTPUT_COUNT_MISMATCH",
    identifier: Some("RunMat:deal:InputOutputCountMismatch"),
    when: "The call has multiple inputs and the requested output count differs from the input count, or has no input and a nonzero output count.",
    message: "deal: the number of outputs must match the number of inputs unless there is exactly one input",
}];

pub const DEAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DEAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEAL_ERRORS,
};

const DEAL_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "deal copies input arrays to outputs without numeric conversion; every integer class, shape, and value is preserved.",
}];

const DEAL_RESIDENT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident integer A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode passes resident integer handles through structurally without gather, allocation, provider dispatch, or floating conversion.",
    }];

pub const DEAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[B1,...,Bn] = deal(integer_A1,...,integer_An) or deal(integer_A)",
        inputs: &DEAL_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Matching inputs are distributed positionally; one input is replicated to every requested output. No integer arithmetic or floating materialization occurs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[B1,...,Bn] = deal(resident_integer_A1,...,resident_integer_An) or deal(resident_integer_A)",
        inputs: &DEAL_RESIDENT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "This RunMat-only form is gated before provider access and retains each input's original resident handle and owner.",
    },
];

#[runmat_macros::runtime_builtin(
    name = "deal",
    descriptor(self::DEAL_DESCRIPTOR),
    extensions(self::DEAL_EXTENSIONS),
    integer_capabilities(self::DEAL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::common::deal"
)]
pub async fn deal_builtin_registered(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::deal_builtin(rest).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, Tensor};

    fn call(inputs: Vec<Value>, outputs: usize) -> crate::BuiltinResult<Value> {
        let _outputs = crate::output_count::push_output_count(Some(outputs));
        block_on(deal_builtin_registered(inputs))
    }

    #[test]
    fn deal_distributes_matching_inputs_and_replicates_one_input() {
        assert_eq!(
            call(vec![Value::Num(3.0)], 3).unwrap(),
            Value::OutputList(vec![Value::Num(3.0); 3])
        );
        assert_eq!(
            call(vec![Value::Num(2.0), Value::from("x")], 2).unwrap(),
            Value::OutputList(vec![Value::Num(2.0), Value::from("x")])
        );
        assert_eq!(call(vec![Value::Num(4.0)], 1).unwrap(), Value::Num(4.0));
    }

    #[test]
    fn deal_enforces_public_input_output_count_rules() {
        for (inputs, outputs) in [
            (vec![Value::Num(1.0), Value::Num(2.0)], 1),
            (vec![Value::Num(1.0), Value::Num(2.0)], 3),
            (Vec::new(), 1),
        ] {
            let error = call(inputs, outputs).expect_err("count mismatch");
            assert_eq!(
                error.identifier(),
                Some("RunMat:deal:InputOutputCountMismatch")
            );
        }
        assert_eq!(call(Vec::new(), 0).unwrap(), Value::OutputList(Vec::new()));
        assert_eq!(
            call(
                vec![Value::Int(runmat_builtins::IntValue::U64(u64::MAX))],
                0
            )
            .unwrap(),
            Value::OutputList(Vec::new())
        );
    }

    #[test]
    fn deal_preserves_all_integer_storage_classes_exactly() {
        let storages = [
            IntegerStorage::I8(vec![-1, 2]),
            IntegerStorage::I16(vec![-1, 2]),
            IntegerStorage::I32(vec![-1, 2]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, 2]),
            IntegerStorage::U16(vec![0, 2]),
            IntegerStorage::U32(vec![0, 2]),
            IntegerStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_993]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap();
            let output = call(vec![Value::Tensor(tensor)], 1).unwrap();
            let Value::Tensor(output) = output else {
                panic!("integer tensor output");
            };
            assert_eq!(output.integer_storage(), Some(&storage));
            assert_eq!(output.shape, vec![1, 2]);
        }
    }

    #[test]
    fn deal_resident_pass_through_is_gated_before_provider_access() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            for storage in [
                IntegerStorage::I8(vec![-1, 2]),
                IntegerStorage::I16(vec![-1, 2]),
                IntegerStorage::I32(vec![-1, 2]),
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                IntegerStorage::U8(vec![0, 2]),
                IntegerStorage::U16(vec![0, 2]),
                IntegerStorage::U32(vec![0, 2]),
                IntegerStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_993]),
            ] {
                let tensor = Tensor::new_integer(storage, vec![1, 2]).unwrap();
                let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                    .expect("upload exact resident integer tensor");
                let integer_type = runmat_accelerate_api::handle_integer_type(&handle)
                    .expect("resident integer class metadata");
                assert!(std::ptr::eq(
                    runmat_accelerate_api::provider_for_handle(&handle).expect("handle owner"),
                    provider
                ));
                {
                    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                    let error =
                        call(vec![Value::GpuTensor(handle.clone())], 2).expect_err("resident gate");
                    assert_eq!(
                        error.identifier(),
                        DEAL_RESIDENT_INPUT_EXTENSION.error_identifier
                    );
                }
                let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
                let result = call(vec![Value::GpuTensor(handle.clone())], 2).unwrap();
                let Value::OutputList(outputs) = result else {
                    panic!("resident replicated outputs");
                };
                assert_eq!(outputs.len(), 2);
                for output in outputs {
                    let Value::GpuTensor(output) = output else {
                        panic!("resident output");
                    };
                    assert_eq!(output, handle);
                    assert_eq!(
                        runmat_accelerate_api::handle_integer_type(&output),
                        Some(integer_type)
                    );
                    assert!(std::ptr::eq(
                        runmat_accelerate_api::provider_for_handle(&output).expect("output owner"),
                        provider
                    ));
                }
                provider
                    .free(&handle)
                    .expect("free resident integer tensor");
            }
        });
    }
}
