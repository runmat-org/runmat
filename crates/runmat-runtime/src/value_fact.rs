use std::collections::BTreeMap;

use runmat_types::{
    AliasFact, CallableFact, CallableIdentity, CellFact, CertaintyFact, ContiguityFact,
    DimensionFact, DynamicReason, ExceptionFact, ExecutionFact, ForeignAffinity,
    ForeignAffinityFact, ForeignFact, ForeignLifetime, ForeignLifetimeFact, ForeignOwnership,
    ForeignOwnershipFact, InvalidationVector, LayoutFact, MutationFact, NumericClass,
    NumericDomain, NumericFact, ObjectFact, OutputListFact, QualifiedName, ResidencyFact,
    ShapeFact, StorageFact, StructFact, SymbolName, ValueFact, ValueKindFact, ViewFact,
};
use runmat_value::{IntValue, NumericDType, Value};

/// Adapt one live runtime value into the dependency-neutral static fact schema.
///
/// This is deliberately owned by the runtime: `runmat-types` never imports or
/// inspects live values. The exhaustive match is also the coverage guard for
/// additions to `Value`.
pub fn value_fact(value: &Value) -> ValueFact {
    match value {
        Value::Int(value) => numeric_scalar(int_class(value), NumericDomain::Real),
        Value::Num(_) => numeric_scalar(NumericClass::Double, NumericDomain::Real),
        Value::Complex(_, _) => numeric_scalar(NumericClass::Double, NumericDomain::Complex),
        Value::Bool(_) => scalar(ValueKindFact::Logical),
        Value::LogicalArray(value) => dense(ValueKindFact::Logical, &value.shape),
        Value::String(_) => scalar(ValueKindFact::String),
        Value::StringArray(value) => dense(ValueKindFact::String, &value.shape),
        Value::CharArray(value) => dense(ValueKindFact::Character, &value.shape),
        Value::Tensor(value) => dense(
            numeric_kind(dtype_class(value.numeric_dtype()), NumericDomain::Real),
            &value.shape,
        ),
        Value::SparseTensor(value) => {
            let kind = value
                .numeric_dtype()
                .map_or(ValueKindFact::Logical, |dtype| {
                    numeric_kind(dtype_class(dtype), NumericDomain::Real)
                });
            sparse(kind, &value.shape())
        }
        Value::ComplexTensor(value) => dense(
            numeric_kind(dtype_class(value.numeric_dtype()), NumericDomain::Complex),
            &value.shape,
        ),
        Value::Symbolic(_) => scalar(ValueKindFact::Symbolic),
        Value::SymbolicArray(value) => dense(ValueKindFact::Symbolic, &value.shape),
        Value::Cell(value) => {
            let elements = value.data.iter().map(value_fact).collect::<Vec<_>>();
            let element = elements
                .iter()
                .cloned()
                .reduce(|left, right| runmat_types::FactJoin::join(&left, &right))
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
            dense(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(element),
                    elements,
                    elements_complete: true,
                }),
                &value.shape,
            )
        }
        Value::Struct(value) => scalar(ValueKindFact::Struct(StructFact {
            fields: value
                .fields
                .iter()
                .map(|(name, value)| (name.clone(), value_fact(value)))
                .collect(),
            fields_complete: true,
        })),
        Value::GpuTensor(value) => {
            let kind = gpu_kind(value);
            ValueFact {
                kind: kind.clone().unwrap_or(ValueKindFact::Unknown),
                shape: shape(&value.shape),
                storage: StorageFact::Opaque,
                layout: LayoutFact::Unknown,
                contiguity: ContiguityFact::Unknown,
                view: ViewFact::Unknown,
                residency: ResidencyFact::Device {
                    provider: Some(format!("device:{}", value.device_id)),
                },
                alias: AliasFact::Identity,
                mutation: MutationFact::HandleSemantics,
                certainty: if kind.is_some() {
                    CertaintyFact::Proven
                } else {
                    CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation)
                },
                invalidation: InvalidationVector::default(),
            }
        }
        Value::Object(value) => object(
            &value.class_name,
            value
                .properties
                .iter()
                .map(|(name, value)| (name.clone(), value_fact(value)))
                .collect(),
            true,
            false,
            ShapeFact::Scalar,
        ),
        Value::ObjectArray(value) => object(
            value.class_name(),
            BTreeMap::new(),
            false,
            value
                .data()
                .iter()
                .any(|element| matches!(element, Value::HandleObject(_))),
            shape(value.shape()),
        ),
        Value::HandleObject(value) => object(
            &value.class_name,
            BTreeMap::new(),
            false,
            true,
            ShapeFact::Scalar,
        ),
        Value::Listener(value) => object(
            &value.target_class_name,
            BTreeMap::new(),
            false,
            true,
            ShapeFact::Scalar,
        ),
        Value::OutputList(values) => scalar(ValueKindFact::OutputList(OutputListFact {
            outputs: values.iter().map(value_fact).collect(),
            variadic: false,
        })),
        Value::FunctionHandle(name) => {
            callable(CallableIdentity::DynamicName(SymbolName(name.clone())))
        }
        Value::ExternalFunctionHandle(name) => {
            callable(CallableIdentity::ExternalName(qualified_name(name)))
        }
        Value::MethodFunctionHandle(name) => callable(CallableIdentity::Method(
            runmat_types::MethodId(name.clone()),
        )),
        Value::BoundFunctionHandle { function, .. } => callable(CallableIdentity::BoundFunction(
            runmat_types::FunctionId(*function),
        )),
        Value::Closure(value) => {
            let mut fact = callable_kind(CallableIdentity::DynamicName(SymbolName(
                value.function_name.clone(),
            )));
            fact.captures = value.captures.iter().map(value_fact).collect();
            scalar(ValueKindFact::Callable(fact))
        }
        Value::ClassRef(name) => scalar(ValueKindFact::ClassReference(
            runmat_types::ClassReferenceFact {
                class: None,
                runtime_class: Some(qualified_name(name)),
            },
        )),
        Value::MException(value) => scalar(ValueKindFact::Exception(ExceptionFact {
            identifier: Some(value.identifier.clone()),
        })),
        Value::Future(_) => execution(ExecutionFact::Future {
            output: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            state: runmat_types::FutureStateFact::Unknown,
        }),
        Value::Task(_) => execution(ExecutionFact::Task {
            output: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
        }),
        Value::Pool(_) => execution(ExecutionFact::Pool),
        Value::Job(_) => execution(ExecutionFact::Job {
            output: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
        }),
        Value::Foreign(reference) => {
            let mut fact = fact(
                ValueKindFact::Foreign(ForeignFact {
                    family: reference.type_identity.family.clone(),
                    type_name: Some(reference.type_identity.name.clone()),
                    type_version: Some(reference.type_identity.version),
                    ownership: match reference.ownership {
                        ForeignOwnership::Borrowed => ForeignOwnershipFact::Borrowed,
                        ForeignOwnership::Owned => ForeignOwnershipFact::Owned,
                        ForeignOwnership::Shared => ForeignOwnershipFact::Shared,
                    },
                    affinity: match reference.affinity {
                        ForeignAffinity::AnyThread => ForeignAffinityFact::AnyThread,
                        ForeignAffinity::OriginThread => ForeignAffinityFact::OriginThread,
                        ForeignAffinity::OriginProcess => ForeignAffinityFact::OriginProcess,
                        ForeignAffinity::RemoteHost => ForeignAffinityFact::RemoteHost,
                    },
                    lifetime: match reference.lifetime {
                        ForeignLifetime::Call => ForeignLifetimeFact::Call,
                        ForeignLifetime::Session => ForeignLifetimeFact::Session,
                        ForeignLifetime::Persistent => ForeignLifetimeFact::Persistent,
                        ForeignLifetime::External => ForeignLifetimeFact::External,
                    },
                }),
                ShapeFact::Scalar,
                StorageFact::Opaque,
            );
            fact.alias = AliasFact::Identity;
            fact.mutation = MutationFact::HandleSemantics;
            fact
        }
    }
}

fn numeric_scalar(class: NumericClass, domain: NumericDomain) -> ValueFact {
    scalar(numeric_kind(class, domain))
}

fn numeric_kind(class: NumericClass, domain: NumericDomain) -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact { class, domain })
}

fn scalar(kind: ValueKindFact) -> ValueFact {
    fact(kind, ShapeFact::Scalar, StorageFact::Scalar)
}

fn dense(kind: ValueKindFact, dimensions: &[usize]) -> ValueFact {
    fact(kind, shape(dimensions), StorageFact::Dense)
}

fn sparse(kind: ValueKindFact, dimensions: &[usize]) -> ValueFact {
    fact(kind, shape(dimensions), StorageFact::Sparse)
}

fn fact(kind: ValueKindFact, shape: ShapeFact, storage: StorageFact) -> ValueFact {
    ValueFact {
        kind,
        shape,
        storage,
        layout: LayoutFact::ColumnMajor,
        contiguity: ContiguityFact::Contiguous,
        view: ViewFact::Materialized,
        residency: ResidencyFact::Host,
        alias: AliasFact::Unique,
        mutation: MutationFact::ValueSemantics,
        certainty: CertaintyFact::Proven,
        invalidation: InvalidationVector::default(),
    }
}

fn object(
    class_name: &str,
    properties: BTreeMap<String, ValueFact>,
    properties_complete: bool,
    handle_semantics: bool,
    shape: ShapeFact,
) -> ValueFact {
    let mut fact = fact(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(qualified_name(class_name)),
            properties,
            properties_complete,
            handle_semantics: Some(handle_semantics),
        }),
        shape,
        StorageFact::Opaque,
    );
    if handle_semantics {
        fact.alias = AliasFact::Identity;
        fact.mutation = MutationFact::HandleSemantics;
    }
    fact
}

fn callable(identity: CallableIdentity) -> ValueFact {
    scalar(ValueKindFact::Callable(callable_kind(identity)))
}

fn callable_kind(identity: CallableIdentity) -> CallableFact {
    CallableFact {
        identity: Some(identity),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: true,
    }
}

fn execution(execution: ExecutionFact) -> ValueFact {
    let mut fact = scalar(ValueKindFact::Execution(execution));
    fact.alias = AliasFact::Identity;
    fact.mutation = MutationFact::HandleSemantics;
    fact
}

fn shape(dimensions: &[usize]) -> ShapeFact {
    ShapeFact::Shaped {
        dims: dimensions
            .iter()
            .copied()
            .map(DimensionFact::Known)
            .collect(),
    }
}

fn qualified_name(name: &str) -> QualifiedName {
    QualifiedName(
        name.split('.')
            .map(|segment| SymbolName(segment.to_owned()))
            .collect(),
    )
}

fn int_class(value: &IntValue) -> NumericClass {
    match value {
        IntValue::I8(_) => NumericClass::Int8,
        IntValue::I16(_) => NumericClass::Int16,
        IntValue::I32(_) => NumericClass::Int32,
        IntValue::I64(_) => NumericClass::Int64,
        IntValue::U8(_) => NumericClass::UInt8,
        IntValue::U16(_) => NumericClass::UInt16,
        IntValue::U32(_) => NumericClass::UInt32,
        IntValue::U64(_) => NumericClass::UInt64,
    }
}

fn dtype_class(value: NumericDType) -> NumericClass {
    match value {
        NumericDType::F64 => NumericClass::Double,
        NumericDType::F32 => NumericClass::Single,
        NumericDType::I8 => NumericClass::Int8,
        NumericDType::I16 => NumericClass::Int16,
        NumericDType::I32 => NumericClass::Int32,
        NumericDType::I64 => NumericClass::Int64,
        NumericDType::U8 => NumericClass::UInt8,
        NumericDType::U16 => NumericClass::UInt16,
        NumericDType::U32 => NumericClass::UInt32,
        NumericDType::U64 => NumericClass::UInt64,
    }
}

fn gpu_kind(handle: &runmat_accelerate_api::GpuTensorHandle) -> Option<ValueKindFact> {
    use runmat_accelerate_api::{GpuTensorStorage, IntegerElementType, ProviderPrecision};

    if runmat_accelerate_api::handle_is_logical(handle) {
        return Some(ValueKindFact::Logical);
    }
    let class = runmat_accelerate_api::handle_integer_type(handle)
        .map(|integer| match integer {
            IntegerElementType::I8 => NumericClass::Int8,
            IntegerElementType::I16 => NumericClass::Int16,
            IntegerElementType::I32 => NumericClass::Int32,
            IntegerElementType::I64 => NumericClass::Int64,
            IntegerElementType::U8 => NumericClass::UInt8,
            IntegerElementType::U16 => NumericClass::UInt16,
            IntegerElementType::U32 => NumericClass::UInt32,
            IntegerElementType::U64 => NumericClass::UInt64,
        })
        .or_else(|| match runmat_accelerate_api::handle_precision(handle) {
            Some(ProviderPrecision::F32) => Some(NumericClass::Single),
            Some(ProviderPrecision::F64) => Some(NumericClass::Double),
            None => match runmat_accelerate_api::handle_class_name(handle).as_deref() {
                Some("single") => Some(NumericClass::Single),
                Some("double") => Some(NumericClass::Double),
                _ => None,
            },
        })?;
    let domain = match runmat_accelerate_api::handle_storage(handle) {
        GpuTensorStorage::Real => NumericDomain::Real,
        GpuTensorStorage::ComplexInterleaved => NumericDomain::Complex,
    };
    Some(numeric_kind(class, domain))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        clear_handle_metadata, GpuTensorHandle, GpuTensorStorage, NumericElementType,
    };
    use runmat_value::{CellArray, ForeignRef, Tensor};

    #[test]
    fn preserves_recursive_numeric_class_and_shape() {
        let value = Value::Cell(
            CellArray::new_with_shape(
                vec![
                    Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                    Value::Tensor(Tensor::from_f32(vec![3.0, 4.0], vec![1, 2]).unwrap()),
                ],
                vec![1, 2],
            )
            .unwrap(),
        );
        let fact = value_fact(&value);
        let ValueKindFact::Cell(cell) = fact.kind else {
            panic!("expected cell fact");
        };
        assert!(matches!(
            cell.element.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real
            })
        ));
        assert_eq!(fact.shape, shape(&[1, 2]));
        assert_eq!(cell.elements.len(), 2);
        assert!(cell.elements_complete);
    }

    #[test]
    fn preserves_device_integer_class_domain_shape_and_residency() {
        let handle = GpuTensorHandle::new(vec![2, 3], 71, 9).with_numeric_descriptor(
            NumericElementType::U64,
            GpuTensorStorage::ComplexInterleaved,
        );
        let fact = value_fact(&Value::GpuTensor(handle.clone()));
        assert_eq!(
            fact.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Complex,
            })
        );
        assert_eq!(fact.shape, shape(&[2, 3]));
        assert_eq!(
            fact.residency,
            ResidencyFact::Device {
                provider: Some("device:71".into())
            }
        );
        assert_eq!(fact.certainty, CertaintyFact::Proven);
        clear_handle_metadata(&handle);
    }

    #[test]
    fn unknown_device_element_type_is_an_explicit_dynamic_boundary() {
        let handle = GpuTensorHandle::new(vec![1, 4], 73, 11);
        clear_handle_metadata(&handle);
        let fact = value_fact(&Value::GpuTensor(handle));
        assert_eq!(fact.kind, ValueKindFact::Unknown);
        assert_eq!(
            fact.certainty,
            CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation)
        );
    }

    #[test]
    fn foreign_reference_preserves_static_type_and_lifecycle_without_live_identity() {
        let value = Value::Foreign(ForeignRef {
            host_identity: "worker-7".into(),
            handle: 91,
            generation: 4,
            type_identity: runmat_types::ForeignTypeIdentity {
                family: "java".into(),
                name: "java.lang.StringBuilder".into(),
                version: 2,
            },
            ownership: ForeignOwnership::Shared,
            affinity: ForeignAffinity::OriginProcess,
            lifetime: ForeignLifetime::Session,
        });

        let fact = value_fact(&value);
        assert_eq!(
            fact.kind,
            ValueKindFact::Foreign(ForeignFact {
                family: "java".into(),
                type_name: Some("java.lang.StringBuilder".into()),
                type_version: Some(2),
                ownership: ForeignOwnershipFact::Shared,
                affinity: ForeignAffinityFact::OriginProcess,
                lifetime: ForeignLifetimeFact::Session,
            })
        );
        assert_eq!(fact.shape, ShapeFact::Scalar);
        assert_eq!(fact.storage, StorageFact::Opaque);
        assert_eq!(fact.alias, AliasFact::Identity);
        assert_eq!(fact.mutation, MutationFact::HandleSemantics);
    }
}
