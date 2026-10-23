use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::Value;

mod control;
pub(crate) mod endpoint;
mod path;

pub use endpoint::{PreparedSequenceEndpoint, SequenceEndpointSpec};
pub use path::{
    assignment_container_in_context, AssignmentStepSpec, PreparedSequenceDestination,
    SequenceDestinationBuilder,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DestinationCardinality {
    Fixed,
    Discard,
    Sequence(usize),
}

impl DestinationCardinality {
    const fn count(self) -> usize {
        match self {
            Self::Fixed | Self::Discard => 1,
            Self::Sequence(count) => count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DestinationRange {
    start: usize,
    len: usize,
}

impl DestinationRange {
    pub const fn start(self) -> usize {
        self.start
    }

    pub const fn len(self) -> usize {
        self.len
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestinationLayout {
    ranges: Vec<DestinationRange>,
    total: usize,
}

impl DestinationLayout {
    pub fn new(
        cardinalities: impl IntoIterator<Item = DestinationCardinality>,
    ) -> Result<Self, RuntimeError> {
        let mut ranges = Vec::new();
        let mut total = 0usize;
        for cardinality in cardinalities {
            let len = cardinality.count();
            ranges.push(DestinationRange { start: total, len });
            total = total.checked_add(len).ok_or_else(|| {
                semantic_error(
                    "DestinationCardinalityOverflow",
                    "output destination cardinality exceeds the supported limit",
                )
            })?;
        }
        Ok(Self { ranges, total })
    }

    pub const fn total(&self) -> usize {
        self.total
    }

    pub fn ranges(&self) -> &[DestinationRange] {
        &self.ranges
    }

    pub fn distribute(self, mut values: Vec<Value>) -> Result<Vec<Vec<Value>>, RuntimeError> {
        if values.len() < self.total {
            return Err(semantic_error(
                "CommaSeparatedListOutputShortage",
                format!(
                    "sequence assignment requires exactly one value per destination element (expected {}, received {})",
                    self.total, values.len()
                ),
            ));
        }
        values.truncate(self.total);
        let mut remaining = values.into_iter();
        self.ranges
            .into_iter()
            .map(|range| Ok(remaining.by_ref().take(range.len).collect()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextual_member_cardinality_preserves_aggregate_count_and_overload_order() {
        use crate::context::{
            host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
            RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
        };
        use futures::executor::block_on;
        use std::{cell::RefCell, rc::Rc, sync::Arc};
        struct Parallel(ManagedHostEffectLedger);
        impl RuntimeParallelService for Parallel {
            fn supports(&self, _: ParallelCapability) -> bool {
                false
            }
            fn placement_resources(&self) -> RuntimeParallelResources {
                let state = self.0.snapshot();
                RuntimeParallelResources {
                    cpu_millicores_available: 1000,
                    memory_available_bytes: Some(state.free_bytes()),
                    epoch: state.epoch,
                }
            }
            fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
                Some(self.0.clone())
            }
        }
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
                );
        let _activation = runtime.enter();
        let record = || {
            let mut record = runmat_value::UnboundedStructValueBuilder::new();
            record.insert("field", Value::Num(1.));
            record.finish()
        };
        for (base, count) in [
            (Value::Struct(record()), 1),
            (
                Value::StructArray(
                    runmat_value::StructArray::new(vec![record(), record()], vec![1, 2]).unwrap(),
                ),
                2,
            ),
        ] {
            let endpoint = block_on(PreparedSequenceEndpoint::prepare_with_context(
                Some(&runtime),
                &base,
                SequenceEndpointSpec::Member("field".into()),
            ))
            .unwrap();
            assert_eq!(endpoint.cardinality(), count);
        }
        let trace = Rc::new(RefCell::new(vec!["before"]));
        let observed = Rc::clone(&trace);
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "member_cardinality321").then_some(32101)
            })))
            .unwrap();
        let _invoker = crate::user_functions::install_local_semantic_function_invoker(Rc::new(
            move |runtime, function, arguments, requested| {
                assert_eq!(function, 32101);
                assert_eq!(requested, 1);
                let supplied = arguments.supplied();
                assert_eq!(supplied.len(), 3);
                assert!(matches!(&supplied[0], Value::Object(object) if object.class_name().display_name() == "MemberCardinality321"));
                let Value::Struct(descriptor) = &supplied[1] else { panic!("scalar substruct") };
                assert_eq!(descriptor.fields.get("type"), Some(&Value::String(".".into())));
                assert_eq!(descriptor.fields.get("subs"), Some(&Value::String("field".into())));
                let Value::Object(context) = &supplied[2] else { panic!("indexing context") };
                assert_eq!(context.property("__enum_member__"), Some(&Value::String("Assignment".into())));
                observed.borrow_mut().push("overload");
                let admission_runtime = runtime.clone();
                crate::user_functions::admit_user_function_with(admission_runtime.as_ref(), move || async move {
                    let _arguments = arguments;
                    crate::execution::RootedValueSequence::in_context(
                        crate::sequence::ValueSequence::single(Value::Num(3.))
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                        runtime.as_ref().expect("selected runtime"),
                    )
                })
            },
        )).unwrap();
        // Fixture hooks install admitted backing on the selected context.
        // Its independent pre-operation charge lasts until that context retires.
        let fixture = ledger.snapshot();
        assert!(fixture.live_claimed_bytes > 0);
        assert_eq!(fixture.live_reserved_bytes, fixture.live_claimed_bytes);
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: "MemberCardinality321".into(),
            parent: None,
            properties: Default::default(),
            methods: [(
                "numArgumentsFromSubscript".into(),
                crate::class_registry::RuntimeMethod {
                    name: "numArgumentsFromSubscript".into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: runmat_types::MemberAccess::Public,
                    function_name: "member_cardinality321".into(),
                    implicit_class_argument: None,
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            )]
            .into(),
        });
        let base = Value::Object(runmat_value::ObjectInstance::new("MemberCardinality321"));
        let endpoint = block_on(PreparedSequenceEndpoint::prepare_with_context(
            Some(&runtime),
            &base,
            SequenceEndpointSpec::Member("field".into()),
        ))
        .unwrap();
        trace.borrow_mut().push("after");
        assert_eq!(endpoint.cardinality(), 3);
        assert_eq!(*trace.borrow(), ["before", "overload", "after"]);
        drop(endpoint);
        assert_eq!(
            ledger.snapshot().live_reserved_bytes,
            fixture.live_reserved_bytes
        );
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            fixture.live_claimed_bytes
        );
        drop(_invoker);
        drop(_resolver);
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            fixture.live_claimed_bytes
        );
        drop(_activation);
        drop(runtime);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[test]
    fn mixed_layout_preserves_target_order_and_zero_length_sequences() {
        let layout = DestinationLayout::new([
            DestinationCardinality::Fixed,
            DestinationCardinality::Sequence(0),
            DestinationCardinality::Discard,
            DestinationCardinality::Sequence(2),
        ])
        .unwrap();
        assert_eq!(layout.total(), 4);
        assert_eq!(layout.ranges()[1].len(), 0);
        assert_eq!(layout.ranges()[3].start(), 2);

        let values = layout
            .distribute(vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
                Value::Num(5.0),
            ])
            .unwrap();
        assert_eq!(values[0], vec![Value::Num(1.0)]);
        assert!(values[1].is_empty());
        assert_eq!(values[2], vec![Value::Num(2.0)]);
        assert_eq!(values[3], vec![Value::Num(3.0), Value::Num(4.0)]);
    }

    #[test]
    fn distribution_reports_shortage_without_partial_output() {
        let layout = DestinationLayout::new([
            DestinationCardinality::Fixed,
            DestinationCardinality::Sequence(2),
        ])
        .unwrap();
        let error = layout
            .distribute(vec![Value::Num(1.0), Value::Num(2.0)])
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:CommaSeparatedListOutputShortage")
        );
        assert!(error
            .to_string()
            .contains("exactly one value per destination element (expected 3, received 2)"));
    }
}
