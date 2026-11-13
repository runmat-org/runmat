use runmat_types::{standard, ClassIdentity};
use runmat_value::{CharArray, UnboundedObjectBuilder, Value};
use thiserror::Error;

use super::is_class_or_subclass;

#[derive(Debug, Error)]
pub enum EventDataError {
    #[error("event data source must be a valid handle object")]
    InvalidSource,
    #[error("notification data must be an event.EventData handle object")]
    InvalidPayload,
    #[error("event data target is not an object")]
    InvalidTarget,
    #[error("event source root failed")]
    SourceRoot(#[source] runmat_gc::GcError),
    #[error("event data root failed")]
    DataRoot(#[source] runmat_gc::GcError),
    #[error("event data allocation failed")]
    Allocation(#[source] runmat_gc::GcError),
    #[error("event data binding failed")]
    Binding(#[source] runmat_gc::GcError),
}

impl EventDataError {
    pub fn is_admission_failure(&self) -> bool {
        matches!(
            self,
            Self::InvalidSource | Self::InvalidPayload | Self::InvalidTarget
        )
    }
}

#[derive(Debug)]
pub struct PreparedEventData {
    value: Value,
    _source_root: runmat_gc::ExplicitRoot,
    _data_root: runmat_gc::ExplicitRoot,
}

impl PreparedEventData {
    pub fn value(&self) -> &Value {
        &self.value
    }
}

/// The documented event data hierarchy is handle-only. A plain object with a
/// matching class spelling is not an instance admitted to `notify`.
pub fn is_event_data_value(value: &Value) -> bool {
    match value {
        Value::HandleObject(handle) => {
            crate::is_handle_valid(handle)
                && runmat_gc::gc_with_value(&handle.target, |value| {
                    let Value::Object(object) = value else {
                        return false;
                    };
                    if object.class_name() != &handle.class_name
                        || object.has_native_payload() && object.selected_class().is_some()
                    {
                        return false;
                    }
                    if let Some(facts) = object.selected_class() {
                        return facts.is_handle_class()
                            && super::checked_receiver_projection(object)
                                .ok()
                                .flatten()
                                .is_some_and(|receiver| {
                                    receiver.class_is_or_subclass(
                                        facts.class(),
                                        &standard::EVENT_DATA.owned(),
                                    )
                                });
                    }
                    is_class_or_subclass(&handle.class_name, &standard::EVENT_DATA.owned())
                })
                .unwrap_or(false)
        }
        _ => false,
    }
}

/// The zero-argument EventData constructor is represented by ordinary class
/// construction. Notification fills its two private-set standard properties.
pub fn default_event_data(
    source: &Value,
    event_name: &str,
) -> Result<PreparedEventData, EventDataError> {
    let Value::HandleObject(source_handle) = source else {
        return Err(EventDataError::InvalidSource);
    };
    if !crate::is_handle_valid(source_handle) {
        return Err(EventDataError::InvalidSource);
    }
    let _source_root =
        runmat_gc::gc_root(source_handle.target).map_err(EventDataError::SourceRoot)?;
    let class_name: ClassIdentity = standard::EVENT_DATA.owned();
    let mut instance = UnboundedObjectBuilder::new(class_name.clone());
    instance.insert("Source".into(), source.clone());
    instance.insert(
        "EventName".into(),
        Value::CharArray(CharArray::new_row(event_name)),
    );
    let data_root = runmat_gc::gc_allocate_rooted(Value::Object(instance.finish()))
        .map_err(EventDataError::Allocation)?;
    let value = Value::HandleObject(runmat_value::HandleRef {
        class_name: class_name.into(),
        target: data_root.handle(),
        valid: true,
    });
    Ok(PreparedEventData {
        value,
        _source_root,
        _data_root: data_root,
    })
}

/// Admit one supplied EventData instance, or construct the standard data
/// object, and bind the notification metadata before callback effects.
pub fn prepare_event_data(
    source: &Value,
    event_name: &str,
    supplied: Option<Value>,
) -> Result<PreparedEventData, EventDataError> {
    let Some(data) = supplied else {
        return default_event_data(source, event_name);
    };
    let Value::HandleObject(source_handle) = source else {
        return Err(EventDataError::InvalidSource);
    };
    if !crate::is_handle_valid(source_handle) || !is_event_data_value(&data) {
        return Err(EventDataError::InvalidPayload);
    }
    let Value::HandleObject(data_handle) = &data else {
        unreachable!("validated EventData is a handle object");
    };
    let _source_root =
        runmat_gc::gc_root(source_handle.target).map_err(EventDataError::SourceRoot)?;
    let _data_root = runmat_gc::gc_root(data_handle.target).map_err(EventDataError::DataRoot)?;
    runmat_gc::gc_with_value_mut(&data_handle.target, |value| {
        let Value::Object(_) = value else {
            return Err(EventDataError::InvalidTarget);
        };
        let replacement = std::mem::replace(value, Value::Num(0.0));
        let Value::Object(object) = replacement else {
            unreachable!("target was validated as an object")
        };
        let mut object = object.into_unbounded_builder().map_err(|object| {
            *value = Value::Object(object);
            EventDataError::InvalidTarget
        })?;
        object.insert("Source".into(), source.clone());
        object.insert(
            "EventName".into(),
            Value::CharArray(CharArray::new_row(event_name)),
        );
        *value = Value::Object(object.finish());
        Ok(())
    })
    .map_err(EventDataError::Binding)??;
    Ok(PreparedEventData {
        value: data,
        _source_root,
        _data_root,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_registry::{register_class, RuntimeClass};
    use runmat_value::ObjectInstance;
    use std::error::Error;

    #[test]
    fn data_errors_preserve_gc_source_and_admission_phase() {
        let cause = runmat_gc::GcError::OutOfMemory("event data fixture".into());
        let error = EventDataError::Allocation(cause);
        assert!(!error.is_admission_failure());
        assert!(error
            .source()
            .and_then(|source| source.downcast_ref::<runmat_gc::GcError>())
            .is_some());
        let runtime = crate::build_runtime_error("event data allocation failed")
            .with_identifier("RunMat:NotifyArgumentLifetime")
            .with_source(error)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        assert!(runtime
            .source()
            .and_then(|source| source.downcast_ref::<EventDataError>())
            .and_then(Error::source)
            .and_then(|source| source.downcast_ref::<runmat_gc::GcError>())
            .is_some());
        assert!(EventDataError::InvalidPayload.is_admission_failure());
    }

    #[test]
    fn standard_data_identity_and_subclass_admission() {
        let name = ClassIdentity::from("EventDataDerivedSourceTest");
        register_class(RuntimeClass {
            name: name.clone(),
            parent: Some(standard::EVENT_DATA.owned()),
            properties: Default::default(),
            methods: Default::default(),
            events: Default::default(),
        });
        let derived = runmat_value::HandleRef {
            class_name: name.into(),
            target: runmat_gc::gc_allocate(Value::Object(ObjectInstance::new(
                "EventDataDerivedSourceTest",
            )))
            .expect("derived data"),
            valid: true,
        };
        assert!(is_event_data_value(&Value::HandleObject(derived)));
        assert!(!is_event_data_value(&Value::Object(ObjectInstance::new(
            standard::EVENT_DATA.owned(),
        ))));
        assert!(!is_event_data_value(&Value::Num(1.0)));
        let mismatched = Value::HandleObject(runmat_value::HandleRef {
            class_name: standard::EVENT_DATA.owned().into(),
            target: runmat_gc::gc_allocate(Value::Object(ObjectInstance::new("ForgedData")))
                .expect("forged object"),
            valid: true,
        });
        assert!(!is_event_data_value(&mismatched));
    }

    #[test]
    fn default_data_binds_actual_source_and_event_name() {
        let source_name = ClassIdentity::from("EventDataSourceTest");
        let source = Value::HandleObject(runmat_value::HandleRef {
            class_name: source_name.clone().into(),
            target: runmat_gc::gc_allocate(Value::Object(ObjectInstance::new(source_name)))
                .expect("source"),
            valid: true,
        });
        let data = default_event_data(&source, "Changed").expect("default event data");
        assert!(is_event_data_value(data.value()));
        let Value::HandleObject(handle) = data.value() else {
            panic!("EventData must be a handle object");
        };
        runmat_gc::gc_with_value(&handle.target, |value| {
            let Value::Object(object) = value else {
                panic!("EventData target must be an object");
            };
            assert_eq!(object.property("Source"), Some(&source));
            assert_eq!(
                object.property("EventName"),
                Some(&Value::CharArray(CharArray::new_row("Changed")))
            );
        })
        .expect("live event data");
        assert!(default_event_data(&Value::Num(1.0), "Changed").is_err());
        assert!(prepare_event_data(&source, "Changed", Some(Value::Num(1.0))).is_err());
        let reused =
            prepare_event_data(&source, "Again", Some(Value::HandleObject(handle.clone())))
                .expect("valid supplied data");
        let Value::HandleObject(handle) = reused.value() else {
            panic!("supplied data remains a handle");
        };
        runmat_gc::gc_with_value(&handle.target, |value| {
            let Value::Object(object) = value else {
                panic!("supplied data target");
            };
            assert_eq!(object.property("Source"), Some(&source));
            assert_eq!(
                object.property("EventName"),
                Some(&Value::CharArray(CharArray::new_row("Again")))
            );
        })
        .expect("live supplied data");
    }
}
