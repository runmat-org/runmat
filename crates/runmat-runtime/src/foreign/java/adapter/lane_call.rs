use super::*;

/// Origin-admitted Java operations. All RunMat values and foreign handles are
/// resolved before this typed, Send-only request reaches the JNI owner.
pub(super) enum JavaLaneCall {
    Construct {
        class: String,
        values: Vec<JavaValue>,
        edt: bool,
    },
    Static {
        class: String,
        method: String,
        values: Vec<JavaValue>,
        edt: bool,
    },
    Qualified {
        name: String,
        values: Vec<JavaValue>,
    },
    Method {
        handle: JavaObjectHandle,
        method: String,
        values: Vec<JavaValue>,
        edt: bool,
    },
    Get {
        handle: JavaObjectHandle,
        field: String,
    },
    Set {
        handle: JavaObjectHandle,
        field: String,
        value: JavaValue,
    },
    AddClasspath {
        entry: String,
        at_end: bool,
    },
    RemoveClasspath {
        entry: String,
    },
    SetClasspath {
        entries: Vec<String>,
    },
    NewArray {
        class: String,
        dimensions: Vec<usize>,
    },
    Classpath,
    Headless,
}

pub(super) enum JavaLaneResult {
    Value(JavaValue),
    Empty,
    Classpath(ClasspathSnapshot),
    Headless(bool),
}

/// The selected target is borrowed from the lane-owned qualified name. It is
/// resolved before the origin marshals arguments or registers callbacks, and
/// remains on this same submitted lane operation until invocation completes.
pub(super) enum QualifiedTarget<'a> {
    Constructor(&'a str),
    Static { class: &'a str, method: &'a str },
}

pub(super) fn select_qualified<'a>(
    owner: &runmat_java::JavaLaneSession<'_>,
    name: &'a str,
) -> Result<QualifiedTarget<'a>, JavaInvocationError> {
    let session = owner.session();
    if session.class_exists(name)? {
        return Ok(QualifiedTarget::Constructor(name));
    }
    let (class, method) = name
        .rsplit_once('.')
        .ok_or(JavaInvocationError::InvalidQualifiedName)?;
    if !session.class_exists(class)? {
        return Err(JavaInvocationError::ClassNotFound(class.into()));
    }
    Ok(QualifiedTarget::Static { class, method })
}

pub(super) fn execute_selected_qualified(
    owner: &runmat_java::JavaLaneSession<'_>,
    target: QualifiedTarget<'_>,
    values: Vec<JavaValue>,
) -> Result<JavaLaneResult, JavaInvocationError> {
    let values = owner.materialize_values(values)?;
    let session = owner.session();
    let result = match target {
        QualifiedTarget::Constructor(name) => session.construct_resolved(name, &values)?,
        QualifiedTarget::Static { class, method } => {
            session.call_static_resolved(class, method, &values)?
        }
    };
    Ok(JavaLaneResult::Value(result))
}

impl JavaLaneCall {
    pub(super) fn execute(
        self,
        owner: &runmat_java::JavaLaneSession<'_>,
    ) -> Result<JavaLaneResult, JavaInvocationError> {
        let session = owner.session();
        match self {
            Self::Construct { class, values, edt } => {
                let values = owner.materialize_values(values)?;
                let result = if edt {
                    session.construct_resolved_on_edt(&class, &values)?
                } else {
                    session.construct_resolved(&class, &values)?
                };
                Ok(JavaLaneResult::Value(result))
            }
            Self::Static {
                class,
                method,
                values,
                edt,
            } => {
                let values = owner.materialize_values(values)?;
                let result = if edt {
                    session.call_static_resolved_on_edt(&class, &method, &values)?
                } else {
                    session.call_static_resolved(&class, &method, &values)?
                };
                Ok(JavaLaneResult::Value(result))
            }
            Self::Qualified { name, values } => {
                let target = select_qualified(owner, &name)?;
                execute_selected_qualified(owner, target, values)
            }
            Self::Method {
                handle,
                method,
                values,
                edt,
            } => {
                let values = owner.materialize_values(values)?;
                let result = if edt {
                    session.call_method_resolved_on_edt(handle, &method, &values)?
                } else {
                    session.call_method_resolved(handle, &method, &values)?
                };
                Ok(JavaLaneResult::Value(result))
            }
            Self::Get { handle, field } => session
                .get_field_resolved(handle, &field)
                .map(JavaLaneResult::Value),
            Self::Set {
                handle,
                field,
                value,
            } => {
                let value = owner
                    .materialize_values(vec![value])?
                    .pop()
                    .expect("one Java assignment");
                session.set_field_resolved(handle, &field, &value)?;
                Ok(JavaLaneResult::Empty)
            }
            Self::AddClasspath { entry, at_end } => {
                session.add_dynamic_classpath_at(entry, at_end)?;
                Ok(JavaLaneResult::Empty)
            }
            Self::RemoveClasspath { entry } => {
                session.remove_dynamic_classpath(std::path::Path::new(&entry))?;
                Ok(JavaLaneResult::Empty)
            }
            Self::SetClasspath { entries } => {
                session
                    .replace_dynamic_classpath(entries.into_iter().map(std::path::PathBuf::from))?;
                Ok(JavaLaneResult::Empty)
            }
            Self::NewArray { class, dimensions } => session
                .new_object_array(&class, &dimensions)
                .map(JavaLaneResult::Value),
            Self::Classpath => Ok(JavaLaneResult::Classpath(session.classpath())),
            Self::Headless => {
                let result = session.call_static_resolved(
                    "java.awt.GraphicsEnvironment",
                    "isHeadless",
                    &[],
                )?;
                let JavaValue::Boolean(headless) = result else {
                    return Err(JavaInvocationError::UnsupportedValue(
                        "Java graphics environment returned an invalid capability value".into(),
                    ));
                };
                Ok(JavaLaneResult::Headless(headless))
            }
        }
    }
}
