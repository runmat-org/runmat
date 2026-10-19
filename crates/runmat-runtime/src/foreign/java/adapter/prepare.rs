use super::lane_call::JavaLaneCall;
use super::*;

pub(super) enum JavaOutput {
    Value,
    Assigned(ForeignRef),
    Empty,
    Classpath(Vec<Value>),
    HasClasspath,
}

pub(super) struct PreparedJavaCall {
    pub(super) command: JavaLaneCall,
    pub(super) output: JavaOutput,
}

impl JavaAdapter {
    pub(super) fn prepare_java_call(
        &self,
        context: &RuntimeContext,
        call: ForeignCall,
    ) -> Result<PreparedJavaCall, RuntimeError> {
        let edt = matches!(
            call.symbol.as_str(),
            "construct_edt" | "call_static_edt" | "invoke_member_edt"
        );
        if edt && !self.desktop_available.get() {
            return Err(build_runtime_error(
                "Java EDT execution requires a Desktop host with Java UI support",
            )
            .with_builtin("java")
            .with_identifier("RunMat:Java:EdtUnavailable")
            .build());
        }
        let mut arguments = call.arguments.into_iter();
        let (command, output) = match call.symbol.as_str() {
            "construct" | "construct_edt" => {
                let class = string_argument(arguments.next(), "Java class")?;
                let values = self.java_arguments(context, arguments)?;
                (
                    JavaLaneCall::Construct { class, values, edt },
                    JavaOutput::Value,
                )
            }
            "call_static" | "call_static_edt" => {
                let class = string_argument(arguments.next(), "Java class")?;
                let method = string_argument(arguments.next(), "Java method")?;
                let values = self.java_arguments(context, arguments)?;
                (
                    JavaLaneCall::Static {
                        class,
                        method,
                        values,
                        edt,
                    },
                    JavaOutput::Value,
                )
            }
            "invoke_qualified" => {
                let name = string_argument(arguments.next(), "qualified Java name")?;
                let values = self.java_arguments(context, arguments)?;
                (JavaLaneCall::Qualified { name, values }, JavaOutput::Value)
            }
            "invoke_member" | "invoke_member_edt" => {
                let reference = foreign_argument(arguments.next())?;
                let method = string_argument(arguments.next(), "Java method")?;
                let handle = self.resolve_java_handle(&reference)?;
                let values = self.java_arguments(context, arguments)?;
                (
                    JavaLaneCall::Method {
                        handle,
                        method,
                        values,
                        edt,
                    },
                    JavaOutput::Value,
                )
            }
            "get_member" => {
                let reference = foreign_argument(arguments.next())?;
                let field = string_argument(arguments.next(), "Java field")?;
                let handle = self.resolve_java_handle(&reference)?;
                (JavaLaneCall::Get { handle, field }, JavaOutput::Value)
            }
            "set_member" => {
                let reference = foreign_argument(arguments.next())?;
                let field = string_argument(arguments.next(), "Java field")?;
                let value = self.argument_to_java(
                    context,
                    arguments.next().ok_or_else(|| {
                        invalid_conversion("Java field assignment requires a value")
                    })?,
                )?;
                let handle = self.resolve_java_handle(&reference)?;
                (
                    JavaLaneCall::Set {
                        handle,
                        field,
                        value,
                    },
                    JavaOutput::Assigned(reference),
                )
            }
            "add_classpath" => {
                let entry = string_argument(arguments.next(), "Java classpath entry")?;
                let position = arguments
                    .next()
                    .map(|value| string_argument(Some(value), "Java classpath position"))
                    .transpose()?;
                let position = match position.as_deref() {
                    None => runmat_builtins::JavaInsertionPosition::Begin,
                    Some(name) => runmat_builtins::JavaInsertionPosition::parse_wire(name)
                        .ok_or_else(|| {
                            invalid_conversion("Java classpath position must be 'begin' or 'end'")
                        })?,
                };
                (
                    JavaLaneCall::AddClasspath {
                        entry,
                        at_end: position == runmat_builtins::JavaInsertionPosition::End,
                    },
                    JavaOutput::Empty,
                )
            }
            "remove_classpath" => {
                let entry = string_argument(arguments.next(), "Java classpath entry")?;
                (JavaLaneCall::RemoveClasspath { entry }, JavaOutput::Empty)
            }
            "set_classpath" => {
                let entries = arguments
                    .map(|value| string_argument(Some(value), "Java classpath entry"))
                    .collect::<Result<Vec<_>, _>>()?;
                (JavaLaneCall::SetClasspath { entries }, JavaOutput::Empty)
            }
            "new_array" => {
                let class = string_argument(arguments.next(), "Java array class")?;
                let dimensions = arguments
                    .map(java_dimension)
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    JavaLaneCall::NewArray { class, dimensions },
                    JavaOutput::Value,
                )
            }
            "classpath" => (
                JavaLaneCall::Classpath,
                JavaOutput::Classpath(arguments.collect()),
            ),
            "has_classpath" => (JavaLaneCall::Classpath, JavaOutput::HasClasspath),
            operation => {
                return Err(foreign_error(
                    ForeignErrorKind::InvalidCall,
                    format!("unsupported Java operation {operation}"),
                ));
            }
        };
        Ok(PreparedJavaCall { command, output })
    }

    fn java_arguments(
        &self,
        context: &RuntimeContext,
        arguments: impl Iterator<Item = Value>,
    ) -> Result<Vec<JavaValue>, RuntimeError> {
        arguments
            .map(|value| self.argument_to_java(context, value))
            .collect()
    }
}
