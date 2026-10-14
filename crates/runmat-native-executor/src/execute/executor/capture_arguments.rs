//! Split a feval/semantic call's flattened capture prefix using verified
//! Native IR identity, before starting a native invocation.

use super::*;
use runmat_native_codegen::NativeLocalKind;
use runmat_runtime::call::lexical::LexicalCapture;

impl NativeExecutor {
    pub fn split_flattened_capture_arguments(
        &self,
        function: ProgramFunctionId,
        arguments: Vec<Value>,
    ) -> NativeExecutorResult<(Vec<LexicalCapture>, Vec<Value>)> {
        let body = self
            .functions
            .iter()
            .find(|body| body.id == function)
            .ok_or_else(|| {
                NativeExecutorError::Host(format!("native function {} is unavailable", function.0))
            })?;
        split_for_function(body, arguments)
    }
}

fn split_for_function(
    body: &runmat_native_codegen::NativeFunction,
    arguments: Vec<Value>,
) -> NativeExecutorResult<(Vec<LexicalCapture>, Vec<Value>)> {
    let bindings = body
        .locals
        .iter()
        .filter(|local| local.kind == NativeLocalKind::Capture)
        .map(|local| {
            local.binding.ok_or_else(|| {
                NativeExecutorError::Host("native capture local has no semantic binding".into())
            })
        })
        .collect::<NativeExecutorResult<Vec<_>>>()?;
    if arguments.len() < bindings.len() {
        return Err(NativeExecutorError::Host(
            "closure invocation omitted lexical captures".into(),
        ));
    }
    let mut values = arguments.into_iter();
    let captures = bindings
        .into_iter()
        .map(|binding| LexicalCapture {
            binding,
            value: values.next().expect("capture prefix was length-checked"),
        })
        .collect();
    Ok((captures, values.collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_native_codegen::{
        NativeBlockId, NativeFunctionAbi, NativeLocalId, NativeLocalMetadata,
    };
    use runmat_types::{BindingId, ProgramSourceId};

    fn body() -> runmat_native_codegen::NativeFunction {
        runmat_native_codegen::NativeFunction {
            lexical_bindings: Vec::new(),
            inherited_storage_bindings: Vec::new(),
            creates_lexical_environment: false,
            id: ProgramFunctionId(4),
            source: ProgramSourceId(0),
            name: "nested".into(),
            source_role: runmat_types::FunctionSourceRole::Unknown,
            lexical_parent: None,
            class_method_owner: None,
            capabilities: Default::default(),
            abi: NativeFunctionAbi {
                fixed_inputs: vec![NativeLocalId(2)],
                varargin: None,
                fixed_outputs: vec![],
                varargout: None,
                implicit_nargin: None,
                implicit_nargout: None,
            },
            argument_validations: vec![],
            locals: vec![
                NativeLocalMetadata {
                    id: NativeLocalId(0),
                    binding: Some(BindingId(7)),
                    name: None,
                    kind: NativeLocalKind::Capture,
                },
                NativeLocalMetadata {
                    id: NativeLocalId(1),
                    binding: Some(BindingId(9)),
                    name: None,
                    kind: NativeLocalKind::Capture,
                },
                NativeLocalMetadata {
                    id: NativeLocalId(2),
                    binding: Some(BindingId(10)),
                    name: None,
                    kind: NativeLocalKind::Parameter,
                },
            ],
            entry: NativeBlockId(0),
            blocks: vec![],
            expected_sites: vec![],
        }
    }

    #[test]
    fn native_capture_layout_splits_typed_prefix_without_guessing_names() {
        let (captures, arguments) = split_for_function(
            &body(),
            vec![Value::Num(17.0), Value::Num(23.0), Value::Num(5.0)],
        )
        .expect("verified capture layout");
        assert_eq!(
            captures,
            vec![
                LexicalCapture {
                    binding: BindingId(7),
                    value: Value::Num(17.0)
                },
                LexicalCapture {
                    binding: BindingId(9),
                    value: Value::Num(23.0)
                },
            ]
        );
        assert_eq!(arguments, [Value::Num(5.0)]);
        assert!(split_for_function(&body(), vec![Value::Num(17.0)]).is_err());
        let mut missing_binding = body();
        missing_binding.locals[1].binding = None;
        assert!(split_for_function(
            &missing_binding,
            vec![Value::Num(17.0), Value::Num(23.0), Value::Num(5.0)],
        )
        .is_err());
    }
}
