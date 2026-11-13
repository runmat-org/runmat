//! Split a feval/semantic call's flattened capture prefix using verified
//! Native IR identity, before starting a native invocation.

use super::super::state::NativeCaptureInput;
use super::*;
use runmat_native_codegen::NativeClosureCaptureDisposition;
use runmat_runtime::call::lexical::LexicalCapture;
use runmat_runtime::call::RootedFunctionArguments;

impl NativeExecutor {
    pub(super) fn retained_capture_arguments(
        &self,
        function: FunctionId,
        arguments: &RootedFunctionArguments,
    ) -> NativeExecutorResult<Vec<NativeCaptureInput>> {
        let body = self
            .functions
            .iter()
            .find(|body| body.id == function)
            .ok_or_else(|| {
                NativeExecutorError::Host(format!("native function {} is unavailable", function.0))
            })?;
        if let Some(Value::Closure(closure)) = arguments.callable().map(|owner| owner.value()) {
            if closure
                .bound_function()
                .is_some_and(|bound| bound != function.0)
            {
                return Err(NativeExecutorError::Host(
                    "selected native body differs from retained closure identity".into(),
                ));
            }
        }
        let actual_capture_count = arguments.lexical_capture_count();
        if body.closure_captures.len() != actual_capture_count {
            let carrier = match arguments.callable().map(|owner| owner.value()) {
                Some(Value::Closure(closure)) => format!(
                    "closure '{}' ({:?}, bound={:?})",
                    closure.function_name,
                    closure.source_role,
                    closure.bound_function()
                ),
                Some(_) => "non-closure callable".to_owned(),
                None => "ordinary arguments".to_owned(),
            };
            return Err(NativeExecutorError::Host(format!(
                "retained closure capture count disagrees with selected native body '{}' \
                     (function {}, expected {}, actual {}, carrier {carrier})",
                body.name,
                function.0,
                body.closure_captures.len(),
                actual_capture_count,
            )));
        }
        body.closure_captures
            .iter()
            .enumerate()
            .filter(|(_, capture)| {
                matches!(
                    capture.disposition,
                    NativeClosureCaptureDisposition::ByValue
                        | NativeClosureCaptureDisposition::TransitSnapshot
                )
            })
            .map(|(index, capture)| {
                let value = arguments.capture_lease(index).ok_or_else(|| {
                    NativeExecutorError::Host("retained closure capture is unavailable".into())
                })?;
                Ok(NativeCaptureInput::Retained {
                    binding: capture.binding,
                    local: capture.local,
                    value,
                })
            })
            .collect()
    }

    pub fn split_flattened_capture_arguments(
        &self,
        function: FunctionId,
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
    if arguments.len() < body.closure_captures.len() {
        return Err(NativeExecutorError::Host(
            "closure invocation omitted lexical captures".into(),
        ));
    }
    let mut values = arguments.into_iter();
    let mut captures = Vec::new();
    for capture in &body.closure_captures {
        let value = values.next().ok_or_else(|| {
            NativeExecutorError::Host("closure invocation omitted lexical captures".into())
        })?;
        if !matches!(
            capture.disposition,
            NativeClosureCaptureDisposition::ByValue
                | NativeClosureCaptureDisposition::TransitSnapshot
        ) {
            return Err(NativeExecutorError::Host(
                "legacy flattened closure invocation cannot retain shared capture authority".into(),
            ));
        }
        captures.push(LexicalCapture {
            binding: capture.binding,
            value,
        });
    }
    Ok((captures, values.collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_native_codegen::{
        NativeBlockId, NativeClosureCapture, NativeFunctionAbi, NativeLocalId, NativeLocalKind,
        NativeLocalMetadata,
    };
    use runmat_types::{BindingId, ProgramSourceId};

    fn body() -> runmat_native_codegen::NativeFunction {
        runmat_native_codegen::NativeFunction {
            spawn_safety: runmat_types::SpawnSafetyFact::SpawnSafe,
            lexical_bindings: Vec::new(),
            inherited_storage_bindings: Vec::new(),
            creates_lexical_environment: false,
            id: FunctionId(4),
            portable_id: runmat_types::ProgramFunctionId(14),
            mir_body_digest: runmat_execution::Digest::sha256(b"fixture:native-capture-arguments"),
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
            binding_locals: vec![
                runmat_native_codegen::NativeBindingLocal {
                    binding: BindingId(7),
                    local: NativeLocalId(0),
                },
                runmat_native_codegen::NativeBindingLocal {
                    binding: BindingId(9),
                    local: NativeLocalId(1),
                },
                runmat_native_codegen::NativeBindingLocal {
                    binding: BindingId(10),
                    local: NativeLocalId(2),
                },
            ],
            closure_captures: vec![
                NativeClosureCapture {
                    binding: BindingId(7),
                    local: NativeLocalId(0),
                    disposition: NativeClosureCaptureDisposition::ByValue,
                },
                NativeClosureCapture {
                    binding: BindingId(9),
                    local: NativeLocalId(1),
                    disposition: NativeClosureCaptureDisposition::ByValue,
                },
            ],
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
            expression_continuation_geometry:
                runmat_native_codegen::NativeExpressionContinuationGeometry {
                    region_capacity: 0,
                    retained_input_capacity: 0,
                },
            operation_contracts: vec![],
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
        let mut shared_capture = body();
        shared_capture.closure_captures[1].disposition = NativeClosureCaptureDisposition::Lexical;
        assert!(split_for_function(
            &shared_capture,
            vec![Value::Num(17.0), Value::Num(23.0), Value::Num(5.0)],
        )
        .is_err());
    }
}
