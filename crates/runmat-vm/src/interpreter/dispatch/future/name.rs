//! Selected named handle creation; the parent owns admission.
use super::*;

pub(super) async fn dispatch(
    mut meta: DispatchMeta<'_>,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    context: &crate::bytecode::program::ExecutionContext,
) -> Result<(), RuntimeError> {
    let _call_arg_spans = meta.call_arg_spans.take();
    use runmat_runtime::user_functions::{
        DynamicFunctionLoadPhase, RuntimeNameCandidate, RuntimeNameFilter, RuntimeNameQuery,
        RuntimeNameRoute,
    };

    let (name, external) = match meta.instr {
        Instr::CreateFunctionHandle(name) => (name, false),
        Instr::CreateExternalFunctionHandle(name) => (name, true),
        _ => unreachable!("name-handle route requires a name-handle opcode"),
    };
    let runtime = context.runtime.clone();
    let selected = runtime
        .clone()
        .scope(async {
            let observed = runmat_runtime::user_functions::observe_runtime_name(
                name,
                RuntimeNameQuery {
                    route: RuntimeNameRoute::NamedFeval,
                    filter: RuntimeNameFilter::Any,
                    all: false,
                },
            )
            .await
            .map_err(|error| {
                runmat_runtime::build_runtime_error(error.to_string())
                    .with_identifier("RunMat:SelectedHandle:NameResolution")
                    .build()
            })?;
            let phase = match observed.selected() {
                Some(RuntimeNameCandidate::RegisteredConstructor(class)) => {
                    Some(DynamicFunctionLoadPhase::Handle {
                        class: Some(class.clone()),
                        selected_source: None,
                    })
                }
                Some(RuntimeNameCandidate::Source(file))
                | Some(RuntimeNameCandidate::ClassFile(file)) => {
                    Some(DynamicFunctionLoadPhase::Handle {
                        class: None,
                        selected_source: Some(file.clone()),
                    })
                }
                _ => None,
            };
            let Some(phase) = phase else {
                return Ok(None);
            };
            let selected = runmat_runtime::user_functions::try_load_and_call_dynamic_function(
                name.clone(),
                runmat_runtime::call::RootedFunctionArguments::ordinary(
                    runmat_runtime::execution::RootedValues::in_context(Vec::new(), &runtime)?,
                ),
                1,
                phase,
                None,
            )
            .await;
            if runtime.search_path().map(|path| path.generation())
                != observed.captured_path_revision
                || runmat_runtime::class_registry::class_registry_generation()
                    != observed.captured_class_revision
            {
                return Err(runmat_runtime::build_runtime_error(
                    "name selection changed while creating a bound handle",
                )
                .with_identifier("RunMat:SelectedHandle:SelectionChanged")
                .build());
            }
            Ok(Some(selected))
        })
        .await?;
    if let Some(std::ops::ControlFlow::Break(result)) = selected {
        let rooted = result?;
        rooted.with_owned(|sequence| {
            if sequence.len() != 1
                || !matches!(
                    sequence.as_slice().first(),
                    Some(Value::BoundFunctionHandle(_))
                )
            {
                return Err(runmat_runtime::build_runtime_error(
                    "selected source did not return one exact bound handle",
                )
                .with_identifier("RunMat:SelectedHandle:InvalidOutput")
                .build());
            }
            let selected = sequence
                .into_output_storage()
                .try_into_first()
                .map_err(|_| {
                    runmat_runtime::sequence::sequence_error_to_runtime(
                        runmat_value::ValueSequenceError::SharedOutputStorage,
                    )
                })?
                .expect("one checked selected handle");
            stack.try_push(selected)?;
            Ok(())
        })?;
    } else if external {
        stack.try_push(Value::external_function_handle(name.clone()))?;
    } else {
        stack.try_push(Value::function_handle(name.clone()))?;
    }
    Ok(())
}
