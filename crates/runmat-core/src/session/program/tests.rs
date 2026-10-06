use super::*;
use crate::{ExecutableSource, RunMatSession};

async fn unit(owner: &str, offset: u8) -> ExecutableUnit {
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let source = ExecutableSource::new(
        owner,
        "entry.m",
        format!("function y = entry(x); y = x + {offset}; end\n"),
    );
    session.compile_executable_unit(source, None).await.unwrap()
}

fn function(unit: &ExecutableUnit) -> ProgramFunctionId {
    ProgramFunctionId(unit.functions().resolve_name("entry").unwrap().0 as u32)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn nested_program_scope_restores_original_callable_and_rejects_expired_authority() {
    let outer = unit("outer-owner", 1).await;
    let inner = unit("inner-owner", 2).await;
    let service = Rc::new(SessionProgramService::default());
    assert!(service
        .freeze_bound_callable(function(&outer), "entry")
        .is_err());
    {
        let _outer = service.enter(&outer);
        let original = service
            .freeze_bound_callable(function(&outer), "entry")
            .unwrap();
        assert_eq!(
            original.executable,
            outer
                .portable_envelope_for_function(function(&outer))
                .unwrap()
        );
        {
            let _inner = service.enter(&inner);
            let active = service
                .freeze_bound_callable(function(&inner), "entry")
                .unwrap();
            assert_eq!(active.owner_identity, "inner-owner");
            assert_ne!(active.source_digest, original.source_digest);
            assert_eq!(
                active.executable,
                inner
                    .portable_envelope_for_function(function(&inner))
                    .unwrap()
            );
        }
        assert_eq!(
            service
                .freeze_bound_callable(function(&outer), "entry")
                .unwrap()
                .executable,
            original.executable
        );
        assert!(service
            .freeze_bound_callable(function(&outer), "foreign-name")
            .is_err());
        assert!(service
            .freeze_bound_callable(ProgramFunctionId(u32::MAX), "entry")
            .is_err());
    }
    assert!(service
        .freeze_bound_callable(function(&outer), "entry")
        .is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn restored_product_freezes_original_analysis_source_and_function_without_recompilation() {
    let original = unit("cold-owner", 7).await;
    let envelope = original.portable_envelope().unwrap();
    let restored = ExecutableUnit::from_portable_envelope(&envelope).unwrap();
    let service = Rc::new(SessionProgramService::default());
    let _scope = service.enter(&restored);
    let captured = service
        .freeze_bound_callable(function(&restored), "entry")
        .unwrap();
    assert_eq!(captured.executable, envelope);
    assert_eq!(
        captured.source_digest,
        runmat_execution::Digest::sha256(original.source().text.as_bytes())
    );
    assert_eq!(
        captured.executable_source_digest.to_string(),
        original.revision().source_digest
    );
    assert_eq!(
        captured.function,
        original
            .function_analysis(function(&original))
            .unwrap()
            .clone()
    );
}

#[cfg(not(target_arch = "wasm32"))]
mod deferred;
