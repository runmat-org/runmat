//! V2 captured-image coordinates never substitute for portable function identity.
use super::{captured_program_revision, DeferredCall, DeferredProgram, Digest, ProgramRevision};
use runmat_types::{FunctionId, ProgramFunctionId};

/// Original source revisions qualify origins. Older V2 embedding products
/// instead bind their complete captured image by content; that digest is not
/// a source revision and must never be compared with a body's original origin.
pub(super) fn revision_owner(
    revision: &ProgramRevision,
    image: &[u8],
) -> Result<Option<Digest>, String> {
    if revision == &captured_program_revision(image) {
        Ok(None)
    } else {
        revision
            .identity_digest()
            .map(Some)
            .map_err(|error| error.to_string())
    }
}

pub(super) fn resolve(
    registry: &crate::FunctionRegistry,
    portable: ProgramFunctionId,
    origin_program: Option<Digest>,
) -> Result<FunctionId, String> {
    let mut selected = None;
    for (local, body) in &registry.functions {
        if body.portable_function != portable {
            continue;
        }
        if body.function != *local {
            return Err(
                "captured function body disagrees with its local registry coordinate".into(),
            );
        }
        if let Some(origin) = body.binding_origin {
            if origin.function != portable {
                return Err("captured function body disagrees with its portable origin".into());
            }
            if origin_program.is_some_and(|program| program != origin.program) {
                continue;
            }
        } else if origin_program.is_some() {
            continue;
        }
        if selected.replace(*local).is_some() {
            return Err("captured portable function has ambiguous original bodies".into());
        }
    }
    selected.ok_or_else(|| "captured portable function has no matching original body".into())
}

pub(super) fn captured_function(
    call: &DeferredCall,
    local: FunctionId,
) -> Result<ProgramFunctionId, String> {
    let DeferredProgram::InterpreterV2(image) = &call.program else {
        return Err("semantic callable requires its exact captured interpreter program".into());
    };
    let registry =
        crate::decode_interpreter_program_v2(image).map_err(|error| error.to_string())?;
    let body = registry
        .get(local)
        .ok_or("semantic callable is absent from its captured program")?;
    let origin_program = call
        .program_revision
        .as_ref()
        .map(|revision| revision_owner(revision, image))
        .transpose()?
        .flatten();
    let portable = body.portable_function;
    if resolve(&registry, portable, origin_program)? != local {
        return Err("semantic callable differs from its original captured body".into());
    }
    Ok(portable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FunctionBytecode;
    use runmat_execution::FunctionBindingOrigin;
    use std::collections::HashMap;

    fn body(
        local: usize,
        portable: u32,
        program: Option<Digest>,
    ) -> (FunctionId, FunctionBytecode) {
        let function = FunctionId(local);
        let portable_function = ProgramFunctionId(portable);
        (
            function,
            FunctionBytecode {
                function,
                portable_function,
                binding_origin: program.map(|program| FunctionBindingOrigin {
                    program,
                    function: portable_function,
                }),
                ..Default::default()
            },
        )
    }

    #[test]
    fn captured_portable_owner_is_not_a_local_ordinal_or_name() {
        let program = Digest::sha256(b"original-revision");
        let other = Digest::sha256(b"other-revision");
        let registry = crate::FunctionRegistry::new(HashMap::from([
            body(77, 7, Some(program)),
            body(7, 8, Some(program)),
            body(88, 7, Some(other)),
        ]));
        assert_eq!(
            resolve(&registry, ProgramFunctionId(7), Some(program)).unwrap(),
            FunctionId(77)
        );
        assert_eq!(
            resolve(&registry, ProgramFunctionId(7), Some(other)).unwrap(),
            FunctionId(88)
        );
        assert!(resolve(&registry, ProgramFunctionId(7), None).is_err());
        assert!(resolve(&registry, ProgramFunctionId(77), Some(program)).is_err());
        let duplicates = crate::FunctionRegistry::new(HashMap::from([
            body(77, 7, Some(program)),
            body(88, 7, Some(program)),
        ]));
        assert!(resolve(&duplicates, ProgramFunctionId(7), Some(program)).is_err());
    }

    #[test]
    fn captured_image_accepts_originless_original_coordinates_but_not_forged_origin() {
        let registry = crate::FunctionRegistry::new(HashMap::from([body(77, 7, None)]));
        assert_eq!(
            resolve(&registry, ProgramFunctionId(7), None).unwrap(),
            FunctionId(77)
        );
        assert!(resolve(
            &registry,
            ProgramFunctionId(7),
            Some(Digest::sha256(b"source revision"))
        )
        .is_err());
        let image = b"exact legacy captured image";
        assert_eq!(
            revision_owner(&captured_program_revision(image), image).unwrap(),
            None
        );
        let (local, mut malformed) = body(77, 7, Some(Digest::sha256(b"revision")));
        malformed.binding_origin.as_mut().unwrap().function = ProgramFunctionId(77);
        let registry = crate::FunctionRegistry::new(HashMap::from([(local, malformed)]));
        assert!(resolve(&registry, ProgramFunctionId(7), None).is_err());
    }
}
