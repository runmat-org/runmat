//! Late names are selected from the admitted source inventory, not function
//! order. Exact lexical calls and handles retain their separate ID/lease path.

use std::path::Path;

use runmat_execution::ExecutableSourceMapV2;
use runmat_native_codegen::aot::{AotProgramFunction, AotProgramManifest};
use runmat_types::{FunctionId, FunctionSourceRole};

pub(super) fn select(
    program: &AotProgramManifest,
    caller: Option<usize>,
    name: &str,
) -> Option<FunctionId> {
    select_scoped_from(&program.functions, &program.source_map, caller, name)
        .or_else(|| program.resolve_name(name))
}

pub(super) fn select_scoped(
    program: &AotProgramManifest,
    caller: Option<usize>,
    name: &str,
) -> Option<FunctionId> {
    select_scoped_from(&program.functions, &program.source_map, caller, name)
}

fn select_scoped_from(
    functions: &[AotProgramFunction],
    map: &ExecutableSourceMapV2,
    caller: Option<usize>,
    name: &str,
) -> Option<FunctionId> {
    let caller = functions
        .iter()
        .find(|function| Some(function.function.0) == caller)?;
    let caller_source = source(map, caller)?;
    let caller_path = &caller_source.relative_path;
    let visibility_dir = private_parent(caller_path).or_else(|| Path::new(caller_path).parent());

    // A local declaration is visible only inside its own source. Nested
    // bodies are deliberately absent: textual feval must not bypass the
    // lexical-environment/handle authority required for nested calls.
    let local = select_unique(functions.iter().filter(|function| {
        function.name == name
            && function.role == FunctionSourceRole::Local
            && function.source == caller.source
    }));
    if local.is_some() {
        return local;
    }

    // The source map carries the immutable package owner and package-relative
    // path. A private source is visible only to callers in its parent folder
    // of the same selected package, never to a same-spelled foreign source.
    select_unique(functions.iter().filter(|function| {
        if function.name != name || function.role != FunctionSourceRole::Primary {
            return false;
        }
        let Some(candidate) = source(map, function) else {
            return false;
        };
        candidate.owner_identity == caller_source.owner_identity
            && private_parent(&candidate.relative_path) == visibility_dir
    }))
}

fn source<'a>(
    map: &'a ExecutableSourceMapV2,
    function: &AotProgramFunction,
) -> Option<&'a runmat_execution::ExecutableSourceRecordV2> {
    let ordinal = usize::try_from(function.source.0).ok()?;
    map.entries
        .get(ordinal)
        .filter(|source| source.source_id == function.source)
}

fn private_parent(path: &str) -> Option<&Path> {
    let parent = Path::new(path).parent()?;
    if parent.file_name()? != "private" {
        return None;
    }
    parent.parent()
}

fn select_unique<'a>(
    mut matches: impl Iterator<Item = &'a AotProgramFunction>,
) -> Option<FunctionId> {
    let selected = matches.next()?.function;
    matches.next().is_none().then_some(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{Digest, ExecutableSourceRecordV2};
    use runmat_types::ProgramSourceId;

    fn source_record(ordinal: u32, owner: &str, path: &str) -> ExecutableSourceRecordV2 {
        ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(ordinal),
            owner_identity: owner.into(),
            relative_path: path.into(),
            content_revision: Digest::sha256(b"function body"),
            display_name: path.into(),
            text: "function body".into(),
        }
    }

    fn function(
        ordinal: u32,
        source: u32,
        name: &str,
        role: FunctionSourceRole,
    ) -> AotProgramFunction {
        AotProgramFunction {
            function: FunctionId(ordinal as usize),
            portable_function: runmat_types::ProgramFunctionId(ordinal),
            source: ProgramSourceId(source),
            name: name.into(),
            role,
        }
    }

    #[test]
    fn private_and_local_names_follow_caller_source_not_manifest_order() {
        let map = ExecutableSourceMapV2 {
            entries: vec![
                source_record(0, "package-a", "lib/caller.m"),
                source_record(1, "package-b", "lib/private/helper.m"),
                source_record(2, "package-a", "lib/private/helper.m"),
                source_record(3, "package-a", "other/caller.m"),
                source_record(4, "package-a", "lib/private/private_caller.m"),
            ],
        };
        let functions = vec![
            function(0, 0, "caller", FunctionSourceRole::Primary),
            function(1, 1, "helper", FunctionSourceRole::Primary),
            function(2, 2, "helper", FunctionSourceRole::Primary),
            function(3, 0, "local", FunctionSourceRole::Local),
            function(4, 3, "other", FunctionSourceRole::Primary),
            function(5, 4, "private_caller", FunctionSourceRole::Primary),
        ];
        assert_eq!(
            select_scoped_from(&functions, &map, Some(0), "helper"),
            Some(FunctionId(2))
        );
        assert_eq!(
            select_scoped_from(&functions, &map, Some(0), "local"),
            Some(FunctionId(3))
        );
        assert_eq!(
            select_scoped_from(&functions, &map, Some(4), "helper"),
            None
        );
        assert_eq!(select_scoped_from(&functions, &map, Some(4), "local"), None);
        assert_eq!(
            select_scoped_from(&functions, &map, Some(5), "helper"),
            Some(FunctionId(2))
        );
    }

    #[test]
    fn textual_nested_name_is_not_scoped_even_for_its_parent() {
        let map = ExecutableSourceMapV2 {
            entries: vec![source_record(0, "package-a", "caller.m")],
        };
        let functions = [
            function(0, 0, "caller", FunctionSourceRole::Primary),
            function(1, 0, "nested", FunctionSourceRole::Nested),
        ];
        assert_eq!(
            select_scoped_from(&functions, &map, Some(0), "nested"),
            None
        );
    }
}
