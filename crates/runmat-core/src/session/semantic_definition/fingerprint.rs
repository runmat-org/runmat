//! Complete canonical semantic definition fingerprint shared by publication and invalidation.
use std::collections::{BTreeMap, BTreeSet};

#[derive(serde::Serialize)]
enum SemanticSourceRevision {
    Program(runmat_types::ProgramSourceId),
    Runtime(runmat_execution::Digest),
}

pub(crate) fn fingerprint(
    function: &runmat_vm::FunctionBytecode,
) -> Result<String, serde_json::Error> {
    // Spell out every semantic field so adding a new FunctionBytecode field makes this
    // destructure fail to compile until its invalidation semantics are chosen. HashMap and
    // HashSet iteration order is not stable, so canonicalize those fields explicitly.
    let runmat_vm::FunctionBytecode {
        function,
        definition_owner: _,
        display_name,
        class_method_owner,
        private_owner_scope,
        source_id,
        // Derived display provenance is assigned after unchanged-definition
        // matching. The source reference is normalized below without using
        // the portable serializer on session-issued authority.
        source_file: _,
        source_role,
        portable_function: _,
        binding_origin: _,
        capabilities,
        instructions,
        catch_scopes,
        instr_spans,
        call_arg_spans,
        coverage_sites,
        var_count,
        input_slots,
        varargin_slot,
        implicit_nargin_slot,
        output_slots,
        varargout_slot,
        implicit_nargout_slot,
        capture_slots,
        var_names,
        initially_unassigned_slots,
        argument_validations,
        resume_points,
        regions,
        parfor_regions,
        spmd_regions,
        distributed_values,
        collective_contracts,
    } = function;
    let var_names = var_names.iter().collect::<BTreeMap<_, _>>();
    let initially_unassigned_slots = initially_unassigned_slots.iter().collect::<BTreeSet<_>>();
    let resume_points = resume_points.iter().collect::<Vec<_>>();
    let mut payload = Vec::new();
    macro_rules! append {
        ($value:expr) => {{
            let bytes = serde_json::to_vec($value)?;
            payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            payload.extend_from_slice(&bytes);
        }};
    }
    append!(function);
    append!(display_name);
    append!(class_method_owner);
    append!(private_owner_scope);
    append!(&source_id.map(|source| match source {
        runmat_vm::BytecodeSourceRef::Program(id) => SemanticSourceRevision::Program(id),
        runmat_vm::BytecodeSourceRef::Runtime(reference) => {
            SemanticSourceRevision::Runtime(reference.semantic_content_revision())
        }
    }));
    append!(source_role);
    append!(capabilities);
    append!(instructions);
    append!(catch_scopes);
    append!(instr_spans);
    append!(call_arg_spans);
    append!(coverage_sites);
    append!(var_count);
    append!(input_slots);
    append!(varargin_slot);
    append!(implicit_nargin_slot);
    append!(output_slots);
    append!(varargout_slot);
    append!(implicit_nargout_slot);
    append!(capture_slots);
    append!(&var_names);
    append!(&initially_unassigned_slots);
    append!(argument_validations);
    append!(&resume_points);
    append!(regions);
    append!(parfor_regions);
    append!(spmd_regions);
    append!(distributed_values);
    append!(collective_contracts);
    Ok(runmat_execution::Digest::sha256(&payload).to_string())
}
