use super::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Install immutable bytecode metadata that native execution does not visit as
/// ordinary MIR sites. Core calls this before entering a native unit so class
/// definitions have the same runtime visibility as interpreter execution.
pub fn prepare_native_execution_metadata(bytecode: &Bytecode) -> VmResult<()> {
    // A function-only source has no synthetic script entrypoint. Bound
    // functions can also be imported or assembled from distinct sources, so
    // neither the first function nor the presence of a top-level declaration
    // proves that all class metadata has been published. Preserve source order
    // in each stream and visit both the executable registry and bound-function
    // product by semantic FunctionId; either can hold a distinct merged class.
    let mut declarations = Vec::new();
    let mut indices = HashMap::new();
    collect_native_class_metadata(
        bytecode.instructions.iter(),
        &mut declarations,
        &mut indices,
    )?;
    let mut functions = bytecode
        .function_registry
        .functions
        .iter()
        .map(|(id, function)| (id.0, 0u8, function))
        .chain(
            bytecode
                .bound_functions
                .iter()
                .map(|(id, function)| (id.0, 1u8, function)),
        )
        .collect::<Vec<_>>();
    functions.sort_by_key(|(id, source, _)| (*id, *source));
    for (_, _, function) in functions {
        collect_native_class_metadata(
            function.instructions.iter(),
            &mut declarations,
            &mut indices,
        )?;
    }
    register_native_class_metadata(declarations, &indices)
}

fn collect_native_class_metadata<'a>(
    instructions: impl IntoIterator<Item = &'a Instr>,
    declarations: &mut Vec<&'a Instr>,
    indices: &mut HashMap<runmat_types::ClassIdentity, usize>,
) -> VmResult<()> {
    for instruction in instructions {
        if let Instr::RegisterClass { name, .. } = instruction {
            if let Some(&index) = indices.get(name) {
                if !same_class_declaration(declarations[index], instruction) {
                    return Err(mex(
                        "ClassMetadataConflict",
                        &format!(
                            "native bytecode has conflicting declarations for class '{}'",
                            name.display_name()
                        ),
                    ));
                }
            } else {
                indices.insert(name.clone(), declarations.len());
                declarations.push(instruction);
            }
        }
    }
    Ok(())
}

fn same_class_declaration(first: &Instr, second: &Instr) -> bool {
    let (
        Instr::RegisterClass {
            name: first_name,
            super_class: first_parent,
            is_sealed: first_sealed,
            is_abstract: first_abstract,
            properties: first_properties,
            methods: first_methods,
            enumerations: first_enumerations,
        },
        Instr::RegisterClass {
            name: second_name,
            super_class: second_parent,
            is_sealed: second_sealed,
            is_abstract: second_abstract,
            properties: second_properties,
            methods: second_methods,
            enumerations: second_enumerations,
        },
    ) = (first, second)
    else {
        return false;
    };
    first_name == second_name
        && first_parent == second_parent
        && first_sealed == second_sealed
        && first_abstract == second_abstract
        && first_enumerations == second_enumerations
        && first_properties == second_properties
        && first_methods == second_methods
}

fn register_native_class_metadata(
    declarations: Vec<&Instr>,
    indices: &HashMap<runmat_types::ClassIdentity, usize>,
) -> VmResult<()> {
    // Dependency edges apply only to declarations in this bytecode product;
    // an external superclass retains the existing registry lookup semantics.
    // Stable ordinal selection otherwise preserves top-level and FunctionId
    // order, without publishing a class twice from repeated function prologues.
    let mut dependents = vec![Vec::new(); declarations.len()];
    let mut indegree = vec![0usize; declarations.len()];
    for (child, instruction) in declarations.iter().enumerate() {
        if let Instr::RegisterClass {
            super_class: Some(parent),
            ..
        } = instruction
        {
            if let Some(&parent_index) = indices.get(parent) {
                dependents[parent_index].push(child);
                indegree[child] += 1;
            }
        }
    }
    let mut ready = BinaryHeap::new();
    for (index, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            ready.push(Reverse(index));
        }
    }
    let mut order = Vec::with_capacity(declarations.len());
    while let Some(Reverse(index)) = ready.pop() {
        order.push(index);
        for &child in &dependents[index] {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                ready.push(Reverse(child));
            }
        }
    }
    if order.len() != declarations.len() {
        return Err(mex(
            "ClassInheritanceCycle",
            "native class metadata contains a cyclic superclass declaration",
        ));
    }
    for index in order {
        register_class_instruction(declarations[index])?;
    }
    Ok(())
}

pub(super) fn register_class_metadata<'a>(
    instructions: impl IntoIterator<Item = &'a Instr>,
) -> VmResult<()> {
    for instruction in instructions {
        register_class_instruction(instruction)?;
    }
    Ok(())
}

fn register_class_instruction(instruction: &Instr) -> VmResult<()> {
    if let Instr::RegisterClass {
        name,
        super_class,
        is_sealed,
        is_abstract,
        properties,
        methods,
        enumerations,
    } = instruction
    {
        interp_dispatch::handle_register_class(
            name.clone(),
            super_class.clone(),
            *is_sealed,
            *is_abstract,
            properties.clone(),
            methods.clone(),
            enumerations.clone(),
        )?;
    }
    Ok(())
}
