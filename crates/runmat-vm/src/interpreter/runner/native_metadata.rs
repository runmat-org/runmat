use super::*;
use crate::bytecode::program::same_class_registration;
use crate::bytecode::FunctionBytecode;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[path = "native_metadata/selected_class.rs"]
mod selected_class;
pub use selected_class::{
    prepare_admitted_execution_metadata_for_class_batch, unresolved_admitted_class_parents_for,
    SelectedClassProduct, SelectedDeclarationProduct,
};

/// Install immutable bytecode metadata that native execution does not visit as
/// ordinary MIR sites. Core calls this before entering a native unit so class
/// definitions have the same runtime visibility as interpreter execution.
pub fn prepare_native_execution_metadata(bytecode: &Bytecode) -> VmResult<()> {
    prepare_execution_metadata(
        bytecode,
        bytecode
            .function_registry
            .functions
            .iter()
            .map(|(id, function)| (id.0, 0u8, function))
            .chain(
                bytecode
                    .bound_functions
                    .iter()
                    .map(|(id, function)| (id.0, 1u8, function)),
            ),
    )
}

/// A portable admitted product keeps class-bearing functions in its separate
/// authenticated registry component, not in the bytecode component. Install
/// metadata from that exact pair without rebuilding a shadow bytecode.
pub fn prepare_admitted_execution_metadata(unit: &crate::AdmittedExecutableUnit) -> VmResult<()> {
    prepare_execution_metadata(
        unit.bytecode(),
        unit.registry()
            .functions
            .iter()
            .map(|(id, function)| (id.0, 0u8, function)),
    )
}

/// Publish the class metadata of one already-admitted selected product graph
/// as a single checked batch. Source and function admission precede this call;
/// no constructor body is executed by metadata publication.
pub fn prepare_admitted_execution_metadata_batch(
    units: &[&crate::AdmittedExecutableUnit],
) -> VmResult<()> {
    let mut declarations = Vec::new();
    let mut indices = HashMap::new();
    for unit in units {
        collect_native_class_metadata(
            unit.bytecode().instructions.iter(),
            &mut declarations,
            &mut indices,
        )?;
        let mut functions = unit.registry().functions.iter().collect::<Vec<_>>();
        functions.sort_by_key(|(id, _)| id.0);
        for (_, function) in functions {
            collect_native_class_metadata(
                function.instructions.iter(),
                &mut declarations,
                &mut indices,
            )?;
        }
    }
    register_native_class_metadata(declarations, &indices, true)
}

/// Typed external class parents required by an admitted product even when its
/// constructor has no explicit superclass-call instruction. The same metadata
/// collector used for publication establishes the product-local declarations.
pub fn unresolved_admitted_class_parents(
    unit: &crate::AdmittedExecutableUnit,
) -> VmResult<Vec<runmat_types::ClassIdentity>> {
    let mut declarations = Vec::new();
    let mut indices = HashMap::new();
    collect_native_class_metadata(
        unit.bytecode().instructions.iter(),
        &mut declarations,
        &mut indices,
    )?;
    let mut functions = unit.registry().functions.iter().collect::<Vec<_>>();
    functions.sort_by_key(|(id, _)| id.0);
    for (_, function) in functions {
        collect_native_class_metadata(
            function.instructions.iter(),
            &mut declarations,
            &mut indices,
        )?;
    }
    let mut parents = Vec::new();
    for declaration in declarations {
        if let Instr::RegisterClass {
            super_class: Some(parent),
            ..
        } = declaration
        {
            if !indices.contains_key(parent)
                && !runmat_runtime::class_registry::has_intrinsic_implicit_constructor(parent)
                && !parents.contains(parent)
            {
                parents.push(parent.clone());
            }
        }
    }
    Ok(parents)
}

pub(super) fn prepare_execution_metadata<'a>(
    bytecode: &'a Bytecode,
    functions: impl Iterator<Item = (usize, u8, &'a FunctionBytecode)>,
) -> VmResult<()> {
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
    let mut functions = functions.collect::<Vec<_>>();
    functions.sort_by_key(|(id, source, _)| (*id, *source));
    for (_, _, function) in functions {
        collect_native_class_metadata(
            function.instructions.iter(),
            &mut declarations,
            &mut indices,
        )?;
    }
    register_native_class_metadata(declarations, &indices, false)
}

fn collect_native_class_metadata<'a>(
    instructions: impl IntoIterator<Item = &'a Instr>,
    declarations: &mut Vec<&'a Instr>,
    indices: &mut HashMap<runmat_types::ClassIdentity, usize>,
) -> VmResult<()> {
    for instruction in instructions {
        if let Instr::RegisterClass { name, .. } = instruction {
            if let Some(&index) = indices.get(name) {
                if !same_class_registration(declarations[index], instruction) {
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

fn register_native_class_metadata(
    declarations: Vec<&Instr>,
    indices: &HashMap<runmat_types::ClassIdentity, usize>,
    closed: bool,
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
    let batch = order
        .into_iter()
        .filter_map(|index| checked_class_instruction(declarations[index]))
        .collect::<Vec<_>>();
    if batch.is_empty() {
        return Ok(());
    }
    if closed {
        runmat_runtime::class_registry::register_checked_classes(batch)
    } else {
        runmat_runtime::class_registry::register_open_class_declarations(batch)
    }
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
    if let Some(checked) = checked_class_instruction(instruction) {
        runmat_runtime::class_registry::register_open_class_declarations(vec![checked])?;
    }
    Ok(())
}

fn checked_class_instruction(
    instruction: &Instr,
) -> Option<runmat_runtime::class_registry::CheckedClassRegistration> {
    if let Instr::RegisterClass {
        name,
        super_class,
        is_sealed,
        is_abstract,
        properties,
        methods,
        events,
        enumerations,
    } = instruction
    {
        Some(interp_dispatch::checked_class_registration(
            crate::object::class_def::ClassRegistration {
                name: name.clone(),
                super_class: super_class.clone(),
                is_sealed: *is_sealed,
                is_abstract: *is_abstract,
                properties: properties.clone(),
                methods: methods.clone(),
                events: events.clone(),
                enumerations: enumerations.clone(),
            },
        ))
    } else {
        None
    }
}
