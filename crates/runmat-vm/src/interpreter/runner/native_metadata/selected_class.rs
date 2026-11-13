use super::*;

#[path = "selected_class/product.rs"]
mod product;
pub use product::{SelectedClassProduct, SelectedDeclarationProduct};

fn selected_declarations<'a>(
    units: &[&'a crate::AdmittedExecutableUnit],
    selected_class: &runmat_types::ClassIdentity,
) -> VmResult<(Vec<&'a Instr>, HashMap<runmat_types::ClassIdentity, usize>)> {
    let instructions = units.iter().flat_map(|unit| {
        unit.bytecode().instructions.iter().chain(
            unit.registry()
                .functions
                .values()
                .flat_map(|body| body.instructions.iter()),
        )
    });
    selected_declarations_from(instructions, selected_class)
}

fn selected_declarations_from<'a>(
    instructions: impl IntoIterator<Item = &'a Instr>,
    selected_class: &runmat_types::ClassIdentity,
) -> VmResult<(Vec<&'a Instr>, HashMap<runmat_types::ClassIdentity, usize>)> {
    let mut by_class: HashMap<runmat_types::ClassIdentity, Vec<&Instr>> = HashMap::new();
    for instruction in instructions {
        if let Instr::RegisterClass { name, .. } = instruction {
            by_class.entry(name.clone()).or_default().push(instruction);
        }
    }
    let mut selected = Vec::new();
    let mut indices = HashMap::new();
    let mut cursor = Some(selected_class.clone());
    while let Some(class) = cursor {
        if indices.contains_key(&class) {
            return Err(mex(
                "ClassInheritanceCycle",
                "selected class declarations contain an inheritance cycle",
            ));
        }
        let Some(candidates) = by_class.get(&class) else {
            break;
        };
        let declaration = candidates[0];
        if candidates
            .iter()
            .skip(1)
            .any(|candidate| !same_class_registration(declaration, candidate))
        {
            return Err(mex(
                "ClassMetadataConflict",
                "selected class has conflicting declarations",
            ));
        }
        indices.insert(class, selected.len());
        cursor = match declaration {
            Instr::RegisterClass { super_class, .. } => super_class.clone(),
            _ => unreachable!("class metadata index contains only class declarations"),
        };
        selected.push(declaration);
    }
    Ok((selected, indices))
}

/// Publish only the exact selected constructor's class ancestry. Companion
/// declarations in the same compiled source are not dependencies of this call.
pub fn prepare_admitted_execution_metadata_for_class_batch(
    units: &[&crate::AdmittedExecutableUnit],
    selected_class: &runmat_types::ClassIdentity,
) -> VmResult<()> {
    let (declarations, indices) = selected_declarations(units, selected_class)?;
    if declarations.is_empty() {
        return Err(mex(
            "ClassMetadataMissing",
            "selected constructor lacks its class declaration",
        ));
    }
    register_native_class_metadata(declarations, &indices, true)
}

/// The one external parent at the edge of the selected local ancestry, if any.
/// Source resolution uses this identity but does not publish unrelated classes.
pub fn unresolved_admitted_class_parents_for(
    unit: &crate::AdmittedExecutableUnit,
    selected_class: &runmat_types::ClassIdentity,
) -> VmResult<Vec<runmat_types::ClassIdentity>> {
    let (declarations, indices) = selected_declarations(&[unit], selected_class)?;
    let Some(last) = declarations.last() else {
        return Err(mex(
            "ClassMetadataMissing",
            "selected constructor lacks its class declaration",
        ));
    };
    let Instr::RegisterClass { super_class, .. } = last else {
        unreachable!("selected class metadata contains only class declarations")
    };
    Ok(super_class
        .iter()
        .filter(|parent| {
            !indices.contains_key(*parent)
                && !runmat_runtime::class_registry::has_intrinsic_implicit_constructor(parent)
        })
        .cloned()
        .collect())
}
