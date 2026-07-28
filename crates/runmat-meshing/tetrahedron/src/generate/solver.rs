use runmat_meshing_core::contracts::ProtectedBoundaryComplex;

use super::{
    generate_convex_polyhedron_tetrahedron_mesh_from_plc,
    generate_holed_polyhedron_tetrahedron_mesh_from_plc,
    generate_nested_tetrahedron_shell_tetrahedron_mesh_from_plc,
    generate_single_tetrahedron_mesh_from_plc,
    generate_star_shaped_polyhedron_tetrahedron_mesh_from_plc,
    generate_structured_box_tetrahedron_mesh_from_plc, TetrahedronGenerationError, TetrahedronMesh,
};

pub fn generate_solver_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    let mut attempted_families = Vec::<&'static str>::new();
    let mut rejected_families = Vec::<(&'static str, &'static str)>::new();

    attempted_families.push("nested_tetrahedron_shell");
    match generate_nested_tetrahedron_shell_tetrahedron_mesh_from_plc(plc) {
        Ok(mut mesh) => {
            record_solver_family_selection(
                &mut mesh,
                "nested_tetrahedron_shell",
                &attempted_families,
                &rejected_families,
            );
            return Ok(mesh);
        }
        Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc) => rejected_families
            .push((
                "nested_tetrahedron_shell",
                "unsupported_nested_tetrahedron_shell_plc",
            )),
        Err(err) => return Err(err),
    }

    attempted_families.push("structured_box");
    match generate_structured_box_tetrahedron_mesh_from_plc(plc) {
        Ok(mut mesh) => {
            record_solver_family_selection(
                &mut mesh,
                "structured_box",
                &attempted_families,
                &rejected_families,
            );
            Ok(mesh)
        }
        Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc) => {
            rejected_families.push(("structured_box", "unsupported_structured_box_plc"));

            attempted_families.push("single_tetrahedron");
            match generate_single_tetrahedron_mesh_from_plc(plc) {
                Ok(mut mesh) => {
                    record_solver_family_selection(
                        &mut mesh,
                        "single_tetrahedron",
                        &attempted_families,
                        &rejected_families,
                    );
                    Ok(mesh)
                }
                Err(TetrahedronGenerationError::UnsupportedSingleTetrahedronPlc) => {
                    rejected_families
                        .push(("single_tetrahedron", "unsupported_single_tetrahedron_plc"));

                    attempted_families.push("convex_polyhedron");
                    match generate_convex_polyhedron_tetrahedron_mesh_from_plc(plc) {
                        Ok(mut mesh) => {
                            record_solver_family_selection(
                                &mut mesh,
                                "convex_polyhedron",
                                &attempted_families,
                                &rejected_families,
                            );
                            Ok(mesh)
                        }
                        Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc) => {
                            rejected_families
                                .push(("convex_polyhedron", "unsupported_convex_polyhedron_plc"));

                            attempted_families.push("holed_polyhedron");
                            match generate_holed_polyhedron_tetrahedron_mesh_from_plc(plc) {
                                Ok(mut mesh) => {
                                    record_solver_family_selection(
                                        &mut mesh,
                                        "holed_polyhedron",
                                        &attempted_families,
                                        &rejected_families,
                                    );
                                    Ok(mesh)
                                }
                                Err(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc) => {
                                    rejected_families.push((
                                        "holed_polyhedron",
                                        "unsupported_holed_polyhedron_plc",
                                    ));

                                    attempted_families.push("star_shaped_polyhedron");
                                    generate_star_shaped_polyhedron_tetrahedron_mesh_from_plc(plc)
                                        .map(|mut mesh| {
                                            record_solver_family_selection(
                                                &mut mesh,
                                                "star_shaped_polyhedron",
                                                &attempted_families,
                                                &rejected_families,
                                            );
                                            mesh
                                        })
                                }
                                Err(err) => Err(err),
                            }
                        }
                        Err(err) => Err(err),
                    }
                }
                Err(err) => Err(err),
            }
        }
        Err(err) => Err(err),
    }
}

fn record_solver_family_selection(
    mesh: &mut TetrahedronMesh,
    selected_family: &'static str,
    attempted_families: &[&'static str],
    rejected_families: &[(&'static str, &'static str)],
) {
    mesh.evidence.entity_counts.insert(
        "solver_generation_attempted_families".to_string(),
        attempted_families.len(),
    );
    mesh.evidence.entity_counts.insert(
        "solver_generation_rejected_families".to_string(),
        rejected_families.len(),
    );
    mesh.evidence
        .entity_counts
        .insert("solver_generation_selected_families".to_string(), 1);
    for (index, family) in attempted_families.iter().enumerate() {
        mesh.evidence
            .entity_counts
            .insert(format!("solver_generation_attempted_{family}"), 1);
        if *family == selected_family {
            mesh.evidence
                .entity_counts
                .insert(format!("solver_generation_selected_{family}"), 1);
            mesh.evidence.entity_counts.insert(
                "solver_generation_selected_family_index".to_string(),
                index + 1,
            );
        }
    }
    for (family, reason) in rejected_families {
        mesh.evidence
            .entity_counts
            .insert(format!("solver_generation_rejected_{family}_{reason}"), 1);
    }
}
