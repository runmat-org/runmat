fn combined(files: &[&str]) -> String {
    files.join("\n")
}

fn count_sections(document: &str) -> usize {
    document
        .lines()
        .filter(|line| line.starts_with("## "))
        .count()
}

fn count_fea_examples(document: &str) -> usize {
    document.lines().filter(|line| *line == "```fea").count()
}

#[test]
fn physics_guides_cover_every_checked_family_and_procedure() {
    let docs = combined(&[
        include_str!("../../../../docs/fea/physics/index.md"),
        include_str!("../../../../docs/fea/physics/structural.md"),
        include_str!("../../../../docs/fea/physics/thermal.md"),
        include_str!("../../../../docs/fea/physics/electromagnetic.md"),
        include_str!("../../../../docs/fea/physics/acoustic.md"),
        include_str!("../../../../docs/fea/physics/fluid.md"),
        include_str!("../../../../docs/fea/physics/thermomechanical.md"),
        include_str!("../../../../docs/fea/physics/electrothermal.md"),
        include_str!("../../../../docs/fea/physics/conjugate-heat-transfer.md"),
        include_str!("../../../../docs/fea/physics/fluid-structure-interaction.md"),
    ]);
    for term in [
        "structural static",
        "structural preload",
        "structural modal",
        "structural transient",
        "structural modal_transient",
        "structural harmonic",
        "structural modal_harmonic",
        "structural prestressed_modal",
        "structural buckling",
        "structural prestressed_harmonic",
        "structural modal_response_spectrum",
        "structural random_response",
        "structural nonlinear",
        "thermal steady",
        "thermal transient",
        "electromagnetic",
        "acoustic",
        "fluid",
        "thermomechanical",
        "electrothermal",
        "conjugate_heat_transfer",
        "fluid_structure_interaction",
    ] {
        assert!(
            docs.contains(term),
            "FEA physics docs do not cover `{term}`"
        );
    }
}

#[test]
fn constraint_guides_cover_every_closed_constraint_and_interface_kind() {
    let docs = combined(&[
        include_str!("../../../../docs/fea/constraints/structural.md"),
        include_str!("../../../../docs/fea/constraints/thermal.md"),
        include_str!("../../../../docs/fea/constraints/electromagnetic.md"),
        include_str!("../../../../docs/fea/constraints/acoustic.md"),
        include_str!("../../../../docs/fea/constraints/fluid.md"),
        include_str!("../../../../docs/fea/constraints/connectors.md"),
        include_str!("../../../../docs/fea/constraints/interfaces.md"),
        include_str!("../../../../docs/fea/constraints/initial-conditions.md"),
    ]);
    for term in [
        "structural.fixed",
        "structural.symmetry",
        "structural.antisymmetry",
        "structural.prescribed",
        "structural.prescribed_motion",
        "structural.normal_displacement",
        "structural.rigid_coupling",
        "structural.distributing_coupling",
        "structural.tie",
        "structural.periodic",
        "structural.equation",
        "thermal.temperature",
        "thermal.periodic",
        "thermal.tie",
        "thermal.affine_jump",
        "electrical.potential",
        "electrical.equipotential",
        "electrical.floating_conductor",
        "electrical.integral_current",
        "electrical.potential_reference",
        "electrical.periodic",
        "magnetic.vector_potential",
        "magnetic.vector_potential_ground",
        "magnetic.coulomb_gauge",
        "magnetic.periodic",
        "magnetic.phase_shifted_periodic",
        "acoustic.pressure",
        "acoustic.periodic",
        "acoustic.phase_shifted_periodic",
        "fluid.velocity",
        "fluid.moving_wall",
        "fluid.no_slip_wall",
        "fluid.slip_wall",
        "fluid.symmetry",
        "fluid.pressure_reference",
        "fluid.periodic",
        "translational_spring",
        "translational_dashpot",
        "elastic_foundation",
        "contact.frictionless",
        "contact.coulomb",
        "fluid_structure.no_slip",
        "conjugate_heat_transfer.perfect",
        "conjugate_heat_transfer.conductance",
        "conjugate_heat_transfer.resistance",
        "angular_velocity",
        "electric_potential",
        "magnetic_vector_potential",
        "fluid_pressure",
        "fluid_velocity",
    ] {
        assert!(
            docs.contains(term),
            "FEA constraint docs do not cover `{term}`"
        );
    }
}

#[test]
fn load_guides_cover_every_closed_excitation_and_natural_boundary_kind() {
    let docs = combined(&[
        include_str!("../../../../docs/fea/loads/structural.md"),
        include_str!("../../../../docs/fea/loads/thermal.md"),
        include_str!("../../../../docs/fea/loads/electromagnetic.md"),
        include_str!("../../../../docs/fea/loads/acoustic.md"),
        include_str!("../../../../docs/fea/loads/fluid.md"),
    ]);
    for term in [
        "force",
        "moment",
        "traction",
        "pressure",
        "body_force",
        "heat_generation",
        "heat_flux",
        "temperature_field",
        "ambient_temperature",
        "terminal_voltage",
        "electric_current",
        "electric_charge",
        "current_density",
        "magnetic_vector_potential",
        "acoustic_pressure",
        "normal_velocity",
        "inlet_velocity",
        "mass_flow",
        "wall_motion",
        "structural.pressure",
        "thermal.heat_flux",
        "thermal.convection",
        "magnetic.insulation",
        "acoustic.rigid_wall",
        "acoustic.radiation",
        "acoustic.impedance",
        "fluid.outlet_pressure",
    ] {
        assert!(docs.contains(term), "FEA load docs do not cover `{term}`");
    }
}

#[test]
fn load_and_constraint_guides_retain_task_depth_and_runmat_examples() {
    let load_guides = [
        (
            "structural",
            include_str!("../../../../docs/fea/loads/structural.md"),
        ),
        (
            "thermal",
            include_str!("../../../../docs/fea/loads/thermal.md"),
        ),
        (
            "electromagnetic",
            include_str!("../../../../docs/fea/loads/electromagnetic.md"),
        ),
        (
            "acoustic",
            include_str!("../../../../docs/fea/loads/acoustic.md"),
        ),
        ("fluid", include_str!("../../../../docs/fea/loads/fluid.md")),
    ];
    for (family, guide) in load_guides {
        assert!(
            count_sections(guide) >= 20,
            "{family} load guide lost task-oriented coverage"
        );
        assert!(
            count_fea_examples(guide) >= 7,
            "{family} load guide lost concrete .fea examples"
        );
    }

    let constraint_guides = [
        (
            "structural",
            include_str!("../../../../docs/fea/constraints/structural.md"),
        ),
        (
            "thermal",
            include_str!("../../../../docs/fea/constraints/thermal.md"),
        ),
        (
            "electromagnetic",
            include_str!("../../../../docs/fea/constraints/electromagnetic.md"),
        ),
        (
            "acoustic",
            include_str!("../../../../docs/fea/constraints/acoustic.md"),
        ),
        (
            "fluid",
            include_str!("../../../../docs/fea/constraints/fluid.md"),
        ),
        (
            "connectors",
            include_str!("../../../../docs/fea/constraints/connectors.md"),
        ),
        (
            "interfaces",
            include_str!("../../../../docs/fea/constraints/interfaces.md"),
        ),
        (
            "initial conditions",
            include_str!("../../../../docs/fea/constraints/initial-conditions.md"),
        ),
    ];
    for (family, guide) in constraint_guides {
        assert!(
            count_sections(guide) >= 15,
            "{family} constraint guide lost task-oriented coverage"
        );
        assert!(
            count_fea_examples(guide) >= 4,
            "{family} constraint guide lost concrete .fea examples"
        );
    }
}
