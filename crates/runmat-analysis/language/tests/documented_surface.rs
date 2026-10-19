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

fn fea_examples(document: &str) -> String {
    let mut examples = String::new();
    let mut in_fea = false;
    for line in document.lines() {
        if line == "```fea" {
            in_fea = true;
            continue;
        }
        if in_fea && line == "```" {
            in_fea = false;
            examples.push('\n');
            continue;
        }
        if in_fea {
            examples.push_str(line);
            examples.push('\n');
        }
    }
    examples
}

#[test]
fn physics_guides_cover_every_checked_family_and_procedure() {
    let docs = combined(&[
        include_str!("../../../../docs/simulation/physics/index.md"),
        include_str!("../../../../docs/simulation/physics/structural.md"),
        include_str!("../../../../docs/simulation/physics/thermal.md"),
        include_str!("../../../../docs/simulation/physics/electromagnetic.md"),
        include_str!("../../../../docs/simulation/physics/acoustic.md"),
        include_str!("../../../../docs/simulation/physics/fluid.md"),
        include_str!("../../../../docs/simulation/physics/thermomechanical.md"),
        include_str!("../../../../docs/simulation/physics/electrothermal.md"),
        include_str!("../../../../docs/simulation/physics/conjugate-heat-transfer.md"),
        include_str!("../../../../docs/simulation/physics/fluid-structure-interaction.md"),
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
        include_str!("../../../../docs/simulation/constraints/structural.md"),
        include_str!("../../../../docs/simulation/constraints/thermal.md"),
        include_str!("../../../../docs/simulation/constraints/electromagnetic.md"),
        include_str!("../../../../docs/simulation/constraints/acoustic.md"),
        include_str!("../../../../docs/simulation/constraints/fluid.md"),
        include_str!("../../../../docs/simulation/constraints/connectors.md"),
        include_str!("../../../../docs/simulation/constraints/interfaces.md"),
        include_str!("../../../../docs/simulation/constraints/initial-conditions.md"),
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
        "axial_spring",
        "axial_dashpot",
        "isotropic_ground_spring",
        "surface_normal_foundation",
        "contact.frictionless",
        "contact.coulomb",
        "fluid_structure.no_slip",
        "conjugate_heat_transfer.perfect",
        "conjugate_heat_transfer.conductance",
        "conjugate_heat_transfer.resistance",
        "angular_velocity",
        "electrical.potential",
        "magnetic.vector_potential",
        "fluid.pressure",
        "fluid.velocity",
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
        include_str!("../../../../docs/simulation/loads/structural.md"),
        include_str!("../../../../docs/simulation/loads/thermal.md"),
        include_str!("../../../../docs/simulation/loads/electromagnetic.md"),
        include_str!("../../../../docs/simulation/loads/acoustic.md"),
        include_str!("../../../../docs/simulation/loads/fluid.md"),
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
        "incident_pressure",
        "normal_velocity",
        "volume_velocity",
        "inlet_velocity",
        "mass_flow",
        "wall_motion",
        "structural.pressure",
        "thermal.heat_flux",
        "thermal.convection",
        "thermal.radiation",
        "magnetic.insulation",
        "acoustic.rigid_wall",
        "acoustic.radiation",
        "acoustic.impedance",
        "fluid.outlet_pressure",
    ] {
        assert!(docs.contains(term), "FEA load docs do not cover `{term}`");
    }

    let examples = fea_examples(&docs);
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
        "incident_pressure",
        "normal_velocity",
        "volume_velocity",
        "inlet_velocity",
        "mass_flow",
        "wall_motion",
        "structural.pressure",
        "thermal.heat_flux",
        "thermal.convection",
        "thermal.radiation",
        "magnetic.insulation",
        "acoustic.rigid_wall",
        "acoustic.radiation",
        "acoustic.impedance",
        "fluid.outlet_pressure",
    ] {
        assert!(
            examples.contains(term),
            "FEA load docs name `{term}` but do not show it in a .fea example"
        );
    }
}

#[test]
fn constraint_guides_show_every_closed_kind_in_fea_syntax() {
    let docs = combined(&[
        include_str!("../../../../docs/simulation/constraints/structural.md"),
        include_str!("../../../../docs/simulation/constraints/thermal.md"),
        include_str!("../../../../docs/simulation/constraints/electromagnetic.md"),
        include_str!("../../../../docs/simulation/constraints/acoustic.md"),
        include_str!("../../../../docs/simulation/constraints/fluid.md"),
        include_str!("../../../../docs/simulation/constraints/connectors.md"),
        include_str!("../../../../docs/simulation/constraints/interfaces.md"),
        include_str!("../../../../docs/simulation/constraints/initial-conditions.md"),
    ]);
    let examples = fea_examples(&docs);
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
        "axial_spring",
        "axial_dashpot",
        "isotropic_ground_spring",
        "surface_normal_foundation",
        "contact.frictionless",
        "contact.coulomb",
        "fluid_structure.no_slip",
        "conjugate_heat_transfer.perfect",
        "conjugate_heat_transfer.conductance",
        "conjugate_heat_transfer.resistance",
        "angular_velocity",
        "electrical.potential",
        "magnetic.vector_potential",
        "fluid.pressure",
        "fluid.velocity",
    ] {
        assert!(
            examples.contains(term),
            "FEA constraint docs name `{term}` but do not show it in a .fea example"
        );
    }
}

#[test]
fn load_and_constraint_guides_retain_task_depth_and_runmat_examples() {
    let load_guides = [
        (
            "structural",
            include_str!("../../../../docs/simulation/loads/structural.md"),
            38,
            15,
        ),
        (
            "thermal",
            include_str!("../../../../docs/simulation/loads/thermal.md"),
            35,
            12,
        ),
        (
            "electromagnetic",
            include_str!("../../../../docs/simulation/loads/electromagnetic.md"),
            36,
            11,
        ),
        (
            "acoustic",
            include_str!("../../../../docs/simulation/loads/acoustic.md"),
            31,
            12,
        ),
        (
            "fluid",
            include_str!("../../../../docs/simulation/loads/fluid.md"),
            33,
            9,
        ),
    ];
    for (family, guide, minimum_sections, minimum_examples) in load_guides {
        assert!(
            count_sections(guide) >= minimum_sections,
            "{family} load guide lost task-oriented coverage"
        );
        assert!(
            count_fea_examples(guide) >= minimum_examples,
            "{family} load guide lost concrete .fea examples"
        );
    }

    let constraint_guides = [
        (
            "structural",
            include_str!("../../../../docs/simulation/constraints/structural.md"),
            41,
            17,
        ),
        (
            "thermal",
            include_str!("../../../../docs/simulation/constraints/thermal.md"),
            27,
            9,
        ),
        (
            "electromagnetic",
            include_str!("../../../../docs/simulation/constraints/electromagnetic.md"),
            21,
            13,
        ),
        (
            "acoustic",
            include_str!("../../../../docs/simulation/constraints/acoustic.md"),
            21,
            9,
        ),
        (
            "fluid",
            include_str!("../../../../docs/simulation/constraints/fluid.md"),
            27,
            9,
        ),
        (
            "connectors",
            include_str!("../../../../docs/simulation/constraints/connectors.md"),
            26,
            5,
        ),
        (
            "interfaces",
            include_str!("../../../../docs/simulation/constraints/interfaces.md"),
            20,
            10,
        ),
        (
            "initial conditions",
            include_str!("../../../../docs/simulation/constraints/initial-conditions.md"),
            21,
            14,
        ),
    ];
    for (family, guide, minimum_sections, minimum_examples) in constraint_guides {
        assert!(
            count_sections(guide) >= minimum_sections,
            "{family} constraint guide lost task-oriented coverage"
        );
        assert!(
            count_fea_examples(guide) >= minimum_examples,
            "{family} constraint guide lost concrete .fea examples"
        );
    }
}

#[test]
fn load_guides_answer_recognizable_engineering_tasks() {
    let required = [
        (
            include_str!("../../../../docs/simulation/loads/structural.md"),
            [
                "How do I load a pressure vessel correctly?",
                "How do I apply a harmonic load with amplitude and phase?",
                "How do I run a base-shake, response-spectrum, or random-vibration study?",
                "How do I load a hole, pin, clevis, or lug without creating a point singularity?",
            ],
        ),
        (
            include_str!("../../../../docs/simulation/loads/thermal.md"),
            [
                "How do I model a thermal soak followed by cooldown?",
                "How do I model heat from a chip, winding, battery cell, or resistor?",
                "How do I model a prescribed temperature map from test or another solver?",
                "How do I represent a heater controlled to constant power or constant temperature?",
            ],
        ),
        (
            include_str!("../../../../docs/simulation/loads/electromagnetic.md"),
            [
                "How do I excite a capacitor or electrostatic actuator?",
                "How do I excite a resistive conductor and compute Joule loss?",
                "How do I excite a coil in a harmonic magnetic analysis?",
                "How do I model several electrical terminals or conductor branches?",
            ],
        ),
        (
            include_str!("../../../../docs/simulation/loads/acoustic.md"),
            [
                "How do I choose between pressure and normal-velocity excitation?",
                "How do I drive a speaker diaphragm from measured motion?",
                "How do I model a muffler, silencer, or lined duct?",
                "How do I model two coherent acoustic sources?",
            ],
        ),
        (
            include_str!("../../../../docs/simulation/loads/fluid.md"),
            [
                "How do I choose a well-posed inlet and outlet combination?",
                "How do I model pipe or duct flow from a known flow rate?",
                "How do I model external aerodynamics around a vehicle or component?",
                "How do I model a manifold with several inlets and outlets?",
            ],
        ),
    ];
    for (guide, tasks) in required {
        for task in tasks {
            assert!(
                guide.contains(task),
                "load guide no longer answers `{task}`"
            );
        }
    }
}

#[test]
fn constraint_guides_answer_recognizable_engineering_tasks() {
    let guides = combined(&[
        include_str!("../../../../docs/simulation/constraints/structural.md"),
        include_str!("../../../../docs/simulation/constraints/thermal.md"),
        include_str!("../../../../docs/simulation/constraints/electromagnetic.md"),
        include_str!("../../../../docs/simulation/constraints/acoustic.md"),
        include_str!("../../../../docs/simulation/constraints/fluid.md"),
        include_str!("../../../../docs/simulation/constraints/connectors.md"),
        include_str!("../../../../docs/simulation/constraints/interfaces.md"),
        include_str!("../../../../docs/simulation/constraints/initial-conditions.md"),
    ]);
    for task in [
        "How do I choose the minimum constraints for a 3D solid?",
        "How do I constrain thermal expansion without creating artificial thermal stress?",
        "How do I model a cold plate, heat sink, or temperature-controlled mounting surface?",
        "How do I distinguish magnetic insulation, vector-potential ground, and Coulomb gauge?",
        "How do I constrain an acoustic exterior domain?",
        "How do I choose CFD boundary conditions that are mathematically well posed?",
        "How do I model an equipment mount with stiffness and damping?",
        "How do I model bolted or clamped contact between parts?",
        "How do I choose initial conditions for a static-to-transient release?",
    ] {
        assert!(
            guides.contains(task),
            "constraint guides no longer answer `{task}`"
        );
    }
}
