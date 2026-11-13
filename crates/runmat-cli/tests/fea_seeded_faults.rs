//! SIM-08: seeded setup faults (units, entity kinds, hand-offs) in the published cantilever
//! study are rejected before any mesh is generated, each with a located diagnostic.
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

struct Fault {
    name: &'static str,
    from: &'static str,
    to: &'static str,
    expected: &'static str,
}

const FAULTS: &[Fault] = &[
    // Units
    Fault {
        name: "traction in newtons",
        from: "region.loaded_end = [0, 0, -100000] Pa",
        to: "region.loaded_end = [0, 0, -100000] N",
        expected: "traction must be force per area, found N",
    },
    Fault {
        name: "modulus in kilograms",
        from: "youngs_modulus = 200 GPa",
        to: "youngs_modulus = 200 kg",
        expected: "youngs modulus must be force per area, found kg",
    },
    Fault {
        name: "density per area",
        from: "mass_density = 7850 kg/m^3",
        to: "mass_density = 7850 kg/m^2",
        expected: "mass density must be mass per volume, found kg/m^2",
    },
    Fault {
        name: "mesh size in seconds",
        from: "target_size = 0.15 m",
        to: "target_size = 0.15 s",
        expected: "target size must be length, found s",
    },
    Fault {
        name: "Poisson ratio with a unit",
        from: "poisson_ratio = 0.3",
        to: "poisson_ratio = 0.3 Pa",
        expected: "`poisson_ratio` requires a dimensionless scalar",
    },
    Fault {
        name: "unknown unit",
        from: "-100000] Pa",
        to: "-100000] psix",
        expected: "unknown unit `psix`",
    },
    // Entity kinds
    Fault {
        name: "traction on a solid",
        from: "traction on geometry.beam.region.loaded_end",
        to: "traction on geometry.beam.region.body",
        expected: "excitation target topology is incompatible",
    },
    Fault {
        name: "material on a face",
        from: "assign material.steel to geometry.beam.region.body",
        to: "assign material.steel to geometry.beam.region.fixed_end",
        expected: "assigns a material to face region `geometry.beam.region.fixed_end`",
    },
    Fault {
        name: "fixed support on a solid",
        from: "target = geometry.beam.region.fixed_end",
        to: "target = geometry.beam.region.body",
        expected: "constraint kind requires boundary topology",
    },
    Fault {
        name: "missing region",
        from: "region.loaded_end = [",
        to: "region.free_end = [",
        expected: "unresolved reference `geometry.beam.region.free_end`",
    },
    Fault {
        name: "selector matches nothing",
        from: "select name \"Loaded end\"",
        to: "select name \"No such face\"",
        expected:
            "selects face name \"No such face\", which `geometry/cantilever.step` does not carry",
    },
    // Hand-offs
    Fault {
        name: "output the step does not produce",
        from: "output displacement = analysis.static.field.node.displacement",
        to: "output displacement = analysis.static.field.node.temperature",
        expected: "does not produce `Temperature`",
    },
    Fault {
        name: "output from an unknown step",
        from: "analysis.static.field.element.von_mises_stress",
        to: "analysis.modes.field.element.von_mises_stress",
        expected: "unknown analysis step `modes`",
    },
    Fault {
        name: "case includes a missing load",
        from: "include load.end_traction",
        to: "include load.end_pressure",
        expected: "unresolved reference `load.end_pressure`",
    },
    Fault {
        name: "study binds a missing model",
        from: "model = model.loaded_beam",
        to: "model = model.unloaded_beam",
        expected: "unresolved reference `model.unloaded_beam`",
    },
    Fault {
        name: "missing material",
        from: "assign material.steel",
        to: "assign material.aluminum",
        expected: "unresolved reference `material.aluminum`",
    },
];

// witness: FEA-AUTHORING-COMPILER-002/check_plan_run @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Seeded setup faults in the cantilever example are rejected by the run command with
//   the expected diagnostic before any mesh artifact is produced.
#[test]
fn seeded_setup_faults_are_rejected_before_meshing() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let base =
        fs::read_to_string(repository.join("docs/simulation/examples/cantilever-static.fea"))
            .unwrap();
    for fault in FAULTS {
        assert_eq!(base.matches(fault.from).count(), 1, "{}", fault.name);
        let project = TempDir::new().unwrap();
        fs::create_dir_all(project.path().join("geometry")).unwrap();
        fs::copy(
            repository.join("docs/simulation/examples/geometry/cantilever.step"),
            project.path().join("geometry/cantilever.step"),
        )
        .unwrap();
        fs::write(
            project.path().join("cantilever.fea"),
            base.replacen(fault.from, fault.to, 1),
        )
        .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_runmat"))
            .current_dir(project.path())
            .args(["--offline", "run", "cantilever.fea"])
            .env_remove("RUNMAT_CONFIG")
            .env("NO_COLOR", "1")
            .env(
                "RUNMAT_PACKAGE_CACHE_DIR",
                project.path().join("package-cache"),
            )
            .env(
                "RUNMAT_EXECUTION_STATE_DIR",
                project.path().join("execution-state"),
            )
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{} was accepted", fault.name);
        assert!(
            stderr.contains(fault.expected),
            "{}: expected `{}` in\n{stderr}",
            fault.name,
            fault.expected
        );
        assert!(
            !project.path().join(".artifacts/meshes").exists(),
            "{} produced a mesh before it was rejected",
            fault.name
        );
    }
}
