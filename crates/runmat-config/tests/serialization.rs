use runmat_config::runtime::{ConfigLoader, RunMatRuntimeConfig};
use tempfile::TempDir;

#[test]
fn toml_round_trip() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("runmat.toml");

    let mut config = RunMatRuntimeConfig::default();
    config.runtime.callstack_limit = 777;
    config.jit.threshold = 25;

    ConfigLoader::save_to_file(&config, &path).unwrap();
    let loaded = ConfigLoader::load_from_file(&path).unwrap();

    assert_eq!(loaded.runtime.callstack_limit, 777);
    assert_eq!(loaded.jit.threshold, 25);
}

#[test]
fn json_round_trip() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("runmat.json");

    let mut config = RunMatRuntimeConfig::default();
    config.plotting.mode = runmat_config::runtime::PlotMode::Headless;
    config.accelerate.enabled = false;

    ConfigLoader::save_to_file(&config, &path).unwrap();
    let loaded = ConfigLoader::load_from_file(&path).unwrap();

    assert_eq!(
        loaded.plotting.mode,
        runmat_config::runtime::PlotMode::Headless
    );
    assert!(!loaded.accelerate.enabled);
}

#[test]
fn save_preserves_unowned_sections() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &path,
        "# keep\n[package]\nname = \"demo\"\n\n[desktop.artifacts]\nroot = \".cache\"\n",
    )
    .unwrap();
    let mut config = RunMatRuntimeConfig::default();
    config.runtime.callstack_limit = 777;
    ConfigLoader::save_to_file(&config, &path).unwrap();
    let source = std::fs::read_to_string(path).unwrap();
    assert!(source.contains("# keep\n"));
    assert!(source.contains("[package]\nname = \"demo\""));
    assert!(source.contains("[desktop.artifacts]\nroot = \".cache\""));
}
