use runmat_config::{ConfigLoader, RunMatConfig};
use tempfile::TempDir;

#[test]
fn toml_round_trip() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("runmat.toml");

    let mut config = RunMatConfig::default();
    config.runtime.timeout = 777;
    config.jit.threshold = 25;

    ConfigLoader::save_to_file(&config, &path).unwrap();
    let loaded = ConfigLoader::load_from_file(&path).unwrap();

    assert_eq!(loaded.runtime.timeout, 777);
    assert_eq!(loaded.jit.threshold, 25);
}

#[test]
fn json_round_trip() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("runmat.json");

    let mut config = RunMatConfig::default();
    config.plotting.mode = runmat_config::PlotMode::Headless;
    config.accelerate.enabled = false;

    ConfigLoader::save_to_file(&config, &path).unwrap();
    let loaded = ConfigLoader::load_from_file(&path).unwrap();

    assert_eq!(loaded.plotting.mode, runmat_config::PlotMode::Headless);
    assert!(!loaded.accelerate.enabled);
}
