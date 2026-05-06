use runmat_config::{ConfigLoader, RunMatConfig};
use tempfile::TempDir;

#[test]
fn file_loading() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join(".runmat.yaml");

    let mut config = RunMatConfig::default();
    config.runtime.timeout = 600;
    config.jit.threshold = 20;

    ConfigLoader::save_to_file(&config, &config_path).unwrap();
    let loaded = ConfigLoader::load_from_file(&config_path).unwrap();

    assert_eq!(loaded.runtime.timeout, 600);
    assert_eq!(loaded.jit.threshold, 20);
}
