use runmat_config::{LanguageCompatMode, PlotMode, RunMatConfig};

#[test]
fn config_defaults() {
    let config = RunMatConfig::default();
    assert_eq!(config.runtime.timeout, 300);
    assert!(config.jit.enabled);
    assert_eq!(config.jit.threshold, 10);
    assert_eq!(config.plotting.mode, PlotMode::Auto);
    assert!(matches!(config.language.compat, LanguageCompatMode::RunMat));
    assert_eq!(config.runtime.error_namespace, "");
}
