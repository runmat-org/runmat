use runmat_config::runtime::{LanguageCompatMode, PlotMode, RunMatRuntimeConfig};

#[test]
fn config_defaults() {
    let config = RunMatRuntimeConfig::default();
    assert_eq!(config.runtime.callstack_limit, 200);
    assert!(config.jit.enabled);
    assert_eq!(config.jit.threshold, 10);
    assert_eq!(config.plotting.mode, PlotMode::Auto);
    assert!(matches!(config.language.compat, LanguageCompatMode::RunMat));
    assert_eq!(config.runtime.error_namespace, "");
}
