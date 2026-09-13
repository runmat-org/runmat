export function factoryCliHelp() {
  const topology = "--topology PATH --candidate PATH --attestation PATH --component-graph PATH --draft PATH --c01-c03-review PATH --c04-c05-review PATH --c06-c07-review PATH --reconciliation PATH --stability-corrections PATH";
  const controlReview = "--control-scaffold PATH --control-review-set PATH --control-candidate PATH --control-attestation PATH";
  return `Usage:\n` +
    `  builtin-migration-factory.mjs inventory|seed-dispositions --compiled-inventory PATH [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs compile-dispositions --review PATH --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs draft-control --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs component-graph --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs compose-topology --baseline-inventory PATH --component-graph PATH --draft PATH --c01-c03-review PATH --c04-c05-review PATH --c06-c07-review PATH --reconciliation PATH --stability-corrections PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs freeze-topology --candidate PATH --attestation PATH --baseline-inventory PATH --component-graph PATH --draft PATH --c01-c03-review PATH --c04-c05-review PATH --c06-c07-review PATH --reconciliation PATH --stability-corrections PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs validate-topology --topology PATH --candidate PATH --attestation PATH --baseline-inventory PATH --component-graph PATH --draft PATH --c01-c03-review PATH --c04-c05-review PATH --c06-c07-review PATH --reconciliation PATH --stability-corrections PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs scaffold-control --baseline-inventory PATH ${topology} [--output PATH]\n` +
    `  builtin-migration-factory.mjs init-control-reviews --baseline-inventory PATH ${topology} --control-scaffold PATH --review-directory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs validate-global-control-review --baseline-inventory PATH ${topology} --control-scaffold PATH --global-review PATH\n` +
    `  builtin-migration-factory.mjs validate-bundle-control-review --baseline-inventory PATH ${topology} --control-scaffold PATH --global-review PATH --bundle-review PATH --bundle ID\n` +
    `  builtin-migration-factory.mjs index-control-reviews --baseline-inventory PATH ${topology} --control-scaffold PATH --review-directory PATH --review-set-directory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs compose-control --baseline-inventory PATH ${topology} --control-scaffold PATH --control-review-set PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs scaffold-control-attestation --baseline-inventory PATH ${topology} --control-scaffold PATH --control-review-set PATH --control-candidate PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs seal-control-attestation --baseline-inventory PATH ${topology} --control-scaffold PATH --control-review-set PATH --control-candidate PATH --attestation-review PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs freeze-control --baseline-inventory PATH ${topology} ${controlReview} [--output PATH]\n` +
    `  builtin-migration-factory.mjs validate-control --control PATH --baseline-inventory PATH ${topology} ${controlReview} [--output PATH]\n` +
    `  builtin-migration-factory.mjs issue-lease --request PATH --control PATH --baseline-inventory PATH --lease-base-inventory PATH --state PATH --queue-checkpoint PATH --trusted-queue-checkpoint-digest SHA256 ${topology} ${controlReview} [--output PATH]\n` +
    `  builtin-migration-factory.mjs produce-gate --compiled-inventory PATH --control PATH --baseline-inventory PATH --lease-base-inventory PATH --lease PATH --state PATH --queue-checkpoint PATH --trusted-queue-checkpoint-digest SHA256 ${topology} ${controlReview} --bundle ID --gate NAME --artifact ID [--inputs PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs materialize-composition --control PATH --baseline-inventory PATH --lease-base-inventory PATH --lease PATH --state PATH --queue-checkpoint PATH --trusted-queue-checkpoint-digest SHA256 ${topology} ${controlReview} [--output PATH]\n` +
    `  builtin-migration-factory.mjs queue --compiled-inventory PATH --control PATH --baseline-inventory PATH ${topology} ${controlReview} [--state PATH --queue-checkpoint PATH --trusted-queue-checkpoint-digest SHA256] [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs prepare NAME --compiled-inventory PATH --control PATH --baseline-inventory PATH --lease-base-inventory PATH ${topology} ${controlReview} --lease PATH --workspace PATH [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs audit --compiled-inventory PATH --control PATH --baseline-inventory PATH --lease-base-inventory PATH ${topology} ${controlReview} --lease PATH --batch PATH --evidence PATH [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs verify --manifest PATH --control PATH --baseline-inventory PATH ${topology} ${controlReview} [--output PATH]\n` +
    `  builtin-migration-factory.mjs seal --manifest PATH --lease PATH --control PATH --baseline-inventory PATH ${topology} ${controlReview} [--output PATH]\n\n` +
    `Generated files are content-addressed development evidence, never production authority.\n`;
}
