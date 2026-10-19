## Temperature, heat transport and conservation

Steady thermal verification checks temperature and heat transport together. An independent conduction reference establishes the expected gradient and flux; balance checks account for applied heat, boundary transfer and internal sources. Refinement compares physical quantities across meshes and element orders, including interfaces and spatially varying conductivity where those lie within the assessed scope.

A temperature field alone cannot establish conservation. A nearly uniform temperature may conceal an incorrect source or boundary sign, while a small algebraic residual establishes only that the assembled equations were solved closely. The field requirements cover temperature, gradient and heat flux with their physical units and original mesh association.

## Execution and durable results

Workflow verification follows an authored thermal study through admission, native execution, output publication and cold reopening. It checks that original temperature and element fields retain their descriptors and provenance. Browser playback of a saved scene is an inspection path; it does not repeat or independently validate the thermal solve.

## Criteria and assessment records

<!-- fea:verification -->

Read each record with its tested identity, geometry, mesh, boundary conditions and limits. The linked checks support their declared obligations; they do not extend an assessment to an untested model.

Start with the [steady heat-transfer guide](/docs/simulation/guides/steady-heat-transfer), and use [validation methodology](/docs/simulation/validation) for the shared approach to evidence and qualification.
