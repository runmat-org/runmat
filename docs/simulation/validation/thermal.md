## Temperature, heat transport and conservation

Steady thermal verification checks temperature and heat transport together. An independent conduction reference establishes the expected gradient and flux; balance checks account for applied heat, boundary transfer and internal sources. Refinement compares physical quantities across meshes and element orders, including interfaces and spatially varying conductivity where those lie within the assessed scope.

A temperature field alone cannot establish conservation. A nearly uniform temperature may conceal an incorrect source or boundary sign, while a small algebraic residual establishes only that the assembled equations were solved closely. The field requirements cover temperature, gradient and heat flux with their physical units and original mesh association.

## Radiation and thermal stress studies

[radiating-bar.fea](../examples/radiating-bar.fea): a 1 m steel bar ($k$ = 45 W/m/K) takes
1000 W/m² on one end and radiates from the other ($\varepsilon$ = 0.8, 300 K surroundings). The
radiating end satisfies $q=\varepsilon\sigma_{SB}(T^4-T_\infty^4)$, and the heated end is $qL/k$
hotter. RunMat gives 416.6790 K and 438.9013 K; the closed form gives 416.6790 K and 438.9013 K
(agreement to 1e-12). The published energy balance closes: 10 W applied, 10 W radiated. The same
check runs as a test on Tet4 and Tet10 meshes
(`radiating_bar_matches_the_gray_surface_closed_form_and_balances_power`).

[thermal-stress-static.fea](../examples/thermal-stress-static.fea): the same bar clamped at both ends
and held at 373.15 K (80 K above its reference) has the axial stress $E\alpha\Delta T$ = 192 MPa away
from the clamps. The volume-mean von Mises stress is 195.9 MPa; the clamps also restrain lateral
expansion, which raises the stress near the ends.

## Execution and durable results

Workflow verification follows an authored thermal study through admission, native execution, output publication and cold reopening. It checks that original temperature and element fields retain their descriptors and provenance. Browser playback of a saved scene is an inspection path; it does not repeat or independently validate the thermal solve.

## Criteria and assessment records

<!-- fea:verification -->

Read each record with its tested identity, geometry, mesh, boundary conditions and limits. The linked checks support their declared obligations; they do not extend an assessment to an untested model.

Start with the [steady heat-transfer guide](/docs/simulation/guides/steady-heat-transfer), and use [validation methodology](/docs/simulation/validation) for the shared approach to evidence and qualification.
