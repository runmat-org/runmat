[radiating-bar.fea](../examples/radiating-bar.fea): a 1 m steel bar ($k$ = 45 W/m/K) takes
1000 W/m² on one end and radiates from the other ($\varepsilon$ = 0.8, 300 K surroundings). The
radiating end satisfies $q=\varepsilon\sigma_{SB}(T^4-T_\infty^4)$, and the heated end is $qL/k$
hotter. RunMat gives 416.6790 K and 438.9013 K; the closed form gives 416.6790 K and 438.9013 K
(agreement to 1e-12). The published energy balance closes: 10 W applied, 10 W radiated. The same
check runs as a test on Tet4 and Tet10 meshes
(`radiating_bar_matches_the_gray_surface_closed_form_and_balances_power`).
