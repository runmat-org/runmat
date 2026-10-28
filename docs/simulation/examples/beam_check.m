% Compare the cantilever examples with Euler-Bernoulli beam theory.
% Tip deflection of a cantilever under an end load: delta = P*L^3 / (3*E*I).
P = 1000;            % N, 100 kPa end traction on a 0.1 m x 0.1 m face
L = 1.0;             % m
E = 200e9;           % Pa
I = 0.1^4 / 12;      % m^4
delta = P * L^3 / (3 * E * I);

studies = ["cantilever-coarse.fea", "cantilever-medium.fea", "cantilever-static.fea"];
for k = 1:numel(studies)
    run = fea.run(studies(k));
    results = fea.results(run);
    u = fea.field(results, fea.StructuralField.displacement());
    peak = max(vecnorm(u.values, 2, 2));
    fprintf("%-24s peak %.5f mm, beam theory %.5f mm, difference %+.2f%%\n", ...
        studies(k), 1e3 * peak, 1e3 * delta, 100 * (peak - delta) / delta);
end
