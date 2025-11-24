type MonteCarloRow = {
  paths: string;
  runMatMs: number;
  vsPyTorch: string;
  vsNumPy: string;
};

const MONTE_CARLO_ROWS: MonteCarloRow[] = [
  { paths: "250k", runMatMs: 109, vsPyTorch: "7.6× faster", vsNumPy: "37.4× faster" },
  { paths: "500k", runMatMs: 136, vsPyTorch: "6.6× faster", vsNumPy: "60.3× faster" },
  { paths: "1M", runMatMs: 188, vsPyTorch: "4.8× faster", vsNumPy: "85.6× faster" },
  { paths: "2M", runMatMs: 298, vsPyTorch: "3.7× faster", vsNumPy: "108.5× faster" },
  { paths: "5M", runMatMs: 607, vsPyTorch: "2.8× faster", vsNumPy: "131.5× faster" },
];

export default function MonteCarloSweep() {
  return (
    <div className="mx-auto w-full max-w-[40rem] rounded-2xl border border-border/60 bg-background/60 overflow-hidden shadow-lg">
      <div className="bg-[#0E1421] px-4 sm:px-6 py-4 border-b border-border/60">
        <div className="text-sm sm:text-base font-semibold uppercase tracking-wide text-gray-200">
          Monte Carlo
        </div>
        <div className="mt-1 text-xs sm:text-sm text-gray-400">
          Path count sweep: 250k → 5M simulations
        </div>
      </div>
      <div className="bg-[#0E1421]">
        <div className="overflow-x-auto">
          <table className="w-full text-sm sm:text-base">
            <thead className="border-b border-border/60 bg-muted/40 text-center">
              <tr>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center text-gray-400 font-medium">
                  Paths (simulations)
                </th>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center font-medium text-gray-400">
                  RunMat (ms)
                </th>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center font-medium text-gray-400">
                  RunMat vs PyTorch
                </th>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center font-medium text-gray-400">
                  RunMat vs NumPy
                </th>
              </tr>
            </thead>
            <tbody className="bg-[#0E1421] text-gray-200">
              {MONTE_CARLO_ROWS.map((row) => (
                <tr key={row.paths} className="border-b border-border/40 last:border-b-0">
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="font-mono text-sm sm:text-base">{row.paths}</span>
                  </td>
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center font-mono text-sm sm:text-base">
                    {row.runMatMs}
                  </td>
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="inline-flex items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 px-3 py-1 text-sm sm:text-base text-purple-100">
                      {row.vsPyTorch}
                    </span>
                  </td>
                    <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="inline-flex items-center justify-center rounded-full border border-blue-500/40 bg-blue-500/10 px-3 py-1 text-sm sm:text-base text-blue-100">
                      {row.vsNumPy}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="bg-gradient-to-r from-blue-500 via-purple-500 to-purple-600 text-white text-sm sm:text-base">
                <td className="px-3 sm:px-6 py-2 sm:py-3 font-medium" colSpan={2}>
                  RunMat speedup range
                </td>
                <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                  2.8× – 7.6× vs PyTorch
                </td>
                <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                  37.4× – 131.5× vs NumPy
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  );
}


