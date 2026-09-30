// bench2.mjs — Phase 2 benchmark: Greedy, SA (multi-start), neal
// Solves directly in-process (imports from qubo.mjs) for greedy & SA;
// neal_solve.py is called via child_process.
// Evaluation is done in-process using evaluateObjectiveFromX / checkBudget.

import { execSync } from 'child_process';
import * as fs from 'fs';
import { greedySolve, simulatedAnnealing, repairSolution, evaluateObjectiveFromX, checkBudget } from './qubo.mjs';

// ──────────────────────────────────────────────
// 1. Helpers
// ──────────────────────────────────────────────

function runCmd(cmd, opts = {}) {
    const t0 = performance.now();
    let stdout;
    try {
        stdout = execSync(cmd, { stdio: ['pipe', 'pipe', 'inherit'], timeout: 300000, ...opts });
    } catch (e) {
        return { ok: false, stdout: e.stdout?.toString() || '', stderr: e.stderr?.toString() || '', runtime_ms: performance.now() - t0, error: e?.message ?? String(e) };
    }
    return { ok: true, stdout: stdout.toString(), runtime_ms: performance.now() - t0 };
}

function loadGraph(quboFile) {
    const data = JSON.parse(fs.readFileSync(quboFile, 'utf8'));
    return {
        n: data.n,
        hypotheses: data.graph.values.map((v, i) => ({ id: i, value: v, cost: data.graph.costs[i] })),
        edges: data.graph.edges,
        budget: data.graph.budget
    };
}

function loadBits(bitsFile) {
    const data = JSON.parse(fs.readFileSync(bitsFile, 'utf8'));
    return data.bits;
}

// Evaluate a bits array in-process using qubo.mjs functions
function evaluateBits(quboFile, bits) {
    const graph = loadGraph(quboFile);
    const objective = evaluateObjectiveFromX(graph, bits);
    const budgetInfo = checkBudget(graph, bits);
    return { objective, feasible: budgetInfo.feasible, used: budgetInfo.used, budget: budgetInfo.budget };
}

// ──────────────────────────────────────────────
// 2. Main benchmark
// ──────────────────────────────────────────────

function runBench2(filterSizes) {
    console.log('=== PHASE 2 BENCHMARK (bench2.mjs) ===\n');

    const allSizes = [100, 500, 2000];
    const sizes = filterSizes && filterSizes.length > 0 ? filterSizes : allSizes;
    const seeds = [42, 12345, 67890];
    const lambdaContra = 1.0;
    const muBudget = 2.0;

    // results-phase2.json structure
    const results = {};

    for (const n of sizes) {
        console.log(`\n--- n = ${n} ---`);
        const nKey = String(n);
        results[nKey] = {};

        for (const seed of seeds) {
            console.log(`  seed = ${seed}`);

            const quboFile = `qubo-${n}-${seed}.json`;
            const bitsGreedyFile = `bits-greedy-${n}-${seed}.json`;
            const bitsSaFile = `bits-sa-${n}-${seed}.json`;
            const bitsNealFile = `bits-neal-${n}-${seed}.json`;
            const bitsGreedyRepFile = `bits-greedy-rep-${n}-${seed}.json`;
            const bitsSaRepFile = `bits-sa-rep-${n}-${seed}.json`;
            const bitsNealRepFile = `bits-neal-rep-${n}-${seed}.json`;

            // --- Export QUBO ---
            if (!fs.existsSync(quboFile)) {
                const r = runCmd(`node qubo.mjs --export-qubo ${n} ${seed}`);
                if (!r.ok) { console.error(`    export FAILED: ${r.error}`); continue; }
            }

            // Load graph once for in-process evaluation
            const graph = loadGraph(quboFile);

            // --- Greedy solve (direct) ---
            const tG0 = performance.now();
            const bitsG = greedySolve(graph);
            const tG = performance.now() - tG0;
            const evalG = evaluateBits(quboFile, bitsG);
            fs.writeFileSync(bitsGreedyFile, JSON.stringify({ bits: bitsG }));

            // Repair greedy
            const bitsGR = repairSolution(graph, bitsG);
            const evalGR = evaluateBits(quboFile, bitsGR);
            fs.writeFileSync(bitsGreedyRepFile, JSON.stringify({ bits: bitsGR }));

            // --- SA solve (direct, with custom params per size) ---
            // For n=2000: 8 restarts × 40000 steps; otherwise 16 restarts (default)
            const saParams = n >= 2000
                ? { restarts: 8, perSteps: 40000 }
                : { restarts: 16 };
            const tSA0 = performance.now();
            const bitsSA = simulatedAnnealing(graph, lambdaContra, muBudget, saParams, seed + 1000);
            const tSA = performance.now() - tSA0;
            const evalSA = evaluateBits(quboFile, bitsSA);
            fs.writeFileSync(bitsSaFile, JSON.stringify({ bits: bitsSA }));

            // Repair SA
            const bitsSAR = repairSolution(graph, bitsSA);
            const evalSAR = evaluateBits(quboFile, bitsSAR);
            fs.writeFileSync(bitsSaRepFile, JSON.stringify({ bits: bitsSAR }));

            // --- neal solve (via CLI) ---
            let evalN = null;
            let rNealRuntime = null;
            let nealError = null;
            try {
                const rNeal = runCmd(`arch -arm64 python3 neal_solve.py ${quboFile} ${bitsNealFile}`);
                if (rNeal.ok) {
                    const bitsN = loadBits(bitsNealFile);
                    evalN = evaluateBits(quboFile, bitsN);
                    rNealRuntime = rNeal.runtime_ms;
                } else {
                    nealError = rNeal.error || '(no error message)';
                    // Try to capture stderr/stdout for diagnostics
                    if (rNeal.stdout) nealError += ' | stdout: ' + rNeal.stdout.slice(0, 200);
                    if (rNeal.stderr) nealError += ' | stderr: ' + rNeal.stderr.slice(0, 200);
                }
            } catch (ex) {
                nealError = String(ex);
            }

            // Repair neal
            let evalNR = null;
            if (evalN) {
                const bitsNR = repairSolution(graph, evalN ? loadBits(bitsNealFile) : []);
                evalNR = evaluateBits(quboFile, bitsNR);
                fs.writeFileSync(bitsNealRepFile, JSON.stringify({ bits: bitsNR }));
            }

            // Store
            const seedKey = String(seed);
            if (!results[nKey][seedKey]) results[nKey][seedKey] = {};

            results[nKey][seedKey].greedy = evalG ? { objective: evalG.objective, feasible: evalG.feasible, runtime_ms: tG } : { objective: null, feasible: false, runtime_ms: null };
            results[nKey][seedKey].greedy_repair = evalGR ? { objective: evalGR.objective, feasible: evalGR.feasible } : { objective: null, feasible: false };
            results[nKey][seedKey].sa = evalSA ? { objective: evalSA.objective, feasible: evalSA.feasible, runtime_ms: tSA } : { objective: null, feasible: false, runtime_ms: null };
            results[nKey][seedKey].sa_repair = evalSAR ? { objective: evalSAR.objective, feasible: evalSAR.feasible } : { objective: null, feasible: false };
            results[nKey][seedKey].neal = evalN ? { objective: evalN.objective, feasible: evalN.feasible, runtime_ms: rNealRuntime } : { objective: null, feasible: false, runtime_ms: null };
            results[nKey][seedKey].neal_repair = evalNR ? { objective: evalNR.objective, feasible: evalNR.feasible } : { objective: null, feasible: false };

            console.log(`    greedy:  obj=${evalG?.objective ?? '?'}, feas=${evalG?.feasible ?? '?'}, ms=${tG.toFixed(1)}` +
                (evalGR && evalGR.objective !== evalG?.objective ? ` -> repair: obj=${evalGR.objective}` : ''));
            console.log(`    sa:      obj=${evalSA?.objective ?? '?'}, feas=${evalSA?.feasible ?? '?'}, ms=${tSA.toFixed(1)}` +
                (evalSAR && evalSAR.objective !== evalSA?.objective ? ` -> repair: obj=${evalSAR.objective}` : ''));
            if (evalN) {
                console.log(`    neal:    obj=${evalN.objective}, feas=${evalN.feasible}, ms=${(rNealRuntime ?? 0).toFixed(1)}` +
                    (evalNR && evalNR.objective !== evalN.objective ? ` -> repair: obj=${evalNR.objective}` : ''));
            } else {
                console.log(`    neal:    FAILED: ${nealError ?? 'unknown'}`);
            }
        }
    }

    // ──────────────────────────────────────────────
    // 4. Aggregate results
    // ──────────────────────────────────────────────
    console.log('\n\n=== PHASE 2 AGGREGATED RESULTS ===\n');

    const solvers = ['greedy', 'greedy_repair', 'sa', 'sa_repair', 'neal', 'neal_repair'];

    const hdr = `${'Solver'.padEnd(16)} | ${'n'.padEnd(6)} | ${'Mean Obj'.padEnd(12)} | ${'Feas %'.padEnd(8)} | ${'Mean ms'.padEnd(12)}`;
    console.log(hdr);
    console.log('─'.repeat(hdr.length));

    const aggregated = {};

    for (const n of sizes) {
        const nKey = String(n);
        aggregated[nKey] = {};

        for (const solver of solvers) {
            const vals = [];
            for (const seed of seeds) {
                const sk = String(seed);
                const r = results[nKey]?.[sk]?.[solver];
                if (r && r.objective !== null && r.objective !== undefined) {
                    vals.push(r);
                }
            }
            if (vals.length === 0) {
                console.log(`${solver.padEnd(16)} | ${String(n).padEnd(6)} | ${'N/A'.padEnd(12)} | ${'N/A'.padEnd(8)} | ${'N/A'.padEnd(12)}`);
                continue;
            }
            const meanObj = vals.reduce((s, v) => s + v.objective, 0) / vals.length;
            const feasCount = vals.filter(v => v.feasible).length;
            const feasPct = (feasCount / vals.length) * 100;
            const meanRt = vals.reduce((s, v) => s + (v.runtime_ms || 0), 0) / vals.length;

            aggregated[nKey][solver] = {
                mean_objective: meanObj,
                feasible_fraction: feasCount / vals.length,
                mean_runtime_ms: meanRt
            };

            console.log(`${solver.padEnd(16)} | ${String(n).padEnd(6)} | ${meanObj.toFixed(2).padEnd(12)} | ${feasPct.toFixed(1).padEnd(8)} | ${meanRt.toFixed(2).padEnd(12)}`);
        }
    }

    // ──────────────────────────────────────────────
    // 5. Summary
    // ──────────────────────────────────────────────
    console.log('\n--- Сводка: средний objective по solver × n ---');
    for (const n of sizes) {
        const nKey = String(n);
        const a = aggregated[nKey];
        if (!a) continue;
        const g = a.greedy?.mean_objective;
        const s = a.sa?.mean_objective;
        const nv = a.neal?.mean_objective;
        const gr = a.greedy_repair?.mean_objective;
        const sr = a.sa_repair?.mean_objective;
        const nr = a.neal_repair?.mean_objective;

        console.log(`\nn=${n}:`);
        if (g !== undefined) console.log(`  greedy:           ${g.toFixed(2)}`);
        if (s !== undefined) console.log(`  sa (16 restart):  ${s.toFixed(2)}  diff_to_greedy=${(s - g).toFixed(2)}`);
        if (nv !== undefined) console.log(`  neal:             ${nv.toFixed(2)}  diff_to_sa=${(nv - s).toFixed(2)}`);
        if (gr !== undefined) console.log(`  greedy+repair:    ${gr.toFixed(2)}`);
        if (sr !== undefined) console.log(`  sa+repair:        ${sr.toFixed(2)}`);
        if (nr !== undefined) console.log(`  neal+repair:      ${nr.toFixed(2)}`);
    }

    // Answer: does our multi-start SA catch neal?
    console.log('\n--- Выводы ---');
    for (const n of sizes) {
        const nKey = String(n);
        const a = aggregated[nKey];
        if (!a) continue;
        const s = a.sa?.mean_objective;
        const nv = a.neal?.mean_objective;
        const g = a.greedy?.mean_objective;
        if (s !== undefined && nv !== undefined) {
            const gap = ((s - nv) / nv * 100).toFixed(1);
            console.log(`n=${n}: SA vs neal: SA=${s.toFixed(2)}, neal=${nv.toFixed(2)}, gap=${gap}%`);
            if (s >= nv) {
                console.log(`  → SA догнал/обогнал neal`);
            } else {
                console.log(`  → SA НЕ догнал neal (отставание ${gap}%)`);
            }
        }
        if (s !== undefined && g !== undefined) {
            console.log(`n=${n}: SA vs greedy: SA=${s.toFixed(2)}, greedy=${g.toFixed(2)}, diff=${(s - g).toFixed(2)}`);
            if (s > g) {
                console.log(`  → SA обгоняет greedy`);
            } else {
                console.log(`  → SA НЕ обгоняет greedy`);
            }
        }
    }

    // ──────────────────────────────────────────────
    // 6. Save results-phase2.json
    // ──────────────────────────────────────────────
    const output = {
        params: { sizes, seeds },
        aggregated,
        per_seed: results
    };
    fs.writeFileSync('results-phase2.json', JSON.stringify(output, null, 2));
    console.log('\nresults-phase2.json written.');
}

// ──────────────────────────────────────────────
// 3. CLI entry
// ──────────────────────────────────────────────
// Accept optional --size N arguments to run only specific sizes.
// Example: node bench2.mjs --size 100 --size 500
const benchArgs = process.argv.slice(2);
const filterSizes = [];
for (let i = 0; i < benchArgs.length; i++) {
    if (benchArgs[i] === '--size' && i + 1 < benchArgs.length) {
        filterSizes.push(parseInt(benchArgs[i + 1]));
        i++;
    }
}
if (filterSizes.length > 0) {
    runBench2(filterSizes);
} else {
    runBench2();
}