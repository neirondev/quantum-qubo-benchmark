// qubo.mjs — QUBO-конвейер для отбора гипотез (UI-005, фазы 1-2)
// Чистый Node, без зависимостей.

import * as fs from 'fs';

// ──────────────────────────────────────────────
// 1. mulberry32 PRNG
// ──────────────────────────────────────────────
function mulberry32(seed) {
    let s = seed | 0;
    return function () {
        s = (s + 0x6d2b79f5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ──────────────────────────────────────────────
// 2. genHypothesisGraph
// ──────────────────────────────────────────────
function genHypothesisGraph(n, seed) {
    const rng = mulberry32(seed);
    const hypotheses = [];
    for (let i = 0; i < n; i++) {
        hypotheses.push({
            id: i,
            value: 1 + Math.floor(rng() * 10),   // 1..10
            cost: 1 + Math.floor(rng() * 5)       // 1..5
        });
    }

    const edges = [];
    const edgeCount = Math.max(0, Math.floor(2 * n)); // ~2n
    // To avoid duplicates, use a Set of sorted pairs "i,j" (i < j)
    const usedPairs = new Set();
    for (let e = 0; e < edgeCount; e++) {
        let i, j, key;
        // Try up to 100 times to find a unique pair
        for (let attempt = 0; attempt < 100; attempt++) {
            i = Math.floor(rng() * n);
            j = Math.floor(rng() * n);
            if (i === j) continue;
            if (i > j) { [i, j] = [j, i]; }
            key = `${i},${j}`;
            if (!usedPairs.has(key)) break;
        }
        if (!usedPairs.has(key)) {
            usedPairs.add(key);
            const type = rng() > 0.5 ? 'contradiction' : 'synergy';
            let weight;
            if (type === 'contradiction') {
                weight = 5 + Math.floor(rng() * 11); // 5..15
            } else {
                weight = 1 + Math.floor(rng() * 5);  // 1..5
            }
            edges.push({ i, j, type, weight });
        }
    }

    const totalCost = hypotheses.reduce((s, h) => s + h.cost, 0);
    const budget = 0.35 * totalCost;

    return { hypotheses, edges, budget, n };
}

// ──────────────────────────────────────────────
// 3. buildQubo — полная матрица (с бюджетом, может быть большой)
// ──────────────────────────────────────────────
function buildQubo(graph, lambdaContra = 1.0, muBudget = 2.0) {
    const { hypotheses, edges, budget, n } = graph;
    const Q = new Map();

    function addWeight(i, j, w) {
        const key = i <= j ? `${i},${j}` : `${j},${i}`;
        const prev = Q.get(key) || 0;
        Q.set(key, prev + w);
    }

    // Целевая часть: -value_i * x_i
    for (let i = 0; i < n; i++) {
        addWeight(i, i, -hypotheses[i].value);
    }

    // Штрафы/бонусы для рёбер
    for (const e of edges) {
        if (e.type === 'contradiction') {
            addWeight(e.i, e.j, lambdaContra * e.weight);
        } else {
            addWeight(e.i, e.j, -e.weight);
        }
    }

    // Бюджетный штраф: mu * (sum(cost_i * x_i) - budget)^2
    // диагональ: mu * cost_i * (cost_i - 2 * budget)
    // попарно: 2 * mu * cost_i * cost_j
    for (let i = 0; i < n; i++) {
        const ci = hypotheses[i].cost;
        addWeight(i, i, muBudget * ci * (ci - 2 * budget));
    }
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            addWeight(i, j, 2 * muBudget * hypotheses[i].cost * hypotheses[j].cost);
        }
    }

    return Q;
}

// buildSparseQubo — без бюджетного штрафа (для быстрого SA)
function buildSparseQubo(graph, lambdaContra = 1.0) {
    const { hypotheses, edges, n } = graph;
    const Q = new Map();

    function addWeight(i, j, w) {
        const key = i <= j ? `${i},${j}` : `${j},${i}`;
        const prev = Q.get(key) || 0;
        Q.set(key, prev + w);
    }

    for (let i = 0; i < n; i++) {
        addWeight(i, i, -hypotheses[i].value);
    }

    for (const e of edges) {
        if (e.type === 'contradiction') {
            addWeight(e.i, e.j, lambdaContra * e.weight);
        } else {
            addWeight(e.i, e.j, -e.weight);
        }
    }

    return Q;
}

// ──────────────────────────────────────────────
// 4. Решатели
// ──────────────────────────────────────────────

// 4a. Greedy: сортируем по value/cost, добавляем пока бюджет позволяет + убираем противоречия
function greedySolve(graph) {
    const { hypotheses, edges, budget } = graph;
    const n = hypotheses.length;

    // Build adjacency for contradiction checks
    const contraAdj = new Map(); // i -> Set of j
    for (const e of edges) {
        if (e.type === 'contradiction') {
            if (!contraAdj.has(e.i)) contraAdj.set(e.i, new Set());
            if (!contraAdj.has(e.j)) contraAdj.set(e.j, new Set());
            contraAdj.get(e.i).add(e.j);
            contraAdj.get(e.j).add(e.i);
        }
    }

    // Sort indices by value/cost descending
    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => {
        const ra = hypotheses[a].value / hypotheses[a].cost;
        const rb = hypotheses[b].value / hypotheses[b].cost;
        return rb - ra;
    });

    const selected = new Set();
    let usedBudget = 0;

    for (const i of indices) {
        const h = hypotheses[i];
        if (usedBudget + h.cost > budget) continue;

        // Check contradictions with already selected
        let hasConflict = false;
        const conflicts = contraAdj.get(i);
        if (conflicts) {
            for (const j of conflicts) {
                if (selected.has(j)) {
                    hasConflict = true;
                    break;
                }
            }
        }
        if (hasConflict) continue;

        selected.add(i);
        usedBudget += h.cost;
    }

    const x = new Array(n).fill(0);
    for (const i of selected) x[i] = 1;
    return x;
}

// 4b. Simulated Annealing (быстрый: sparse Q + аналитический бюджет)
// Поддерживает restarts: при restarts>1 использует T0=15, cooling=0.9995,
// steps = 30000 для n≤500, 60000 для n=2000 на каждый прогон.
function simulatedAnnealing(graph, lambdaContra, muBudget, params, seed) {
    const { hypotheses, edges, budget, n } = graph;
    const restarts = params.restarts || 1;

    // Параметры на прогон: при мультистарте — свои, иначе — из params
    const perSteps = restarts > 1
        ? (params.perSteps || (n <= 500 ? 30000 : 60000))
        : (params.steps || 20000);
    const T0 = restarts > 1 ? 15 : (params.T0 || 10);
    const cooling = restarts > 1 ? 0.9995 : (params.cooling || 0.999);

    // Sparse graph Q (без бюджетного штрафа)
    const sparseQ = buildSparseQubo(graph, lambdaContra);

    // Предвычисляем budget terms
    const mu = muBudget;
    const costs = hypotheses.map(h => h.cost);
    const diagBudget = costs.map(c => mu * c * (c - 2 * budget));

    // Полная оценка: sparse часть + бюджетная часть
    function evaluate(x) {
        let val = 0;
        for (const [key, w] of sparseQ) {
            const [i, j] = key.split(',').map(Number);
            if (x[i] && x[j]) val += w;
        }
        let sumCost = 0;
        for (let i = 0; i < n; i++) {
            if (x[i]) sumCost += costs[i];
        }
        const over = sumCost - budget;
        val += mu * over * over;
        return val;
    }

    // deltaE для флипа бита k: O(граф + 1)
    function deltaE(x, k, sumCost) {
        let graphContrib = 0;
        for (const [key, w] of sparseQ) {
            const [i, j] = key.split(',').map(Number);
            if (i === k && j === k) {
                graphContrib += w;
            } else if (i === k && x[j] === 1) {
                graphContrib += w;
            } else if (j === k && x[i] === 1) {
                graphContrib += w;
            }
        }

        const sumCostWO = sumCost - (x[k] ? costs[k] : 0);
        const budgetContrib = diagBudget[k] + 2 * mu * costs[k] * sumCostWO;

        const total = graphContrib + budgetContrib;
        return x[k] === 0 ? total : -total;
    }

    let globalBestX = null;
    let globalBestVal = Infinity;

    for (let r = 0; r < restarts; r++) {
        const restartSeed = seed + r * 7919;  // простой большой сдвиг для разных траекторий
        const rng = mulberry32(restartSeed);

        let x = new Array(n).fill(0).map(() => (rng() < 0.5 ? 1 : 0));
        let currentVal = evaluate(x);
        let sumCost = 0;
        for (let i = 0; i < n; i++) if (x[i]) sumCost += costs[i];

        let T = T0;
        let bestX = [...x];
        let bestVal = currentVal;

        for (let step = 0; step < perSteps; step++) {
            const k = Math.floor(rng() * n);
            const dE = deltaE(x, k, sumCost);

            if (dE < 0 || rng() < Math.exp(-dE / T)) {
                x[k] = 1 - x[k];
                currentVal += dE;
                if (x[k]) sumCost += costs[k];
                else sumCost -= costs[k];

                if (currentVal < bestVal) {
                    bestVal = currentVal;
                    bestX = [...x];
                }
            }

            T *= cooling;
            if (T < 1e-10) break;
        }

        if (bestVal < globalBestVal) {
            globalBestVal = bestVal;
            globalBestX = bestX;
        }
    }

    return globalBestX;
}

// ──────────────────────────────────────────────
// 4c. REPAIR — пост-обработка для соблюдения бюджета
// ──────────────────────────────────────────────
function repairSolution(graph, x) {
    const { hypotheses, edges, budget, n } = graph;
    const contraAdj = new Map();
    for (const e of edges) {
        if (e.type === 'contradiction') {
            if (!contraAdj.has(e.i)) contraAdj.set(e.i, new Set());
            if (!contraAdj.has(e.j)) contraAdj.set(e.j, new Set());
            contraAdj.get(e.i).add(e.j);
            contraAdj.get(e.j).add(e.i);
        }
    }
    const selected = new Set();
    for (let i = 0; i < n; i++) if (x[i]) selected.add(i);
    let totalCost = 0;
    for (const i of selected) totalCost += hypotheses[i].cost;
    while (totalCost > budget + 1e-9) {
        let worstIdx = -1;
        let worstRatio = Infinity;
        for (const i of selected) {
            const ratio = hypotheses[i].value / hypotheses[i].cost;
            if (ratio < worstRatio) { worstRatio = ratio; worstIdx = i; }
        }
        if (worstIdx === -1) break;
        selected.delete(worstIdx);
        totalCost -= hypotheses[worstIdx].cost;
    }
    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => {
        const ra = hypotheses[a].value / hypotheses[a].cost;
        const rb = hypotheses[b].value / hypotheses[b].cost;
        return rb - ra;
    });
    for (const i of indices) {
        if (selected.has(i)) continue;
        const h = hypotheses[i];
        if (totalCost + h.cost > budget + 1e-9) continue;
        let hasConflict = false;
        const conflicts = contraAdj.get(i);
        if (conflicts) {
            for (const j of conflicts) {
                if (selected.has(j)) { hasConflict = true; break; }
            }
        }
        if (hasConflict) continue;
        selected.add(i);
        totalCost += h.cost;
    }
    const result = new Array(n).fill(0);
    for (const i of selected) result[i] = 1;
    return result;
}

// ──────────────────────────────────────────────
// 4d. exportQuboJson — экспорт задачи в JSON
// ──────────────────────────────────────────────
function exportQuboJson(n, seed, lambdaContra = 1.0, muBudget = 2.0) {
    const graph = genHypothesisGraph(n, seed);
    const Q = buildQubo(graph, lambdaContra, muBudget);
    const entries = [];
    for (const [key, w] of Q) {
        const [i, j] = key.split(',').map(Number);
        entries.push([i, j, w]);
    }
    const output = { n, Q: entries, graph: { values: graph.hypotheses.map(h => h.value), costs: graph.hypotheses.map(h => h.cost), budget: graph.budget, edges: graph.edges.map(e => ({ i: e.i, j: e.j, type: e.type, weight: e.weight })) } };
    const filename = `qubo-${n}-${seed}.json`;
    fs.writeFileSync(filename, JSON.stringify(output, null, 2));
    console.log(`[EXPORT] Written ${filename}`);
    return filename;
}

// ──────────────────────────────────────────────
// 4e. evaluateFromFile — оценка решения из файла
// ──────────────────────────────────────────────
function evaluateFromFile(quboFile, bitsFile) {
    const quboData = JSON.parse(fs.readFileSync(quboFile, 'utf8'));
    const bitsData = JSON.parse(fs.readFileSync(bitsFile, 'utf8'));
    const bits = bitsData.bits;
    const graph = { n: quboData.n, hypotheses: quboData.graph.values.map((v, i) => ({ id: i, value: v, cost: quboData.graph.costs[i] })), edges: quboData.graph.edges, budget: quboData.graph.budget };
    const objective = evaluateObjectiveFromX(graph, bits);
    const budgetInfo = checkBudget(graph, bits);
    const ones = bits.reduce((s, v) => s + v, 0);
    const result = { objective, x_ones: ones, feasible: budgetInfo.feasible, used: budgetInfo.used, budget: budgetInfo.budget };
    console.log(JSON.stringify(result, null, 2));
    return result;
}

export { greedySolve, simulatedAnnealing, repairSolution, evaluateObjectiveFromX, checkBudget, genHypothesisGraph, buildQubo, exportQuboJson, evaluateFromFile };

// ──────────────────────────────────────────────
// 5. Max-Cut санити-тест
// ──────────────────────────────────────────────
// Квадрат 4 вершины, рёбра сторон. Max-Cut = 4 (разбиение по диагоналям).
function makeMaxCutGraph() {
    const n = 4;
    const hypotheses = [
        { id: 0, value: 0, cost: 1 },
        { id: 1, value: 0, cost: 1 },
        { id: 2, value: 0, cost: 1 },
        { id: 3, value: 0, cost: 1 }
    ];
    // Edges: sides of the square (0-1, 1-2, 2-3, 3-0)
    // Max-Cut formulation: max sum (w_ij * (x_i XOR x_j))
    // As minimization QUBO: min sum (-w_ij * (x_i XOR x_j))
    // Since (x_i XOR x_j) = x_i + x_j - 2*x_i*x_j (when x_i,x_j in {0,1})
    // But we want: -w * (x_i + x_j - 2*x_i*x_j) = -w*x_i - w*x_j + 2w*x_i*x_j
    // Diagonal: -w for each endpoint; off-diagonal: +2w
    // We'll use buildQubo-like approach but with zero value hypotheses
    const edges = [
        { i: 0, j: 1, type: 'contradiction', weight: 0 },  // weight=0 means just track connectivity
        { i: 1, j: 2, type: 'contradiction', weight: 0 },
        { i: 2, j: 3, type: 'contradiction', weight: 0 },
        { i: 3, j: 0, type: 'contradiction', weight: 0 }
    ];
    // Actually, let's build MaxCut QUBO directly:
    // max sum w*(x_i XOR x_j) = max sum w*(x_i + x_j - 2*x_i*x_j)
    // As min: min sum -w*(x_i + x_j - 2*x_i*x_j)
    // Q_i,i += -w for each edge incident to i
    // Q_i,j += 2w for each edge
    // w = 1 for simplicity
    const Q = new Map();
    const w = 1;
    for (const e of edges) {
        const key1 = `${e.i},${e.i}`;
        const key2 = `${e.j},${e.j}`;
        const pairKey = e.i <= e.j ? `${e.i},${e.j}` : `${e.j},${e.i}`;
        Q.set(key1, (Q.get(key1) || 0) - w);
        Q.set(key2, (Q.get(key2) || 0) - w);
        Q.set(pairKey, (Q.get(pairKey) || 0) + 2 * w);
    }
    return { Q, n };
}

function runSanityCheck() {
    const n = 4;
    const w = 1; // Max-Cut edge weight

    // Graph compatible with simulatedAnnealing(graph, lambdaContra, muBudget, params, seed):
    //   - each vertex has value = sum_of_incident_weights = 2 (degree 2 in 4-cycle)
    //   - each edge is 'contradiction' with weight = 2*w -> off-diagonal Q[i][j] = 1 * 2 = 2*w
    //   - muBudget = 0 disables budget penalty (not applicable to Max-Cut)
    const hypotheses = Array.from({ length: n }, (_, i) => ({ id: i, value: 2, cost: 1 }));
    const edges = [
        { i: 0, j: 1, type: 'contradiction', weight: 2 * w },
        { i: 1, j: 2, type: 'contradiction', weight: 2 * w },
        { i: 2, j: 3, type: 'contradiction', weight: 2 * w },
        { i: 3, j: 0, type: 'contradiction', weight: 2 * w }
    ];
    const graph = { hypotheses, edges, budget: n, n };

    const seed = 42;
    const params = { steps: 20000, T0: 10, cooling: 0.999 };
    const lambdaContra = 1.0;
    const muBudget = 0.0;

    const x = simulatedAnnealing(graph, lambdaContra, muBudget, params, seed);

    // Evaluate cut size directly
    let cutValue = 0;
    const edgeList = [[0, 1], [1, 2], [2, 3], [3, 0]];
    for (const [i, j] of edgeList) {
        if (x[i] !== x[j]) cutValue += w;
    }

    console.log(`[SANITY] Max-Cut 4-cycle: cut = ${cutValue} (expected: 4)`);
    console.log(`[SANITY] Solution: ${JSON.stringify(x)}`);
    if (cutValue !== 4) {
        console.log(`[SANITY] WARNING: Sanity check FAILED (cut=${cutValue}) — retrying with more steps...`);
        const x2 = simulatedAnnealing(graph, lambdaContra, muBudget, { steps: 100000, T0: 20, cooling: 0.9995 }, seed + 1);
        let cutValue2 = 0;
        for (const [i, j] of edgeList) {
            if (x2[i] !== x2[j]) cutValue2 += w;
        }
        console.log(`[SANITY] Retry cut = ${cutValue2}`);
        if (cutValue2 !== 4) {
            throw new Error(`Max-Cut sanity FAILED: cut=${cutValue2}, expected 4`);
        }
        return { cut: cutValue2, x: x2 };
    }
    return { cut: cutValue, x };
}

// ──────────────────────────────────────────────
// 6. Бенчмарк
// ──────────────────────────────────────────────
function evaluateObjectiveFromX(graph, x) {
    // Original problem objective: sum(value_i * x_i) - sum(contradiction_penalties) + sum(synergy_bonuses)
    // Where contradiction penalty = lambdaContra * weight_ij * x_i * x_j
    // synergy bonus = weight_ij (added to value) when both selected
    const { hypotheses, edges } = graph;
    let val = 0;
    for (let i = 0; i < graph.n; i++) {
        if (x[i]) val += hypotheses[i].value;
    }
    for (const e of edges) {
        if (x[e.i] && x[e.j]) {
            if (e.type === 'contradiction') {
                val -= e.weight; // penalty (subtract)
            } else {
                val += e.weight; // bonus (add to value)
            }
        }
    }
    return val;
}

function checkBudget(graph, x) {
    let used = 0;
    for (let i = 0; i < graph.n; i++) {
        if (x[i]) used += graph.hypotheses[i].cost;
    }
    return { used, budget: graph.budget, feasible: used <= graph.budget + 1e-9 };
}

function runBenchmark() {
    console.log('\n=== БЕНЧМАРК ===\n');
    const sizes = [100, 500, 2000];
    const seeds = [12345, 67890, 11111];

    // results[solver][n] = { objective, feasible, runtime }
    const results = { greedy: {}, sa: {} };
    const params = { steps: 20000, T0: 10, cooling: 0.999 };
    const lambdaContra = 1.0;
    const muBudget = 2.0;

    for (const n of sizes) {
        console.log(`\nn = ${n}...`);
        for (const seed of seeds) {
            const graph = genHypothesisGraph(n, seed);
            const Q = buildQubo(graph, lambdaContra, muBudget);

            // Greedy
            const t0 = performance.now();
            const xG = greedySolve(graph);
            const tG = performance.now() - t0;
            const objG = evaluateObjectiveFromX(graph, xG);
            const budgetG = checkBudget(graph, xG);

            // SA
            const t1 = performance.now();
            const xSA = simulatedAnnealing(graph, lambdaContra, muBudget, params, seed + 1000);
            const tSA = performance.now() - t1;
            const objSA = evaluateObjectiveFromX(graph, xSA);
            const budgetSA = checkBudget(graph, xSA);

            if (!results.greedy[n]) results.greedy[n] = [];
            if (!results.sa[n]) results.sa[n] = [];
            results.greedy[n].push({ objective: objG, feasible: budgetG.feasible, runtime: tG, seed });
            results.sa[n].push({ objective: objSA, feasible: budgetSA.feasible, runtime: tSA, seed });
        }
    }

    // Print table
    console.log('\n--- Результаты бенчмарка ---\n');
    const header = `${'Solver'.padEnd(10)} | ${'n'.padEnd(6)} | ${'Objective'.padEnd(12)} | ${'Feasible'.padEnd(10)} | ${'Runtime ms'.padEnd(12)}`;
    console.log(header);
    console.log('─'.repeat(header.length));
    for (const n of sizes) {
        for (let s = 0; s < seeds.length; s++) {
            const gr = results.greedy[n][s];
            const sa = results.sa[n][s];
            console.log(`${'greedy'.padEnd(10)} | ${String(n).padEnd(6)} | ${gr.objective.toFixed(2).padEnd(12)} | ${String(gr.feasible).padEnd(10)} | ${gr.runtime.toFixed(2).padEnd(12)}`);
            console.log(`${'SA'.padEnd(10)} | ${String(n).padEnd(6)} | ${sa.objective.toFixed(2).padEnd(12)} | ${String(sa.feasible).padEnd(10)} | ${sa.runtime.toFixed(2).padEnd(12)}`);
            // Check SA >= greedy
            if (sa.objective < gr.objective - 1e-6) {
                console.log(`  ⚠ SA worse than greedy at n=${n}, seed=${seeds[s]}: SA=${sa.objective} < greedy=${gr.objective}`);
            }
        }
    }

    // Summary: where SA beats greedy
    console.log('\n--- Сводка: где SA выигрывает ---');
    for (const n of sizes) {
        const grAvgs = results.greedy[n].map(r => r.objective);
        const saAvgs = results.sa[n].map(r => r.objective);
        const avgG = grAvgs.reduce((a,b) => a+b, 0) / grAvgs.length;
        const avgSA = saAvgs.reduce((a,b) => a+b, 0) / saAvgs.length;
        console.log(`n=${n}: greedy avg=${avgG.toFixed(2)}, SA avg=${avgSA.toFixed(2)}, diff=${(avgSA - avgG).toFixed(2)}`);
    }

    // Write results.json
    const output = {
        sanity: { test: 'Max-Cut 4-cycle', result: 'PASSED' },
        benchmark: {
            params: { lambdaContra, muBudget },
            sa_params: params,
            results: { greedy: results.greedy, sa: results.sa }
        }
    };
    fs.writeFileSync('results.json', JSON.stringify(output, null, 2));
    console.log('\nresults.json written.');
    return results;
}

// ──────────────────────────────────────────────
// 7. Phase 2 benchmark: greedy / SA / SA+repair / neal / neal+repair
// ──────────────────────────────────────────────
function runPhase2Benchmark() {
    console.log('\n=== PHASE 2 BENCHMARK ===\n');
    const sizes = [100, 500, 2000];
    const seed = 42;
    const params = { steps: 20000, T0: 10, cooling: 0.999 };
    const lambdaContra = 1.0;
    const muBudget = 2.0;

    const results = {};

    for (const n of sizes) {
        console.log(`\n--- n = ${n} ---`);
        const graph = genHypothesisGraph(n, seed);
        const sizeKey = String(n);
        results[sizeKey] = {};

        // 1. Greedy
        const t0 = performance.now();
        const xG = greedySolve(graph);
        const tG = performance.now() - t0;
        const objG = evaluateObjectiveFromX(graph, xG);
        const budG = checkBudget(graph, xG);
        const xGR = repairSolution(graph, xG);
        const objGR = evaluateObjectiveFromX(graph, xGR);
        const budGR = checkBudget(graph, xGR);
        results[sizeKey].greedy = { objective: objG, feasible: budG.feasible, runtime_ms: tG };
        results[sizeKey].greedy_repair = { objective: objGR, feasible: budGR.feasible };

        // 2. SA
        const t1 = performance.now();
        const xSA = simulatedAnnealing(graph, lambdaContra, muBudget, params, seed + 1000);
        const tSA = performance.now() - t1;
        const objSA = evaluateObjectiveFromX(graph, xSA);
        const budSA = checkBudget(graph, xSA);
        const xSAR = repairSolution(graph, xSA);
        const objSAR = evaluateObjectiveFromX(graph, xSAR);
        const budSAR = checkBudget(graph, xSAR);
        results[sizeKey].sa = { objective: objSA, feasible: budSA.feasible, runtime_ms: tSA };
        results[sizeKey].sa_repair = { objective: objSAR, feasible: budSAR.feasible };

        // 3. Export QUBO for neal
        const quboFile = exportQuboJson(n, seed, lambdaContra, muBudget);

        // 4. Run neal via python
        const bitsFile = `neal-${n}-${seed}.json`;
        const pyCmd = `arch -arm64 python3 neal_solve.py ${quboFile} ${bitsFile}`;
        console.log(`[PHASE2] Running: ${pyCmd}`);
        const { execSync } = require('child_process');
        try {
            execSync(pyCmd, { stdio: 'inherit', timeout: 300000 });
        } catch (e) {
            console.error(`[PHASE2] neal failed for n=${n}: ${e.message}`);
            results[sizeKey].neal = { objective: null, feasible: false, runtime_ms: null, error: e.message };
            results[sizeKey].neal_repair = { objective: null, feasible: false };
            continue;
        }

        // 5. Evaluate neal solution
        const nealBits = JSON.parse(fs.readFileSync(bitsFile, 'utf8'));
        const objNeal = evaluateObjectiveFromX(graph, nealBits.bits);
        const budNeal = checkBudget(graph, nealBits.bits);
        const xNealR = repairSolution(graph, nealBits.bits);
        const objNealR = evaluateObjectiveFromX(graph, xNealR);
        const budNealR = checkBudget(graph, xNealR);
        results[sizeKey].neal = { objective: objNeal, feasible: budNeal.feasible, runtime_ms: nealBits.runtime_ms };
        results[sizeKey].neal_repair = { objective: objNealR, feasible: budNealR.feasible };
    }

    // Print table
    console.log('\n\n=== PHASE 2 RESULTS TABLE ===\n');
    const hdr = `${'Solver'.padEnd(16)} | ${'n'.padEnd(6)} | ${'Objective'.padEnd(12)} | ${'Feasible'.padEnd(10)} | ${'Runtime ms'.padEnd(12)}`;
    console.log(hdr);
    console.log('─'.repeat(hdr.length));
    for (const n of sizes) {
        const sk = String(n);
        for (const solver of ['greedy', 'greedy_repair', 'sa', 'sa_repair', 'neal', 'neal_repair']) {
            const r = results[sk]?.[solver];
            if (!r) continue;
            const objStr = r.objective !== null && r.objective !== undefined ? r.objective.toFixed(2).padEnd(12) : 'null'.padEnd(12);
            const rtStr = r.runtime_ms !== null && r.runtime_ms !== undefined ? r.runtime_ms.toFixed(2).padEnd(12) : 'null'.padEnd(12);
            console.log(`${solver.padEnd(16)} | ${String(n).padEnd(6)} | ${objStr} | ${String(r.feasible).padEnd(10)} | ${rtStr}`);
        }
    }

    // Save
    const output = {
        params: { sizes, seed, lambdaContra, muBudget, sa_params: params },
        results
    };
    fs.writeFileSync('results-phase2.json', JSON.stringify(output, null, 2));
    console.log('\nresults-phase2.json written.');
    return results;
}

// ──────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────
const isMain = process.argv[1] && (
    import.meta.url === `file://${process.argv[1]}` ||
    import.meta.url.endsWith('/qubo.mjs')
);
if (isMain) {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.log('=== QUBO-конвейер: UI-005 ===\n');
        const sanity = runSanityCheck();
        const results = runBenchmark();
    } else if (args[0] === '--export-qubo') {
        const n = parseInt(args[1]);
        const seed = parseInt(args[2]);
        if (isNaN(n) || isNaN(seed)) {
            console.error('Usage: node qubo.mjs --export-qubo N SEED');
            process.exit(1);
        }
        exportQuboJson(n, seed);
    } else if (args[0] === '--evaluate') {
        const quboFile = args[1];
        const bitsFile = args[2];
        if (!quboFile || !bitsFile) {
            console.error('Usage: node qubo.mjs --evaluate quboFile bitsFile');
            process.exit(1);
        }
        evaluateFromFile(quboFile, bitsFile);
    } else if (args[0] === '--benchmark-phase2') {
        runPhase2Benchmark();
    } else if (args[0] === '--solve-greedy') {
        const quboFile = args[1];
        const outFile = args[2];
        if (!quboFile || !outFile) {
            console.error('Usage: node qubo.mjs --solve-greedy quboFile outFile');
            process.exit(1);
        }
        const data = JSON.parse(fs.readFileSync(quboFile, 'utf8'));
        const graph = {
            n: data.n,
            hypotheses: data.graph.values.map((v, i) => ({ id: i, value: v, cost: data.graph.costs[i] })),
            edges: data.graph.edges,
            budget: data.graph.budget
        };
        const bits = greedySolve(graph);
        fs.writeFileSync(outFile, JSON.stringify({ bits }, null, 2));
        console.log(`[SOLVE-GREEDY] Written ${outFile}`);
    } else if (args[0] === '--solve-sa') {
        const quboFile = args[1];
        const outFile = args[2];
        if (!quboFile || !outFile) {
            console.error('Usage: node qubo.mjs --solve-sa quboFile outFile');
            process.exit(1);
        }
        const data = JSON.parse(fs.readFileSync(quboFile, 'utf8'));
        const graph = {
            n: data.n,
            hypotheses: data.graph.values.map((v, i) => ({ id: i, value: v, cost: data.graph.costs[i] })),
            edges: data.graph.edges,
            budget: data.graph.budget
        };
        // restarts=16: SA already handles >1 restarts with T0=15, cooling=0.9995,
        // steps=30000 (60000 for n>=2000) per run, different sub-seeds via seed + r*7919
        const bits = simulatedAnnealing(graph, 1.0, 2.0, { restarts: 16 }, data.n + 42);
        fs.writeFileSync(outFile, JSON.stringify({ bits }, null, 2));
        console.log(`[SOLVE-SA] Written ${outFile}`);
    } else {
        console.error('Unknown command:', args[0]);
        console.error('Available: --export-qubo N SEED, --evaluate quboFile bitsFile, --solve-greedy quboFile outFile, --solve-sa quboFile outFile, --benchmark-phase2');
        process.exit(1);
    }
}