"use client";

import { useState, useEffect } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell, Area, AreaChart,
} from "recharts";
import jsPDF from "jspdf";
import {
  BENCHMARK_DATA as FALLBACK_BENCHMARK_DATA,
  avg,
  type BenchmarkConfig,
  type BenchmarkMetadata,
} from "@/lib/benchmark-data";

type RunSummary = {
  id: string;
  label: string;
  date: string;
  model: string;
  gpu: string;
  framework: string;
  avgP99: number;
  avgThroughput: number;
  speedup: number;
  peakThroughput: number;
};

type ComparisonProfile = {
  id: string;
  label: string;
  model: string;
  gpu: string;
  framework: string;
  avgP99: number;
  avgThroughput: number;
  peakThroughput: number;
};

type BenchmarkApiPayload = BenchmarkMetadata & {
  runHistory?: RunSummary[];
  comparisonCandidates?: ComparisonProfile[];
};

function summarizeCurrentRun(data: BenchmarkMetadata): RunSummary {
  const fp16 = data.configs.filter((c) => c.quant === "FP16");
  const int4 = data.configs.filter((c) => c.quant === "INT4-AWQ");
  const fp16Throughput = avg(fp16.map((c) => c.throughput));
  const int4Throughput = avg(int4.map((c) => c.throughput));
  return {
    id: data.date,
    label: `${data.date} ${data.gpu}`,
    date: data.date,
    model: data.model,
    gpu: data.gpu,
    framework: data.framework,
    avgP99: avg(data.configs.map((c) => c.p99)),
    avgThroughput: avg(data.configs.map((c) => c.throughput)),
    speedup: Number((int4Throughput / fp16Throughput).toFixed(2)),
    peakThroughput: Math.max(...data.configs.map((c) => c.throughput)),
  };
}

// ── Hooks ──
function useTyping(text: string, speed = 50, delay = 0) {
  const [displayed, setDisplayed] = useState("");
  useEffect(() => {
    let i = 0;
    const timeout = setTimeout(() => {
      const interval = setInterval(() => {
        if (i < text.length) { setDisplayed(text.slice(0, i + 1)); i++; }
        else clearInterval(interval);
      }, speed);
      return () => clearInterval(interval);
    }, delay);
    return () => clearTimeout(timeout);
  }, []);
  return displayed;
}

// ── Sub-components ──
function HackerTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="border border-[#00ff41] px-4 py-3 shadow-lg shadow-[#00ff41]/10"
      style={{ background: "rgba(0,8,4,0.96)", fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
      <div className="text-[#00ff41] font-bold text-xs mb-1.5">{`> ${label}`}</div>
      {payload.map((p: any, i: number) => (
        <div key={i} style={{ color: p.color }} className="mt-1">
          {`  ${p.name}: ${typeof p.value === 'number' ? p.value.toFixed(1) : p.value}`}
        </div>
      ))}
    </div>
  );
}

function StatCard({ label, value, unit, sub, accent = "#00ff41", delay = 0 }: {
  label: string; value: string; unit: string; sub?: string; accent?: string; delay?: number;
}) {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), delay); return () => clearTimeout(t); }, [delay]);
  return (
    <div className="w-full sm:flex-1 sm:min-w-[200px] transition-all duration-600"
      style={{
        background: `linear-gradient(135deg, ${accent}06, ${accent}02)`,
        border: `1px solid ${accent}22`, padding: 24,
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(12px)",
        transitionTimingFunction: "cubic-bezier(0.16,1,0.3,1)",
      }}>
      <div className="text-[10px] tracking-[3px] uppercase" style={{ color: "#4a7c59" }}>{label}</div>
      <div className="mt-2 leading-none" style={{ fontSize: 40, fontWeight: 800, color: accent }}>
        {value}<span className="ml-1" style={{ fontSize: 18, color: `${accent}77` }}>{unit}</span>
      </div>
      {sub && <div className="mt-2 text-[10px]" style={{ color: "#4a7c59" }}>{sub}</div>}
    </div>
  );
}

function Tab({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className="text-[10px] sm:text-[11px] tracking-[1.5px] uppercase cursor-pointer transition-all duration-200 px-3 sm:px-6 py-2.5 border-none whitespace-nowrap"
      style={{
        background: active ? "rgba(0,255,65,0.08)" : "transparent",
        borderBottom: active ? "2px solid #00ff41" : "2px solid transparent",
        color: active ? "#00ff41" : "#3a6a45",
        fontFamily: "'JetBrains Mono', monospace",
      }}>{label}</button>
  );
}

function ChartBox({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <div className="p-4 sm:p-7" style={{ background: "linear-gradient(180deg, rgba(0,255,65,0.02) 0%, rgba(0,255,65,0.005) 100%)", border: "1px solid #0d2e15" }}>
      {title && (
        <h3 className="text-xs tracking-[2px] uppercase font-semibold mt-0 mb-4" style={{ color: "#2a7a3a" }}>
          <span className="text-[#00ff41]">$</span> {title}
        </h3>
      )}
      {children}
    </div>
  );
}

// ── Chart constants ──
const gc = { grid: "#0d2e15", axis: "#0d2e15" };
const tickStyle = { fill: "#4a7c59", fontSize: 10, fontFamily: "JetBrains Mono" };
const tickStyleSm = { ...tickStyle, fontSize: 9 };

// ── Main Dashboard ──
export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedBatch, setSelectedBatch] = useState("all");
  const [selectedTokens, setSelectedTokens] = useState("all");
  const [hoveredConfig, setHoveredConfig] = useState<number | null>(null);
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkMetadata>(FALLBACK_BENCHMARK_DATA);
  const [runHistory, setRunHistory] = useState<RunSummary[]>([summarizeCurrentRun(FALLBACK_BENCHMARK_DATA)]);
  const [comparisonCandidates, setComparisonCandidates] = useState<ComparisonProfile[]>([]);
  const [selectedComparison1Id, setSelectedComparison1Id] = useState<string>("");
  const [selectedComparison2Id, setSelectedComparison2Id] = useState<string>("");
  const title = useTyping("llm-inference-bench", 55, 200);
  const subtitle = useTyping("Mistral-7B // vLLM 0.16.0 // NVIDIA L4 24GB // AWQ-Marlin", 25, 1400);

  useEffect(() => {
    let mounted = true;

    const loadData = async () => {
      try {
        const response = await fetch("/api/benchmark-data", { cache: "no-store" });
        if (!response.ok) return;
        const data = (await response.json()) as BenchmarkApiPayload;
        if (!mounted || !data?.configs?.length) return;
        setBenchmarkData(data);
        setRunHistory(data.runHistory?.length ? data.runHistory : [summarizeCurrentRun(data)]);
        setComparisonCandidates(data.comparisonCandidates ?? []);
      } catch {
        // Keep fallback dataset if API read fails.
      }
    };

    loadData();
    return () => {
      mounted = false;
    };
  }, []);

  const BENCHMARK_DATA = benchmarkData;
  const fp16Configs = BENCHMARK_DATA.configs.filter((c) => c.quant === "FP16");
  const int4Configs = BENCHMARK_DATA.configs.filter((c) => c.quant === "INT4-AWQ");

  const fp16Avg = {
    p50: avg(fp16Configs.map((c) => c.p50)),
    p99: avg(fp16Configs.map((c) => c.p99)),
    throughput: avg(fp16Configs.map((c) => c.throughput)),
  };

  const int4Avg = {
    p50: avg(int4Configs.map((c) => c.p50)),
    p99: avg(int4Configs.map((c) => c.p99)),
    throughput: avg(int4Configs.map((c) => c.throughput)),
  };

  const speedup = (int4Avg.throughput / fp16Avg.throughput).toFixed(1);
  const p99Reduction = ((1 - int4Avg.p99 / fp16Avg.p99) * 100).toFixed(1);
  const peakThroughput = Math.max(...int4Configs.map((c) => c.throughput)).toFixed(1);

  const batchScaling = [1, 4, 8].map((b) => ({
    batch: `BS=${b}`,
    FP16: avg(fp16Configs.filter((c) => c.batch === b).map((c) => c.throughput)),
    "INT4-AWQ": avg(int4Configs.filter((c) => c.batch === b).map((c) => c.throughput)),
  }));

  const tokenScaling = [128, 256, 512].map((t) => ({
    tokens: `${t} tok`,
    FP16: avg(fp16Configs.filter((c) => c.tokens === t).map((c) => c.throughput)),
    "INT4-AWQ": avg(int4Configs.filter((c) => c.tokens === t).map((c) => c.throughput)),
  }));

  const speedupByBatch = [1, 4, 8].map((b) => {
    const fp = fp16Configs.filter((c) => c.batch === b);
    const iq = int4Configs.filter((c) => c.batch === b);
    return {
      batch: `Batch ${b}`,
      speedup: Number((avg(iq.map((c) => c.throughput)) / avg(fp.map((c) => c.throughput))).toFixed(2)),
    };
  });

  const rpsByBatch = [1, 4, 8].map((b) => ({
    batch: `BS=${b}`,
    FP16: avg(fp16Configs.filter((c) => c.batch === b).map((c) => c.rps)),
    "INT4-AWQ": avg(int4Configs.filter((c) => c.batch === b).map((c) => c.rps)),
  }));

  const filteredData = BENCHMARK_DATA.configs.filter((c) => {
    if (selectedBatch !== "all" && c.batch !== Number(selectedBatch)) return false;
    if (selectedTokens !== "all" && c.tokens !== Number(selectedTokens)) return false;
    return true;
  });

  const latencyData = [
    { name: "FP16", P50: fp16Avg.p50, P99: fp16Avg.p99 },
    { name: "INT4-AWQ", P50: int4Avg.p50, P99: int4Avg.p99 },
  ];

  const throughputData = [
    { name: "FP16", value: fp16Avg.throughput },
    { name: "INT4-AWQ", value: int4Avg.throughput },
  ];

  const headToHead = [1, 4, 8].flatMap((b) =>
    [128, 256, 512].map((t) => {
      const fp16 = BENCHMARK_DATA.configs.find((c) => c.quant === "FP16" && c.batch === b && c.tokens === t);
      const int4 = BENCHMARK_DATA.configs.find((c) => c.quant === "INT4-AWQ" && c.batch === b && c.tokens === t);
      return { name: `B${b}×T${t}`, FP16: fp16?.p50 || 0, "INT4-AWQ": int4?.p50 || 0 };
    })
  );

  const currentSummary = summarizeCurrentRun(BENCHMARK_DATA);
  
  const selectedComparison1 = comparisonCandidates.find((c) => c.id === selectedComparison1Id);
  const selectedComparison2 = comparisonCandidates.find((c) => c.id === selectedComparison2Id);
  
  // Build comparison table data with win/loss indicators
  const buildComparisonTable = (p1: typeof selectedComparison1, p2: typeof selectedComparison2) => {
    if (!p1 || !p2) return [];
    return [
      {
        metric: "Throughput (tok/s)",
        p1Value: p1.avgThroughput,
        p2Value: p2.avgThroughput,
        higherIsBetter: true,
      },
      {
        metric: "P99 Latency (ms)",
        p1Value: p1.avgP99,
        p2Value: p2.avgP99,
        higherIsBetter: false,
      },
      {
        metric: "Peak Throughput (tok/s)",
        p1Value: p1.peakThroughput,
        p2Value: p2.peakThroughput,
        higherIsBetter: true,
      },
    ];
  };
  
  const comparisonTable = buildComparisonTable(selectedComparison1, selectedComparison2);

  const applyScenarioPreset = (preset: "low-latency" | "max-throughput" | "balanced" | "cost-efficient") => {
    if (preset === "low-latency") {
      setSelectedBatch("1");
      setSelectedTokens("128");
      return;
    }
    if (preset === "max-throughput") {
      setSelectedBatch("8");
      setSelectedTokens("128");
      return;
    }
    if (preset === "balanced") {
      setSelectedBatch("4");
      setSelectedTokens("256");
      return;
    }
    setSelectedBatch("4");
    setSelectedTokens("128");
  };

  const exportPdfReport = () => {
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const marginX = 40;
    let y = 50;

    const addLine = (text: string, size = 10) => {
      doc.setFontSize(size);
      doc.text(text, marginX, y);
      y += size + 8;
      if (y > 780) {
        doc.addPage();
        y = 50;
      }
    };

    addLine("LLM Inference Benchmark Report", 16);
    addLine(`Date: ${BENCHMARK_DATA.date}`);
    addLine(`Model: ${BENCHMARK_DATA.model}`);
    addLine(`GPU: ${BENCHMARK_DATA.gpu}`);
    addLine(`Framework: ${BENCHMARK_DATA.framework}`);
    y += 6;

    addLine("Headline Metrics", 12);
    addLine(`Throughput speedup (INT4 vs FP16): ${speedup}x`);
    addLine(`P99 latency reduction: ${p99Reduction}%`);
    addLine(`Peak throughput: ${peakThroughput} tok/s`);
    y += 6;

    addLine("Filter State", 12);
    addLine(`Batch filter: ${selectedBatch}`);
    addLine(`Tokens filter: ${selectedTokens}`);
    y += 6;

    addLine("Config Table (filtered)", 12);
    filteredData.forEach((c: BenchmarkConfig) => {
      addLine(
        `${c.quant} | B${c.batch} T${c.tokens} | P50 ${c.p50.toFixed(1)} | P90 ${c.p90.toFixed(1)} | P99 ${c.p99.toFixed(1)} | tok/s ${c.throughput.toFixed(2)} | req/s ${c.rps.toFixed(2)}`
      );
    });
    y += 6;

    addLine("Run History", 12);
    runHistory.forEach((r) => {
      addLine(
        `${r.date} | ${r.gpu} | ${r.model} | speedup ${r.speedup.toFixed(2)}x | avg p99 ${r.avgP99.toFixed(1)} ms | avg throughput ${r.avgThroughput.toFixed(1)} tok/s`
      );
    });

    doc.save(`benchmark-report-${BENCHMARK_DATA.date}.pdf`);
  };

  return (
    <div className="min-h-screen relative" style={{ background: "#010a04", color: "#00ff41", fontFamily: "'JetBrains Mono', monospace" }}>
      <div className="scanlines" />
      <div className="grid-bg" />

      <div className="max-w-[1200px] mx-auto px-4 sm:px-6 py-8 sm:py-12 relative z-[1]">
        {/* ── Header ── */}
        <header className="mb-14 pb-9" style={{ borderBottom: "1px solid #0d2e15" }}>
          <div className="text-[10px] tracking-[4px] uppercase mb-4" style={{ color: "#1a5c2a" }}>
            Bereket Lemma // ML Systems Engineering
          </div>
          <h1 className="text-[30px] sm:text-[44px] font-extrabold m-0 leading-none -tracking-[1px]" style={{ color: "#00ff41" }}>
            {title}<span style={{ animation: "blink 1s step-end infinite", fontWeight: 400 }}>_</span>
          </h1>
          <p className="text-[13px] mt-3.5 min-h-[20px] tracking-[0.5px]" style={{ color: "#2a7a3a" }}>{subtitle}</p>
          <div className="flex gap-5 mt-6 text-[10px] flex-wrap tracking-[1px]" style={{ color: "#1a5c2a" }}>
            {[`GPU: ${BENCHMARK_DATA.gpu}`, `Engine: ${BENCHMARK_DATA.framework}`, `Configs: ${BENCHMARK_DATA.configs.length}`, `Runs: 10/config`, `Warmup: 3 iters`, `Date: ${BENCHMARK_DATA.date}`].map((item, i) => (
              <span key={i}>{i > 0 && <span className="mr-5" style={{ color: "#0d2e15" }}>|</span>}{item}</span>
            ))}
          </div>
          
          {/* Quick Explanation */}
          <div className="mt-6 p-4 rounded" style={{ background: "rgba(0,255,65,0.06)", border: "1px solid rgba(0,255,65,0.15)" }}>
            <div className="text-[10px] tracking-[1px] uppercase mb-2" style={{ color: "#2a7a3a" }}>What You're Looking At</div>
            <div className="text-[11px] leading-relaxed" style={{ color: "#5a9a69" }}>
              This benchmark compares two ways to run AI models: <span style={{ color: "#00ff41" }}>FP16 (full precision)</span> vs <span style={{ color: "#00d4ff" }}>INT4-AWQ (compressed)</span>. Compression makes the model run faster and use less memory, but we test if quality is affected. Below you'll see speed (throughput), response time (latency), and how performance scales with different loads.
            </div>
          </div>
        </header>

        {/* ── Stats ── */}
        <div className="flex gap-3 flex-wrap mb-4 items-center">
          <span className="text-[10px] uppercase tracking-[2px]" style={{ color: "#4a7c59" }}>Scenario Presets</span>
          <button onClick={() => applyScenarioPreset("low-latency")} className="text-[10px] px-3 py-1.5" style={{ border: "1px solid #0d2e15", color: "#00ff41", background: "transparent" }}>Low Latency</button>
          <button onClick={() => applyScenarioPreset("max-throughput")} className="text-[10px] px-3 py-1.5" style={{ border: "1px solid #0d2e15", color: "#00ff41", background: "transparent" }}>Max Throughput</button>
          <button onClick={() => applyScenarioPreset("balanced")} className="text-[10px] px-3 py-1.5" style={{ border: "1px solid #0d2e15", color: "#00ff41", background: "transparent" }}>Balanced</button>
          <button onClick={() => applyScenarioPreset("cost-efficient")} className="text-[10px] px-3 py-1.5" style={{ border: "1px solid #0d2e15", color: "#00ff41", background: "transparent" }}>Cost Efficient</button>
          <button onClick={exportPdfReport} className="text-[10px] px-3 py-1.5 ml-auto" style={{ border: "1px solid #00ff41", color: "#00ff41", background: "rgba(0,255,65,0.08)" }}>
            Export PDF
          </button>
        </div>

        <div className="flex gap-4 flex-wrap mb-12">
          <StatCard label="Throughput Speedup" value={speedup} unit="x" sub="Compressed version is this many times faster" accent="#00ff41" delay={200} />
          <StatCard label="P99 Latency Improved" value={p99Reduction} unit="%" sub="Worst-case response time got faster" accent="#00d4ff" delay={400} />
          <StatCard label="Peak Speed" value={peakThroughput} unit="tok/s" sub="Max words/sec the model can generate" accent="#ff6b35" delay={600} />
          <StatCard label="Test Scenarios" value={String(BENCHMARK_DATA.configs.length)} unit="" sub="Different batch sizes & input lengths" accent="#a855f7" delay={800} />
        </div>

        {/* ── Tabs ── */}
        <nav className="flex gap-0 mb-9 overflow-x-auto" style={{ borderBottom: "1px solid #0d2e15" }}>
          {["overview", "latency", "throughput", "scaling", "history", "compare", "explorer"].map((tab) => (
            <Tab key={tab} label={tab} active={activeTab === tab} onClick={() => setActiveTab(tab)} />
          ))}
        </nav>

        {/* ── Overview ── */}
        {activeTab === "overview" && (
          <div className="flex flex-col gap-7">
            {/* Tab Description */}
            <div className="p-4 rounded" style={{ background: "rgba(0,212,255,0.06)", border: "1px solid rgba(0,212,255,0.15)" }}>
              <div className="text-[10px] tracking-[1px] uppercase mb-2" style={{ color: "#2a7a3a" }}>Overview Tab</div>
              <div className="text-[11px] leading-relaxed" style={{ color: "#5a9a69" }}>
                <div><strong style={{ color: "#00ff41" }}>Latency (response time):</strong> How long it takes the model to respond. Lower is better for interactive apps.</div>
                <div className="mt-1"><strong style={{ color: "#00d4ff" }}>Throughput (speed):</strong> How many words the model generates per second. Higher is better for processing large batches.</div>
                <div className="mt-1"><strong>P50 vs P99:</strong> P50 is typical performance, P99 is worst-case. Production systems care about P99.</div>
              </div>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ChartBox title="Avg Latency (ms)">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={latencyData} barGap={6}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                    <XAxis dataKey="name" tick={{ ...tickStyle, fontSize: 11 }} axisLine={{ stroke: gc.axis }} />
                    <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}s`} />
                    <Tooltip content={<HackerTooltip />} />
                    <Bar dataKey="P50" fill="#00ff41" radius={[2, 2, 0, 0]} name="P50 (ms)" />
                    <Bar dataKey="P99" fill="#ff4444" radius={[2, 2, 0, 0]} name="P99 (ms)" />
                  </BarChart>
                </ResponsiveContainer>
                <div className="text-[10px] mt-2" style={{ color: "#1a5c2a" }}>↓ Lower is better</div>
              </ChartBox>

              <ChartBox title="Avg Throughput (tok/s)">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={throughputData} barGap={6}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                    <XAxis dataKey="name" tick={{ ...tickStyle, fontSize: 11 }} axisLine={{ stroke: gc.axis }} />
                    <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} />
                    <Tooltip content={<HackerTooltip />} />
                    <Bar dataKey="value" name="Tokens/sec" radius={[2, 2, 0, 0]}>
                      <Cell fill="#00ff41" /><Cell fill="#00d4ff" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="text-[10px] mt-2" style={{ color: "#1a5c2a" }}>↑ Higher is better — {speedup}x with AWQ-Marlin</div>
              </ChartBox>
            </div>

            <ChartBox title="Throughput Scaling by Batch Size">
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={batchScaling}>
                  <defs>
                    <linearGradient id="gF" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#00ff41" stopOpacity={0.25} /><stop offset="100%" stopColor="#00ff41" stopOpacity={0} /></linearGradient>
                    <linearGradient id="gI" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#00d4ff" stopOpacity={0.25} /><stop offset="100%" stopColor="#00d4ff" stopOpacity={0} /></linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                  <XAxis dataKey="batch" tick={{ ...tickStyle, fontSize: 11 }} axisLine={{ stroke: gc.axis }} />
                  <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} />
                  <Tooltip content={<HackerTooltip />} />
                  <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11, color: "#4a7c59" }} />
                  <Area type="monotone" dataKey="FP16" stroke="#00ff41" strokeWidth={2.5} fill="url(#gF)" dot={{ fill: "#00ff41", r: 5, strokeWidth: 0 }} name="FP16 (tok/s)" />
                  <Area type="monotone" dataKey="INT4-AWQ" stroke="#00d4ff" strokeWidth={2.5} fill="url(#gI)" dot={{ fill: "#00d4ff", r: 5, strokeWidth: 0 }} name="INT4-AWQ (tok/s)" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartBox>
          </div>
        )}

        {/* ── Latency ── */}
        {activeTab === "latency" && (
          <div className="flex flex-col gap-7">
            {/* Tab Description */}
            <div className="p-4 rounded" style={{ background: "rgba(255,107,53,0.06)", border: "1px solid rgba(255,107,53,0.15)" }}>
              <div className="text-[10px] tracking-[1px] uppercase mb-2" style={{ color: "#2a7a3a" }}>Latency Tab</div>
              <div className="text-[11px] leading-relaxed" style={{ color: "#5a9a69" }}>
                <div><strong>Time to first response:</strong> How long users wait before seeing the model start responding.</div>
                <div className="mt-1"><strong>P50:</strong> What 50% of requests experience (typical case)</div>
                <div className="mt-1"><strong>P99:</strong> What the slowest 1% of requests experience (worst case - still matters for user experience)</div>
                <div className="mt-1">Green <span style={{color: "#00ff41"}}>FP16</span> vs Blue <span style={{color: "#00d4ff"}}>INT4-AWQ</span> - can compression stay fast?</div>
              </div>
            </div>
            <ChartBox title="P50 Latency: FP16 vs INT4-AWQ">
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={headToHead} barGap={2}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                  <XAxis dataKey="name" tick={tickStyleSm} axisLine={{ stroke: gc.axis }} angle={-45} textAnchor="end" height={70} />
                  <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}s`} />
                  <Tooltip content={<HackerTooltip />} />
                  <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 10 }} />
                  <Bar dataKey="FP16" fill="#00ff4166" radius={[2, 2, 0, 0]} name="FP16 (ms)" />
                  <Bar dataKey="INT4-AWQ" fill="#00d4ff88" radius={[2, 2, 0, 0]} name="INT4-AWQ (ms)" />
                </BarChart>
              </ResponsiveContainer>
            </ChartBox>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {(["FP16", "INT4-AWQ"] as const).map((q) => (
                <ChartBox key={q} title={`${q} — P99 Tail Latency`}>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={BENCHMARK_DATA.configs.filter((c) => c.quant === q).map((c) => ({ name: `B${c.batch}×T${c.tokens}`, P99: c.p99 }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                      <XAxis dataKey="name" tick={tickStyleSm} axisLine={{ stroke: gc.axis }} angle={-45} textAnchor="end" height={60} />
                      <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}s`} />
                      <Tooltip content={<HackerTooltip />} />
                      <Bar dataKey="P99" fill={q === "FP16" ? "#00ff4166" : "#00d4ff88"} radius={[2, 2, 0, 0]} name="P99 (ms)" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartBox>
              ))}
            </div>
          </div>
        )}

        {/* ── Throughput ── */}
        {activeTab === "throughput" && (
          <div className="flex flex-col gap-7">
            {/* Tab Description */}
            <div className="p-4 rounded" style={{ background: "rgba(168,85,247,0.06)", border: "1px solid rgba(168,85,247,0.15)" }}>
              <div className="text-[10px] tracking-[1px] uppercase mb-2" style={{ color: "#2a7a3a" }}>Throughput Tab</div>
              <div className="text-[11px] leading-relaxed" style={{ color: "#5a9a69" }}>
                <div><strong style={{ color: "#00ff41" }}>Tokens/second:</strong> How many words/pieces of text the model generates per second. Higher = more productive.</div>
                <div className="mt-1"><strong>Speedup Ratio:</strong> If compressed is 2x faster, you can serve 2x more users on same hardware = cost savings.</div>
                <div className="mt-1">Test setup varies batch sizes and input lengths to show how performance changes with different workloads.</div>
              </div>
            </div>
            <ChartBox title="Throughput by Sequence Length">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={tokenScaling} barGap={8}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                  <XAxis dataKey="tokens" tick={{ ...tickStyle, fontSize: 11 }} axisLine={{ stroke: gc.axis }} />
                  <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} />
                  <Tooltip content={<HackerTooltip />} />
                  <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                  <Bar dataKey="FP16" fill="#00ff41" radius={[2, 2, 0, 0]} name="FP16 (tok/s)" />
                  <Bar dataKey="INT4-AWQ" fill="#00d4ff" radius={[2, 2, 0, 0]} name="INT4-AWQ (tok/s)" />
                </BarChart>
              </ResponsiveContainer>
            </ChartBox>
            <ChartBox title="Speedup Ratio per Batch Size">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={speedupByBatch}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                  <XAxis dataKey="batch" tick={{ ...tickStyle, fontSize: 11 }} axisLine={{ stroke: gc.axis }} />
                  <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} domain={[0, 5]} tickFormatter={(v) => `${v}x`} />
                  <Tooltip content={<HackerTooltip />} />
                  <Bar dataKey="speedup" name="Speedup (x)" radius={[2, 2, 0, 0]}>
                    <Cell fill="#ff6b35" /><Cell fill="#00ff41" /><Cell fill="#00d4ff" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartBox>
          </div>
        )}

        {/* ── Scaling ── */}
        {activeTab === "scaling" && (
          <div className="flex flex-col gap-7">
            {/* Tab Description */}
            <div className="p-4 rounded" style={{ background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.15)" }}>
              <div className="text-[10px] tracking-[1px] uppercase mb-2" style={{ color: "#2a7a3a" }}>Scaling Tab</div>
              <div className="text-[11px] leading-relaxed" style={{ color: "#5a9a69" }}>
                <div><strong>Batch Size:</strong> How many requests the system processes together. Batching makes GPU more efficient.</div>
                <div className="mt-1"><strong>Scaling behavior:</strong> Does compressed format follow same pattern as original? Ideally both should scale well.</div>
                <div className="mt-1"><strong>Requests/sec:</strong> Real-world metric - how many users can you serve simultaneously without queuing delays.</div>
              </div>
            </div>
            <ChartBox title="Throughput vs Batch Size">
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={batchScaling}>
                  <defs>
                    <linearGradient id="gFs" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#00ff41" stopOpacity={0.3} /><stop offset="100%" stopColor="#00ff41" stopOpacity={0} /></linearGradient>
                    <linearGradient id="gIs" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#00d4ff" stopOpacity={0.3} /><stop offset="100%" stopColor="#00d4ff" stopOpacity={0} /></linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                  <XAxis dataKey="batch" tick={{ ...tickStyle, fontSize: 11 }} axisLine={{ stroke: gc.axis }} />
                  <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} />
                  <Tooltip content={<HackerTooltip />} />
                  <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                  <Area type="monotone" dataKey="FP16" stroke="#00ff41" strokeWidth={2.5} fill="url(#gFs)" dot={{ fill: "#00ff41", r: 6 }} />
                  <Area type="monotone" dataKey="INT4-AWQ" stroke="#00d4ff" strokeWidth={2.5} fill="url(#gIs)" dot={{ fill: "#00d4ff", r: 6 }} />
                </AreaChart>
              </ResponsiveContainer>
            </ChartBox>
            <ChartBox title="Requests/sec Scaling">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={rpsByBatch}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                  <XAxis dataKey="batch" tick={{ ...tickStyle, fontSize: 11 }} axisLine={{ stroke: gc.axis }} />
                  <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} />
                  <Tooltip content={<HackerTooltip />} />
                  <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                  <Line type="monotone" dataKey="FP16" stroke="#00ff41" strokeWidth={2.5} dot={{ fill: "#00ff41", r: 5 }} />
                  <Line type="monotone" dataKey="INT4-AWQ" stroke="#00d4ff" strokeWidth={2.5} dot={{ fill: "#00d4ff", r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </ChartBox>
          </div>
        )}

        {/* ── History ── */}
        {activeTab === "history" && (
          <div className="flex flex-col gap-7">
            <ChartBox title="Run History Trends">
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={runHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                  <XAxis dataKey="date" tick={{ ...tickStyle, fontSize: 11 }} axisLine={{ stroke: gc.axis }} />
                  <YAxis yAxisId="left" tick={tickStyle} axisLine={{ stroke: gc.axis }} />
                  <YAxis yAxisId="right" orientation="right" tick={tickStyle} axisLine={{ stroke: gc.axis }} />
                  <Tooltip content={<HackerTooltip />} />
                  <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                  <Line yAxisId="left" type="monotone" dataKey="avgThroughput" stroke="#00ff41" strokeWidth={2.5} name="Avg Throughput" dot={{ fill: "#00ff41", r: 4 }} />
                  <Line yAxisId="left" type="monotone" dataKey="peakThroughput" stroke="#00d4ff" strokeWidth={2.5} name="Peak Throughput" dot={{ fill: "#00d4ff", r: 4 }} />
                  <Line yAxisId="right" type="monotone" dataKey="speedup" stroke="#ff6b35" strokeWidth={2.5} name="INT4/FP16 Speedup" dot={{ fill: "#ff6b35", r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </ChartBox>

            <ChartBox title="Run History Table">
              <div className="overflow-auto" style={{ border: "1px solid #0d2e15" }}>
                <table className="w-full" style={{ borderCollapse: "collapse", fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #0d2e15" }}>
                      {["Date", "GPU", "Model", "Speedup", "Avg P99 (ms)", "Avg Throughput", "Peak Throughput"].map((h) => (
                        <th key={h} className="text-left text-[10px] tracking-[1.5px] uppercase font-semibold px-4 py-3.5" style={{ color: "#2a7a3a" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {runHistory.map((r) => (
                      <tr key={r.id} style={{ borderBottom: "1px solid #081a0c" }}>
                        <td className="px-4 py-3" style={{ color: "#5a9a69" }}>{r.date}</td>
                        <td className="px-4 py-3" style={{ color: "#5a9a69" }}>{r.gpu}</td>
                        <td className="px-4 py-3" style={{ color: "#5a9a69" }}>{r.model}</td>
                        <td className="px-4 py-3" style={{ color: "#00ff41" }}>{r.speedup.toFixed(2)}x</td>
                        <td className="px-4 py-3" style={{ color: "#ff5555" }}>{r.avgP99.toFixed(1)}</td>
                        <td className="px-4 py-3" style={{ color: "#00d4ff" }}>{r.avgThroughput.toFixed(1)}</td>
                        <td className="px-4 py-3" style={{ color: "#00d4ff" }}>{r.peakThroughput.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {runHistory.length < 2 && (
                <div className="mt-3 text-[10px]" style={{ color: "#4a7c59" }}>
                  Add JSON snapshots under <span style={{ color: "#00ff41" }}>backend/results/history</span> to see multi-run trends.
                </div>
              )}
            </ChartBox>
          </div>
        )}

        {/* ── Compare ── */}
        {activeTab === "compare" && (
          <div className="flex flex-col gap-7">
            {/* Tab Description */}
            <div className="p-4 rounded" style={{ background: "rgba(59,130,246,0.06)", border: "1px solid rgba(59,130,246,0.15)" }}>
              <div className="text-[10px] tracking-[1px] uppercase mb-2" style={{ color: "#2a7a3a" }}>Compare Tab</div>
              <div className="text-[11px] leading-relaxed" style={{ color: "#5a9a69" }}>
                <div><strong>Choose 2 profiles</strong> below and see which performs better for your use case.</div>
                <div className="mt-1"><strong>Green bar wins:</strong> If Profile 1 is taller = more throughput OR shorter P99 = it's better for that metric.</div>
                <div className="mt-1">Use this to decide: Should I upgrade hardware, change the model, or keep current setup?</div>
              </div>
            </div>
            <ChartBox title="Head-to-Head Profile Comparison">
              <div className="flex flex-col gap-5 mb-6">
                {/* Profile Selectors */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="text-[10px] tracking-[1px] uppercase font-semibold block mb-2" style={{ color: "#2a7a3a" }}>Profile 1 (Left Side)</label>
                    <select
                      value={selectedComparison1Id}
                      onChange={(e) => setSelectedComparison1Id(e.target.value)}
                      className="text-[11px] px-3 py-2 w-full"
                      style={{ background: "#021108", color: "#00ff41", border: "1px solid #0d2e15" }}
                    >
                      <option value="">Select first profile</option>
                      {comparisonCandidates.map((c) => (
                        <option key={c.id} value={c.id}>{c.label}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] tracking-[1px] uppercase font-semibold block mb-2" style={{ color: "#2a7a3a" }}>Profile 2 (Right Side)</label>
                    <select
                      value={selectedComparison2Id}
                      onChange={(e) => setSelectedComparison2Id(e.target.value)}
                      className="text-[11px] px-3 py-2 w-full"
                      style={{ background: "#021108", color: "#00d4ff", border: "1px solid #0d2e15" }}
                    >
                      <option value="">Select second profile</option>
                      {comparisonCandidates.map((c) => (
                        <option key={c.id} value={c.id}>{c.label}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Profile Details */}
                {selectedComparison1 && selectedComparison2 && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 py-3">
                    <div className="p-3 rounded" style={{ background: "rgba(0,255,65,0.08)", border: "1px solid rgba(0,255,65,0.2)" }}>
                      <div className="text-[10px] tracking-[1px] uppercase mb-1" style={{ color: "#2a7a3a" }}>Profile 1</div>
                      <div className="text-[12px] font-bold mb-1" style={{ color: "#00ff41" }}>{selectedComparison1.label}</div>
                      <div className="text-[9px]" style={{ color: "#5a9a69" }}>
                        <div>{selectedComparison1.gpu}</div>
                        <div>{selectedComparison1.model}</div>
                      </div>
                    </div>
                    <div className="p-3 rounded" style={{ background: "rgba(0,212,255,0.08)", border: "1px solid rgba(0,212,255,0.2)" }}>
                      <div className="text-[10px] tracking-[1px] uppercase mb-1" style={{ color: "#2a7a3a" }}>Profile 2</div>
                      <div className="text-[12px] font-bold mb-1" style={{ color: "#00d4ff" }}>{selectedComparison2.label}</div>
                      <div className="text-[9px]" style={{ color: "#5a9a69" }}>
                        <div>{selectedComparison2.gpu}</div>
                        <div>{selectedComparison2.model}</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {selectedComparison1 && selectedComparison2 ? (
                <>
                  {/* Comparison Chart */}
                  <div className="mb-6 pb-6" style={{ borderBottom: "1px solid #0d2e15" }}>
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart data={[
                        { metric: "Throughput", p1: selectedComparison1.avgThroughput, p2: selectedComparison2.avgThroughput },
                        { metric: "Peak Thru", p1: selectedComparison1.peakThroughput, p2: selectedComparison2.peakThroughput },
                        { metric: "P99 (ms)", p1: selectedComparison1.avgP99, p2: selectedComparison2.avgP99 },
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" stroke={gc.grid} />
                        <XAxis dataKey="metric" tick={{ ...tickStyle, fontSize: 11 }} axisLine={{ stroke: gc.axis }} />
                        <YAxis tick={tickStyle} axisLine={{ stroke: gc.axis }} />
                        <Tooltip content={<HackerTooltip />} />
                        <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                        <Bar dataKey="p1" name="Profile 1" fill="#00ff41" radius={[2, 2, 0, 0]} />
                        <Bar dataKey="p2" name="Profile 2" fill="#00d4ff" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Detailed Comparison Table */}
                  <div className="overflow-auto">
                    <table className="w-full" style={{ borderCollapse: "collapse", fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid #0d2e15" }}>
                          <th className="text-left text-[10px] tracking-[1.5px] uppercase font-semibold px-4 py-3" style={{ color: "#2a7a3a" }}>Metric</th>
                          <th className="text-left text-[10px] tracking-[1.5px] uppercase font-semibold px-4 py-3" style={{ color: "#00ff41" }}>Profile 1 Value</th>
                          <th className="text-center text-[10px] tracking-[1.5px] uppercase font-semibold px-4 py-3" style={{ color: "#4a7c59" }}>Difference</th>
                          <th className="text-left text-[10px] tracking-[1.5px] uppercase font-semibold px-4 py-3" style={{ color: "#00d4ff" }}>Profile 2 Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonTable.map((row) => {
                          const higherIsBetter = row.higherIsBetter;
                          const diff = row.p2Value - row.p1Value;
                          const pctDiff = ((diff / row.p1Value) * 100).toFixed(1);
                          const p2Wins = (higherIsBetter && diff > 0) || (!higherIsBetter && diff < 0);
                          const p1Wins = (higherIsBetter && diff < 0) || (!higherIsBetter && diff > 0);
                          return (
                            <tr key={row.metric} style={{ borderBottom: "1px solid #081a0c" }}>
                              <td className="px-4 py-3 text-[11px]" style={{ color: "#5a9a69" }}>{row.metric}</td>
                              <td className="px-4 py-3 text-[11px]" style={{ color: p1Wins ? "#66ff00" : "#ff9999" }}>
                                {row.p1Value.toFixed(1)} {p1Wins ? "✓ BETTER" : ""}
                              </td>
                              <td className="px-4 py-3 text-[11px] text-center" style={{ color: p2Wins ? "#00ff41" : p1Wins ? "#ff5555" : "#5a9a69" }}>
                                {p2Wins ? "+" : ""}{pctDiff}%
                              </td>
                              <td className="px-4 py-3 text-[11px]" style={{ color: p2Wins ? "#66ff00" : "#ff9999" }}>
                                {row.p2Value.toFixed(1)} {p2Wins ? "✓ BETTER" : ""}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Winner Summary */}
                  <div className="mt-6 pt-4" style={{ borderTop: "1px solid #0d2e15" }}>
                    <div className="text-[10px] tracking-[1.5px] uppercase font-semibold mb-3" style={{ color: "#2a7a3a" }}>Performance Summary</div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div className="p-3 rounded" style={{ background: "rgba(0,255,65,0.08)", border: "1px solid rgba(0,255,65,0.2)" }}>
                        <div className="text-[10px]" style={{ color: "#00ff41" }}>✓ Profile 1 Better at:</div>
                        {comparisonTable.some((row) => ((row.higherIsBetter && row.p1Value > row.p2Value) || (!row.higherIsBetter && row.p1Value < row.p2Value))) ? (
                          <div className="text-[9px] mt-1" style={{ color: "#5a9a69" }}>
                            {comparisonTable.filter((row) => ((row.higherIsBetter && row.p1Value > row.p2Value) || (!row.higherIsBetter && row.p1Value < row.p2Value))).map((row) => row.metric).join(", ")}
                          </div>
                        ) : (
                          <div className="text-[9px] mt-1" style={{ color: "#5a9a69" }}>No advantages</div>
                        )}
                      </div>
                      <div className="p-3 rounded" style={{ background: "rgba(0,212,255,0.08)", border: "1px solid rgba(0,212,255,0.2)" }}>
                        <div className="text-[10px]" style={{ color: "#00d4ff" }}>✓ Profile 2 Better at:</div>
                        {comparisonTable.some((row) => ((row.higherIsBetter && row.p2Value > row.p1Value) || (!row.higherIsBetter && row.p2Value < row.p1Value))) ? (
                          <div className="text-[9px] mt-1" style={{ color: "#5a9a69" }}>
                            {comparisonTable.filter((row) => ((row.higherIsBetter && row.p2Value > row.p1Value) || (!row.higherIsBetter && row.p2Value < row.p1Value))).map((row) => row.metric).join(", ")}
                          </div>
                        ) : (
                          <div className="text-[9px] mt-1" style={{ color: "#5a9a69" }}>No advantages</div>
                        )}
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-[11px] py-6 text-center" style={{ color: "#4a7c59" }}>
                  Select two profiles above to compare them side-by-side
                </div>
              )}
            </ChartBox>
          </div>
        )}

        {/* ── Explorer ── */}
        {activeTab === "explorer" && (
          <div>
            {/* Tab Description */}
            <div className="p-4 rounded mb-6" style={{ background: "rgba(236,72,153,0.06)", border: "1px solid rgba(236,72,153,0.15)" }}>
              <div className="text-[10px] tracking-[1px] uppercase mb-2" style={{ color: "#2a7a3a" }}>Explorer Tab</div>
              <div className="text-[11px] leading-relaxed" style={{ color: "#5a9a69" }}>
                <div><strong>Deep dive into raw data:</strong> Filter by batch size and input length (tokens) to see exact metrics.</div>
                <div className="mt-1"><strong>What to look for:</strong> Green FP16 vs Blue INT4 - does compression stay fast across all scenarios?</div>
              </div>
            </div>
            <div className="flex gap-4 mb-6 flex-wrap items-center">
              <span className="text-[11px] font-semibold" style={{ color: "#2a7a3a" }}>$ FILTER:</span>
              {([
                ["Batch", ["all", "1", "4", "8"], selectedBatch, setSelectedBatch],
                ["Tokens", ["all", "128", "256", "512"], selectedTokens, setSelectedTokens],
              ] as const).map(([label, opts, val, setVal]: any) => (
                <div key={label} className="flex gap-1.5 flex-wrap">
                  <span className="text-[10px] self-center mr-1" style={{ color: "#4a7c59" }}>{label}</span>
                  {opts.map((o: string) => (
                    <button key={o} onClick={() => setVal(o)} className="text-[10px] cursor-pointer transition-all"
                      style={{
                        background: val === o ? "rgba(0,255,65,0.12)" : "transparent",
                        border: val === o ? "1px solid #00ff4144" : "1px solid #0d2e15",
                        color: val === o ? "#00ff41" : "#3a6a45",
                        padding: "4px 14px", fontFamily: "'JetBrains Mono', monospace",
                      }}>{o}</button>
                  ))}
                </div>
              ))}
              <span className="text-[10px] ml-2" style={{ color: "#1a5c2a" }}>[{filteredData.length} configs]</span>
            </div>

            <div className="overflow-auto" style={{ background: "rgba(0,255,65,0.015)", border: "1px solid #0d2e15" }}>
              <table className="w-full" style={{ borderCollapse: "collapse", fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #0d2e15" }}>
                    {["Quant", "Batch", "Tokens", "P50 (ms)", "P90 (ms)", "P99 (ms)", "Tok/s", "Req/s"].map((h) => (
                      <th key={h} className="text-left text-[10px] tracking-[1.5px] uppercase font-semibold px-4 py-3.5" style={{ color: "#2a7a3a" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((c, i) => (
                    <tr key={i}
                      onMouseEnter={() => setHoveredConfig(i)}
                      onMouseLeave={() => setHoveredConfig(null)}
                      className="transition-colors"
                      style={{
                        borderBottom: "1px solid #081a0c",
                        background: hoveredConfig === i ? "rgba(0,255,65,0.04)" : "transparent",
                      }}>
                      <td className="px-4 py-3 font-bold" style={{ color: c.quant === "FP16" ? "#00ff41" : "#00d4ff" }}>{c.quant}</td>
                      <td className="px-4 py-3" style={{ color: "#5a9a69" }}>{c.batch}</td>
                      <td className="px-4 py-3" style={{ color: "#5a9a69" }}>{c.tokens}</td>
                      <td className="px-4 py-3" style={{ color: "#00ff41" }}>{c.p50.toFixed(1)}</td>
                      <td className="px-4 py-3" style={{ color: "#ccaa00" }}>{c.p90.toFixed(1)}</td>
                      <td className="px-4 py-3" style={{ color: "#ff5555" }}>{c.p99.toFixed(1)}</td>
                      <td className="px-4 py-3 font-bold" style={{ color: "#00d4ff" }}>{c.throughput.toFixed(1)}</td>
                      <td className="px-4 py-3" style={{ color: "#5a9a69" }}>{c.rps.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Methodology ── */}
        <div className="mt-14 pt-9" style={{ borderTop: "1px solid #0d2e15" }}>
          <h3 className="text-xs tracking-[2px] uppercase font-semibold mt-0 mb-5" style={{ color: "#2a7a3a" }}>
            <span className="text-[#00ff41]">$</span> Methodology
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { l: "Warmup", v: "3 iterations", d: "Cold-start elimination" },
              { l: "Measurement", v: "10 runs/config", d: "Statistical stability" },
              { l: "Decoding", v: "Greedy (T=0.0)", d: "Deterministic output" },
              { l: "Engine", v: "vLLM 0.16.0", d: "PagedAttention + continuous batching" },
              { l: "Quantization", v: "AWQ-Marlin", d: "INT4 Marlin kernel (fast path)" },
              { l: "Hardware", v: "NVIDIA L4 24GB", d: "Google Cloud us-west1-a" },
            ].map((m) => (
              <div key={m.l} className="p-5" style={{ background: "rgba(0,255,65,0.015)", border: "1px solid #0d2e15" }}>
                <div className="text-[9px] tracking-[3px] uppercase" style={{ color: "#1a5c2a" }}>{m.l}</div>
                <div className="text-[15px] font-bold mt-1.5" style={{ color: "#00ff41" }}>{m.v}</div>
                <div className="text-[9px] mt-1.5" style={{ color: "#3a6a45" }}>{m.d}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Footer ── */}
        <footer className="mt-14 pt-7 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4" style={{ borderTop: "1px solid #0d2e15" }}>
          <div className="text-[10px] tracking-[1px]" style={{ color: "#1a5c2a" }}>
            Built by <span className="font-semibold" style={{ color: "#00ff41" }}>Bereket Lemma</span> // bereketlemma.com
          </div>
          <div className="flex gap-5 text-[10px]">
            <a href="https://github.com/bereketlemma/llm-inference-bench" target="_blank" rel="noopener noreferrer"
              className="no-underline pb-0.5" style={{ color: "#2a7a3a", borderBottom: "1px solid #0d2e15" }}>Source Code ↗</a>
            <a href="https://bereketlemma.com" target="_blank" rel="noopener noreferrer"
              className="no-underline pb-0.5" style={{ color: "#2a7a3a", borderBottom: "1px solid #0d2e15" }}>Portfolio ↗</a>
          </div>
        </footer>
      </div>
    </div>
  );
}
