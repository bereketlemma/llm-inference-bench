import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";
import { BENCHMARK_DATA as FALLBACK_BENCHMARK_DATA } from "../../../lib/benchmark-data";

type BenchmarkConfig = {
  quant: string;
  batch: number;
  tokens: number;
  p50: number;
  p90: number;
  p99: number;
  throughput: number;
  rps: number;
};

type BenchmarkMetadata = {
  model: string;
  gpu: string;
  framework: string;
  date: string;
  configs: BenchmarkConfig[];
};

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

const DEFAULT_COMPARISON_PROFILES: ComparisonProfile[] = [
  {
    id: "compare-a10g-mistral",
    label: "NVIDIA A10G - Mistral-7B",
    model: "Mistral-7B-v0.1",
    gpu: "NVIDIA A10G 24GB",
    framework: "vLLM 0.16.0",
    avgP99: 6120.0,
    avgThroughput: 182.4,
    peakThroughput: 540.8,
  },
  {
    id: "compare-l4-llama3-8b",
    label: "NVIDIA L4 - Llama-3 8B",
    model: "Llama-3-8B-Instruct",
    gpu: "NVIDIA L4 24GB",
    framework: "vLLM 0.16.0",
    avgP99: 7010.0,
    avgThroughput: 165.2,
    peakThroughput: 497.6,
  },
  {
    id: "compare-t4-mistral",
    label: "NVIDIA T4 - Mistral-7B",
    model: "Mistral-7B-v0.1",
    gpu: "NVIDIA T4 16GB",
    framework: "vLLM 0.16.0",
    avgP99: 10320.0,
    avgThroughput: 96.1,
    peakThroughput: 301.4,
  },
];

function parseCsv(text: string): Array<Record<string, string>> {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(",").map((v) => v.trim());
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = values[i] ?? "";
    });
    return row;
  });
}

function avg(nums: number[]) {
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function summarizeRun(base: BenchmarkMetadata): RunSummary {
  const fp16 = base.configs.filter((c) => c.quant === "FP16");
  const int4 = base.configs.filter((c) => c.quant === "INT4-AWQ");
  const fp16Throughput = avg(fp16.map((c) => c.throughput));
  const int4Throughput = avg(int4.map((c) => c.throughput));
  return {
    id: base.date,
    label: `${base.date} ${base.gpu}`,
    date: base.date,
    model: base.model,
    gpu: base.gpu,
    framework: base.framework,
    avgP99: avg(base.configs.map((c) => c.p99)),
    avgThroughput: avg(base.configs.map((c) => c.throughput)),
    speedup: Number((int4Throughput / fp16Throughput).toFixed(2)),
    peakThroughput: Math.max(...base.configs.map((c) => c.throughput)),
  };
}

function comparisonFromRun(run: RunSummary): ComparisonProfile {
  return {
    id: `run-${run.id}`,
    label: `${run.date} ${run.gpu}`,
    model: run.model,
    gpu: run.gpu,
    framework: run.framework,
    avgP99: run.avgP99,
    avgThroughput: run.avgThroughput,
    peakThroughput: run.peakThroughput,
  };
}

async function readTextFromFirstExistingPath(filePaths: string[]): Promise<string | null> {
  for (const filePath of filePaths) {
    try {
      return await fs.readFile(filePath, "utf8");
    } catch {
      // Try next candidate path.
    }
  }
  return null;
}

async function loadJsonArray<T>(dirPath: string): Promise<T[]> {
  try {
    const files = await fs.readdir(dirPath);
    const jsonFiles = files.filter((f) => f.toLowerCase().endsWith(".json"));
    const records = await Promise.all(
      jsonFiles.map(async (name) => {
        const raw = await fs.readFile(path.join(dirPath, name), "utf8");
        return JSON.parse(raw) as T;
      })
    );
    return records;
  } catch {
    return [];
  }
}

export async function GET() {
  try {
    const cwd = process.cwd();
    const candidateDataDirs = [
      path.join(cwd, "backend", "results", "Data"),
      path.join(cwd, "results", "Data"),
      path.join(cwd, "..", "backend", "results", "Data"),
      path.join(cwd, "..", "results", "Data"),
    ];

    const fp16Raw = await readTextFromFirstExistingPath(
      candidateDataDirs.map((dir) => path.join(dir, "fp16_results.csv"))
    );
    const awqRaw = await readTextFromFirstExistingPath(
      candidateDataDirs.map((dir) => path.join(dir, "awq_marlin_results.csv"))
    );

    const hasCsvData = Boolean(fp16Raw && awqRaw);

    const payload: BenchmarkMetadata = hasCsvData
      ? {
          model: "Mistral-7B-v0.1",
          gpu: "NVIDIA L4 24GB",
          framework: "vLLM 0.16.0",
          date: new Date().toISOString().slice(0, 10),
          configs: [
            ...parseCsv(fp16Raw as string).map((r) => ({
              quant: "FP16",
              batch: Number(r.batch_size),
              tokens: Number(r.output_tokens),
              p50: Number(r.p50_latency_ms),
              // fp16 CSV has no p90 column, so we surface p99 in both p90/p99 slots.
              p90: Number(r.p99_latency_ms),
              p99: Number(r.p99_latency_ms),
              throughput: Number(r.tokens_per_second),
              rps: Number(r.requests_per_second),
            })),
            ...parseCsv(awqRaw as string).map((r) => ({
              quant: "INT4-AWQ",
              batch: Number(r.batch_size),
              tokens: Number(r.output_tokens),
              p50: Number(r.p50_latency_ms),
              p90: Number(r.p90_latency_ms),
              p99: Number(r.p99_latency_ms),
              throughput: Number(r.tokens_per_second),
              rps: Number(r.requests_per_second),
            })),
          ],
        }
      : {
          ...FALLBACK_BENCHMARK_DATA,
          date: new Date().toISOString().slice(0, 10),
        };

    const currentSummary = summarizeRun(payload);
    const historyDirs = [
      path.join(cwd, "backend", "results", "history"),
      path.join(cwd, "results", "history"),
      path.join(cwd, "..", "backend", "results", "history"),
      path.join(cwd, "..", "results", "history"),
    ];
    const comparisonDirs = [
      path.join(cwd, "backend", "results", "comparisons"),
      path.join(cwd, "results", "comparisons"),
      path.join(cwd, "..", "backend", "results", "comparisons"),
      path.join(cwd, "..", "results", "comparisons"),
    ];

    const historyBatches = await Promise.all(historyDirs.map((dir) => loadJsonArray<RunSummary>(dir)));
    const comparisonBatches = await Promise.all(
      comparisonDirs.map((dir) => loadJsonArray<ComparisonProfile>(dir))
    );
    const historyFromDisk = historyBatches.flat();
    const comparisonsFromDisk = comparisonBatches.flat();

    const runHistory: RunSummary[] =
      historyFromDisk.length > 0
        ? [...historyFromDisk, currentSummary]
            .filter((r, i, arr) => arr.findIndex((x) => x.id === r.id) === i)
            .sort((a, b) => a.date.localeCompare(b.date))
        : [currentSummary];

    const comparisonCandidates: ComparisonProfile[] = [
      comparisonFromRun(currentSummary),
      ...DEFAULT_COMPARISON_PROFILES,
      ...comparisonsFromDisk,
    ].filter((c, i, arr) => arr.findIndex((x) => x.id === c.id) === i);

    return NextResponse.json({
      ...payload,
      runHistory,
      comparisonCandidates,
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Failed to load benchmark CSV data",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
