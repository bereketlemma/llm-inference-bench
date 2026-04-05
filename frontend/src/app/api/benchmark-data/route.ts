import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

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
    const root = path.resolve(process.cwd(), "..");
    const dataDir = path.join(root, "backend", "results", "Data");
    const fp16Path = path.join(dataDir, "fp16_results.csv");
    const awqPath = path.join(dataDir, "awq_marlin_results.csv");

    const [fp16Raw, awqRaw] = await Promise.all([
      fs.readFile(fp16Path, "utf8"),
      fs.readFile(awqPath, "utf8"),
    ]);

    const fp16Rows = parseCsv(fp16Raw).map((r) => ({
      quant: "FP16",
      batch: Number(r.batch_size),
      tokens: Number(r.output_tokens),
      p50: Number(r.p50_latency_ms),
      // fp16 CSV has no p90 column, so we surface p99 in both p90/p99 slots.
      p90: Number(r.p99_latency_ms),
      p99: Number(r.p99_latency_ms),
      throughput: Number(r.tokens_per_second),
      rps: Number(r.requests_per_second),
    }));

    const awqRows = parseCsv(awqRaw).map((r) => ({
      quant: "INT4-AWQ",
      batch: Number(r.batch_size),
      tokens: Number(r.output_tokens),
      p50: Number(r.p50_latency_ms),
      p90: Number(r.p90_latency_ms),
      p99: Number(r.p99_latency_ms),
      throughput: Number(r.tokens_per_second),
      rps: Number(r.requests_per_second),
    }));

    const payload: BenchmarkMetadata = {
      model: "Mistral-7B-v0.1",
      gpu: "NVIDIA L4 24GB",
      framework: "vLLM 0.16.0",
      date: new Date().toISOString().slice(0, 10),
      configs: [...fp16Rows, ...awqRows],
    };

    const currentSummary = summarizeRun(payload);
    const historyDir = path.join(root, "backend", "results", "history");
    const comparisonsDir = path.join(root, "backend", "results", "comparisons");
    const historyFromDisk = await loadJsonArray<RunSummary>(historyDir);
    const comparisonsFromDisk = await loadJsonArray<ComparisonProfile>(comparisonsDir);

    const runHistory: RunSummary[] =
      historyFromDisk.length > 0
        ? [...historyFromDisk, currentSummary]
            .filter((r, i, arr) => arr.findIndex((x) => x.id === r.id) === i)
            .sort((a, b) => a.date.localeCompare(b.date))
        : [currentSummary];

    const comparisonCandidates: ComparisonProfile[] = [
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
