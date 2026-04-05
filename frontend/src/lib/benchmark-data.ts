export interface BenchmarkConfig {
  quant: string;
  batch: number;
  tokens: number;
  p50: number;
  p90: number;
  p99: number;
  throughput: number;
  rps: number;
}

export interface BenchmarkMetadata {
  model: string;
  gpu: string;
  framework: string;
  date: string;
  configs: BenchmarkConfig[];
}

// ── Real benchmark data from NVIDIA L4 24GB, vLLM 0.16.0, April 5 2026 ──
export const BENCHMARK_DATA: BenchmarkMetadata = {
  model: "Mistral-7B-v0.1",
  gpu: "NVIDIA L4 24GB",
  framework: "vLLM 0.16.0",
  date: "2026-04-05",
  configs: [
    // FP16 — mistralai/Mistral-7B-v0.1
    // NOTE: fp16_results.csv includes p50/p99 but not p90, so p90 is set equal to p99.
    { quant: "FP16", batch: 1, tokens: 128, p50: 3587.0, p90: 3590.0, p99: 3590.0, throughput: 17.9, rps: 1.0 },
    { quant: "FP16", batch: 1, tokens: 256, p50: 7169.7, p90: 7176.2, p99: 7176.2, throughput: 17.9, rps: 0.5 },
    { quant: "FP16", batch: 1, tokens: 512, p50: 14353.3, p90: 14355.3, p99: 14355.3, throughput: 17.8, rps: 0.25 },
    { quant: "FP16", batch: 4, tokens: 128, p50: 1872.0, p90: 1880.0, p99: 1880.0, throughput: 68.5, rps: 2.14 },
    { quant: "FP16", batch: 4, tokens: 256, p50: 3740.0, p90: 3760.0, p99: 3760.0, throughput: 68.3, rps: 1.07 },
    { quant: "FP16", batch: 4, tokens: 512, p50: 7520.0, p90: 7540.0, p99: 7540.0, throughput: 68.0, rps: 0.53 },
    { quant: "FP16", batch: 8, tokens: 128, p50: 1920.0, p90: 1940.0, p99: 1940.0, throughput: 133.9, rps: 4.17 },
    { quant: "FP16", batch: 8, tokens: 256, p50: 3840.0, p90: 3860.0, p99: 3860.0, throughput: 133.5, rps: 2.08 },
    { quant: "FP16", batch: 8, tokens: 512, p50: 30590.4, p90: 30600.1, p99: 30600.1, throughput: 133.9, rps: 0.26 },

    // INT4 AWQ-Marlin — TheBloke/Mistral-7B-v0.1-AWQ (Marlin kernel)
    { quant: "INT4-AWQ", batch: 1, tokens: 128, p50: 2084.0713044999575, p90: 2085.5602032996103, p99: 2086.536158529161, throughput: 61.418114129200504, rps: 0.479828775803786 },
    { quant: "INT4-AWQ", batch: 1, tokens: 256, p50: 4201.152579498739, p90: 4204.540233201624, p99: 4205.02444471942, throughput: 60.94668651082105, rps: 0.238072812611235 },
    { quant: "INT4-AWQ", batch: 1, tokens: 512, p50: 8460.707731999719, p90: 8464.054168999428, p99: 8464.155994100365, throughput: 60.52256440402549, rps: 0.11820811263080327 },
    { quant: "INT4-AWQ", batch: 4, tokens: 128, p50: 2180.0984455003345, p90: 2180.385091400967, p99: 2180.513520141976, throughput: 234.8717381943008, rps: 1.834935367434046 },
    { quant: "INT4-AWQ", batch: 4, tokens: 256, p50: 4393.61407350043, p90: 4394.748404600978, p99: 4395.082024158619, throughput: 233.07076694134471, rps: 0.9104326248442998 },
    { quant: "INT4-AWQ", batch: 4, tokens: 512, p50: 8990.094441500332, p90: 8990.671367699179, p99: 8990.931164668473, throughput: 227.8061661788544, rps: 0.44493391694846324 },
    { quant: "INT4-AWQ", batch: 8, tokens: 128, p50: 2264.0232930007187, p90: 2264.8089762031304, p99: 2265.7407775214597, throughput: 452.2787145126649, rps: 3.5334269852156557 },
    { quant: "INT4-AWQ", batch: 8, tokens: 256, p50: 4589.950338500785, p90: 4590.335008700276, p99: 4590.359500669474, throughput: 446.20374091604015, rps: 1.7429833459986221 },
    { quant: "INT4-AWQ", batch: 8, tokens: 512, p50: 9544.76591850289, p90: 9547.16771179883, p99: 9547.95848077978, throughput: 429.117713725238, rps: 0.8381205191857305 },
  ],
};

// ── Computed summaries ──
export const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

const byQuant = (quant: string) => BENCHMARK_DATA.configs.filter((c) => c.quant === quant);

export const fp16Configs = byQuant("FP16");
export const int4Configs = byQuant("INT4-AWQ");

export const fp16Avg = {
  p50: avg(fp16Configs.map((c) => c.p50)),
  p99: avg(fp16Configs.map((c) => c.p99)),
  throughput: avg(fp16Configs.map((c) => c.throughput)),
};

export const int4Avg = {
  p50: avg(int4Configs.map((c) => c.p50)),
  p99: avg(int4Configs.map((c) => c.p99)),
  throughput: avg(int4Configs.map((c) => c.throughput)),
};

export const speedup = (int4Avg.throughput / fp16Avg.throughput).toFixed(1);
export const p99Reduction = ((1 - int4Avg.p99 / fp16Avg.p99) * 100).toFixed(1);
export const p50Reduction = ((1 - int4Avg.p50 / fp16Avg.p50) * 100).toFixed(1);
export const peakThroughput = Math.max(...int4Configs.map((c) => c.throughput)).toFixed(1);

export const batchScaling = [1, 4, 8].map((b) => ({
  batch: `BS=${b}`,
  FP16: avg(fp16Configs.filter((c) => c.batch === b).map((c) => c.throughput)),
  "INT4-AWQ": avg(int4Configs.filter((c) => c.batch === b).map((c) => c.throughput)),
}));

export const tokenScaling = [128, 256, 512].map((t) => ({
  tokens: `${t} tok`,
  FP16: avg(fp16Configs.filter((c) => c.tokens === t).map((c) => c.throughput)),
  "INT4-AWQ": avg(int4Configs.filter((c) => c.tokens === t).map((c) => c.throughput)),
}));

export const speedupByBatch = [1, 4, 8].map((b) => {
  const fp = fp16Configs.filter((c) => c.batch === b);
  const iq = int4Configs.filter((c) => c.batch === b);
  return {
    batch: `Batch ${b}`,
    speedup: Number((avg(iq.map((c) => c.throughput)) / avg(fp.map((c) => c.throughput))).toFixed(2)),
  };
});

export const rpsByBatch = [1, 4, 8].map((b) => ({
  batch: `BS=${b}`,
  FP16: avg(fp16Configs.filter((c) => c.batch === b).map((c) => c.rps)),
  "INT4-AWQ": avg(int4Configs.filter((c) => c.batch === b).map((c) => c.rps)),
}));
