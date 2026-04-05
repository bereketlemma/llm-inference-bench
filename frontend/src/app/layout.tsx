import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'llm-inference-bench | Bereket Lemma',
  description: 'Interactive benchmark dashboard — Mistral-7B inference performance across FP16 and INT4 AWQ-Marlin quantization on NVIDIA L4. Built with vLLM 0.16.0.',
  openGraph: {
    title: 'llm-inference-bench',
    description: 'Mistral-7B inference benchmarks — 3.3x throughput with AWQ-Marlin INT4 quantization',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
