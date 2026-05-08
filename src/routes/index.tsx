import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ClientOnly } from "@tanstack/react-router";
import { compute, type EcoInputs } from "@/lib/resilience-engine";
import { EarthHero } from "@/components/eco/EarthHero";
import { ResilienceGauge } from "@/components/eco/ResilienceGauge";
import { EcoSlider } from "@/components/eco/EcoSlider";
import { FeatureImportance } from "@/components/eco/FeatureImportance";
import { SensitivityHeatmap } from "@/components/eco/SensitivityHeatmap";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Eco-Pulse — The Rainfall Resilience Engine" },
      { name: "description", content: "Multi-source environmental intelligence: detect rainfall-response anomalies and model ecosystem recovery in real time." },
    ],
  }),
  component: Index,
});

const DEFAULT: EcoInputs = {
  rainfall: 420,
  infiltration: 0.55,
  slope: 12,
  diversity: 0.6,
  degradation: 0.35,
};

const PITCH_FRAMES: EcoInputs[] = [
  { rainfall: 200, infiltration: 0.7, slope: 5, diversity: 0.75, degradation: 0.15 },
  { rainfall: 600, infiltration: 0.55, slope: 18, diversity: 0.6, degradation: 0.35 },
  { rainfall: 950, infiltration: 0.25, slope: 32, diversity: 0.3, degradation: 0.75 },
  { rainfall: 1200, infiltration: 0.15, slope: 38, diversity: 0.2, degradation: 0.85 },
  { rainfall: 500, infiltration: 0.5, slope: 15, diversity: 0.55, degradation: 0.4 },
];

function Index() {
  const [inputs, setInputs] = useState<EcoInputs>(DEFAULT);
  const [pitch, setPitch] = useState(false);
  const frameRef = useRef(0);

  const out = useMemo(() => compute(inputs), [inputs]);

  useEffect(() => {
    if (!pitch) return;
    const id = setInterval(() => {
      frameRef.current = (frameRef.current + 1) % PITCH_FRAMES.length;
      setInputs(PITCH_FRAMES[frameRef.current]);
    }, 2200);
    return () => clearInterval(id);
  }, [pitch]);

  const update = (k: keyof EcoInputs) => (v: number) => setInputs((s) => ({ ...s, [k]: v }));

  return (
    <main className="min-h-screen text-foreground">
      {/* HERO */}
      <section className="relative overflow-hidden border-b border-border">
        <div className="absolute inset-0 grid-bg opacity-30" />
        <div className="relative mx-auto max-w-7xl px-6 pt-10 pb-20 grid lg:grid-cols-2 gap-10 items-center">
          <div>
            <motion.div
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-[10px] uppercase tracking-[0.3em] text-muted-foreground"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              Environmental Intelligence · v0.1
            </motion.div>
            <motion.h1
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
              className="mt-6 font-display text-5xl md:text-7xl font-bold leading-[1.05] tracking-tight"
            >
              Eco-<span className="neon-text">Pulse</span>
              <span className="block text-foreground/80 text-3xl md:text-4xl mt-3 font-medium">
                The Rainfall Resilience Engine
              </span>
            </motion.h1>
            <motion.p
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25 }}
              className="mt-6 max-w-xl text-muted-foreground text-base leading-relaxed"
            >
              Why does identical rainfall heal one ecosystem and break another?
              Eco-Pulse fuses hydrology, biodiversity, and land-use signals into a
              single Resilience Score — and surfaces the anomalies before they cascade.
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
              className="mt-8 flex flex-wrap gap-3"
            >
              <a href="#simulator" className="px-5 py-3 rounded-lg bg-primary text-primary-foreground font-medium neon-glow hover:opacity-90 transition">
                Launch Simulator →
              </a>
              <button
                onClick={() => setPitch((p) => !p)}
                className={`px-5 py-3 rounded-lg border border-border font-medium transition ${pitch ? "bg-primary/20 neon-text" : "hover:bg-secondary/40"}`}
              >
                {pitch ? "■ Stop Pitch Mode" : "▶ Pitch Mode"}
              </button>
            </motion.div>

            <div className="mt-10 grid grid-cols-3 gap-4 max-w-md">
              {[
                { k: "Resilience", v: out.resilienceScore.toFixed(0) },
                { k: "Runoff Risk", v: out.runoffRisk.toFixed(0) },
                { k: "Recovery", v: out.recoveryIndex.toFixed(0) },
              ].map((s) => (
                <div key={s.k} className="glass rounded-xl p-3">
                  <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">{s.k}</div>
                  <div className="font-mono text-2xl mt-1 neon-text">{s.v}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="relative h-[420px] md:h-[520px] rounded-2xl glass overflow-hidden">
            <ClientOnly fallback={<div className="w-full h-full grid place-items-center text-muted-foreground text-sm">Initializing biosphere…</div>}>
              <EarthHero intensity={out.runoffRisk / 50 + 0.5} />
            </ClientOnly>
            <div className="absolute bottom-3 left-4 right-4 flex justify-between text-[10px] uppercase tracking-[0.25em] text-muted-foreground/80 font-mono">
              <span>LIVE BIOSPHERE</span>
              <span>RAIN INTENSITY · {(out.runoffRisk).toFixed(0)}</span>
            </div>
          </div>
        </div>
      </section>

      {/* SIMULATOR */}
      <section id="simulator" className="mx-auto max-w-7xl px-6 py-16">
        <div className="flex items-end justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold">Simulator Dashboard</h2>
            <p className="text-muted-foreground text-sm mt-1">Move sliders. Watch the ecosystem respond.</p>
          </div>
          <AnimatePresence>
            {pitch && (
              <motion.span
                initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
                className="text-[10px] uppercase tracking-[0.3em] neon-text font-mono"
              >
                ● Pitch Mode Auto-Run
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Sliders */}
          <div className="glass rounded-2xl p-6 space-y-6">
            <h3 className="text-sm uppercase tracking-[0.25em] text-muted-foreground">Environment</h3>
            <EcoSlider label="Rainfall" value={inputs.rainfall} onChange={update("rainfall")} min={0} max={1500} unit=" mm" hint="Annual precipitation" />
            <EcoSlider label="Soil Infiltration" value={inputs.infiltration} onChange={update("infiltration")} min={0.05} max={1} step={0.01} hint="Capacity to absorb water" />
            <EcoSlider label="Slope" value={inputs.slope} onChange={update("slope")} min={0} max={45} unit="°" hint="Terrain gradient" />
            <EcoSlider label="Biodiversity" value={inputs.diversity} onChange={update("diversity")} min={0} max={1} step={0.01} hint="Species richness index" />
            <EcoSlider label="Land Degradation" value={inputs.degradation} onChange={update("degradation")} min={0} max={1} step={0.01} hint="Human / erosion impact" />
          </div>

          {/* Gauge + anomaly */}
          <div className="glass rounded-2xl p-6 flex flex-col items-center justify-center">
            <ResilienceGauge score={out.resilienceScore} anomaly={out.anomalyFlag} />
            <div className="mt-6 w-full grid grid-cols-2 gap-3 text-center">
              <div className="rounded-lg bg-secondary/40 p-3">
                <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Runoff Risk</div>
                <div className="font-mono text-xl mt-1">{out.runoffRisk.toFixed(0)}</div>
              </div>
              <div className="rounded-lg bg-secondary/40 p-3">
                <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Recovery Δ vs hist.</div>
                <div className={`font-mono text-xl mt-1 ${out.anomalyDelta < 0 ? "text-destructive" : "neon-text"}`}>
                  {out.anomalyDelta > 0 ? "+" : ""}{out.anomalyDelta.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Feature importance */}
          <div className="glass rounded-2xl p-6">
            <h3 className="text-sm uppercase tracking-[0.25em] text-muted-foreground mb-2">Explainable AI · Drivers</h3>
            <p className="text-xs text-muted-foreground/70 mb-4">Which factor dominates the current ecosystem response?</p>
            <FeatureImportance data={out.featureImportance} />
            <div className="mt-4 text-[11px] font-mono text-muted-foreground">
              Primary driver: <span className="neon-text">{out.featureImportance[0]?.name}</span> ({out.featureImportance[0]?.value}%)
            </div>
          </div>
        </div>
      </section>

      {/* HEATMAP + DRIVERS */}
      <section className="mx-auto max-w-7xl px-6 pb-20">
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6">
            <h3 className="text-sm uppercase tracking-[0.25em] text-muted-foreground">Sensitivity Heatmap</h3>
            <p className="text-xs text-muted-foreground/70 mt-1 mb-4">
              Resilience across (Land Degradation × Slope) — current rainfall & biodiversity held constant.
            </p>
            <SensitivityHeatmap inputs={inputs} />
          </div>

          <div className="glass rounded-2xl p-6">
            <h3 className="text-sm uppercase tracking-[0.25em] text-muted-foreground">Recovery Drivers</h3>
            <ul className="mt-4 space-y-3">
              {out.drivers.map((d) => (
                <li key={d.factor}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-foreground/90">{d.factor}</span>
                    <span className={`font-mono ${d.direction === "positive" ? "neon-text" : "text-destructive"}`}>
                      {d.direction === "positive" ? "+" : "−"}{d.impact}
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-secondary/60 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }} animate={{ width: `${d.impact}%` }}
                      transition={{ duration: 0.6 }}
                      className="h-full"
                      style={{ background: d.direction === "positive" ? "var(--neon)" : "var(--danger)" }}
                    />
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-10 text-center text-[10px] uppercase tracking-[0.4em] text-muted-foreground font-mono">
          Eco-Pulse · Hackathon MVP · Synthetic ensemble model
        </div>
      </section>
    </main>
  );
}
