import { motion } from "framer-motion";

export function ResilienceGauge({ score, anomaly }: { score: number; anomaly: boolean }) {
  const pct = Math.max(0, Math.min(100, score));
  const radius = 90;
  const circ = 2 * Math.PI * radius;
  const offset = circ - (pct / 100) * circ;
  const color = anomaly ? "var(--danger)" : pct > 65 ? "var(--neon)" : pct > 35 ? "var(--warn)" : "var(--danger)";

  return (
    <div className="relative flex flex-col items-center justify-center">
      <svg width={220} height={220} className="-rotate-90">
        <circle cx={110} cy={110} r={radius} stroke="var(--border)" strokeWidth={14} fill="none" />
        <motion.circle
          cx={110} cy={110} r={radius}
          stroke={color} strokeWidth={14} fill="none"
          strokeLinecap="round"
          strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: offset }}
          transition={{ type: "spring", stiffness: 60, damping: 18 }}
          style={{ filter: `drop-shadow(0 0 12px ${color})` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.div
          key={pct}
          initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
          className="font-mono text-5xl font-bold neon-text"
        >
          {pct.toFixed(0)}
        </motion.div>
        <div className="text-xs uppercase tracking-[0.25em] text-muted-foreground mt-1">
          Resilience
        </div>
        {anomaly && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="mt-2 text-[10px] font-mono uppercase tracking-widest text-destructive"
          >
            ⚠ Anomaly Detected
          </motion.div>
        )}
      </div>
    </div>
  );
}
