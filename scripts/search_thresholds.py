import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_unified_system import evaluate_system

print("Searching thresholds for Event Recall >= 8/9:")
for alpha in [0.5, 0.6, 0.7]:
    for ph in [70, 75, 80, 85, 90, 92, 94]:
        for pl in [50, 60, 70]:
            if pl >= ph:
                continue
            r = evaluate_system(alpha=alpha, p_high=ph, p_low=pl, H=15)
            if r["caught_events"] >= 8:
                print(f"a={alpha} ph={ph} pl={pl} | EvRec: {r['caught_events']}/9 ({r['event_recall']*100:.1f}%) | Prec: {r['precision']*100:.1f}% | Rec: {r['recall']*100:.1f}% | FPR: {r['fpr']*100:.2f}% | F1: {r['f1']:.4f} | Delay: {r['mean_delay_sec']:.2f}s")
