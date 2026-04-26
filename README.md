# Satellite Imaging Planner 

## Team Members: Nova
Satya Pranavi Vemula

Kundana Priya

Muvva Sirimalli

## Overview
This project implements a **case-adaptive imaging scheduler** for an Earth observation satellite. The goal is to maximize AOI coverage while respecting spacecraft constraints and improving control-effort efficiency (η_E).

---

## Key Idea

From the scoring function:

- Coverage (C) is dominant  
- η_E depends on total momentum change (ΔH_used)

High peak angular velocity during fast slews → large ΔH_used → η_E ≈ 0

### Strategy:
> **Spread imaging over time to reduce peak angular velocity**

---

## Approach

### Case-Adaptive Planning

- **Case 1 & 2 (Broad visibility)**  
  - Use **time-slot scheduling**  
  - Larger `min_gap` (≈ 2.5s)  
  - Frames distributed across the pass  
  - Reduces peak slew rate → improves η_E  

- **Case 3 (Narrow visibility)**  
  - Use **greedy best-time scheduling (v5.3 style)**  
  - Smaller `min_gap` (≈ 0.9s)  
  - Focus on maximizing coverage  
  - η_E sacrificed due to tight window  

---

## Path Optimization Pipeline

Before scheduling (Cases 1 & 2):

1. **Strip-snake ordering** → spatial coherence  
2. **Nearest-neighbour reordering** → minimize angular jumps  
3. **Local path improvement** → reduce total slew  

This reduces total rotation → lowers control effort.

---

## Attitude Strategy

Each frame uses:

- Pre-hold: 50 ms  
- Shutter: 120 ms (constant attitude → zero smear)  
- Post-hold: 50 ms  
- Slew: smooth interpolation between frames  

---

## Constraints Handling

All constraints are strictly enforced:

- Smear constraint (|ω| ≤ 0.05°/s) → constant attitude during imaging  
- Off-nadir ≤ 60° (with safety margin)  
- No shutter overlap (min_gap enforced)  
- Monotonic attitude timeline  

Wheel limits are handled indirectly by:
- limiting slew rates  
- spreading maneuvers over time  

---

## Key Insight

> η_E is improved not by minimizing motion completely, but by  
> **reducing peak angular velocity through time spacing**

---

## Expected Behavior

- Case 1 & 2:  
  - ~49 frames  
  - improved η_E (~0.3–0.4)

- Case 3:  
  - ~45–47 frames  
  - η_E ≈ 0 (due to geometry limits)

---

## Summary

This solution balances:
- **High coverage (primary objective)**
- **Controlled motion (secondary optimization)**

using a hybrid of greedy scheduling and time-distributed planning.

---
