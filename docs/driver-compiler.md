# Driver Compiler Investigation (Workstream D — research only, 2026-11-19)

No drivers, firmware, or settings were changed. Sources: Intel linux-npu-driver
release notes (GitHub, fetched 2026-11-19) and the OpenVINO issue tracker.

## Question 1: Is the driver-compiler failure (ZE_RESULT_ERROR_UNKNOWN 0x7ffffffe,
even on trivial graphs, platform 3720) acknowledged anywhere?

**Yes — as a compiler-selection defect, not a driver bug.**

- OpenVINO issue **#38738** "[Bug]: NPU3720 uses the driver compiler by default on
  Linux" (Oct 2026, closed *not planned*): platform 3720 is missing from the list in
  `compiler_adapter_factory` that selects the plugin compiler, so Linux defaults 3720
  to the **driver** compiler; compilation then fails. The reporter confirms the same
  model compiles and runs with `compiler_type=PLUGIN`. No fix was merged into
  OpenVINO; the reporter published a private fix branch only.
- This matches our source reading exactly: the plugin compiler is the default only
  for NPU4000/5010/5020/6010; 3720 falls through to the driver compiler.
- Related evidence that the driver-compiler path is generally unreliable:
  **#37438** (driver compiler Q/DQ Concat produces zeroed output channels, closed),
  **#39890**, **#39379** (wrong RoPE results on NPU 3720).

## Question 2: Which driver release fixes it?

**None documented.** Intel linux-npu-driver release notes through **v1.38.0**
(current latest; our machine has Fedora's 1.32.0 + level-zero 26.35.39758.21) contain
no note about driver-compiler compilation failures, `ZE_RESULT_ERROR_UNKNOWN`, or
compiler selection. The only known-issue notes concern unrelated firmware/GPU test
failures. Since the root cause is OpenVINO's platform-default selection (per #38738),
a driver upgrade is not the identified remedy.

## Question 3: Documented compiler-selection requirements for platform 3720?

- No official requirement documented; the selection is an OpenVINO-side default
  (arguably a bug per #38738, closed without action).
- Practical requirement on this machine, verified by us and by #38738's reporter:
  **`IE_NPU_COMPILER_TYPE=PLUGIN`** (or `NPU_COMPILER_TYPE=PLUGIN` in the config
  file) is mandatory for NPU work on platform 3720 on Linux.

## Bonus finding (same class as our original failure)

OpenVINO issue **#39184** (open): "Qwen3.5-4B INT8 IR fails on Intel AI Boost
NPU37XX due to dynamic hybrid KV cache shapes" — the same failure class as our
original stateful-IR problem (dynamic ReadValue shapes), confirming it is a known
NPU37XX limitation rather than something machine-specific.

## Conclusion

Keep `IE_NPU_COMPILER_TYPE=PLUGIN` as the standing workaround (already in `env.sh`).
No driver upgrade, firmware change, or OpenVINO patch is justified by the available
evidence. Re-check the linux-npu-driver release notes if a future driver release
explicitly mentions compiler fixes.

*Caveat: issue contents were retrieved via automated page fetch; re-verify issue
numbers/statuses before citing externally.*
