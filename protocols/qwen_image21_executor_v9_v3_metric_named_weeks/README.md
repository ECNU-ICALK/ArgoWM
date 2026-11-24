# Executor prompt v3: named metrics and rounded weeks

Biological descriptions come from the retained CoT v3 teacher run. The copied
`cot/` protocol keeps its biological version and opts into the independent
`metric_named_v1` evidence format. Only Growth plan numeric evidence gets one
`Metric = {display_name}; ` immediately before Source / Target / Change.
Target phenotype remains qualitative. The original provider and protocols are
read-only inputs. The exporter validates the original protocol hash, exact
numeric evidence, and byte-identical recovery after removing metric prefixes.

Time uses `max(1, (actual_interval_days + 3) // 7)` weeks. Numeric time features
use seven times these rounded weeks. The Simple prefix and all other wording
remain the same as executor prompt v2 weeks. Legacy formats remain supported.
