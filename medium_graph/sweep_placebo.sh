#!/bin/bash
# Kept for backward compatibility — the placebo sweep is now a special case of
# sweep_conditions.sh (all three transforms + oracle). See PLACEBO_README.md.
exec "$(dirname "$0")/sweep_conditions.sh" "$1" "${2:-results/placebo}" "none shuffle homophily" 1 1
