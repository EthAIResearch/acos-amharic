#!/usr/bin/env bash
# Run once, from inside the repo (requires: gh auth login already done)
set -e

# type: labels
gh label create "type:feature"    --color 0E8A16 --description "New capability"
gh label create "type:bug"        --color D93F0B --description "Something broken"
gh label create "type:experiment" --color 5319E7 --description "Training run / research experiment"
gh label create "type:data"       --color FBCA04 --description "Schema/annotation issue"
gh label create "type:docs"       --color C5DEF5 --description "Documentation"

# stage: labels (blue family, lightening by stage)
gh label create "stage:1-tagging"   --color 08306B --description "ATE + OTE tagging"
gh label create "stage:2-pairing"   --color 2171B5 --description "Aspect-opinion pairing"
gh label create "stage:3-category"  --color 4292C6 --description "Category classification"
gh label create "stage:4-sentiment" --color 6BAED6 --description "Sentiment classification"
gh label create "stage:5-implicit"  --color 9ECAE1 --description "Implicit aspect/opinion detection"
gh label create "stage:6-eval"      --color C6DBEF --description "End-to-end quad evaluation"

# priority: labels (grey scale, distinct from type/stage families)
gh label create "p:high"   --color 4A4A4A --description "High priority"
gh label create "p:medium" --color 9E9E9E --description "Medium priority"
gh label create "p:low"    --color D9D9D9 --description "Low priority"

echo "Done -- 14 labels created."
