#!/bin/bash
# Fast code2flow analysis for large projects

echo "🚀 Fast Code2Flow Analysis"
echo "=========================="

# Set environment variables for performance
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Fast analysis with optimizations
echo "1️⃣ Quick analysis (spring layout, limited depth)"
time code2flow ../src/nlp2cmd/ \
    -v \
    -o ./output_quick \
    --layout spring \
    --max-depth 2 \
    --max-nodes 1000 \
    --exclude "*test*" \
    --exclude "*__pycache__*"

echo ""
echo "2️⃣ Ultra-fast analysis (dot layout, minimal depth)"
time code2flow ../src/nlp2cmd/ \
    -v \
    -o ./output_ultra \
    --layout dot \
    --max-depth 1 \
    --max-nodes 500

echo ""
echo "3️⃣ Module-by-module analysis"
for module in adapters automation llm generation pipeline_runner web_schema; do
    if [ -d "../src/nlp2cmd/$module" ]; then
        echo "📦 Analyzing $module..."
        time code2flow "../src/nlp2cmd/$module" \
            -v \
            -o "./output_modules/$module" \
            --layout dot \
            --max-depth 2
    fi
done

echo ""
echo "✅ Analysis complete!"
echo "Check output directories:"
echo "  - ./output_quick/"
echo "  - ./output_ultra/" 
echo "  - ./output_modules/"
