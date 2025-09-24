#!/usr/bin/env python3
"""
Plot Overall Subspace Overlap

This script scans subspace overlap result JSONs produced by subspace_overlap_unified.py
and generates bar plots showing the overall subspace similarity between all pairs of
methods (tok_vs_act, tok_vs_tna, tok_vs_a2t, act_vs_tna, act_vs_a2t, tna_vs_a2t) at
each N value. You can filter by model ids and datasets.

Input JSON format expected (per file):
{
  "overlap_results": {
    N: { pair_name: {"mean": float, "std": float}, ... },
    ...
  },
  "config": { optional metadata saved by the generator }
}

Output: One bar plot per N value per selection group. By default, one plot per JSON.
You can choose to aggregate across multiple JSONs by using --aggregate and filters.
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Headless backend
import matplotlib
matplotlib.use('Agg')


PAIR_ORDER = [
    # 'tok_vs_act', 'tok_vs_tna', 'tok_vs_a2t',
    # 'act_vs_tna', 'act_vs_a2t', 'tna_vs_a2t'
    'tok_vs_act', 'tok_vs_a2t', 'act_vs_a2t'
]


def discover_result_files(base_dir: str) -> List[str]:
    matches = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith('subspace_analysis_results_') and f.endswith('.json'):
                matches.append(os.path.join(root, f))
    return sorted(matches)


def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r') as fp:
            return json.load(fp)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return {}


def extract_config_from_filename(filename: str) -> Tuple[str, str, str, str]:
    # subspace_analysis_results_{trained_dataset}_{eval_dataset}_{icl_source}_{model}.json
    config = json.load(open(filename, 'r'))['config']
    return config['dataset'], config['eval_dataset'], config['icl_source'], config['model_id'].split('/')[-1]
    


def filter_file(path: str, args: argparse.Namespace) -> bool:
    trained, evald, icl, model = extract_config_from_filename(path)
    if args.trained_datasets and trained not in set(args.trained_datasets):
        return False
    if args.eval_datasets and evald not in set(args.eval_datasets):
        return False
    if args.icl_sources and icl not in set(args.icl_sources):
        return False
    if args.models:
        # match either exact model file token or substring
        ok = any(m in model for m in args.models)
        if not ok:
            return False
    return True


def aggregate_overlaps(json_datas: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, float]]]:
    # Returns: { N: { pair_name: {mean: float, std: float} } }
    collector: Dict[int, Dict[str, List[float]]] = {}
    for data in json_datas:
        overlaps = data.get('overlap_results', {})
        for n_str, pairs in overlaps.items():
            try:
                n_val = int(n_str)
            except Exception:
                # allow ints as keys too
                n_val = n_str if isinstance(n_str, int) else None
            if n_val is None:
                continue
            if n_val not in collector:
                collector[n_val] = {p: [] for p in PAIR_ORDER}
            for pair_name, stats in pairs.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    collector[n_val].setdefault(pair_name, []).append(stats['mean'])
                elif isinstance(stats, (int, float)):
                    collector[n_val].setdefault(pair_name, []).append(float(stats))

    aggregated: Dict[int, Dict[str, Dict[str, float]]] = {}
    for n_val, pair_map in collector.items():
        aggregated[n_val] = {}
        for pair_name, values in pair_map.items():
            if values:
                aggregated[n_val][pair_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
    return aggregated


def plot_bars_for_overlaps(overlaps: Dict[int, Dict[str, Dict[str, float]]], title: str, out_path: str) -> None:
    if not overlaps:
        print("No overlaps to plot.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # One figure per N
    for n_val in sorted(overlaps.keys()):
        pairs = overlaps[n_val]
        labels = [p for p in PAIR_ORDER if p in pairs]
        means = [pairs[p]['mean'] for p in labels]
        stds = [pairs[p]['std'] for p in labels]

        plt.figure(figsize=(12, 6))
        x = np.arange(len(labels))
        bars = plt.bar(x, means, yerr=stds, capsize=4, alpha=0.8, color='skyblue', edgecolor='navy')
        plt.xticks(x, labels, rotation=20)
        plt.ylim(0.0, 1.0)
        plt.ylabel('Subspace Overlap (mean ± std)')
        plt.title(f"{title} | N={n_val}")
        plt.grid(True, axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + (stds[i] if i < len(stds) else 0) + 0.01,
                     f"{means[i]:.3f}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        file_path = out_path.replace('.png', f"_N{n_val}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {file_path}")


def plot_combined_grouped(overlaps: Dict[int, Dict[str, Dict[str, float]]], title: str, out_path: str) -> None:
    if not overlaps:
        print("No overlaps to plot (combined).")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_values = sorted(overlaps.keys())
    # Determine union of pairs across Ns, preserve declared order
    pair_labels = [p for p in PAIR_ORDER if any(p in overlaps[n] for n in n_values)]
    if not pair_labels:
        print("No pair labels found to plot (combined).")
        return

    x = np.arange(len(pair_labels))
    num_groups = len(n_values)
    width = 0.8 / max(1, num_groups)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf']

    plt.figure(figsize=(14, 7))
    bars_for_labels = []
    for i, n_val in enumerate(n_values):
        means = []
        stds = []
        for pair in pair_labels:
            stats = overlaps.get(n_val, {}).get(pair)
            if stats:
                means.append(stats.get('mean', 0.0))
                stds.append(stats.get('std', 0.0))
            else:
                means.append(0.0)
                stds.append(0.0)
        positions = x + (i - (num_groups - 1) / 2) * width
        bars = plt.bar(positions, means, width, yerr=stds, capsize=3,
                       label=f'N={n_val}', color=colors[i % len(colors)], alpha=0.85,
                       edgecolor='black', linewidth=0.5)
        bars_for_labels.append(bars)

    plt.xticks(x, pair_labels, rotation=15)
    plt.ylabel('Subspace Overlap (mean ± std)')
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title='Training Examples (N)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {out_path}")


def collect_pair_values_across_datas(json_datas: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    # Collect raw overlap means across all Ns and files for each pair
    pair_to_values: Dict[str, List[float]] = {p: [] for p in PAIR_ORDER}
    for data in json_datas:
        overlaps = data.get('overlap_results', {})
        for _, pairs in overlaps.items():
            for pair_name, stats in pairs.items():
                if pair_name not in pair_to_values:
                    continue
                if isinstance(stats, dict) and 'mean' in stats:
                    try:
                        pair_to_values[pair_name].append(float(stats['mean']))
                    except Exception:
                        pass
                elif isinstance(stats, (int, float)):
                    pair_to_values[pair_name].append(float(stats))
    # Remove empty entries
    return {k: v for k, v in pair_to_values.items() if v}


def plot_violin_for_pairs(pair_values: Dict[str, List[float]], title: str, out_path: str) -> None:
    if not pair_values:
        print("No overlaps to plot (violin).")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    labels = [p for p in PAIR_ORDER if p in pair_values]
    data = [pair_values[p] for p in labels]

    plt.figure(figsize=(4, 4))
    parts = plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    # Style
    for pc in parts['bodies']:
        pc.set_facecolor('#87CEEB')
        pc.set_edgecolor('#1f77b4')
        pc.set_alpha(0.7)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('#d62728')
        parts['cmeans'].set_linewidth(1.5)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('#2ca02c')
        parts['cmedians'].set_linewidth(1.5)

    x_positions = np.arange(1, len(labels) + 1)
    plt.xticks(x_positions, labels, rotation=15)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Subspace Overlap')
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)

    # Overlay mean ± std as error bars and annotate mean/median
    means = [float(np.mean(vals)) if len(vals) > 0 else 0.0 for vals in data]
    stds = [float(np.std(vals)) if len(vals) > 0 else 0.0 for vals in data]
    medians = [float(np.median(vals)) if len(vals) > 0 else 0.0 for vals in data]
    plt.errorbar(x_positions, means, yerr=stds, fmt='o', color='#d62728', ecolor='#d62728', capsize=4, label='Mean ± STD')
    plt.scatter(x_positions, medians, color='#2ca02c', marker='D', label='Median')

    # Annotate values slightly above the points
    for i, x in enumerate(x_positions):
        y_mean = means[i]
        y_med = medians[i]
        plt.text(x - 0.15, min(0.98, y_mean + 0.03), f"{y_mean:.3f}", color='#d62728', ha='right', va='bottom', fontsize=8)
        plt.text(x + 0.15, min(0.98, y_med + 0.03), f"{y_med:.3f}", color='#2ca02c', ha='left', va='bottom', fontsize=8)

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot: {out_path}")


def plot_boxplot_for_pairs(pair_values: Dict[str, List[float]], title: str, out_path: str) -> None:
    if not pair_values:
        print("No overlaps to plot (boxplot).")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def replace_name(name: str) -> str:
        name = name.replace('tok_vs_act', 'SFT only\nvs\nIA2 only')
        name = name.replace('tok_vs_a2t', 'SFT only\nvs\nIA2 → SFT')
        name = name.replace('act_vs_a2t', 'IA2 only\nvs\nIA2 → SFT')
        return name

    out_path = out_path.replace('.png', '.pdf')
    labels = [p for p in PAIR_ORDER if p in pair_values]
    data = [pair_values[p] for p in labels]

    plot_labels = [replace_name(label) for label in labels]

    plt.figure(figsize=(4, 4))
    box = plt.boxplot(data, patch_artist=True, showmeans=True, meanline=True)
    # Style
    for patch in box['boxes']:
        patch.set_facecolor('#ADD8E6')
        patch.set_edgecolor('#1f77b4')
    for whisker in box['whiskers']:
        whisker.set_color('#1f77b4')
    for cap in box['caps']:
        cap.set_color('#1f77b4')
    for median in box['medians']:
        median.set_color('#2ca02c')
        median.set_linewidth(1.5)
    if 'means' in box:
        for mean_line in box['means']:
            mean_line.set_color('#d62728')
            mean_line.set_linewidth(1.5)

    x_positions = np.arange(1, len(labels) + 1)
    plt.xticks(x_positions, plot_labels, rotation=0)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Subspace Overlap')
    # plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)

    # Compute and annotate mean/median/std
    means = [float(np.mean(vals)) if len(vals) > 0 else 0.0 for vals in data]
    stds = [float(np.std(vals)) if len(vals) > 0 else 0.0 for vals in data]
    medians = [float(np.median(vals)) if len(vals) > 0 else 0.0 for vals in data]
    for i, x in enumerate(x_positions):
        # plt.text(x, min(0.98, max(means[i], medians[i]) + 0.1), f"μ={means[i]:.3f}\nσ={stds[i]:.3f}\nmed={medians[i]:.3f}",
        plt.text(x, 0.75, f"μ={means[i]:.3f}\nσ={stds[i]:.3f}\nmed={medians[i]:.3f}",
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot overall subspace overlaps as bar charts.')
    parser.add_argument('--results_base', type=str, default='../plots/subspace_analysis',
                        help='Base directory to search for subspace_analysis_results_*.json')
    parser.add_argument('--output_dir', type=str, default='../plots/subspace_overlap_bars',
                        help='Where to save the bar plots')
    # Filters
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Filter by model identifiers (substring match on model token in filename)')
    parser.add_argument('--trained_datasets', nargs='+', type=str, default=None,
                        help='Filter by trained datasets')
    parser.add_argument('--eval_datasets', nargs='+', type=str, default=None,
                        help='Filter by evaluation datasets')
    parser.add_argument('--icl_sources', nargs='+', type=str, default=None,
                        help='Filter by ICL source datasets')
    parser.add_argument('--aggregate', action='store_true', default=True,
                        help='If set, aggregate across all matched files into a single set of plots')
    parser.add_argument('--label_types', nargs='+', type=str, default=None,
                        choices=['icl_outputs', 'ground_truth'],
                        help='If provided, restrict to these label types; otherwise plot available ones')

    args = parser.parse_args()

    files = discover_result_files(args.results_base)
    matched = [p for p in files if filter_file(p, args)]
    if not matched:
        print('No result files matched the filters.')
        return

    # Show configurations used
    configs_used = []
    labeled_files = {'icl_outputs': [], 'ground_truth': []}
    for p in matched:
        data = load_json(p)
        if not data:
            continue
        cfg = data.get('config', {})
        lt = cfg.get('label_type', 'unknown')
        trained = cfg.get('dataset', 'unknown')
        evald = cfg.get('eval_dataset', 'unknown')
        icl = cfg.get('icl_source', 'unknown')
        model = cfg.get('model_id', 'unknown').split('/')[-1]
        configs_used.append({'trained_dataset': trained, 'eval_dataset': evald, 'icl_source': icl, 'model': model, 'label_type': lt, 'file': p})
        if lt in labeled_files:
            labeled_files[lt].append(p)
    print("Configurations used:")
    for c in configs_used:
        print(f"  model={c['model']}, train={c['trained_dataset']}, eval={c['eval_dataset']}, icl={c['icl_source']}, label={c.get('label_type','unknown')}")

    # Determine which label types to process
    label_types_to_process = ['icl_outputs', 'ground_truth']
    if args.label_types:
        label_types_to_process = [lt for lt in label_types_to_process if lt in set(args.label_types)]

    if args.aggregate:
        os.makedirs(args.output_dir, exist_ok=True)
        for lt in label_types_to_process:
            group_files = labeled_files.get(lt, [])
            if not group_files:
                continue
            datas = [load_json(p) for p in group_files]
            datas = [d for d in datas if d]
            aggregated = aggregate_overlaps(datas)
            title = f'Subspace Overlaps'
            out_base = os.path.join(args.output_dir, f'aggregated_overlaps_{lt}.png')
            plot_bars_for_overlaps(aggregated, title, out_base)
            # Combined grouped plot across Ns
            out_combined = os.path.join(args.output_dir, f'aggregated_overlaps_combined_{lt}.png')
            plot_combined_grouped(aggregated, title + ' (Combined)', out_combined)
            # Violin plot across all files and Ns for each pair
            pair_values = collect_pair_values_across_datas(datas)
            out_violin = os.path.join(args.output_dir, f'aggregated_overlaps_violin_{lt}.png')
            plot_violin_for_pairs(pair_values, title, out_violin)
            # Boxplot with the same aggregated data
            out_box = os.path.join(args.output_dir, f'aggregated_overlaps_box_{lt}.png')
            plot_boxplot_for_pairs(pair_values, title, out_box)
            # Save configurations summary per label type
            with open(os.path.join(args.output_dir, f'aggregated_configs_used_{lt}.json'), 'w') as fp:
                json.dump({'matched_files': group_files, 'configs': [c for c in configs_used if c.get('label_type') == lt]}, fp, indent=2)
        return

    # Otherwise, one plot set per file
    for path in matched:
        data = load_json(path)
        if not data:
            continue
        overlaps = data.get('overlap_results', {})
        # Normalize structure: ensure dict with means/stds
        normalized: Dict[int, Dict[str, Dict[str, float]]] = {}
        for n_str, pairs in overlaps.items():
            try:
                n_val = int(n_str)
            except Exception:
                n_val = n_str if isinstance(n_str, int) else None
            if n_val is None:
                continue
            normalized[n_val] = {}
            for pair_name, stats in pairs.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    normalized[n_val][pair_name] = {
                        'mean': float(stats['mean']),
                        'std': float(stats.get('std', 0.0))
                    }
                elif isinstance(stats, (int, float)):
                    normalized[n_val][pair_name] = {'mean': float(stats), 'std': 0.0}

        # Since extract_config_from_filename may not include label, get from loaded data
        cfg = data.get('config', {})
        trained = cfg.get('dataset', 'unknown')
        evald = cfg.get('eval_dataset', 'unknown')
        icl = cfg.get('icl_source', 'unknown')
        model = cfg.get('model_id', 'unknown').split('/')[-1]
        lt = cfg.get('label_type', 'unknown')
        if args.label_types and lt not in set(args.label_types):
            continue
        title = f"{model} | train={trained}, eval={evald}, icl={icl} | {lt}"
        out_name = f"bars_{trained}_{evald}_{icl}_{lt}_{model}.png"
        out_path = os.path.join(args.output_dir, out_name)
        plot_bars_for_overlaps(normalized, title, out_path)
        # Combined grouped plot for this file
        out_combined = os.path.join(args.output_dir, f"bars_combined_{trained}_{evald}_{icl}_{lt}_{model}.png")
        plot_combined_grouped(normalized, title + ' (Combined)', out_combined)
        # Save per-file config detail
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"configs_used_{trained}_{evald}_{icl}_{lt}_{model}.json"), 'w') as fp:
            json.dump({'file': path, 'config': {'trained_dataset': trained, 'eval_dataset': evald, 'icl_source': icl, 'label_type': lt, 'model': model}}, fp, indent=2)


if __name__ == '__main__':
    main()


