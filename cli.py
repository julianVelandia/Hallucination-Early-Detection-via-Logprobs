"""
Benchmark CLI — Logprob collection for hallucination detection research.

Usage examples:
  python cli.py --help
  python cli.py config show
  python cli.py config set --model gpt-4o-mini --turns 2
  python cli.py explore
  python cli.py explore --dataset bullshit-v2 --refresh
  python cli.py run --questions 10 --turns 1
  python cli.py run --questions 20 --turns 2 --domain legal
  python cli.py run --questions 0 --turns 1          # 0 = all questions
  python cli.py status
  python cli.py reset --question-id leg_pnf_01
  python cli.py reset --all
  python cli.py label run                             # Tarea 2: label all turns
  python cli.py label run --labeler-model gpt-4o-mini --limit 20
  python cli.py label export --format csv
  python cli.py label status
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model": "gpt-4o-mini",
    "output": "data/results.json",
    "default_dataset": "bullshit-v2",
    "turns": 1,
    "temperature": 0.0,
    "top_logprobs": 5,
}


def load_config(config_file: str) -> dict:
    path = Path(config_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**DEFAULT_CONFIG, **data}
    return dict(DEFAULT_CONFIG)


def save_config(config_file: str, cfg: dict):
    path = Path(config_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "-c", "--config",
    "config_file",
    default="config.json",
    show_default=True,
    help="Path to configuration file.",
)
@click.pass_context
def cli(ctx, config_file):
    """
    \b
    Logprob Benchmark CLI
    =====================
    Collects LLM reasoning traces and token logprobs from benchmark datasets.
    Designed for hallucination detection research (PhD thesis, UNAL 2026).

    \b
    Quick start:
      1. cp .env.example .env  and add your OPENAI_API_KEY
      2. python cli.py explore
      3. python cli.py run --questions 10
      4. python cli.py status
    """
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file
    ctx.obj["config"] = load_config(config_file)


# ---------------------------------------------------------------------------
# config command
# ---------------------------------------------------------------------------

@cli.group()
@click.pass_context
def config(ctx):
    """Manage configuration settings."""
    pass


@config.command("show")
@click.pass_context
def config_show(ctx):
    """Display current configuration."""
    cfg = ctx.obj["config"]
    table = Table(title="Current Configuration", box=box.ROUNDED)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    for k, v in cfg.items():
        table.add_row(k, str(v))
    console.print(table)


@config.command("set")
@click.option("--model", help="OpenAI model name (e.g. gpt-4o-mini, gpt-4o).")
@click.option("--output", help="Output JSON file path.")
@click.option("--dataset", "default_dataset", help="Default dataset key.")
@click.option("--turns", type=int, help="Default conversation turns per question.")
@click.option("--temperature", type=float, help="Sampling temperature (0.0 = deterministic).")
@click.option("--top-logprobs", type=int, help="Number of top token alternatives to capture (1-20).")
@click.pass_context
def config_set(ctx, model, output, default_dataset, turns, temperature, top_logprobs):
    """Update one or more configuration values."""
    cfg = ctx.obj["config"]
    updates = {
        k: v for k, v in {
            "model": model,
            "output": output,
            "default_dataset": default_dataset,
            "turns": turns,
            "temperature": temperature,
            "top_logprobs": top_logprobs,
        }.items() if v is not None
    }
    if not updates:
        console.print("[yellow]No values provided. Use --help to see options.[/yellow]")
        return
    cfg.update(updates)
    save_config(ctx.obj["config_file"], cfg)
    console.print("[green]Configuration updated:[/green]")
    for k, v in updates.items():
        console.print(f"  {k} = {v}")


# ---------------------------------------------------------------------------
# explore command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--dataset", "dataset_key", default=None, help="Dataset key to explore.")
@click.option("--refresh", is_flag=True, help="Force re-download even if cached.")
@click.pass_context
def explore(ctx, dataset_key, refresh):
    """
    Fetch and explore a benchmark dataset.

    Shows dataset statistics and a sample of questions.
    """
    from src.dataset import DATASETS, fetch_dataset, extract_questions

    cfg = ctx.obj["config"]
    dataset_key = dataset_key or cfg["default_dataset"]

    # List available datasets
    console.print(Panel.fit(
        "\n".join(f"  [cyan]{k}[/cyan]: {v['description']}" for k, v in DATASETS.items()),
        title="Available Datasets",
        border_style="blue",
    ))

    console.print(f"\n[bold]Fetching:[/bold] {dataset_key}")
    with console.status("Downloading dataset..."):
        raw = fetch_dataset(dataset_key, force_refresh=refresh)

    questions = extract_questions(raw, dataset_key)
    console.print(f"[green]Loaded {len(questions)} questions.[/green]\n")

    # Domain breakdown
    domain_counts = {}
    technique_counts = {}
    control_count = 0
    for q in questions:
        dg = q.get("domain_group", "unknown")
        domain_counts[dg] = domain_counts.get(dg, 0) + 1
        t = q.get("technique", "unknown")
        technique_counts[t] = technique_counts.get(t, 0) + 1
        if q.get("is_control"):
            control_count += 1

    # Domain table
    t1 = Table(title="Questions by Domain", box=box.SIMPLE_HEAVY)
    t1.add_column("Domain Group", style="cyan")
    t1.add_column("Count", justify="right", style="green")
    for domain, cnt in sorted(domain_counts.items()):
        t1.add_row(domain, str(cnt))
    t1.add_row("[bold]TOTAL[/bold]", f"[bold]{len(questions)}[/bold]")
    console.print(t1)

    # Technique table
    t2 = Table(title="Questions by Technique", box=box.SIMPLE_HEAVY)
    t2.add_column("Technique", style="cyan")
    t2.add_column("Count", justify="right", style="green")
    for tech, cnt in sorted(technique_counts.items(), key=lambda x: -x[1]):
        t2.add_row(tech, str(cnt))
    console.print(t2)

    console.print(f"\nControl questions (real, non-nonsense): [yellow]{control_count}[/yellow]")

    # Sample questions
    console.print("\n[bold]Sample questions (first 3):[/bold]")
    for q in questions[:3]:
        console.print(Panel(
            f"[bold]ID:[/bold] {q['id']}\n"
            f"[bold]Domain:[/bold] {q['domain_group']} / {q['domain']}\n"
            f"[bold]Technique:[/bold] {q['technique']}\n"
            f"[bold]Control:[/bold] {q['is_control']}\n\n"
            f"[italic]{q['question'][:300]}[/italic]",
            border_style="dim",
        ))


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "-q", "--questions",
    type=int,
    default=10,
    show_default=True,
    help="Number of questions to process. 0 = all.",
)
@click.option(
    "-t", "--turns",
    type=int,
    default=None,
    help="Conversation turns per question (overrides config).",
)
@click.option(
    "--dataset",
    "dataset_key",
    default=None,
    help="Dataset key to use (overrides config).",
)
@click.option(
    "--domain",
    default=None,
    help="Filter by domain group (e.g. legal, medical, finance, software, physics).",
)
@click.option(
    "--technique",
    default=None,
    help="Filter by technique type.",
)
@click.option(
    "--model",
    default=None,
    help="OpenAI model to use (overrides config).",
)
@click.option(
    "--include-controls",
    is_flag=True,
    default=False,
    help="Include control (non-nonsense) questions.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without making API calls.",
)
@click.pass_context
def run(ctx, questions, turns, dataset_key, domain, technique, model, include_controls, dry_run):
    """
    Run benchmark data collection.

    Processes QUESTIONS questions with TURNS conversation turns each,
    capturing logprobs at every token. Already-processed questions
    (same question_id + model) are skipped automatically.

    \b
    Examples:
      python cli.py run --questions 10 --turns 1
      python cli.py run --questions 20 --turns 2 --domain legal
      python cli.py run --questions 0               # all questions
      python cli.py run --questions 5 --dry-run
    """
    from src.dataset import fetch_dataset, extract_questions, filter_questions
    from src.runner import run_question
    from src.storage import ResultStorage

    cfg = ctx.obj["config"]
    model = model or cfg["model"]
    turns = turns if turns is not None else cfg["turns"]
    dataset_key = dataset_key or cfg["default_dataset"]
    output_path = cfg["output"]

    if not os.environ.get("OPENAI_API_KEY") and not dry_run:
        console.print("[red]Error:[/red] OPENAI_API_KEY not set. Add it to .env or set the environment variable.")
        sys.exit(1)

    # Load dataset
    with console.status("Loading dataset..."):
        raw = fetch_dataset(dataset_key)
    all_questions = extract_questions(raw, dataset_key)

    # Filter
    filtered = filter_questions(
        all_questions,
        domain_group=domain,
        technique=technique,
        exclude_controls=not include_controls,
    )

    # Skip already processed
    storage = ResultStorage(output_path)
    processed_ids = storage.get_processed_ids(model=model)
    pending = [q for q in filtered if q["id"] not in processed_ids]

    if questions > 0:
        pending = pending[:questions]

    console.print(
        f"\n[bold]Run configuration[/bold]\n"
        f"  Model:    [cyan]{model}[/cyan]\n"
        f"  Dataset:  [cyan]{dataset_key}[/cyan]\n"
        f"  Turns:    [cyan]{turns}[/cyan]\n"
        f"  Output:   [cyan]{output_path}[/cyan]\n"
        f"  Domain:   [cyan]{domain or 'all'}[/cyan]\n"
        f"  Pending:  [cyan]{len(pending)}[/cyan] questions "
        f"([dim]{len(processed_ids)} already done[/dim])\n"
    )

    if not pending:
        console.print("[yellow]Nothing to process. All matching questions are already in the dataset.[/yellow]")
        return

    if dry_run:
        console.print("[bold yellow]Dry run — no API calls will be made.[/bold yellow]")
        t = Table(title="Would Process", box=box.SIMPLE)
        t.add_column("ID", style="cyan")
        t.add_column("Domain", style="green")
        t.add_column("Technique")
        for q in pending:
            t.add_row(q["id"], q["domain_group"], q["technique"])
        console.print(t)
        return

    # Process
    errors = []
    for i, q in enumerate(pending, 1):
        label = f"[{i}/{len(pending)}] {q['id']} ({q['domain_group']})"
        with console.status(f"Processing {label}..."):
            try:
                turn_results = run_question(
                    question_data=q,
                    model=model,
                    turns=turns,
                    temperature=cfg["temperature"],
                    top_logprobs=cfg["top_logprobs"],
                )
                storage.add_result(
                    question_id=q["id"],
                    model=model,
                    question_meta=q,
                    turns=turn_results,
                )
                total_tokens = sum(t.get("total_tokens") or 0 for t in turn_results)
                console.print(
                    f"[green]✓[/green] {label} "
                    f"[dim]({total_tokens} tokens, {len(turn_results)} turn(s))[/dim]"
                )
            except Exception as e:
                console.print(f"[red]✗[/red] {label} — {e}")
                errors.append({"id": q["id"], "error": str(e)})

    # Summary
    stats = storage.get_stats(model=model)
    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"Processed {len(pending) - len(errors)} questions "
        f"({len(errors)} errors). "
        f"Total in dataset: {stats['total_entries']}."
    )
    if errors:
        console.print("[yellow]Failed:[/yellow]", [e["id"] for e in errors])


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", default=None, help="Filter stats by model.")
@click.pass_context
def status(ctx, model):
    """Show collection progress and dataset statistics."""
    from src.storage import ResultStorage

    cfg = ctx.obj["config"]
    storage = ResultStorage(cfg["output"])
    stats = storage.get_stats(model=model)

    console.print(Panel.fit(
        f"[bold]Output file:[/bold] {stats['output_file']}\n"
        f"[bold]Total entries:[/bold] {stats['total_entries']}\n"
        f"[bold]Unique questions:[/bold] {stats['unique_questions']}",
        title="Dataset Status",
        border_style="green",
    ))

    if stats["by_model"]:
        tm = Table(title="Entries by Model", box=box.SIMPLE_HEAVY)
        tm.add_column("Model", style="cyan")
        tm.add_column("Count", justify="right", style="green")
        for m, c in sorted(stats["by_model"].items()):
            tm.add_row(m, str(c))
        console.print(tm)

    if stats["by_domain"]:
        t = Table(title="Entries by Domain", box=box.SIMPLE_HEAVY)
        t.add_column("Domain Group", style="cyan")
        t.add_column("Count", justify="right", style="green")
        for d, c in sorted(stats["by_domain"].items()):
            t.add_row(d, str(c))
        console.print(t)

    if stats["by_technique"]:
        t2 = Table(title="Entries by Technique", box=box.SIMPLE_HEAVY)
        t2.add_column("Technique", style="cyan")
        t2.add_column("Count", justify="right", style="green")
        for tech, cnt in sorted(stats["by_technique"].items(), key=lambda x: -x[1]):
            t2.add_row(tech, str(cnt))
        console.print(t2)


# ---------------------------------------------------------------------------
# reset command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--question-id", default=None, help="Reset a specific question by ID.")
@click.option("--model", default=None, help="Limit reset to a specific model.")
@click.option("--all", "reset_all", is_flag=True, help="Reset the entire dataset (irreversible).")
@click.pass_context
def reset(ctx, question_id, model, reset_all):
    """
    Reset processed questions so they can be re-collected.

    \b
    Examples:
      python cli.py reset --question-id leg_pnf_01
      python cli.py reset --all
    """
    from src.storage import ResultStorage

    cfg = ctx.obj["config"]
    storage = ResultStorage(cfg["output"])

    if reset_all:
        click.confirm(
            "This will delete ALL collected data. Are you sure?",
            abort=True,
        )
        storage.reset_all()
        console.print("[green]Dataset reset.[/green]")
    elif question_id:
        removed = storage.reset_question(question_id, model=model)
        console.print(f"[green]Removed {removed} entr(ies) for question '{question_id}'.[/green]")
    else:
        console.print("[yellow]Specify --question-id ID or --all.[/yellow]")


# ---------------------------------------------------------------------------
# label command group — Tarea 2: Etiquetado de Alucinaciones
# ---------------------------------------------------------------------------

@cli.group()
def label():
    """
    \b
    Hallucination labeling pipeline (Tarea 2).
    ============================================
    Uses an LLM judge to score each agent response:
      0.0  No hallucination — agent correctly rejected the false premise.
      0.5  Partial / ambiguous — neither confirms nor denies.
      1.0  Full hallucination — agent invented reasoning for the false concept.

    \b
    Subcommands:
      label run     — Score unlabeled turns via the LLM judge.
      label export  — Export labeled dataset to CSV / Excel.
      label status  — Show labeling progress.
    """
    pass


@label.command("run")
@click.option(
    "--labeler-model",
    default="gpt-4o",
    show_default=True,
    help="OpenAI model used as the LLM judge.",
)
@click.option(
    "--input", "input_path",
    default=None,
    help="Path to results JSON (overrides config output path).",
)
@click.option(
    "--labels-output",
    default="data/labels.json",
    show_default=True,
    help="Path to store the incremental label file.",
)
@click.option(
    "--export/--no-export",
    "do_export",
    default=True,
    show_default=True,
    help="Export to CSV/Excel after labeling completes.",
)
@click.option(
    "--export-path",
    default="data/labeled_dataset",
    show_default=True,
    help="Base path for exported files (extensions added automatically).",
)
@click.option(
    "--format", "export_fmt",
    type=click.Choice(["csv", "xlsx", "both"]),
    default="both",
    show_default=True,
    help="Export format.",
)
@click.option(
    "--limit",
    type=int,
    default=0,
    help="Max entries to label in this run (0 = all unlabeled).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be labeled without making API calls.",
)
@click.pass_context
def label_run(ctx, labeler_model, input_path, labels_output, do_export,
              export_path, export_fmt, limit, dry_run):
    """
    Score agent responses with an LLM judge.

    \b
    Already-labeled turns are skipped automatically (incremental).
    Examples:
      python cli.py label run
      python cli.py label run --labeler-model gpt-4o-mini --limit 20
      python cli.py label run --no-export
    """
    import os
    from src.labeler import (
        compute_logprob_stats, label_turn, export_labels, LabelStorage
    )
    from src.storage import ResultStorage
    from openai import OpenAI

    cfg = ctx.obj["config"]
    input_path = input_path or cfg["output"]

    if not os.environ.get("OPENAI_API_KEY") and not dry_run:
        console.print("[red]Error:[/red] OPENAI_API_KEY not set.")
        sys.exit(1)

    # Load raw results
    raw_storage = ResultStorage(input_path)
    all_entries = raw_storage._load()["processed"]

    if not all_entries:
        console.print("[yellow]No entries found in results dataset. Run `python cli.py run` first.[/yellow]")
        return

    label_storage = LabelStorage(labels_output)

    # Build list of (entry_key, turn_dict) that need labeling
    pending = []
    for entry_key, entry in all_entries.items():
        for turn in entry.get("turns", []):
            turn_num = turn["turn"]
            if not label_storage.is_labeled(entry["question_id"], entry["model"], turn_num):
                pending.append((entry_key, entry, turn))

    if limit > 0:
        pending = pending[:limit]

    console.print(
        f"\n[bold]Label run configuration[/bold]\n"
        f"  Labeler model:  [cyan]{labeler_model}[/cyan]\n"
        f"  Input dataset:  [cyan]{input_path}[/cyan]\n"
        f"  Labels file:    [cyan]{labels_output}[/cyan]\n"
        f"  Pending turns:  [cyan]{len(pending)}[/cyan]\n"
        f"  Export after:   [cyan]{do_export}[/cyan]\n"
    )

    if not pending:
        console.print("[yellow]All turns are already labeled. Use `label export` to regenerate files.[/yellow]")
        if do_export:
            _run_export(label_storage, export_path, export_fmt)
        return

    if dry_run:
        console.print("[bold yellow]Dry run — no API calls will be made.[/bold yellow]")
        t = Table(title="Would Label", box=box.SIMPLE)
        t.add_column("Entry key", style="cyan")
        t.add_column("Turn", justify="right")
        t.add_column("Domain", style="green")
        for entry_key, entry, turn in pending[:20]:
            t.add_row(entry_key, str(turn["turn"]), entry.get("domain_group", ""))
        if len(pending) > 20:
            t.add_row(f"... and {len(pending) - 20} more", "", "")
        console.print(t)
        return

    client = OpenAI()
    errors = []

    for i, (entry_key, entry, turn) in enumerate(pending, 1):
        q_id = entry["question_id"]
        turn_num = turn["turn"]
        label_str = f"[{i}/{len(pending)}] {q_id} turn={turn_num}"

        with console.status(f"Labeling {label_str}..."):
            try:
                # Score the response
                lbl = label_turn(
                    client=client,
                    question=entry["question"],
                    nonsensical_element=entry.get("nonsensical_element", ""),
                    response_text=turn["response_text"],
                    turn=turn_num,
                    labeler_model=labeler_model,
                )
                # Compute logprob stats
                stats = compute_logprob_stats(turn.get("logprobs", []))

                record = {
                    "question_id": q_id,
                    "model": entry["model"],
                    "turn": turn_num,
                    "domain_group": entry.get("domain_group", ""),
                    "domain": entry.get("domain", ""),
                    "technique": entry.get("technique", ""),
                    "difficulty_label": entry.get("difficulty_label", ""),
                    "is_control": entry.get("is_control", False),
                    "question": entry.get("question", ""),
                    "nonsensical_element": entry.get("nonsensical_element", ""),
                    "response_text": turn["response_text"],
                    "logprob_n_tokens": stats["n_tokens"],
                    "logprob_mean": stats["mean"],
                    "logprob_std": stats["std"],
                    "logprob_min": stats["min"],
                    "logprob_max": stats["max"],
                    "logprob_p10": stats["p10"],
                    "logprob_p90": stats["p90"],
                    "logprob_mean_top_entropy": stats["mean_top_entropy"],
                    "hallucination_score": lbl["score"],
                    "label_justification": lbl["justification"],
                    "labeler_model": labeler_model,
                    "labeled_at": datetime.utcnow().isoformat() + "Z",
                }
                label_storage.add_label(q_id, entry["model"], turn_num, record)

                score_color = {0.0: "green", 0.5: "yellow", 1.0: "red"}.get(
                    lbl["score"], "white"
                )
                console.print(
                    f"[{score_color}]OK[/{score_color}] {label_str} "
                    f"score=[bold {score_color}]{lbl['score']}[/bold {score_color}]  "
                    f"[dim]{lbl['justification'][:80]}[/dim]"
                )

            except Exception as e:
                console.print(f"[red]ERR[/red] {label_str} - {e}")
                errors.append({"key": entry_key, "turn": turn_num, "error": str(e)})

    # Summary
    stats_summary = label_storage.get_stats()
    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"Labeled {len(pending) - len(errors)} turns "
        f"({len(errors)} errors). "
        f"Total labeled: {stats_summary['total_labeled']}."
    )
    if errors:
        console.print("[yellow]Errors:[/yellow]", [e["key"] for e in errors])

    if do_export:
        _run_export(label_storage, export_path, export_fmt)


@label.command("export")
@click.option(
    "--labels-input",
    default="data/labels.json",
    show_default=True,
    help="Path to the labels JSON file.",
)
@click.option(
    "--output",
    default="data/labeled_dataset",
    show_default=True,
    help="Base path for export (extensions added automatically).",
)
@click.option(
    "--format", "export_fmt",
    type=click.Choice(["csv", "xlsx", "both"]),
    default="both",
    show_default=True,
    help="Export format.",
)
def label_export(labels_input, output, export_fmt):
    """
    Export the labeled dataset to CSV and/or Excel.

    \b
    Examples:
      python cli.py label export
      python cli.py label export --format csv
      python cli.py label export --output data/my_dataset
    """
    from src.labeler import export_labels, LabelStorage

    storage = LabelStorage(labels_input)
    _run_export(storage, output, export_fmt)


@label.command("status")
@click.option(
    "--input", "input_path",
    default=None,
    help="Results JSON path (overrides config).",
)
@click.option(
    "--labels-input",
    default="data/labels.json",
    show_default=True,
    help="Labels JSON file.",
)
@click.pass_context
def label_status(ctx, input_path, labels_input):
    """Show labeling progress."""
    from src.labeler import LabelStorage
    from src.storage import ResultStorage

    cfg = ctx.obj["config"]
    input_path = input_path or cfg["output"]

    raw_storage = ResultStorage(input_path)
    all_entries = raw_storage._load()["processed"]
    total_turns = sum(len(e.get("turns", [])) for e in all_entries.values())

    label_storage = LabelStorage(labels_input)
    stats = label_storage.get_stats()

    pending = total_turns - stats["total_labeled"]

    console.print(Panel.fit(
        f"[bold]Results entries:[/bold] {len(all_entries)}\n"
        f"[bold]Total turns:[/bold] {total_turns}\n"
        f"[bold]Labeled turns:[/bold] {stats['total_labeled']}\n"
        f"[bold]Pending:[/bold] {pending}",
        title="Labeling Status",
        border_style="blue",
    ))

    if stats["by_score"]:
        t = Table(title="Labels by Score", box=box.SIMPLE_HEAVY)
        t.add_column("Score", style="cyan")
        t.add_column("Meaning", style="dim")
        t.add_column("Count", justify="right", style="green")
        meanings = {0.0: "No hallucination", 0.5: "Partial / ambiguous", 1.0: "Full hallucination"}
        for score in sorted(stats["by_score"]):
            t.add_row(str(score), meanings.get(score, ""), str(stats["by_score"][score]))
        console.print(t)


# ---------------------------------------------------------------------------
# train command group — Tarea 3: Pipeline Predictivo
# ---------------------------------------------------------------------------

@cli.group()
def train():
    """
    \b
    ML training pipeline (Tarea 3).
    =================================
    Builds and evaluates hallucination classifiers from the labeled dataset.
    Models: LightGBM, XGBoost, Logistic Regression.

    \b
    Subcommands:
      train run        — Full pipeline: features, train, evaluate, benchmark.
      train report     — Display the last generated report.
      train benchmark  — Re-run latency benchmark on saved models.
    """
    pass


@train.command("run")
@click.option(
    "--csv", "csv_path",
    default="data/labeled_dataset.csv",
    show_default=True,
    help="Labeled dataset CSV (output of Tarea 2).",
)
@click.option(
    "--results-json", "results_json",
    default="data/results.json",
    show_default=True,
    help="Raw results JSON for extended logprob features.",
)
@click.option(
    "--models-dir",
    default="models",
    show_default=True,
    help="Directory to save trained model .pkl files.",
)
@click.option(
    "--output-dir",
    default="data",
    show_default=True,
    help="Directory for reports, plots, and feature CSV.",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    show_default=True,
    help="K for stratified K-fold cross-validation.",
)
@click.option(
    "--benchmark-runs",
    type=int,
    default=200,
    show_default=True,
    help="Number of single-sample passes for latency measurement.",
)
@click.option(
    "--features",
    type=click.Choice(["base", "extended", "all"]),
    default="all",
    show_default=True,
    help="Feature set to use: base (CSV stats only), extended (raw seq only), all.",
)
def train_run(csv_path, results_json, models_dir, output_dir, cv_folds,
              benchmark_runs, features):
    """
    Run the full ML pipeline.

    \b
    Steps:
      1. Build feature matrix (base + extended logprob features).
      2. Train LightGBM, XGBoost, Logistic Regression.
      3. K-Fold cross-validation + metrics.
      4. Latency benchmark (single-sample inference).
      5. Save models to --models-dir.
      6. Export report to --output-dir.

    \b
    Examples:
      python cli.py train run
      python cli.py train run --cv-folds 10 --features all
      python cli.py train run --csv data/labeled_dataset.csv
    """
    import os
    from src.features import build_feature_matrix, ALL_FEATURE_COLS, BASE_FEATURE_COLS, EXTENDED_FEATURE_COLS
    from src.ml_pipeline import run_pipeline, generate_report

    if not os.path.exists(csv_path):
        console.print(f"[red]CSV not found:[/red] {csv_path}. Run `label run` first.")
        sys.exit(1)

    feature_set = {
        "base": BASE_FEATURE_COLS,
        "extended": EXTENDED_FEATURE_COLS,
        "all": ALL_FEATURE_COLS,
    }[features]

    # --- Build feature matrix ---
    with console.status("Building feature matrix..."):
        df = build_feature_matrix(csv_path, results_json)

    features_csv = f"{output_dir}/features.csv"
    df.to_csv(features_csv, index=False)
    console.print(f"Features saved: {features_csv}  ({df.shape[0]} rows x {len(feature_set)} features)")

    # --- Show feature stats ---
    t = Table(title="Feature Matrix Preview", box=box.SIMPLE_HEAVY)
    t.add_column("Feature", style="cyan")
    t.add_column("Mean", justify="right")
    t.add_column("Std", justify="right")
    t.add_column("Min", justify="right")
    t.add_column("Max", justify="right")
    for col in feature_set:
        if col in df.columns:
            s = df[col].dropna()
            t.add_row(col,
                      f"{s.mean():.4f}" if len(s) else "N/A",
                      f"{s.std():.4f}" if len(s) > 1 else "N/A",
                      f"{s.min():.4f}" if len(s) else "N/A",
                      f"{s.max():.4f}" if len(s) else "N/A")
    console.print(t)

    # --- Run pipeline ---
    console.print(f"\n[bold]Training models...[/bold] (CV folds={cv_folds}, features={features})\n")
    with console.status("Running ML pipeline..."):
        result = run_pipeline(
            feature_df=df,
            feature_cols=feature_set,
            target_col="hallucination_score",
            models_dir=models_dir,
            output_dir=output_dir,
            cv_folds=cv_folds,
            benchmark_runs=benchmark_runs,
        )

    summary = result["summary"]

    if summary.get("warning"):
        console.print(f"\n[yellow][WARNING][/yellow] {summary['warning']}\n")

    # --- Metrics table ---
    m_table = Table(title="Classification Metrics", box=box.SIMPLE_HEAVY)
    m_table.add_column("Model", style="cyan")
    m_table.add_column("Accuracy", justify="right")
    m_table.add_column("Precision", justify="right")
    m_table.add_column("Recall", justify="right")
    m_table.add_column("F1", justify="right")
    m_table.add_column("AUC", justify="right")

    for name, res in result["models"].items():
        def _fmt(k):
            v = res.get(k)
            return f"{v:.4f}" if v is not None else "N/A"
        m_table.add_row(
            name, _fmt("accuracy_mean"), _fmt("precision_mean"),
            _fmt("recall_mean"), _fmt("f1_mean"), _fmt("auc_mean"),
        )
    console.print(m_table)

    # --- Latency table ---
    l_table = Table(title="Inference Latency (ms)", box=box.SIMPLE_HEAVY)
    l_table.add_column("Model", style="cyan")
    l_table.add_column("Mean", justify="right")
    l_table.add_column("p50", justify="right")
    l_table.add_column("p95", justify="right")
    l_table.add_column("p99", justify="right")

    for name, res in result["models"].items():
        lat = res.get("latency", {})
        l_table.add_row(
            name,
            f"{lat.get('mean_ms', 0):.4f}",
            f"{lat.get('p50_ms', 0):.4f}",
            f"{lat.get('p95_ms', 0):.4f}",
            f"{lat.get('p99_ms', 0):.4f}",
        )
    console.print(l_table)

    # --- Feature importances ---
    if result.get("feature_importances"):
        for name, imp in result["feature_importances"].items():
            fi_table = Table(title=f"Feature Importance — {name}", box=box.SIMPLE)
            fi_table.add_column("Feature", style="cyan")
            fi_table.add_column("Importance", justify="right", style="green")
            for feat, score in list(imp.items())[:10]:
                fi_table.add_row(feat, f"{score:.4f}")
            console.print(fi_table)

    # --- Generate report ---
    report_path = generate_report(result, f"{output_dir}/ml_report")
    console.print(f"\n[green]Report saved:[/green] {report_path}")
    console.print(f"[green]Metrics CSV:[/green] {report_path.replace('.txt', '.csv')}")

    for path in result.get("plots", []):
        console.print(f"[green]Plot saved:[/green] {path}")

    for name, res in result["models"].items():
        console.print(f"[green]Model saved:[/green] {res['model_path']}")


@train.command("report")
@click.option(
    "--output-dir",
    default="data",
    show_default=True,
    help="Directory where ml_report.txt was saved.",
)
def train_report(output_dir):
    """Display the last generated ML report."""
    report_path = Path(output_dir) / "ml_report.txt"
    if not report_path.exists():
        console.print(f"[yellow]Report not found:[/yellow] {report_path}. Run `train run` first.")
        return
    with open(report_path, encoding="utf-8") as f:
        console.print(f.read())


@train.command("benchmark")
@click.option(
    "--models-dir",
    default="models",
    show_default=True,
    help="Directory with saved .pkl model files.",
)
@click.option(
    "--csv", "csv_path",
    default="data/labeled_dataset.csv",
    show_default=True,
    help="Labeled dataset CSV.",
)
@click.option(
    "--results-json", "results_json",
    default="data/results.json",
    show_default=True,
)
@click.option(
    "--runs",
    type=int,
    default=500,
    show_default=True,
    help="Number of single-sample passes per model.",
)
def train_benchmark(models_dir, csv_path, results_json, runs):
    """
    Re-run inference latency benchmark on saved models.

    \b
    Example:
      python cli.py train benchmark --runs 1000
    """
    import pickle
    from src.features import build_feature_matrix, ALL_FEATURE_COLS
    from src.ml_pipeline import benchmark_latency
    import numpy as np
    from sklearn.impute import SimpleImputer

    models_path = Path(models_dir)
    imputer_file = models_path / "imputer.pkl"

    if not imputer_file.exists():
        console.print("[red]No saved models found.[/red] Run `train run` first.")
        sys.exit(1)

    with open(imputer_file, "rb") as f:
        imp_data = pickle.load(f)
    imputer = imp_data["imputer"]
    feature_cols = imp_data["feature_cols"]

    df = build_feature_matrix(csv_path, results_json)
    X = imputer.transform(df[feature_cols]).astype(np.float32)

    pkl_files = sorted(models_path.glob("*.pkl"))
    pkl_files = [p for p in pkl_files if p.name != "imputer.pkl"]

    if not pkl_files:
        console.print("[yellow]No model .pkl files found.[/yellow]")
        return

    t = Table(title=f"Latency Benchmark ({runs} runs per model)", box=box.SIMPLE_HEAVY)
    t.add_column("Model", style="cyan")
    t.add_column("Mean ms", justify="right")
    t.add_column("Std ms", justify="right")
    t.add_column("p50 ms", justify="right")
    t.add_column("p95 ms", justify="right")
    t.add_column("p99 ms", justify="right")

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        with console.status(f"Benchmarking {pkl_path.stem}..."):
            lat = benchmark_latency(model, X, n_runs=runs)
        t.add_row(
            pkl_path.stem,
            f"{lat['mean_ms']:.4f}",
            f"{lat['std_ms']:.4f}",
            f"{lat['p50_ms']:.4f}",
            f"{lat['p95_ms']:.4f}",
            f"{lat['p99_ms']:.4f}",
        )
    console.print(t)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run_export(label_storage, export_path, export_fmt):
    from src.labeler import export_labels
    labels = label_storage.get_all_labels()
    if not labels:
        console.print("[yellow]No labels to export.[/yellow]")
        return
    with console.status("Exporting dataset..."):
        written = export_labels(labels, export_path, fmt=export_fmt)
    for path in written:
        console.print(f"[green]Exported:[/green] {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
