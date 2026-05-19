import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    runpy.run_path(str(ROOT / "scripts" / "10_final_report_outputs.py"), run_name="__main__")
    print("PASS recovery diagnostics output generated in results/main_pipeline_final/tables/")


if __name__ == "__main__":
    main()
