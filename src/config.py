from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataConfig:
    # Directories
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # HRdata2 location/pattern
    hrdata2_subdir: str = "HRdata2"
    hrdata2_pattern: str = "*.csv"

    # Column names in HRdata2
    index_col: str = "Unnamed: 0"
    round_col: str = "Round"
    phase_col: str = "Phase"
    individual_col: str = "Individual"
    puzzler_col: str = "Puzzler"
    original_id_col: str = "original_ID"
    raw_path_col: str = "raw_data_path"
    team_id_col: str = "Team_ID"
    cohort_col: str = "Cohort"

    # Emotion questionnaire / targets
    frustration_col: str = "Frustrated"
    panas_cols: List[str] = field(
        default_factory=lambda: [
            "upset",
            "hostile",
            "alert",
            "ashamed",
            "inspired",
            "nervous",
            "attentive",
            "afraid",
            "active",
            "determined",
        ]
    )

    def metadata_columns(self) -> List[str]:
        """Columns that describe subject/phase/questionnaires, not raw features."""
        return [
            self.round_col,
            self.phase_col,
            self.individual_col,
            self.puzzler_col,
            self.original_id_col,
            self.raw_path_col,
            self.team_id_col,
            self.cohort_col,
            self.frustration_col,
            *self.panas_cols,
        ]

    def is_metadata(self, col: str) -> bool:
        return col in self.metadata_columns() or col == self.index_col


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig) # changed initialization to use default_factory

    def to_dict(self) -> Dict:
        return {
            "data": {
                "raw_dir": str(self.data.raw_dir),
                "interim_dir": str(self.data.interim_dir),
                "processed_dir": str(self.data.processed_dir),
            }
        }


DEFAULT_CONFIG = Config()
